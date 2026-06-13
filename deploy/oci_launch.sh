#!/usr/bin/env bash
# Idempotent provision-and-launch for the sfci box on Oracle Always Free ARM.
#
# What it does, in order:
#  1. Creates a VCN ("sfci-vcn", 10.0.0.0/16) + internet gateway + a default route via the IGW
#     + a security list that allows 22/80/443 inbound, if any of these don't already exist.
#  2. Creates a public subnet ("sfci-subnet", 10.0.0.0/24) in that VCN, if missing.
#  3. Looks up the latest Oracle Linux 9 aarch64 image automatically.
#  4. Loops attempting to launch VM.Standard.A1.Flex (1 OCPU / 6 GB) across all ADs in the
#     region every $RETRY_SECS until ONE succeeds -- that's the A1 capacity squeeze workaround.
#  5. On success: prints the public IP, instance OCID, ssh command. Saves them to
#     $INSTANCE_FILE for the deploy step next.
#
# Re-runnable. If the instance already exists (by display name), it just prints its details.
# Requires: oci CLI configured at ~/.oci/config; SSH public key at $SSH_KEY.

set -euo pipefail

# ---- knobs ----------------------------------------------------------------------------
# All overridable so a second poller (e.g., parallel SJC region) writes to a distinct file
# and doesn't collide with this region's resources/result:
#   OCI_CLI_REGION=us-sanjose-1 NAME=sfci-sjc INSTANCE_FILE=~/.sfci_instance_sjc.txt ./oci_launch.sh
NAME="${NAME:-sfci}"
SHAPE="VM.Standard.A1.Flex"
OCPUS=1
MEM_GB=6
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519.pub}"
INSTANCE_FILE="${INSTANCE_FILE:-$HOME/.sfci_instance.txt}"
RETRY_SECS="${RETRY_SECS:-300}"              # wait between capacity-retry rounds (5 min default
                                             # since the recent Oracle pattern is "accept then
                                             # immediately terminate" — slow churn avoids tripping
                                             # OCI anti-abuse + lets per-round attempts breathe).
MAX_ROUNDS="${MAX_ROUNDS:-288}"              # ~24 h at 5 min/round
COMPARTMENT="$(grep '^tenancy' ~/.oci/config | cut -d= -f2)"

[[ -f "$SSH_KEY" ]] || { echo "[!] SSH key not found at $SSH_KEY"; exit 1; }

# ---- helpers --------------------------------------------------------------------------
log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

parse_instance_ocid() {                      # parse_instance_ocid <out-file>
  # The OCI CLI prepends prose ("Action completed. Waiting until..." / "Failed to wait...")
  # before the JSON, and stderr is merged into the capture — so jq chokes on the raw file.
  # Strip everything before the first standalone `{` line; fallback: greedy grep for
  # "ocid1.instance.*". Never fails under set -e (both substitutions end in `|| true`).
  local id
  id="$(sed -n '/^{/,$p' "$1" | jq -r '.data.id // empty' 2>/dev/null || true)"
  if [[ -z "$id" ]]; then
    id="$(grep -oE '"ocid1\.instance\.oc1\.[a-z0-9._-]+"' "$1" | head -1 | tr -d '"' || true)"
  fi
  printf '%s' "$id"
}

find_or_create() {                           # find_or_create <kind-display> <list-cmd...> -- <create-cmd...>
  local kind="$1"; shift
  local list=(); while [[ "$1" != "--" ]]; do list+=("$1"); shift; done; shift
  local create=("$@")
  local id; id="$(oci "${list[@]}" --query 'data[0].id' --raw-output 2>/dev/null || true)"
  if [[ -n "$id" && "$id" != "null" ]]; then
    log "$kind already exists: ${id:0:50}..."
  else
    log "creating $kind..."
    id="$(oci "${create[@]}" --query 'data.id' --raw-output)"
  fi
  printf '%s' "$id"
}

# ---- 0) idempotency: if an instance named $NAME exists & running, just print + exit ---
EXISTING_INSTANCE="$(oci compute instance list \
  --compartment-id "$COMPARTMENT" --display-name "$NAME" --lifecycle-state RUNNING \
  --query 'data[0].id' --raw-output 2>/dev/null || true)"
if [[ -n "$EXISTING_INSTANCE" && "$EXISTING_INSTANCE" != "null" ]]; then
  log "instance \"$NAME\" already RUNNING: $EXISTING_INSTANCE"
  PUB_IP="$(oci compute instance list-vnics --instance-id "$EXISTING_INSTANCE" \
            --query 'data[0]."public-ip"' --raw-output 2>/dev/null)"
  log "public IP: $PUB_IP"
  echo "$EXISTING_INSTANCE $PUB_IP" > "$INSTANCE_FILE"
  echo "ssh ubuntu@$PUB_IP   (or opc@ for Oracle Linux)"
  exit 0
fi

# ---- 1) VCN ----------------------------------------------------------------------------
VCN_ID="$(find_or_create "VCN" \
  network vcn list --compartment-id "$COMPARTMENT" --display-name "$NAME-vcn" \
  -- \
  network vcn create --compartment-id "$COMPARTMENT" --display-name "$NAME-vcn" --cidr-block 10.0.0.0/16 --wait-for-state AVAILABLE)"
[[ -n "$VCN_ID" && "$VCN_ID" != "null" ]] || { log "VCN id empty after create"; exit 1; }

# ---- 2) Internet gateway --------------------------------------------------------------
IGW_ID="$(find_or_create "internet gateway" \
  network internet-gateway list --compartment-id "$COMPARTMENT" --vcn-id "$VCN_ID" --display-name "$NAME-igw" \
  -- \
  network internet-gateway create --compartment-id "$COMPARTMENT" --vcn-id "$VCN_ID" --display-name "$NAME-igw" --is-enabled true --wait-for-state AVAILABLE)"

# ---- 3) Default route table -> route 0.0.0.0/0 via IGW --------------------------------
DRT_ID="$(oci network vcn get --vcn-id "$VCN_ID" --query 'data."default-route-table-id"' --raw-output)"
HAS_DEFAULT_ROUTE="$(oci network route-table get --rt-id "$DRT_ID" \
  --query "data.\"route-rules\"[?\"network-entity-id\"==\`$IGW_ID\`]|length(@)" --raw-output 2>/dev/null || echo 0)"
if [[ "$HAS_DEFAULT_ROUTE" == "0" ]]; then
  log "adding default 0.0.0.0/0 route via IGW to default route table..."
  oci network route-table update --rt-id "$DRT_ID" --force \
    --route-rules "[{\"destination\":\"0.0.0.0/0\",\"destinationType\":\"CIDR_BLOCK\",\"networkEntityId\":\"$IGW_ID\"}]" >/dev/null
fi

# ---- 4) Default security list: allow inbound 22, 80, 443 ------------------------------
SL_ID="$(oci network vcn get --vcn-id "$VCN_ID" --query 'data."default-security-list-id"' --raw-output)"
INGRESS_RULES='[
  {"source":"0.0.0.0/0","protocol":"6","isStateless":false,"tcpOptions":{"destinationPortRange":{"min":22,"max":22}}},
  {"source":"0.0.0.0/0","protocol":"6","isStateless":false,"tcpOptions":{"destinationPortRange":{"min":80,"max":80}}},
  {"source":"0.0.0.0/0","protocol":"6","isStateless":false,"tcpOptions":{"destinationPortRange":{"min":443,"max":443}}}
]'
EGRESS_RULES='[{"destination":"0.0.0.0/0","protocol":"all","isStateless":false}]'
log "ensuring default security list allows 22/80/443 inbound..."
oci network security-list update --security-list-id "$SL_ID" --force \
  --ingress-security-rules "$INGRESS_RULES" --egress-security-rules "$EGRESS_RULES" >/dev/null

# ---- 5) Public subnet -----------------------------------------------------------------
SUB_ID="$(find_or_create "public subnet" \
  network subnet list --compartment-id "$COMPARTMENT" --vcn-id "$VCN_ID" --display-name "$NAME-subnet" \
  -- \
  network subnet create --compartment-id "$COMPARTMENT" --vcn-id "$VCN_ID" --display-name "$NAME-subnet" --cidr-block 10.0.0.0/24 --prohibit-public-ip-on-vnic false --wait-for-state AVAILABLE)"

# ---- 6) Latest Oracle Linux 9 ARM image -----------------------------------------------
IMAGE_ID="$(oci compute image list --compartment-id "$COMPARTMENT" \
  --operating-system 'Oracle Linux' --operating-system-version '9' --shape "$SHAPE" \
  --sort-by TIMECREATED --sort-order DESC --limit 1 \
  --query 'data[0].id' --raw-output)"
log "image: $IMAGE_ID"

# ---- 7) ADs in this region -------------------------------------------------------------
# bash 3.2 (macOS default) lacks `mapfile`; word-split the cleaned newline list into the array
ADS=( $(oci iam availability-domain list --query 'data[*].name' --raw-output 2>/dev/null | tr -d ' ",[]' | grep -v '^$') )
log "availability domains: ${ADS[*]}"

# ---- 8) Capacity-retry launch loop -----------------------------------------------------
SHAPE_CFG="{\"ocpus\":$OCPUS,\"memoryInGBs\":$MEM_GB}"
SSH_KEY_CONTENT="$(cat "$SSH_KEY")"

for round in $(seq 1 "$MAX_ROUNDS"); do
  for AD in "${ADS[@]}"; do
    log "round $round/$MAX_ROUNDS  trying $AD..."
    OUT_FILE="$(mktemp)"
    if oci compute instance launch \
        --compartment-id "$COMPARTMENT" \
        --availability-domain "$AD" \
        --shape "$SHAPE" \
        --shape-config "$SHAPE_CFG" \
        --image-id "$IMAGE_ID" \
        --subnet-id "$SUB_ID" \
        --assign-public-ip true \
        --display-name "$NAME" \
        --metadata "$(jq -n --arg k "$SSH_KEY_CONTENT" '{ssh_authorized_keys:$k}')" \
        --wait-for-state RUNNING \
        --max-wait-seconds 300 \
        >"$OUT_FILE" 2>&1; then
      # Same prose-prefixed output as the recovery path (stderr is merged into OUT_FILE),
      # so use the shared parser — raw `jq` here used to die under set -e on success.
      INSTANCE_ID="$(parse_instance_ocid "$OUT_FILE")"
      if [[ -z "$INSTANCE_ID" ]]; then
        log "[!] launch reported success but OCID could not be parsed from output:"
        cat "$OUT_FILE" >&2
        rm -f "$OUT_FILE"
        exit 1
      fi
      log "[OK] launched in $AD: $INSTANCE_ID"
      sleep 5    # give the VNIC a moment to bind
      # list-vnics can transiently 404 right after launch — non-fatal, like the recovery path.
      PUB_IP="$(oci compute instance list-vnics --instance-id "$INSTANCE_ID" \
                --query 'data[0]."public-ip"' --raw-output 2>/dev/null || true)"
      PUB_IP="${PUB_IP:-unknown}"
      log "public IP: $PUB_IP"
      echo "$INSTANCE_ID $PUB_IP" > "$INSTANCE_FILE"
      echo
      echo "======================================================================"
      echo "  Instance up.  ssh opc@$PUB_IP"
      echo "  (saved to $INSTANCE_FILE for the deploy step)"
      echo "======================================================================"
      rm -f "$OUT_FILE"
      exit 0
    fi
    # Transient (capacity / rate-limit / network timeout)? -> next AD, sleep, retry.
    # Wait-for-state timeout? Launch was ACCEPTED -- recover by polling state ourselves.
    # Real error (auth, bad params, missing perm)? -> bail with the message so the user can fix.
    if grep -qiE 'Out of (host )?capacity|InternalError.*capacity|TooManyRequests|RequestException|connection.*timed? out|EndpointConnectionError|Read timed out|ServiceTimeoutException' "$OUT_FILE"; then
      reason="capacity"; grep -qiE 'timed? out|RequestException|ConnectionError' "$OUT_FILE" && reason="network timeout"
      log "   (transient: $reason in $AD -- trying next)"
    elif grep -qE 'Failed to wait until the resource entered the specified state' "$OUT_FILE"; then
      # Launch was accepted; the CLI's --wait-for-state RUNNING just gave up. Pull the OCID,
      # poll lifecycle-state ourselves, then either claim it (RUNNING) or release it (TERMINATED
      # or stuck-in-PROVISIONING). DISABLE set -e inside this block: oci CLI calls can return
      # transient non-zero (404 on race, throttle), and we already handle each via `|| true`.
      set +e
      RECOVER_ID="$(parse_instance_ocid "$OUT_FILE")"
      log "   (wait-state timeout in $AD; OCID parsed: [${RECOVER_ID:-EMPTY}])"
      if [[ -n "$RECOVER_ID" ]]; then
        log "   polling lifecycle-state up to 20 min (40 x 30s)..."
        recovered=0
        for poll in $(seq 1 40); do
          STATE="$(oci compute instance get --instance-id "$RECOVER_ID" --query 'data."lifecycle-state"' --raw-output 2>/dev/null)"
          STATE="${STATE:-UNKNOWN}"
          log "      poll $poll/40: state=$STATE"
          if [[ "$STATE" == "RUNNING" ]]; then
            sleep 5
            PUB_IP="$(oci compute instance list-vnics --instance-id "$RECOVER_ID" --query 'data[0]."public-ip"' --raw-output 2>/dev/null)"
            PUB_IP="${PUB_IP:-unknown}"
            log "[OK] recovered $RECOVER_ID in $AD -- public IP: $PUB_IP"
            echo "$RECOVER_ID $PUB_IP" > "$INSTANCE_FILE"
            echo
            echo "======================================================================"
            echo "  Instance up.  ssh opc@$PUB_IP"
            echo "  (saved to $INSTANCE_FILE for the deploy step)"
            echo "======================================================================"
            rm -f "$OUT_FILE"
            exit 0
          fi
          if [[ "$STATE" == "TERMINATED" || "$STATE" == "TERMINATING" ]]; then
            log "   (Oracle dropped the instance; trying next AD)"
            recovered=1
            break
          fi
          sleep 30
        done
        # Clean up a stuck instance so it doesn't pin compute quota.
        if [[ "$recovered" -eq 0 ]]; then
          FINAL_STATE="$(oci compute instance get --instance-id "$RECOVER_ID" --query 'data."lifecycle-state"' --raw-output 2>/dev/null)"
          FINAL_STATE="${FINAL_STATE:-UNKNOWN}"
          if [[ "$FINAL_STATE" != "TERMINATED" && "$FINAL_STATE" != "TERMINATING" ]]; then
            log "   (terminating stuck instance $RECOVER_ID in state $FINAL_STATE)"
            oci compute instance terminate --instance-id "$RECOVER_ID" --force --preserve-boot-volume false >/dev/null 2>&1
          fi
        fi
      fi
      set -e
    else
      echo "[!] launch failed with non-transient error:" >&2
      cat "$OUT_FILE" >&2
      rm -f "$OUT_FILE"
      exit 1
    fi
    rm -f "$OUT_FILE"
  done
  log "no capacity in any AD this round -- sleeping ${RETRY_SECS}s before round $((round+1))..."
  sleep "$RETRY_SECS"
done

log "gave up after $MAX_ROUNDS rounds (~$((MAX_ROUNDS * RETRY_SECS / 60)) min). Re-run to keep trying."
exit 1
