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
#     ~/.sfci_instance.txt for the deploy step next.
#
# Re-runnable. If the instance already exists (by display name), it just prints its details.
# Requires: oci CLI configured at ~/.oci/config; SSH public key at $SSH_KEY.

set -euo pipefail

# ---- knobs ----------------------------------------------------------------------------
NAME="sfci"
SHAPE="VM.Standard.A1.Flex"
OCPUS=1
MEM_GB=6
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519.pub}"
RETRY_SECS=30                                # wait between capacity-retry rounds
MAX_ROUNDS="${MAX_ROUNDS:-720}"              # ~6 hours @ 30s
COMPARTMENT="$(grep '^tenancy' ~/.oci/config | cut -d= -f2)"
REGION="$(grep '^region' ~/.oci/config | cut -d= -f2)"

[[ -f "$SSH_KEY" ]] || { echo "[!] SSH key not found at $SSH_KEY"; exit 1; }

# ---- helpers --------------------------------------------------------------------------
log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
ocid() { oci --query "$2" --raw-output "$@" 2>/dev/null || true; }

find_or_create() {                           # find_or_create <kind-display> <list-cmd...> -- <create-cmd...>
  local kind="$1"; shift
  local list=(); while [[ "$1" != "--" ]]; do list+=("$1"); shift; done; shift
  local create=("$@")
  local id; id="$(oci ${list[@]} --query 'data[0].id' --raw-output 2>/dev/null || true)"
  if [[ -n "$id" && "$id" != "null" ]]; then
    log "$kind already exists: ${id:0:50}..."
  else
    log "creating $kind..."
    id="$(oci ${create[@]} --query 'data.id' --raw-output)"
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
  echo "$EXISTING_INSTANCE $PUB_IP" > ~/.sfci_instance.txt
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
        --hostname-label "$NAME" \
        --metadata "{\"ssh_authorized_keys\":\"$SSH_KEY_CONTENT\"}" \
        --wait-for-state RUNNING \
        >"$OUT_FILE" 2>&1; then
      INSTANCE_ID="$(jq -r '.data.id' "$OUT_FILE")"
      log "[OK] launched in $AD: $INSTANCE_ID"
      sleep 5    # give the VNIC a moment to bind
      PUB_IP="$(oci compute instance list-vnics --instance-id "$INSTANCE_ID" \
                --query 'data[0]."public-ip"' --raw-output)"
      log "public IP: $PUB_IP"
      echo "$INSTANCE_ID $PUB_IP" > ~/.sfci_instance.txt
      echo
      echo "======================================================================"
      echo "  Instance up.  ssh opc@$PUB_IP"
      echo "  (saved to ~/.sfci_instance.txt for the deploy step)"
      echo "======================================================================"
      rm -f "$OUT_FILE"
      exit 0
    fi
    # Transient (capacity / rate-limit / network timeout)? -> next AD, sleep, retry.
    # Real error (auth, bad params, missing perm)? -> bail with the message so the user can fix.
    if grep -qiE 'Out of (host )?capacity|InternalError.*capacity|TooManyRequests|RequestException|connection.*timed? out|EndpointConnectionError|Read timed out|ServiceTimeoutException' "$OUT_FILE"; then
      reason="capacity"; grep -qiE 'timed? out|RequestException|ConnectionError' "$OUT_FILE" && reason="network timeout"
      log "   (transient: $reason in $AD -- trying next)"
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
