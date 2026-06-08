#!/usr/bin/env bash
# Anti-reclamation CPU burst for Oracle Always Free A1.Flex (1 OCPU = 1 vCPU).
#
# Oracle reclaims Always Free instances whose 95th-percentile CPU stays below 20% over 7 days.
# We need to push the 95th-percentile sample ABOVE 20%, not the average. ONE worker at 22% load
# for 20 min/hour ~= ~7.3% averaged, but every 1-min Oracle sample during that 20-min window
# reads ~22% — easily clearing the 20% bar at the p95. Spreading 20 min instead of 10 min keeps
# the burst gentler so real /compute / /variance requests barely notice.
#
# Earlier `--cpu 2 --cpu-load 80` saturated the single vCPU for 10 min/hour, doubling API
# latency during the burst window. This version keeps that under 25% utilization.
#
# Nice 19 + ionice idle means real requests preempt the burst instantly.

set -euo pipefail
exec nice -n 19 ionice -c3 stress-ng --cpu 1 --cpu-load 22 --timeout 1200s --metrics-brief 2>&1 \
  | logger -t sfci-keepalive
