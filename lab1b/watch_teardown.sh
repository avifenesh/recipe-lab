#!/bin/bash
# Standing watcher: polls all live 1B boxes; when a box flags
# READY_FOR_TEARDOWN, pulls its results and tears it down.
# Exits when all listed boxes are gone.
KEY=~/.ssh/fpv-drone-audio-train-20260701.pem
declare -A BOXES=(
  ["3.142.237.74"]="i-0b13c096b159cd84e V_cur"
  ["3.145.30.229"]="i-055c377a5846b74b2 L_std"
  ["18.189.14.169"]="i-051aa8e6aadcc6fdb L_cur"
)
while [ ${#BOXES[@]} -gt 0 ]; do
  sleep 300
  for ip in "${!BOXES[@]}"; do
    set -- ${BOXES[$ip]}
    iid=$1; cell=$2
    ready=$(ssh -o ConnectTimeout=15 -o BatchMode=yes -i $KEY ubuntu@$ip \
            'ls ~/READY_FOR_TEARDOWN 2>/dev/null' 2>/dev/null)
    if [ -n "$ready" ]; then
      echo "$(date -u +%H:%M) $cell ready — tearing down $iid"
      bash "$(dirname "$0")/teardown_box.sh" $ip $iid $cell && unset "BOXES[$ip]"
    fi
  done
done
echo ALL_BOXES_DOWN
