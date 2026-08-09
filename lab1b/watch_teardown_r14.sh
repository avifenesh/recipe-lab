#!/bin/bash
# Round-14 standing watcher: pulls artifacts and tears down each box when it
# flags READY_FOR_TEARDOWN. Cancel spot request BEFORE terminate.
# IPs are the post-spot-resume ones (both boxes moved when capacity returned).
KEY=~/.ssh/fpv-drone-audio-train-20260701.pem
DEST=/home/avifenesh/projects/recipe-lab/lab1b/results_1b
mkdir -p "$DEST"
declare -A BOXES=(
  # boxE (i-0d957666ce29242dc) torn down 2026-08-09 02:52 after L_std_ext8
  # finished and both 8-epoch eval suites validated.
  ["18.219.6.126"]="i-00b627d9917dcfc57"
)
while [ ${#BOXES[@]} -gt 0 ]; do
  sleep 300
  for ip in "${!BOXES[@]}"; do
    iid=${BOXES[$ip]}
    ready=$(ssh -o ConnectTimeout=15 -o BatchMode=yes -i $KEY ubuntu@$ip \
            'ls ~/READY_FOR_TEARDOWN 2>/dev/null' 2>/dev/null)
    [ -z "$ready" ] && continue
    echo "$(date -u +%H:%M) $ip ready — pulling + teardown $iid"
    rsync -az -e "ssh -o ConnectTimeout=20 -i $KEY" \
      "ubuntu@$ip:~/recipe-lab/lab1b/results/*.json" "$DEST/" 2>/dev/null
    rsync -az -e "ssh -o ConnectTimeout=20 -i $KEY" \
      "ubuntu@$ip:~/recipe-lab/lab1b/evals_*.json" "$DEST/" 2>/dev/null
    rsync -az -e "ssh -o ConnectTimeout=20 -i $KEY" \
      "ubuntu@$ip:~/recipe-lab/lab1b/ckpts/*_best.pt" "$DEST/" 2>/dev/null
    # only tear down if the artifacts actually landed locally
    got=$(ls "$DEST"/evals_*.json 2>/dev/null | wc -l)
    if [ "$got" -lt 1 ]; then
      echo "WARN $ip flagged READY but no evals pulled — leaving box up"
      continue
    fi
    SIR=$(AWS_PROFILE=default aws ec2 describe-instances --region us-east-2 \
      --instance-ids $iid --query 'Reservations[].Instances[].SpotInstanceRequestId' --output text)
    [ -n "$SIR" ] && [ "$SIR" != "None" ] && \
      AWS_PROFILE=default aws ec2 cancel-spot-instance-requests --region us-east-2 \
        --spot-instance-request-ids $SIR >/dev/null
    AWS_PROFILE=default aws ec2 terminate-instances --region us-east-2 \
      --instance-ids $iid >/dev/null
    echo "TORN_DOWN $iid"
    unset "BOXES[$ip]"
  done
done
echo R14_ALL_DOWN
