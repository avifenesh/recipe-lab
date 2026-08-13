#!/bin/bash
# Round-20 standing watcher: pull artifacts and tear down the p5 when it
# flags READY_FOR_TEARDOWN. Cancel spot request BEFORE terminate.
KEY=~/.ssh/fpv-drone-audio-train-20260701.pem
DEST=/home/avifenesh/projects/recipe-lab/lab1b/results_1b
# IP moved 18.188.240.110 -> 16.59.123.191 after a spot marked-for-stop
# reclaim (capacity returned in minutes; ckpt at step 10000 intact).
IP=16.59.123.191
IID=i-016a0cbeda671f516
mkdir -p "$DEST"
while true; do
  sleep 300
  ready=$(ssh -o ConnectTimeout=15 -o BatchMode=yes -i $KEY ubuntu@$IP \
          'ls ~/READY_FOR_TEARDOWN 2>/dev/null' 2>/dev/null)
  [ -z "$ready" ] && continue
  echo "$(date -u +%H:%M) $IP ready — pulling + teardown $IID"
  rsync -az -e "ssh -o ConnectTimeout=20 -i $KEY" \
    "ubuntu@$IP:~/recipe-lab/lab1b/results/r20_*.json" "$DEST/" 2>/dev/null
  rsync -az -e "ssh -o ConnectTimeout=20 -i $KEY" \
    "ubuntu@$IP:~/recipe-lab/lab1b/evals_r20_*.json" "$DEST/" 2>/dev/null
  for c in fm_s7 wl_s13; do
    rsync -az -e "ssh -o ConnectTimeout=20 -i $KEY" \
      "ubuntu@$IP:~/recipe-lab/lab1b/ckpts/r20_V_${c}_best.pt" "$DEST/" 2>/dev/null
  done
  # exact filenames only — never a glob a sibling can satisfy (V_std lesson)
  ok=1
  for f in evals_r20_V_fm_s7.json evals_r20_V_wl_s13.json \
           r20_V_fm_s7.json r20_V_wl_s13.json \
           r20_V_fm_s7_best.pt r20_V_wl_s13_best.pt; do
    [ -s "$DEST/$f" ] || { echo "WARN missing $f — leaving box up"; ok=0; }
  done
  [ "$ok" -eq 1 ] || continue
  SIR=$(AWS_PROFILE=default aws ec2 describe-instances --region us-east-2 \
    --instance-ids $IID --query 'Reservations[].Instances[].SpotInstanceRequestId' --output text)
  [ -n "$SIR" ] && [ "$SIR" != "None" ] && \
    AWS_PROFILE=default aws ec2 cancel-spot-instance-requests --region us-east-2 \
      --spot-instance-request-ids $SIR >/dev/null
  AWS_PROFILE=default aws ec2 terminate-instances --region us-east-2 \
    --instance-ids $IID >/dev/null
  echo "TORN_DOWN $IID"
  break
done
echo R20_DOWN
