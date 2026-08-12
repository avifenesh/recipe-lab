#!/bin/bash
# Round-19 standing watcher: pull artifacts and tear down the p5 when it
# flags READY_FOR_TEARDOWN. Cancel spot request BEFORE terminate.
KEY=~/.ssh/fpv-drone-audio-train-20260701.pem
DEST=/home/avifenesh/projects/recipe-lab/lab1b/results_1b
# IP moved 3.143.244.208 -> 18.223.125.221 after a brief spot blip (box
# came back `pending` within minutes, same instance id, ckpt intact).
IP=18.223.125.221
IID=i-07f0606ab0caf1b40
mkdir -p "$DEST"
while true; do
  sleep 300
  ready=$(ssh -o ConnectTimeout=15 -o BatchMode=yes -i $KEY ubuntu@$IP \
          'ls ~/READY_FOR_TEARDOWN 2>/dev/null' 2>/dev/null)
  [ -z "$ready" ] && continue
  echo "$(date -u +%H:%M) $IP ready — pulling + teardown $IID"
  rsync -az -e "ssh -o ConnectTimeout=20 -i $KEY" \
    "ubuntu@$IP:~/recipe-lab/lab1b/results/r19_*.json" "$DEST/" 2>/dev/null
  rsync -az -e "ssh -o ConnectTimeout=20 -i $KEY" \
    "ubuntu@$IP:~/recipe-lab/lab1b/evals_r19_*.json" "$DEST/" 2>/dev/null
  rsync -az -e "ssh -o ConnectTimeout=20 -i $KEY" \
    "ubuntu@$IP:~/recipe-lab/lab1b/ckpts/r19_V_pm_best.pt" "$DEST/" 2>/dev/null
  rsync -az -e "ssh -o ConnectTimeout=20 -i $KEY" \
    "ubuntu@$IP:~/recipe-lab/lab1b/ckpts/r19_V_wl_best.pt" "$DEST/" 2>/dev/null
  # tear down only when the exact artifacts landed locally (no globs a
  # sibling can satisfy — the V_std lesson)
  ok=1
  for f in evals_r19_V_pm.json evals_r19_V_wl.json r19_V_pm.json \
           r19_V_wl.json r19_V_pm_best.pt r19_V_wl_best.pt; do
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
echo R19_DOWN
