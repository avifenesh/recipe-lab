#!/bin/bash
# Round-16p standing watcher: pull artifacts and tear down the p5 when it
# flags READY_FOR_TEARDOWN. Cancel spot request BEFORE terminate.
KEY=~/.ssh/fpv-drone-audio-train-20260701.pem
DEST=/home/avifenesh/projects/recipe-lab/lab1b/results_1b
# IP moved 18.219.42.215 -> 13.59.82.196 after the 09:32Z spot interruption
# (capacity-not-available ~3.5h); same instance id, root volume survived.
IP=13.59.82.196
IID=i-06af18789e8e85239
mkdir -p "$DEST"
while true; do
  sleep 300
  ready=$(ssh -o ConnectTimeout=15 -o BatchMode=yes -i $KEY ubuntu@$IP \
          'ls ~/READY_FOR_TEARDOWN 2>/dev/null' 2>/dev/null)
  [ -z "$ready" ] && continue
  echo "$(date -u +%H:%M) $IP ready — pulling + teardown $IID"
  rsync -az -e "ssh -o ConnectTimeout=20 -i $KEY" \
    "ubuntu@$IP:~/recipe-lab/lab1b/results/r16p_*.json" "$DEST/" 2>/dev/null
  rsync -az -e "ssh -o ConnectTimeout=20 -i $KEY" \
    "ubuntu@$IP:~/recipe-lab/lab1b/evals_r16p_*.json" "$DEST/" 2>/dev/null
  rsync -az -e "ssh -o ConnectTimeout=20 -i $KEY" \
    "ubuntu@$IP:~/recipe-lab/lab1b/ckpts/r16p_HV_cur_best.pt" "$DEST/" 2>/dev/null
  # tear down only when the exact artifacts landed locally (no globs a
  # sibling can satisfy — the V_std lesson)
  ok=1
  for f in evals_r16p_HV_cur.json r16p_HV_cur.json r16p_HV_cur_best.pt; do
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
echo R16P_DOWN
