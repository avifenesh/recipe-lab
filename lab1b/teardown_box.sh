#!/bin/bash
# Pull results off a finished 1B box, then cancel its spot request and
# terminate. Usage: bash teardown_box.sh <ip> <instance-id> <cell>
# Order matters: cancel the persistent spot request BEFORE terminating,
# otherwise the request re-launches a fresh instance.
set -euo pipefail
IP=$1; IID=$2; CELL=$3
KEY=~/.ssh/fpv-drone-audio-train-20260701.pem
DEST=/home/avifenesh/projects/recipe-lab/lab1b/results_1b
mkdir -p "$DEST"

rsync -az -e "ssh -o ConnectTimeout=20 -i $KEY" \
  ubuntu@$IP:~/recipe-lab/lab1b/results/${CELL}.json \
  ubuntu@$IP:~/recipe-lab/lab1b/evals_${CELL}.json \
  ubuntu@$IP:~/recipe-lab/lab1b/ckpts/${CELL}_best.pt \
  "$DEST/" 2>/dev/null || rsync -az -e "ssh -o ConnectTimeout=20 -i $KEY" \
  ubuntu@$IP:~/recipe-lab/lab1b/results/${CELL}.json "$DEST/"

SIR=$(AWS_PROFILE=default aws ec2 describe-instances --region us-east-2 \
  --instance-ids $IID --query 'Reservations[].Instances[].SpotInstanceRequestId' --output text)
[ -n "$SIR" ] && [ "$SIR" != "None" ] && \
  AWS_PROFILE=default aws ec2 cancel-spot-instance-requests --region us-east-2 \
    --spot-instance-request-ids $SIR --query 'CancelledSpotInstanceRequests[0].State' --output text
AWS_PROFILE=default aws ec2 terminate-instances --region us-east-2 \
  --instance-ids $IID --query 'TerminatingInstances[0].CurrentState.Name' --output text
echo "TORN_DOWN $IID ($CELL) — results in $DEST"
