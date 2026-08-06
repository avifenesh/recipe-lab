#!/bin/bash
# Provision the p5.48xlarge spot instance for the 1B factorial.
# Usage: AWS_PROFILE=default bash launch_p5.sh
# Requires: fpv SG + keypair reused; data1b synced from the fpv box after prep.
set -euo pipefail

REGION=us-east-2
AMI=ami-032b37e4db407994d           # DLAMI PyTorch 2.7, Ubuntu 22.04
ITYPE=p5.48xlarge
KEY=fpv-drone-audio-train-20260701  # same keypair as the fpv box
SG=sg-0f3217e681280a905             # same SG (port 22 allowlisted)
AZ=us-east-2c                       # cheapest spot at launch time

aws ec2 run-instances --region $REGION \
  --image-id $AMI --instance-type $ITYPE --key-name $KEY \
  --security-group-ids $SG --placement AvailabilityZone=$AZ \
  --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=persistent,InstanceInterruptionBehavior=stop}' \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=400,VolumeType=gp3,Throughput=500,DeleteOnTermination=false}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=recipe-lab-1b}]' \
  --query 'Instances[0].InstanceId' --output text
# persistent spot + stop-on-interrupt + non-deleting EBS: interruption pauses,
# capacity return resumes, train1b.py picks up from its checkpoint.
