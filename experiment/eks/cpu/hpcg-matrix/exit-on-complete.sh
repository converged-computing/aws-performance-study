#!/bin/bash
instance=${1}
count=${2:-126}
while true
  do
  count=$(ls logs/hpcg/$instance/ | wc -l)
  if [[ $count -eq $count ]]; 
    then
    echo "Reached completion state"
    sleep 180
    eksctl delete cluster --config-file ../aws-performance-study/experiment/eks/cpu/hpcg-matrix/cfg/eks-config-$instance.yaml --wait
    break
  fi
  echo "Current count is $count, sleeping"
  sleep 30
done
