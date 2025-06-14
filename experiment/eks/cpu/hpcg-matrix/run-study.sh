#!/bin/bash

instance=${1}
tasks=${2}

mkdir -p ./logs/hpcg/$instance
helm dependency update hpcg-matrix/
for tag in $(cat ./tags.txt); do
  outdir=./logs/hpcg/$instance/$tag
  mkdir -p $outdir
  if [[ -f "$outdir/hpcg.out" ]]; then
    echo "$tag was already run"
    continue
  fi
  helm install \
  --set experiment.nodes=1 \
  --set minicluster.size=1 \
  --set minicluster.tasks=$tasks \
  --set experiment.tasks=$tasks \
  --set minicluster.save_logs=true \
  --set monitor.multiple="cpu|futex" \
  --set minicluster.image=ghcr.io/converged-computing/hpcg-matrix:$tag \
  --set monitor.installer="ghcr.io/converged-computing/kernel-header-installer:fedora43" \
  --set experiment.iterations=3 \
  hpcg ./hpcg-matrix
  sleep 3
  time kubectl wait --for=condition=ready pod -l job-name=hpcg --timeout=600s
  pod=$(kubectl get pods -o json | jq  -r .items[0].metadata.name)
  kubectl logs ${pod} -c hpcg -f |& tee $outdir/hpcg.out
  # Let's get the eBPF even if it fails, could be interesting.
  for prog in cpu futex
    do
      kubectl logs ${pod} -c bcc-monitor-$prog -f |& tee $outdir/${prog}.out    
  done
  helm uninstall hpcg
done
