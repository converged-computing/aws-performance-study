# MLServer with Cached Features on EKS

This experiment will compare two cases in an effort to maximize FOM.

- running with a model to select instance type to maximize fom or units of fom per dollar
- running choosing at random.

For this experiment, we would want to show that using the model improves the overall FOM at a rate that is significant. First, create the cluster that can autoscale. It has one set of "sticky nodes" to install things to.

## 1. Create Cluster

```bash
eksctl create cluster --config-file ./eks-config.yaml 
aws eks update-kubeconfig --region us-east-1 --name nfd-cluster
```

Note that we are using an artifact that specifies random is OK. This is how the selection artifact was generated. It has 2 models, and we will use one for figure of merit (fom) and then one that specifies random selection.

```bash
oras push ghcr.io/compspec/ocifit-k8s-compatibility:ml-example-with-random ./compatibility-artifact.json:application/vnd.oci.image.model-compatibilities.v1+json
```

## 2. Enable Autoscaling

Install the autoscaler:

```bash
kubectl apply -f eks-autoscaler.yaml
```

Instance node selectors will be added that trigger an autoscaler. 

## 3. Setup Cluster

The ocifit-k8s images should already be built and deployed to a public registry. You'll need the certificate manager.

```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.17.2/cert-manager.yaml

# Create the configmap with cached node features. This will allow us to select instances that don't exist yet.
kubectl create configmap node-features --from-file=./cached-features/node-features.json

# Install the flux operator and node feature discovery.
kubectl apply -f https://raw.githubusercontent.com/flux-framework/flux-operator/refs/heads/main/examples/dist/flux-operator.yaml
kubectl apply -k https://github.com/kubernetes-sigs/node-feature-discovery/deployment/overlays/default?ref=v0.17.3

# And install the deployment manifest (assuming you are sitting in the cloned repository)
kubectl apply -f ./webhook-mlserver-with-cache.yaml
```

Make sure everything is running:

```bash
$ kubectl get pods
NAME                                     READY   STATUS    RESTARTS   AGE
ocifit-k8s-deployment-86c7fcbbfb-nfs52   2/2     Running   0          39s
```

## 4. Experiments

### Experiment to Test Random Runs

This will be orchestrated by helm. 

```bash
git clone https://github.com/converged-computing/flux-apps-helm
cd flux-apps-helm
mkdir -p ./logs/hpcg/random
helm dependency update hpcg-matrix/
for i in $(seq 1 30); do
  outdir=./logs/hpcg/random/$i
  mkdir -p $outdir
  if [[ -f "$outdir/hpcg.out" ]]; then
    echo "$i was already run for random"
    continue
  fi
  helm install \
  --set experiment.nodes=1 \
  --set minicluster.size=1 \
  --set minicluster.save_logs=true \
  --set minicluster.image=placeholder:latest \
  --set experiment.iterations=3 \
  --set experiment.exclusive=true \
  --set "label.oci\.image\.compatibilities\.selection/enabled"=true \
  --set "annotation.oci\.image\.compatibilities\.selection/model"=random \
  --set "annotation.oci\.image\.compatibilities\.selection/image-ref"=ghcr.io/compspec/ocifit-k8s-compatibility:ml-example-with-random \
  hpcg ./hpcg-matrix
  sleep 3
  time kubectl wait --for=condition=ready pod -l job-name=hpcg --timeout=600s
  kubectl get pods > $outdir/pod-time.txt
  pod=$(kubectl get pods -o json | jq  -r .items[0].metadata.name)
  kubectl logs ${pod} -c hpcg -f |& tee $outdir/hpcg.out
  kubectl get pod ${pod} -o json > $outdir/pod.json
  helm uninstall hpcg
  sleep 3
done
```

### Experiment to FOM Selection

```bash
mkdir -p ./logs/hpcg/fom-model
for i in $(seq 1 30); do
  outdir=./logs/hpcg/fom-model/$i
  mkdir -p $outdir
  if [[ -f "$outdir/hpcg.out" ]]; then
    echo "$i was already run for fom-model"
    continue
  fi
  helm install \
  --set experiment.nodes=1 \
  --set minicluster.size=1 \
  --set experiment.tasks='$(nproc)' \
  --set minicluster.save_logs=true \
  --set minicluster.image=placeholder:latest \
  --set experiment.iterations=3 \
  --set experiment.exclusive=true \
  --set "label.oci\.image\.compatibilities\.selection/enabled"=true \
  --set "annotation.oci\.image\.compatibilities\.selection/model"=fom \
  --set "annotation.oci\.image\.compatibilities\.selection/image-ref"=ghcr.io/compspec/ocifit-k8s-compatibility:ml-example-with-random \
  hpcg ./hpcg-matrix
  sleep 3
  time kubectl wait --for=condition=ready pod -l job-name=hpcg --timeout=600s
  kubectl get pods > $outdir/pod-time.txt
  pod=$(kubectl get pods -o json | jq  -r .items[0].metadata.name)
  kubectl logs ${pod} -c hpcg -f |& tee $outdir/hpcg.out
  kubectl get pod ${pod} -o json > $outdir/pod.json
  helm uninstall hpcg
  sleep 3
done
```

### Experiment to FOM Per Dollar Selection

```bash
mkdir -p ./logs/hpcg/fom-per-dollar
for i in $(seq 1 30); do
  outdir=./logs/hpcg/fom-per-dollar/$i
  mkdir -p $outdir
  if [[ -f "$outdir/hpcg.out" ]]; then
    echo "$i was already run for fom-per-dollar"
    continue
  fi
  helm install \
  --set experiment.nodes=1 \
  --set minicluster.size=1 \
  --set experiment.tasks='$(nproc)' \
  --set minicluster.save_logs=true \
  --set minicluster.image=placeholder:latest \
  --set experiment.iterations=3 \
  --set experiment.exclusive=true \
  --set "label.oci\.image\.compatibilities\.selection/enabled"=true \
  --set "annotation.oci\.image\.compatibilities\.selection/model"=fom_per_dollar \
  --set "annotation.oci\.image\.compatibilities\.selection/image-ref"=ghcr.io/compspec/ocifit-k8s-compatibility:ml-example-with-random \
  hpcg ./hpcg-matrix
  sleep 3
  time kubectl wait --for=condition=ready pod -l job-name=hpcg --timeout=600s
  kubectl get pods > $outdir/pod-time.txt
  pod=$(kubectl get pods -o json | jq  -r .items[0].metadata.name)
  kubectl logs ${pod} -c hpcg -f |& tee $outdir/hpcg.out
  kubectl get pod ${pod} -o json > $outdir/pod.json
  helm uninstall hpcg
  sleep 3
done
```


That's it! We can compare the data (FOM values) to see if our model did better. This is a simple example, but could be expanded to be more advanced.

```bash
eksctl delete cluster --config-file ./eks-config.yaml  --wait
```
