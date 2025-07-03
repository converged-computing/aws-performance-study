# Node Feature Discovery Data

Here we want to extract NFD labels across different instance types. We can then design different containers to run lammps on the nodes (or some other app) and determine if we can create compatibility manifests for CPU. Since we want to keep costs down, we are going to stick to small instance types and simple ways to assess. Create the cluster.

```bash
eksctl create cluster --config-file ./eks-config.yaml
aws eks update-kubeconfig --region us-east-1 --name nfd-cluster
```

Install node feature discovery:

```bash
kubectl apply -k https://github.com/kubernetes-sigs/node-feature-discovery/deployment/overlays/default?ref=v0.17.3
```

Edit the configmap to be:

```
kind: ConfigMap
apiVersion: v1
data:
  nfd-worker.conf: |
    core:
      labelWhiteList:
        ".*": "true"
```
```
kubectl edit configmap -n node-feature-discovery nfd-worker-conf-c2mbm9t788 
```

Recreate pods:

```
kubectl delete pods -n node-feature-discovery -l app=nfd-worker
```

Note that the update above only provides us with ~127 features, so not enough or the full set.

```bash
kubectl get nodes -o json | jq '.items[].metadata.labels' > node-features.json
kubectl get nodes -o json > nodes.json
```

## Node Raw Features

```bash
# For each of these files, do the sequence below
kubectl apply -f ./deploy/nfd-installer.yaml
kubectl delete -f ./deploy/nfd-installer.yaml
kubectl apply -f ./deploy/nfd-installer-arm.yaml
```
```
for pod in $(kubectl get pods -o json | jq -r .items[].metadata.name)
  do
  mkdir -p ./features/${pod}/
  kubectl cp ${pod}:/opt/shared/ ./features/${pod}/
  kubectl get pod -o json $pod > ./features/${pod}/pod.json
  node=$(kubectl get pods -o=custom-columns=PODNAME:.metadata.name,NODENAME:.spec.nodeName $pod | awk 'NR > 1 {print $2}')
  kubectl get node $node -o json > ./features/$pod/node.json
done
```

## Clean Up

When you are done:

```bash
eksctl delete cluster --config-file ./eks-config.yaml --wait
```
