import os
import json


def read_json(filename):
    with open(filename, "r") as fd:
        content = json.loads(fd.read())
    return content


def write_json(obj, filename):
    with open(filename, "w") as fd:
        fd.write(json.dumps(obj, indent=4))


def flatten_features(node_features):
    flat = {}
    for section, items in node_features["flags"].items():
        for key, value in items["elements"].items():
            flat[f"{section}.{key}"] = True
    for section, items in node_features["attributes"].items():
        for value in items["elements"]:
            if isinstance(value, dict):
                name = value["attributes"]["name"]
                for k, v in value["attributes"].items():
                    if k == "name":
                        continue
                    flat[f"{section}.{name}.{k}"] = v
            else:
                flat[f"{section}.{key}"] = value

    for section, items in node_features["instances"].items():
        for value in items["elements"]:
            if isinstance(value, dict):
                if "name" not in value["attributes"]:
                    name = None
                else:
                    name = value["attributes"]["name"]
                for k, v in value["attributes"].items():
                    if k == "name":
                        continue
                    if name is not None:
                        flat[f"{section}.{name}.{k}"] = v
                    else:
                        flat[f"{section}.{k}"] = v
            else:
                flat[f"{section}.{key}"] = value
    return flat


outdir = "./results"
if not os.path.exists(outdir):
    os.makedirs(outdir)

for pod in os.listdir("./features"):
    if pod.startswith("_"):
        continue
    path = os.path.join("./features", pod)
    pod_meta = read_json(os.path.join(path, "pod.json"))
    node_meta = read_json(os.path.join(path, "node.json"))
    node_name = node_meta["metadata"]["labels"]["beta.kubernetes.io/instance-type"]
    print(path)
    features_file = [x for x in os.listdir(path) if "features" in x][0]
    node_features = read_json(os.path.join(path, features_file))
    flat = flatten_features(node_features)
    node_outdir = os.path.join(outdir, node_name)
    if not os.path.exists(node_outdir):
        os.makedirs(node_outdir)
    write_json(flat, os.path.join(node_outdir, "features.json"))
