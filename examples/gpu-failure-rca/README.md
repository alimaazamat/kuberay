# Correlating GPU (Xid) faults with Ray "application" failures on AKS

This example shows how a GPU-level fault (NVIDIA **Xid** event) propagates up
the stack and surfaces to the user as a Ray *application-level* failure, and
how adding **AKS Node Problem Detector (NPD)** with Xid rules plus a small
stand-alone correlator script turns the same failure into a clear
hardware-level root cause.

> **Status:** today this is a prototype that runs **outside** of KubeRay:
> NPD publishes node events; a Python script (run on demand or as a sidecar)
> joins those events to a failed Ray job. There is no upstream KubeRay
> change required to use it. A possible upstream design that would build
> the same join into the operator itself is sketched in
> [FUTURE_WORK.md](FUTURE_WORK.md).

The local simulator [simulate_xid_to_ray.py](simulate_xid_to_ray.py) models
every layer (GPU driver, kernel log, NPD, kube API, Ray worker, correlator)
in-process so you can see the full chain without a GPU. This README walks
through reproducing the same demo end-to-end on a real AKS GPU cluster.

```text
GPU hardware fault
  -> NVIDIA driver writes Xid to kernel log (/dev/kmsg, journalctl -k)
  -> NPD kernel-monitor matches the Xid pattern
  -> NPD sets NodeCondition GPUHealthy=False + Warning Event
  -> Ray worker's CUDA call raises RuntimeError
  -> Ray surfaces RayTaskError to the user        <-- looks like an app bug
  -> Correlator joins Ray failure with NPD event  <-- reveals real cause
```

---

## 0. Prerequisites

- Azure CLI logged in (`az login`) with a subscription that has GPU quota.
- `kubectl`, `helm`, `jq` installed locally.
- Python 3.10+ (for the local simulator and correlator script).
- Quota for at least one `Standard_NC*` SKU (e.g. `Standard_NC6s_v3` or
  `Standard_NC24ads_A100_v4`).

> Tip: if you do not have GPU quota, you can still demo the full chain by
> running the local simulator (`python3 simulate_xid_to_ray.py`) and the
> "fault injection via /dev/kmsg" path described in step 6 on **any** node
> (no real GPU required).

---

## 1. Create the AKS cluster with a GPU node pool

```bash
RG=kuberay-xid-rca
LOC=eastus
CLUSTER=kuberay-xid
GPU_POOL=gpunp
GPU_SKU=Standard_NC6s_v3        # or Standard_NC24ads_A100_v4

az group create -n "$RG" -l "$LOC"

az aks create \
  -g "$RG" -n "$CLUSTER" \
  --node-count 1 \
  --node-vm-size Standard_D4s_v5 \
  --enable-addons monitoring \
  --generate-ssh-keys

az aks nodepool add \
  -g "$RG" --cluster-name "$CLUSTER" \
  -n "$GPU_POOL" \
  --node-count 1 \
  --node-vm-size "$GPU_SKU" \
  --node-taints sku=gpu:NoSchedule \
  --labels accelerator=nvidia

az aks get-credentials -g "$RG" -n "$CLUSTER" --overwrite-existing
```

Install the NVIDIA device plugin so the GPU is exposed to pods:

```bash
kubectl create ns gpu-resources
kubectl apply -n gpu-resources -f \
  https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.5/nvidia-device-plugin.yml
```

Verify the node reports GPUs:

```bash
kubectl get nodes -L accelerator
kubectl describe node -l accelerator=nvidia | grep nvidia.com/gpu
```

---

## 2. Install KubeRay

```bash
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update

helm install kuberay-operator kuberay/kuberay-operator \
  --version 1.4.0 \
  -n kuberay-system --create-namespace

kubectl wait --for=condition=Available \
  deploy/kuberay-operator -n kuberay-system --timeout=180s
```

---

## 3. Install Node Problem Detector with Xid rules

NPD is **not** installed on AKS by default. We deploy the upstream chart and
mount a custom `kernel-monitor` config that matches the same Xid codes the
local simulator handles (13, 31, 43, 63, 64, 74, 79).

Save the config:

```bash
cat > /tmp/xid-monitor.json <<'EOF'
{
  "plugin": "kmsg",
  "logPath": "/dev/kmsg",
  "lookback": "5m",
  "bufferSize": 10,
  "source": "kernel-monitor",
  "conditions": [
    {
      "type": "GPUHealthy",
      "reason": "GPUHealthy",
      "message": "GPU is healthy"
    }
  ],
  "rules": [
    {
      "type": "permanent",
      "condition": "GPUHealthy",
      "reason": "Xid79",
      "pattern": "NVRM: Xid \\(PCI:[^)]+\\): 79,.*"
    },
    {
      "type": "permanent",
      "condition": "GPUHealthy",
      "reason": "Xid74",
      "pattern": "NVRM: Xid \\(PCI:[^)]+\\): 74,.*"
    },
    {
      "type": "permanent",
      "condition": "GPUHealthy",
      "reason": "Xid64",
      "pattern": "NVRM: Xid \\(PCI:[^)]+\\): 64,.*"
    },
    {
      "type": "permanent",
      "condition": "GPUHealthy",
      "reason": "Xid63",
      "pattern": "NVRM: Xid \\(PCI:[^)]+\\): 63,.*"
    },
    {
      "type": "temporary",
      "reason": "Xid43",
      "pattern": "NVRM: Xid \\(PCI:[^)]+\\): 43,.*"
    },
    {
      "type": "temporary",
      "reason": "Xid31",
      "pattern": "NVRM: Xid \\(PCI:[^)]+\\): 31,.*"
    },
    {
      "type": "temporary",
      "reason": "Xid13",
      "pattern": "NVRM: Xid \\(PCI:[^)]+\\): 13,.*"
    }
  ]
}
EOF

kubectl create ns npd
kubectl create configmap node-problem-detector-config \
  -n npd \
  --from-file=kernel-monitor.json=/tmp/xid-monitor.json
```

Install the chart and point it at the configmap:

```bash
helm repo add deliveryhero https://charts.deliveryhero.io/
helm repo update

helm upgrade --install npd deliveryhero/node-problem-detector \
  -n npd \
  --set settings.custom_plugin_monitor=null \
  --set settings.system_log_monitor[0]=/config/kernel-monitor.json \
  --set-string tolerations[0].key=sku \
  --set-string tolerations[0].value=gpu \
  --set-string tolerations[0].effect=NoSchedule \
  --set extraVolumes[0].name=npd-config \
  --set extraVolumes[0].configMap.name=node-problem-detector-config \
  --set extraVolumeMounts[0].name=npd-config \
  --set extraVolumeMounts[0].mountPath=/config

kubectl -n npd rollout status ds/npd
```

Confirm NPD is running on the GPU node:

```bash
kubectl -n npd get pods -o wide
kubectl describe node -l accelerator=nvidia | grep -A2 GPUHealthy
```

You should see `GPUHealthy=True` (i.e. NPD has registered the condition,
default state is healthy).

---

## 4. Submit a Ray training job that uses CUDA

Save as `rayjob-cuda.yaml`:

```yaml
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: cuda-train
spec:
  entrypoint: python /home/ray/train.py
  runtimeEnvYAML: |
    pip:
      - torch==2.3.0
  rayClusterSpec:
    rayVersion: "2.34.0"
    headGroupSpec:
      rayStartParams: {}
      template:
        spec:
          containers:
            - name: ray-head
              image: rayproject/ray:2.34.0
    workerGroupSpecs:
      - groupName: gpu-workers
        replicas: 1
        minReplicas: 1
        maxReplicas: 1
        rayStartParams:
          num-gpus: "1"
        template:
          spec:
            tolerations:
              - key: sku
                value: gpu
                effect: NoSchedule
            nodeSelector:
              accelerator: nvidia
            containers:
              - name: ray-worker
                image: rayproject/ray-ml:2.34.0-gpu
                resources:
                  limits:
                    nvidia.com/gpu: 1
                volumeMounts:
                  - name: code
                    mountPath: /home/ray
            volumes:
              - name: code
                configMap:
                  name: cuda-train-code
```

A trivial training entrypoint that loops on the GPU forever:

```bash
cat > /tmp/train.py <<'EOF'
import time, torch
print("CUDA available:", torch.cuda.is_available())
x = torch.ones(1024, 1024, device="cuda")
i = 0
while True:
    x = x @ x.t() + 1e-6
    if i % 100 == 0:
        print("step", i, "norm", float(x.norm()))
    i += 1
    time.sleep(0.05)
EOF
kubectl create configmap cuda-train-code --from-file=train.py=/tmp/train.py
kubectl apply -f rayjob-cuda.yaml
```

Wait until the worker is running on the GPU node:

```bash
kubectl get pods -l ray.io/cluster=cuda-train -o wide
WORKER=$(kubectl get pod -l ray.io/node-type=worker,ray.io/cluster=cuda-train -o name | head -1)
NODE=$(kubectl get "$WORKER" -o jsonpath='{.spec.nodeName}')
echo "Ray worker on node: $NODE"
```

---

## 5. Scenario A — no Xid awareness

Temporarily stop NPD so the kernel-log signal cannot reach Kubernetes:

```bash
kubectl -n npd scale ds/npd --replicas=0   # (or `kubectl scale` per-node)
```

Inject the fault (step 6) and observe Ray. With NPD disabled, the user only
sees a CUDA-level `RayTaskError` and the node looks healthy:

```bash
kubectl logs -l ray.io/cluster=cuda-train -c ray-worker --tail=50
kubectl describe node "$NODE" | grep -A2 GPUHealthy   # condition stale or absent
```

Then re-enable NPD before scenario B:

```bash
kubectl -n npd rollout restart ds/npd
```

---

## 6. Inject a synthetic Xid event

You have three options, in order of "realness":

### 6a. Real GPU fault (only on bare metal / passthrough)

Run a workload that actually faults the GPU (sustained ECC errors, NVLink
disconnect, power loss). Not generally reproducible on AKS managed nodes.

### 6b. Use `nvidia-smi` to reset the GPU under load

On some SKUs, `nvidia-smi --gpu-reset -i 0` while the worker is using the
GPU produces an Xid 13/43. Run inside a privileged debug pod on the GPU
node:

```bash
kubectl debug node/"$NODE" -it --image=ubuntu --profile=sysadmin -- bash
# inside the debug pod:
chroot /host
nvidia-smi --gpu-reset -i 0 || true
journalctl -k | grep -i nvrm | tail
exit; exit
```

### 6c. Synthetic kmsg injection (works everywhere, including no-GPU nodes)

Write a fake Xid line directly into the kernel ring buffer. NPD's
`kernel-monitor` cannot tell the difference; this is the cleanest way to
demo the chain deterministically.

```bash
kubectl debug node/"$NODE" -it --image=ubuntu --profile=sysadmin -- bash
# inside the debug pod:
chroot /host
echo "<3>NVRM: Xid (PCI:0000:00:00): 79, pid=12345, GPU has fallen off the bus" \
  > /dev/kmsg
journalctl -k | tail -2
exit; exit
```

---

## 7. Scenario B — NPD + correlator surface the root cause

Within a few seconds, NPD picks up the Xid line and updates the node:

```bash
kubectl describe node "$NODE" | grep -A3 GPUHealthy
# Conditions:
#   Type        Status  Reason  Message
#   GPUHealthy  False   Xid79   NVIDIA Xid 79: GPU has fallen off the bus

kubectl get events --field-selector involvedObject.name="$NODE" \
  --sort-by=.lastTimestamp | grep -i xid
# Warning  NvidiaXid79  node/...  Xid 79 ... detected on ...
```

Meanwhile the Ray worker's next CUDA call raises and Ray reports a
`RayTaskError`:

```bash
kubectl logs -l ray.io/cluster=cuda-train -c ray-worker --tail=20
# ray.exceptions.RayTaskError(RuntimeError): CUDA error: unspecified launch failure
```

Run the correlator against the running cluster. A minimal version that
mirrors the logic in [simulate_xid_to_ray.py](simulate_xid_to_ray.py):

```bash
cat > /tmp/correlate.py <<'EOF'
import json, subprocess, sys, datetime as dt

JOB = sys.argv[1] if len(sys.argv) > 1 else "cuda-train"

# 1. find the worker pod and its node
pods = json.loads(subprocess.check_output([
    "kubectl","get","pod","-l",f"ray.io/cluster={JOB}",
    "-o","json"]))
worker = next(p for p in pods["items"]
              if p["metadata"]["labels"].get("ray.io/node-type") == "worker")
node = worker["spec"]["nodeName"]
pod  = worker["metadata"]["name"]

# 2. find the Ray failure time (use pod logs for simplicity)
logs = subprocess.check_output(
    ["kubectl","logs",pod,"-c","ray-worker","--tail=200"]).decode()
fail_line = next((l for l in logs.splitlines()
                  if "RayTaskError" in l or "CUDA error" in l), None)
if not fail_line:
    print("no Ray failure detected"); sys.exit(0)
fail_ts = dt.datetime.now(dt.timezone.utc)  # in real life: parse from log

# 3. pull node events within +/- 30s and look for NvidiaXid*
evs = json.loads(subprocess.check_output([
    "kubectl","get","events","-A",
    "--field-selector",f"involvedObject.name={node}",
    "-o","json"]))["items"]
window = dt.timedelta(seconds=30)
xid = [e for e in evs
       if e.get("reason","").startswith("NvidiaXid")
       and abs(dt.datetime.fromisoformat(
           e["lastTimestamp"].replace("Z","+00:00")) - fail_ts) <= window]

print("Ray failure  :", fail_line.strip())
print("Pod / Node   :", pod, "/", node)
if xid:
    e = xid[0]
    print(f"Verdict      : GPU/HARDWARE ROOT CAUSE ({e['reason']})")
    print(f"NPD message  : {e['message']}")
else:
    print("Verdict      : AMBIGUOUS - LIKELY APP, COULD BE GPU "
          "(no NPD/Xid event near failure)")
EOF
python3 /tmp/correlate.py cuda-train
```

Expected output:

```text
Ray failure  : ray.exceptions.RayTaskError(RuntimeError): CUDA error: unspecified launch failure
Pod / Node   : cuda-train-worker-...  /  aks-gpunp-...-vmss000000
Verdict      : GPU/HARDWARE ROOT CAUSE (NvidiaXid79)
NPD message  : Xid 79 (GPU has fallen off the bus) detected on aks-gpunp-...-vmss000000
```

That same pair of pieces — the NPD condition/event and the correlator —
turns an opaque "RayTaskError: CUDA error" into "GPU 0 fell off the bus".

---

## 8. Cleanup

```bash
kubectl delete rayjob cuda-train --ignore-not-found
kubectl delete configmap cuda-train-code --ignore-not-found
helm uninstall npd -n npd || true
helm uninstall kuberay-operator -n kuberay-system || true
az group delete -n "$RG" --yes --no-wait
```

---

## Mapping back to the local simulator

| Real component                        | Simulator equivalent in `simulate_xid_to_ray.py` |
| ------------------------------------- | ------------------------------------------------ |
| NVIDIA driver writing to `/dev/kmsg`  | `GPU.cuda_launch()` -> `KernelLog.emit()`        |
| `journalctl -k`                       | `KernelLog.lines`                                |
| NPD kernel-monitor JSON rule          | `NodeProblemDetector.XID_REGEX`                  |
| `kubectl describe node` conditions    | `KubeAPI.node_conditions`                        |
| `kubectl get events`                  | `KubeAPI.events`                                 |
| Ray worker raising `RayTaskError`     | `RayWorker.run_task()`                           |
| Stand-alone correlator script         | `XidCorrelator.explain()`                        |

The simulator is the spec; the AKS steps above are the real-world wiring.
For what an upstream KubeRay-native version of the correlator could look
like, see [FUTURE_WORK.md](FUTURE_WORK.md).
