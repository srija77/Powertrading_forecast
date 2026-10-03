# ArgoCD + Kubernetes — Install Runbook

Companion to [2_DEMO_MASTER_RUNBOOK.md](2_DEMO_MASTER_RUNBOOK.md). Run this once to stand up the
cluster + ArgoCD before Section 4 (CI/CD → GitOps). If the status checks there show the
`gmr-mlops` app or the `mlops` pods missing, set them up here, then go back.

End state: a local minikube cluster running ArgoCD (namespace `argocd`) that auto-deploys
the app stack — gmr-app, postgres, prometheus, grafana — into namespace `mlops`, synced
from `k8s/` on `main`. MLflow runs on the host (not in-cluster).

---

## 1. Prerequisites

> **Already worked through the Installation Guide (`1_MLOps_Installation_Guide.docx`)?**
> You don't need to reinstall anything. Just run that guide's **Verification Checklist**
> section to confirm Docker, kubectl, and minikube are installed and on your PATH — then
> install only whatever it reports as missing, and continue below.

- **Docker Desktop** running.
- **kubectl** — https://kubernetes.io/docs/tasks/tools/
- **minikube** — https://minikube.sigs.k8s.io/docs/start/

```powershell
docker version          # daemon reachable
kubectl version --client
minikube version
```

## 2. Start the cluster

```powershell
minikube start --memory=4096 --cpus=2
minikube addons enable metrics-server
kubectl cluster-info                    # API server reachable
minikube status                         # host/kubelet/apiserver = Running
```

## 3. Start MLflow on the host

The in-cluster app loads its model from MLflow at `host.docker.internal:5000`, so MLflow
must run on the host (not in the cluster).

```powershell
docker compose up -d mlflow postgres
curl.exe http://localhost:5000/health   # expect: OK
```

## 4. Install ArgoCD

```powershell
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
kubectl wait --for=condition=available deployment/argocd-server -n argocd --timeout=180s
kubectl get pods -n argocd               # all Running (esp. argocd-repo-server, argocd-server)
```

## 5. Get the ArgoCD login (username + password)

- **Username:** `admin` (always)
- **Password:** auto-generated, stored base64-encoded in the `argocd-initial-admin-secret`.
  Decode it:

```powershell
$pw = kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}"
[System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($pw))   # user: admin
```

One-liner version:

```powershell
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | ForEach-Object { [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($_)) }
```

This is the **initial** password. If it was changed in the UI/CLI, the secret no longer
reflects it. Save it somewhere — the secret can be deleted after first login (then the
commands above stop working).

## 6. Create the app namespace + secrets

```powershell
kubectl create namespace mlops

kubectl create secret generic mlops-secrets -n mlops `
  --from-literal=api-key="YOUR_API_KEY" `
  --from-literal=grafana-password="admin123" `
  --from-literal=postgres-user="mlops" `
  --from-literal=postgres-password="mlops" `
  --from-literal=postgres-db="mlops" `
  --dry-run=client -o yaml | kubectl apply -f -
```

Replace `YOUR_API_KEY` with the real key (keep it out of Git — it lives in `.env` locally).

## 7. Give ArgoCD access to the Git repo (SSH deploy key)

**Required whenever the repo is private** — `srija77/my-mlops-project` is. ArgoCD does *not*
inherit your Git credentials: it runs in the cluster with none of your logins. Skip this and
the app sits at `Unknown / Healthy` forever with:

```
failed to list refs: authentication required: Repository not found.
```

> **"But `git clone` works on my laptop!"** That proves nothing. Windows Git Credential
> Manager silently supplies your saved login. To test what ArgoCD will actually experience:
>
> ```powershell
> git -c credential.helper= ls-remote https://github.com/srija77/my-mlops-project.git HEAD
> ```
>
> Exit 128 / *"could not read Username"* means private → you need this section.

We use an **SSH deploy key** rather than a personal access token. A deploy key belongs to the
*repository* rather than to a person, is read-only by checkbox, and never expires — so it
does not break when someone rotates a password or leaves.

### 7.1 Generate the key pair

```powershell
ssh-keygen -t ed25519 -C "argocd@gmr-mlops" -f "$env:USERPROFILE\.ssh\argocd_gmr_mlops" -N '""'
```

`-N '""'` sets an empty passphrase — mandatory, as ArgoCD cannot type one.

### 7.2 Add the PUBLIC half to GitHub

```powershell
Get-Content "$env:USERPROFILE\.ssh\argocd_gmr_mlops.pub"      # copy the whole line
start "https://github.com/srija77/my-mlops-project/settings/keys/new"
```

- **Title:** `argocd-minikube`
- **Key:** paste the entire `.pub` line, including the `ssh-ed25519 ` prefix
- **Allow write access:** leave **UNCHECKED** — ArgoCD only reads

Use the **repo's** Deploy keys page, not your account's SSH keys page. They look alike; only
the repo one creates a deploy key.

### 7.3 Give the PRIVATE half to ArgoCD

```powershell
kubectl -n argocd create secret generic repo-gmr-mlops `
  --from-literal=type=git `
  --from-literal=url=git@github.com:srija77/my-mlops-project.git `
  --from-file=sshPrivateKey=$env:USERPROFILE\.ssh\argocd_gmr_mlops

kubectl -n argocd label secret repo-gmr-mlops argocd.argoproj.io/secret-type=repository
```

Two things silently break this if you get them wrong:

- **The label is mandatory.** Without `argocd.argoproj.io/secret-type=repository`, ArgoCD
  never looks at the secret and you get the exact same "Repository not found" error.
- **`url` must match the Application's `repoURL` character for character**, including the
  `.git` suffix. A mismatch means the secret is ignored.

### 7.4 Point the Application at the SSH URL

`repoURL` in `k8s/argocd-app.yaml` must be the `git@` form, not `https://`:

```yaml
  source:
    repoURL: git@github.com:srija77/my-mlops-project.git
```

⚠️ **Commit and push this change.** ArgoCD watches `k8s/`, which contains its own
Application manifest — so it manages itself, and `selfHeal` reverts any live `kubectl`
edit back to whatever Git says (within ~20s). See Troubleshooting.

### 7.5 Verify the key works before moving on

```powershell
$env:GIT_SSH_COMMAND = "ssh -i $env:USERPROFILE/.ssh/argocd_gmr_mlops -o IdentitiesOnly=yes"
git ls-remote git@github.com:srija77/my-mlops-project.git HEAD    # prints a SHA = working
```

`Permission denied (publickey)` means step 7.2 didn't take. Once this prints a SHA, delete
the local private key — it now lives in the cluster:

```powershell
Remove-Item "$env:USERPROFILE\.ssh\argocd_gmr_mlops"
```

## 8. Deploy the ArgoCD Application

This registers the `gmr-mlops` app — it tells ArgoCD to watch `k8s/` on `main` and auto-sync
(`prune` + `selfHeal`) into namespace `mlops`.

```powershell
kubectl apply -f k8s/argocd-app.yaml
kubectl get applications -n argocd       # gmr-mlops appears -> Synced / Healthy once it rolls out
```

## 9. Access the UIs

Each port-forward blocks — run each in its own terminal tab.

```powershell
kubectl port-forward svc/argocd-server -n argocd 8080:443     # ArgoCD  -> https://localhost:8080 (admin / step 5)
kubectl port-forward svc/gmr-app-svc   -n mlops  8000:8000    # App     -> http://localhost:8000
kubectl port-forward svc/grafana-svc   -n mlops  3000:3000    # Grafana -> http://localhost:3000
kubectl port-forward svc/prometheus-svc -n mlops 9090:9090    # Prom    -> http://localhost:9090
```

## 10. Verify everything

```powershell
kubectl get pods -n mlops                # gmr-app, postgres, prometheus, grafana - all Running
kubectl get svc  -n mlops                # services + ports
kubectl get applications -n argocd       # gmr-mlops: Synced / Healthy
kubectl logs deployment/gmr-app -n mlops --tail=50
```

---

## Troubleshooting

Work top-down: a broken layer below makes every layer above it look broken too. All five
below were hit for real on 2026-08-12; the symptoms are quoted verbatim.

### A. `kubectl` — "TLS handshake timeout"

```
couldn't get current server API group list: Get "https://127.0.0.1:PORT/api?timeout=32s":
net/http: TLS handshake timeout
```

**This is almost never Kubernetes.** It means Docker is wedged: the WSL2 machine that hosts
the Docker daemon has stopped, so minikube has nowhere to run. Docker Desktop's processes
stay alive and look healthy in Task Manager, which is what makes it misleading.

```powershell
wsl -l -v            # docker-desktop "Stopped" = confirmed (should be "Running")
docker ps            # hangs forever instead of erroring = confirmed
```

Recovery, in order:

```powershell
Get-Process "Docker Desktop","com.docker.backend" | Stop-Process -Force
wsl.exe --shutdown
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
# wait for `docker ps` to return promptly, then:
minikube start
```

Takes ~5 min. Pods come back `Error` at first and self-heal to `Running` within ~1 minute —
**don't delete them prematurely.**

### B. minikube container `Exited (255)`

Normal after Docker wedges or the host is shut down ungracefully. `minikube start` restores
it; cluster state lives in a Docker volume and survives.

### C. ArgoCD — "authentication required: Repository not found"

```
ComparisonError: Failed to load target state: ... failed to list refs:
authentication required: Repository not found.
```

Or, in the UI: *"revision main must be resolved"* and *"could not read Username for
'https://github.com': terminal prompts disabled"*.

The repo is **private** and ArgoCD has no credentials. GitHub reports private repos as
*"not found"* rather than *"access denied"*, so this reads like a typo in `repoURL` when it
isn't. Fix: **step 7** above.

```powershell
# is a repo credential registered at all?
kubectl -n argocd get secrets -l argocd.argoproj.io/secret-type=repository

# what ArgoCD actually thinks the error is
kubectl get app gmr-mlops -n argocd -o "jsonpath={range .status.conditions[*]}{.type}: {.message}{'\n'}{end}"

# is it a network problem instead? (should resolve an IP)
kubectl -n argocd exec deploy/argocd-repo-server -- getent hosts github.com
```

A secondary *"error acquiring repo lock"* is downstream noise — repo-server holds a lock
while retrying the doomed fetch. It clears on its own; `rollout restart` clears it now.

### D. Your `kubectl` edits keep reverting after ~20 seconds

**This is `selfHeal` working as designed, not a bug.** The `gmr-mlops` app watches `k8s/`,
which contains its own `argocd-app.yaml` — so ArgoCD manages itself and forces the cluster
back to whatever **Git** says.

```powershell
# confirm: the Application is in its own managed-resource list
kubectl get app gmr-mlops -n argocd -o "jsonpath={range .status.resources[*]}{.kind}/{.name}{'\n'}{end}"
```

> **The rule: with ArgoCD, fix things in Git — never with `kubectl`.** Commit and push.
> A live patch buys you ~20 seconds. If you must patch live (e.g. to break a chicken-and-egg
> during setup), push the same change to Git *first*, then patch.

### E. Pod `CrashLoopBackOff` with `exit code 3`

```
FileNotFoundError: [Errno 2] No such file or directory: '/tmp/prometheus_multiproc/counter_7.db'
  at app/app.py -> Counter("inference_requests_total", ...)
```

`Dockerfile` sets `PROMETHEUS_MULTIPROC_DIR` unconditionally, so `prometheus_client` starts
in multiprocess mode — but the directory is created only by `on_starting()` in
`app/gunicorn.conf.py`. If gunicorn is launched **without `-c app/gunicorn.conf.py`**, that
hook never runs and the worker dies on import.

Gunicorn auto-loads `./gunicorn.conf.py`, but the file is at `app/gunicorn.conf.py`, so it is
**not** picked up automatically. `k8s/deployment.yaml` must pass it explicitly:

```yaml
          args:
            - "-c"
            - "app/gunicorn.conf.py"
            - "-w"
            - "1"
```

CLI flags win over config values, so `-w 1` still caps workers at one. Docker Compose was
never affected because it always passed `-c`.

Because of `maxUnavailable: 0`, the previous pod keeps serving while the new one crashloops —
the app stays up, the rollout just never completes.

### Generic commands

```powershell
# Pod CrashLoopBackOff - why?
kubectl describe pod <pod-name> -n mlops
kubectl logs <pod-name> -n mlops --previous       # --previous = the crashed container

# App stuck OutOfSync / Unknown - force a sync
kubectl patch application gmr-mlops -n argocd --type merge -p '{\"operation\":{\"sync\":{\"force\":true}}}'
kubectl -n argocd annotate app gmr-mlops argocd.argoproj.io/refresh=hard --overwrite

# argocd-repo-server not Running -> app shows Unknown; restart it
kubectl -n argocd rollout restart deploy/argocd-repo-server

# Restart the app deployment
kubectl rollout restart deployment/gmr-app -n mlops
kubectl rollout status  deployment/gmr-app -n mlops

# Nuke and recreate the app stack (keeps ArgoCD + the repo secret)
kubectl delete -f k8s/argocd-app.yaml
kubectl delete namespace mlops
# then redo from step 6
```

> First image pull is slow (gmr-app ~1.8 GB): 3–10 min on a fresh node.

> **kubectl version skew:** Docker Desktop ships kubectl v1.30.x while minikube runs
> Kubernetes v1.35.x — well beyond the supported ±1 minor. If you hit odd API errors, use
> the matching client: `minikube kubectl -- get pods -A`.

## Cleanup

```powershell
minikube stop                            # stop the cluster (keeps it)
docker compose down                      # stop host MLflow/postgres
minikube delete                          # fully remove the cluster
```

---

## Reference

- `k8s/argocd-app.yaml` — ArgoCD Application: what to watch (`k8s/` on `main`), where to
  deploy (`mlops`), sync policy (`prune` + `selfHeal`).
- `k8s/deployment.yaml` — all app-stack resources (gmr-app, prometheus, grafana, postgres).
- `k8s/namespace.yaml` — the `mlops` namespace.
- **Secret `repo-gmr-mlops`** (namespace `argocd`, step 7) — the SSH deploy key ArgoCD uses
  to read the private repo. Deliberately **not** in Git. It lives only in the cluster, so
  `minikube delete` destroys it and step 7 must be redone. To make rebuilds reproducible,
  seal it instead: `kubectl -n argocd get secret repo-gmr-mlops -o yaml | kubeseal --format yaml`
  and commit the SealedSecret (never the plain Secret — CI runs gitleaks and will fail).

```
GitHub (main) -> ArgoCD watches k8s/ -> auto-sync -> mlops namespace
  gmr-app (Flask+Gunicorn) -> host.docker.internal:5000 (MLflow on host)
  prometheus | grafana | postgres
```
