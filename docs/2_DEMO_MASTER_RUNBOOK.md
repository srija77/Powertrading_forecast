# MLOps Open Source — Master Runbook

End-to-end demo for the DAM MCP forecasting pipeline:
ingestion → data validation → feature engineering → model training → deployment -> model serving → monitoring.

Four sections, run in order or jump to one:

1. [Ingestion → Validation](#section-1--ingestion--validation)
2. [Features → model](#section-2--features--model)
3. [Monitoring and inference services on Docker](#section-3--monitoring-services-on-docker)
4. [CI/CD → GitOps](#section-4--cicd--gitops)

---

## Section 1 — Ingestion → Validation

### Setup & conventions

Run every command from the project root in PowerShell. First create and activate the venv


```powershell
python -m venv venv                     # one-time: create the virtual environment
.\venv\Scripts\Activate.ps1             # activate it (every new terminal)
$env:OPENLINEAGE_DISABLED = "true"      # keep set for every stage except the lineage demo
```

### Ingestion - Writes raw Bronze (`data/raw`) and cleaned Silver (`data/processed`) in one pass.

Two ways to get data into `data/raw/`:

- **Option A** — scrape fresh live data for any date range you choose.
- **Option B** — load the provided March 2025 data and skip scraping (recommended for the demo).

#### Option A — get recent data (work with fresh, live data) -Point each source at the dates you want. These hit live sources, so coverage depends on what the upstream sites are serving that day.

```powershell
# Ingestion - REST + HTML/PDF scrapers
pip install pandas==2.3.3 requests==2.32.5 beautifulsoup4==4.14.3 pdfplumber==0.11.9 selenium==4.41.0 playwright==1.58.0 boto3==1.42.72 python-dotenv==1.2.2
python -m playwright install chromium 


```powershell
# Weather - REST API, no browser or key needed
python src/1_ingestion/3_ingest_weather_local.py --start 2025-06-01 --end 2025-06-30

# DAM + RTM - one day, force re-scrape (Selenium -> IEX, needs Chrome)
python src/1_ingestion/1_ingest_dam_rtm_local.py --date 2025-06-15 --force

# Generation - one day (Playwright: python -m playwright install chromium)
python src/1_ingestion/2_ingest_generation_local.py --start 2025-06-15 --end 2025-06-15
```

#### Option B — for the demo (use the provided March 2025 data)

> **Recommended — populate `data/raw/` from the provided March 2025 data.**
> Instead of re-scraping every source, run the populate script. It reads the bundled
> March 2025 dataset (`inputdata/march_2025_data.zip`) and writes it into `data/raw/` in
> the same `year=/month=/date=` layout the ingestion scripts produce (dam, rtm, weather,
> calendar) — then continue straight to validation.
>
> ```powershell
> python scripts/populate_raw_data.py
> ```


### Version raw data in DVC - After data lands in `data/raw`, refresh its `.dvc` pointer and inspect it.

```powershell
dvc add data/processed/weather                       # refresh the .dvc pointer
git add data/processed/weather.dvc data/raw/.gitignore
Get-Content data\raw\weather.dvc               # show the .dvc file (md5 hash -> tracked data)
```

### Great Expectations - Row-level validation splits each dataset into valid / invalid rows and writes a report plus GE Data Docs.

```powershell
# install packages 
pip install great-expectations==0.17.22 SQLAlchemy==2.0.48 psycopg2-binary==2.9.11
```

```powershell
# First run / full month - validate the entire dataset
python src/2_validation/5_gx_validate_local.py --start 2025-03-01 --end 2025-03-31 --dataset all

# Quick demo - one day, all datasets
python src/2_validation/5_gx_validate_local.py --date 2025-03-15 --dataset all
```

```powershell
start gx\uncommitted\data_docs\local_site\index.html           # GE Data Docs (best visual)
```

---

## Section 2 — Features → model

### Two kinds of run - The pipeline has two kinds of run and this is the core mental model for everything below.
> Independent runs 
> DVC pipeline runs

### Start MLflow (before running the scripts below) - Train, evaluate, and promote all talk to an MLflow server at `127.0.0.1:5000` — the one part of the local demo that needs Docker.

```powershell
docker compose up -d mlflow                    # start ONLY MLflow, not the whole stack
curl.exe http://localhost:5000/health          # expect: OK
start http://localhost:5000                    # MLflow UI
```

### Run the pipeline

#### Independent runs

Run each stage as its own script. After each, the comment says where to check the output.

```powershell
python src/3_feature_engineering/1_build_features.py   # -> data/features/march_2025_eda/march_2025_prepared.parquet
python "src\3_feature_engineering\2_prepare_feast_data.py"    # -> Feast source parquet
python src/4_training/1_train.py                 # MLflow: Experiments > dam_mcp_forecast - compare the 5 runs by rmse
python src/5_evaluation/1_evaluate.py              # MLflow: the run tagged passed_eval - eval_rmse / eval_mae / eval_mape
python src/6_promotion/1_promote_model.py               # MLflow: Models > dam_mcp_forecast - the champion alias on the new version
```

#### DVC run

Reproduce the same stages through DVC. `dvc repro` skips the frozen `build_features` stage
and runs the four downstream stages from `march_2025_prepared.parquet`.

```powershell
dvc dag                             # show the 5-stage graph
dvc status                          # which stages are stale
# Check dvc.yaml file and observe stages that dvc repro runs
dvc repro                           # prepare_feast_data -> train -> evaluate -> promote

                                    #   MLflow: train -> 5 runs by rmse; evaluate -> passed_eval; promote -> champion alias
dvc repro -f <stage_name>           # force re-run a stage + downstream eg: dvc repro -f train
dvc metrics show --json                    # metrics + eval_metrics + promote_report
```

### Refresh the online store - Run this so the Feast online store serves the latest features — predict-by-block_id reads from it, so without a refresh it returns stale (or missing) data after features change.

```powershell
python src/3_feature_engineering/3_refresh_online_store.py         # build_features -> prepare_feast_data -> feast apply -> materialize
```
```powershell
python src/3_feature_engineering/4_query_online_store.py # inspect the online store
```
### Predict in the CLI - Predict a DAM MCP price by `block_id` (a 15-min slot) straight from the online store.

```powershell
# Predict one or more 15-min blocks (needs MLflow up to load the model)
python src/7_prediction/1_predict.py --online --block-id 2025-03-20T18:00 --block-id 2025-03-15T09:00



### Predictions in the UI

Run the app in Docker, then open the UI in a browser.

```powershell
docker compose up -d app                       # app + its deps (MLflow, Postgres); serves on :8000
start http://localhost:8000                     # open the UI
```

- **Batch Predict (CSV Upload) tab** — upload `sample_data/batch_sample.csv` → predictions
  table plus RMSE / MAE / MAPE (the sample ships real March 2025 actuals).
- **Single Prediction tab** — all 37 feature fields come pre-filled with realistic March
  2025 default values, so just click Predict for a real-time prediction without typing.
- **Predict-by-block box** (top of the Single tab) — enter a `block_id` (e.g.
  `2025-03-20T18:00`) to predict straight from the Feast online store.

---

## Section 3 — Monitoring services on Docker

Live inference health (Prometheus + Grafana) plus data lineage (OpenLineage → Marquez). Run following

- **App** (metrics source) — http://localhost:8000/metrics
- **Prometheus** — http://localhost:9090
- **Grafana** — http://localhost:3000 (login `admin` / `admin123`)
- **Marquez UI** — http://localhost:3001
- **Marquez API** — http://localhost:5002

### Check the monitoring config files (before starting)

Prometheus and Grafana are configured from files in the repo, not from the UI. Confirm those
files exist and parse before starting the stack, so a typo does not silently break a scrape
target, a dashboard, or an alert rule.

```powershell
# Prometheus scrape config (jobs and targets)
Get-Content monitoring/prometheus.yml
python -c "import yaml; yaml.safe_load(open('monitoring/prometheus.yml')); print('prometheus.yml OK')"
```

```powershell
# Grafana provisioning YAMLs (datasource, dashboard loader, alert rules)
Get-ChildItem monitoring/provisioning -Recurse -Filter *.yml
python -c "import yaml, glob; [yaml.safe_load(open(f, encoding='utf-8')) for f in glob.glob('monitoring/provisioning/**/*.yml', recursive=True)]; print('grafana yaml OK')"
```

```powershell
# Grafana dashboard JSONs (one file per dashboard)
Get-ChildItem monitoring/provisioning/dashboards -Filter *.json
python -c "import json, glob; [json.load(open(f, encoding='utf-8')) for f in glob.glob('monitoring/provisioning/dashboards/*.json')]; print('dashboards json OK')"
```

```powershell
# The compose file that starts the stack
docker compose config --quiet    # no output and exit 0 means docker-compose.yml is valid
```

### Start the stacks

```powershell
docker compose up -d app prometheus grafana          # metrics stack
docker compose -f marquez-docker-compose.yml up -d   # Marquez (lineage), separate stack
```

```powershell
start http://localhost:9090        # Prometheus (Status -> Targets: gmr-app UP; mlflow DOWN is expected)
start http://localhost:3000        # Grafana (dashboards + alerts auto-provisioned)
start http://localhost:3001        # Marquez (search a job, e.g. "train", to see the lineage graph)
```

### Generate traffic

Metrics only move when the app serves predictions — and `rate()`/`histogram_quantile()`
only show data while the counters are **actively increasing within the query window**. A
one-off burst spikes then decays back to zero (and P95 latency reads `NaN`, panels show
"No data") within ~1–2 minutes. For a demo, run the **sustained** generator in its own tab
so the Prometheus graphs and Grafana panels stay populated the whole time:

```powershell
# Recommended — steady load (~5 req/s for 5 min). Leave it running while you demo.
python scripts/generate_traffic.py --seconds 300 --rps 5
```

```powershell
# Quick one-off burst (spikes then decays — fine for a single Execute, not for graphs)
1..20 | ForEach-Object { curl.exe -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{}" | Out-Null }
```

### Prometheus — two queries

Open http://localhost:9090, paste a query into the expression bar, click **Execute**, then
switch to the **Graph** tab to see it over time. **Keep the sustained traffic generator
above running** — with an idle app these return `0`/`NaN` and the graph looks empty (that is
the #1 reason "Prometheus shows no data": there is nothing wrong with the stack, the counter
just isn't moving). The `[5m]` window keeps data visible for ~5 min after traffic stops.

```promql
rate(inference_requests_total[5m])                                              # request rate (req/s)
histogram_quantile(0.95, rate(inference_latency_seconds_bucket[5m]))            # P95 latency
```

---

## Section 4 — CI/CD → GitOps

- **Repo** — `github.com/your-org/my-mlops-project`
- **Registry** — `ghcr.io/your-org/gmr-app`, `ghcr.io/your-org/mlflow`
- **ArgoCD app** — `gmr-mlops` (namespace `argocd`) → namespace `mlops`
- **Watched path** — `k8s/` on branch `main`
- Contact — mlops-team@example.com
- Note — replace `your-org` with your own GitHub username or organisation

### Pre-flight — Git & remote

Confirm Git is installed, the repo is initialized, and it is connected to GitHub.

```powershell
git --version                       # is Git installed?
git rev-parse --is-inside-work-tree # is this a Git repo? (expect: true)
git remote -v                       # which remote/repo is it connected to?
git ls-remote origin HEAD           # can we reach the remote? (lists the main SHA on success)
```

If it is not a repo yet, initialize it and connect the remote:

```powershell
git init
git remote add origin https://github.com/your-org/my-mlops-project.git
git branch -M main
```

### Check .gitignore before staging

Make sure secrets and junk (`.env`, `venv/`, `mlruns/`, large `*.db`) are ignored — the CI
`test` job runs gitleaks and a leaked secret fails the whole pipeline.

```powershell
Get-Content .gitignore              # confirm .env, venv/, mlruns/, *.db etc. are listed
git status --short                  # review what WOULD be committed - nothing sensitive/large
git check-ignore .env           # prints each path if it is ignored (silent = NOT ignored)
```

### Push to main (this DEPLOYS)

```powershell
git add -A
git commit -m "demo: bump demo_version"
git push origin main
git rev-parse --short HEAD          # the NEW SHA -> becomes the new image tag
```
```powershell
gh run watch                        # test (~1m) -> build-push (~4m) -> deploy (~30s)
gh run view --log-failed            # logs of just the failed steps, if any
```
### Take a look at the CI/CD workflow — `.github/workflows/ml-ci.yml`

The workflow does the build + deploy for you. It is triggered **only** by a push to `main`
or `develop`, or a PR to `main` — a push to any other branch runs nothing.

- **test** (always) — gitleaks secret scan + ruff lint + pytest.
- **build-push** (push to `main`) — build the app + MLflow images, push to GHCR tagged with
  the commit SHA + `latest`.
- **deploy** (after build-push) — write the new image tag into `k8s/deployment.yaml`, then
  commit + push it back to `main`.
- **train** (manual only, `run_training=true`) — `dvc repro` → eval gate → promote champion.

That `deploy` commit to `k8s/deployment.yaml` is the hand-off to GitOps — ArgoCD takes it
from there (next step). You can also follow the run from the terminal with `gh run watch`
instead of the UI.

### Automated retraining on drift — `.github/workflows/retrain-on-drift.yml`

A second workflow closes the loop from drift detection to retraining, so you do not have to
start training by hand. On a schedule it runs the drift check against the training reference
baseline; if drift is found it dispatches the `train` job above, which then evaluates,
promotes, and flows through the same build and deploy chain to the cluster.

Trigger it yourself instead of waiting for the schedule:

```powershell
gh workflow run retrain-on-drift.yml -f force_retrain=true    # retrain now, skip the drift gate (demo)
gh workflow run retrain-on-drift.yml                          # check drift first, retrain only if drifted
gh run watch                                                  # follow the drift gate, then the dispatched train run
```

For real use, edit CURRENT_DATA at the top of the workflow to point at your logged inference
data; the demo default scores sample_data/batch_sample.csv so a run always has data.

### Demo it: make a small change → see the workflow run

Make a visible edit, push it, and watch the pipeline carry it to the cluster from GitHub.

> ⚠️ **Be on `main` first.** The workflow fires only on a push to `main`/`develop` or a PR
> to `main`. Pushing from a `feature/...` branch triggers nothing — that is the usual reason
> the Actions tab stays empty. Check before you push:
>
> ```powershell
> git branch --show-current           # must be "main" for the push below to deploy
> ```
>
> On a feature branch instead? Push it and open a PR — that runs the `test` job only; the
> full build-push → deploy happens when the PR merges to `main`:
>
> ```powershell
> git push -u origin (git branch --show-current)
> gh pr create --fill --base main
> ```

```powershell
git switch main                     # the workflow only deploys from main
git pull origin main                # get current first - else the push is rejected as "tip is behind"

# Bump a version number in the page heading (auto-increments, so it is repeatable)
$f = "app/templates/index.html"
$c = Get-Content $f -Raw
$m = [regex]::Match($c, 'Energy Market Forecasting(?: v(\d+))?</h1>')
$n = if ($m.Groups[1].Success) { [int]$m.Groups[1].Value + 1 } else { 2 }
(Get-Content $f -Raw) -replace 'Energy Market Forecasting(?: v\d+)?</h1>', "Energy Market Forecasting v$n</h1>" | Set-Content $f

git add app/templates/index.html
git add app/templates/index.html
git pull origin main 
git add app/templates/index.html                # rejected as "tip is behind"? run: git pull origin main   (then push again)
git rev-parse --short HEAD          # the NEW SHA -> becomes the new image tag
```

> If `git push` is rejected with **"tip is behind its remote counterpart"**, your local
> `main` is stale — run `git pull origin main` and push again. (This usually means you were
> on a feature branch and pushed a stale local `main`.)

Then watch it on the GitHub UI:

- **Actions** tab — a run appears for this commit; watch **test → build-push → deploy** go green.


### ArgoCD syncs Git → cluster

First check the current ArgoCD and Kubernetes status (the baseline before the sync):

```powershell
kubectl get applications -n argocd                   # gmr-mlops: Synced/OutOfSync + Healthy
kubectl get pods -n mlops                            # app stack: gmr-app, postgres, prometheus, grafana - all Running
```

> If the `gmr-mlops` app or the `mlops` pods don't exist (cluster or ArgoCD not installed
> yet), set them up first — see [3_ARGOCD_K8S_INSTALL_RUNBOOK.md](3_ARGOCD_K8S_INSTALL_RUNBOOK.md)
> — then come back here.

**CD is automatic** — you do not run the deploy. ArgoCD reads the repo from GitHub  and its policy is `automated` + `selfHeal`, so once the `deploy` job pushes the
image-tag commit to `main`, ArgoCD syncs and rolls out the new pods on its own (within its
~3-min poll). Just watch it happen:

```powershell
kubectl get pods -n mlops -w                          # old gmr-app Terminating, new one Running (Ctrl+C to stop)
```


```powershell
# Speed it up - nudge ArgoCD to reconcile NOW instead of waiting the ~3-min poll
kubectl -n argocd annotate app gmr-mlops argocd.argoproj.io/refresh=hard --overwrite
```

### Port-forwards — reach the cluster services

The cluster runs inside the minikube Docker container, so nothing in `mlops` / `argocd` is
reachable from your browser until you forward a port. **Each command blocks** — give it its
own terminal tab and leave it running for the whole demo.

```powershell
kubectl port-forward svc/argocd-server  -n argocd 8080:443     # ArgoCD     -> https://localhost:8080
kubectl port-forward svc/gmr-app-svc    -n mlops  8000:8000    # App UI     -> http://localhost:8000
kubectl port-forward svc/grafana-svc    -n mlops  3000:3000    # Grafana    -> http://localhost:3000 (admin/admin123)
kubectl port-forward svc/prometheus-svc -n mlops  9090:9090    # Prometheus -> http://localhost:9090
kubectl port-forward svc/postgres-svc   -n mlops  5432:5432    # Postgres   -> localhost:5432 (only if you need the DB)
```

> ⚠️ **Port clash with Section 3.** The Docker stack binds 8000 / 3000 / 9090 on localhost
> already. With both up, the forward fails with *"Only one usage of each socket address"*.
> Either free the ports first, or forward the cluster to different local ports:
>
> ```powershell
> docker compose down                                        # free 8000/3000/9090, then forward as above
> ```
> ```powershell
> # ...or run both side by side - LOCAL port on the left, cluster port on the right
> kubectl port-forward svc/gmr-app-svc -n mlops 18000:8000   # cluster app     -> http://localhost:18000
> kubectl port-forward svc/grafana-svc -n mlops 13000:3000   # cluster Grafana -> http://localhost:13000
> ```

Prefer background jobs over juggling four tabs:

```powershell
Start-Job { kubectl port-forward svc/argocd-server  -n argocd 8080:443 }
Start-Job { kubectl port-forward svc/gmr-app-svc    -n mlops  8000:8000 }
Start-Job { kubectl port-forward svc/grafana-svc    -n mlops  3000:3000 }
Start-Job { kubectl port-forward svc/prometheus-svc -n mlops  9090:9090 }
Get-Job                                              # all four should be Running
```

Confirm they are live, and tear them down when done:

```powershell
Get-NetTCPConnection -LocalPort 8080,8000,3000,9090 -State Listen -EA SilentlyContinue |
  Select-Object LocalPort, OwningProcess             # which forwards are actually listening
curl.exe -s -o NUL -w "%{http_code}`n" http://localhost:8000/health   # 200 = forward works
Get-Job | Stop-Job; Get-Job | Remove-Job             # stop every background forward
```

> **A port-forward dies whenever its pod restarts** — which is exactly what an ArgoCD sync
> does. `lost connection to pod` mid-demo is expected, not a fault: Ctrl+C (or re-run
> `Start-Job`) to re-attach to the new pod. Forwarding `svc/...` rather than `pod/...`
> already picks the new pod for you on reconnect.

### watch in the ArgoCD UI - open it, then log in

With the `argocd-server` forward above running, open the UI and accept the self-signed
certificate warning. Login is user `admin` + the auto-generated password:

```powershell
start https://localhost:8080

# Login = user "admin" + the auto-generated password. This command prints the password:
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | ForEach-Object { [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($_)) }
```

### Prove the new image is live

Three different identifiers are in play — don't confuse them:

- **commit SHA** of your push (e.g. `736d28d`) — becomes the image **tag**.
- **image tag** in `k8s/deployment.yaml` (e.g. `gmr-app:736d28d`) — what ArgoCD deploys.
- **image ID** = the `sha256:...` digest — the real identity of the running image.



```powershell
# 1. The tag ArgoCD is deploying (the source of truth - lives in Git, set by the deploy job)
git show origin/main:k8s/deployment.yaml | Select-String "gmr-app:"      # -> ...gmr-app:736d28d

# 2. The tag the cluster's Deployment is set to - must equal #1
kubectl get deploy gmr-app -n mlops -o "jsonpath={.spec.template.spec.containers[0].image}"

# 3. The tag + sha256 image ID the running POD actually pulled
kubectl get pods -n mlops -l app=gmr-app -o "jsonpath={range .items[*]}{.metadata.name}{'  '}{.spec.containers[0].image}{'  '}{.status.containerStatuses[0].imageID}{'\n'}{end}"
```

Confirm it is fully rolled out and serving the new code:

```powershell
kubectl rollout status deploy/gmr-app -n mlops          # "successfully rolled out"
kubectl get pods -n mlops -l app=gmr-app                # one new pod Running; old one gone

kubectl port-forward svc/gmr-app-svc -n mlops 8000:8000 # skip if already forwarded above; re-run if the sync killed it
curl.exe http://localhost:8000/health                   # the NEW demo_version
start http://localhost:8000                             # the heading shows the new version
```


### Quick checks — Kubernetes

```powershell
minikube status                                      # cluster up? (if Stopped: minikube start)
kubectl get pods -n mlops                            # app stack: gmr-app, postgres, prometheus, grafana
kubectl get svc  -n mlops                            # service names + ports (the args for port-forward)
kubectl rollout status deploy/gmr-app -n mlops       # block until rollout finishes
kubectl logs -n mlops deploy/gmr-app --tail=50       # app logs
kubectl get events -n mlops --sort-by=.lastTimestamp # recent cluster events
```

### Quick checks — ArgoCD

```powershell
kubectl get applications -n argocd                                              # gmr-mlops: Synced/OutOfSync + Healthy
kubectl get app gmr-mlops -n argocd -o jsonpath="{.status.sync.status} / {.status.health.status}"
```

### Quick checks — GitHub Actions

```powershell
gh run list --workflow ml-ci.yml -L 5                # recent runs + status
gh run watch                                         # live-follow the latest run
```

> Notes: first image pull is slow (gmr-app ~1.8 GB, 3–10 min on a fresh node). `git push`
> says "everything up to date"? Bump the `demo_version` string in `app/app.py` (`/health`
> route) to make a fresh change. MLflow is NOT in-cluster — it runs on the host via
> docker-compose; the in-cluster app reaches it at `host.docker.internal:5000`.
