"""
generate_traffic.py — sustained load generator for the monitoring demo.

Prometheus rate()/histogram_quantile() only show data when the counters are
*actively increasing* within the query window. A one-off burst spikes then
decays to zero; this sends steady traffic so the Grafana panels and Prometheus
graphs stay populated while you demo.

Usage:
    python scripts/generate_traffic.py                 # 5 min, ~4 req/s
    python scripts/generate_traffic.py --seconds 120 --rps 8
    python scripts/generate_traffic.py --url http://localhost:8000
"""
import argparse
import time
import urllib.request

def hit(url: str) -> int:
    req = urllib.request.Request(
        url + "/predict",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--seconds", type=int, default=300)
    ap.add_argument("--rps", type=float, default=4.0)
    args = ap.parse_args()

    interval = 1.0 / args.rps
    end = time.monotonic() + args.seconds
    sent = ok = 0
    print(f"Sending ~{args.rps} req/s to {args.url}/predict for {args.seconds}s ...")
    while time.monotonic() < end:
        code = hit(args.url)
        sent += 1
        ok += (code == 200)
        if sent % 20 == 0:
            print(f"  sent={sent} ok={ok} last={code}")
        time.sleep(interval)
    print(f"Done. sent={sent} ok={ok}")

if __name__ == "__main__":
    main()
