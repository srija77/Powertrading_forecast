"""
ingest_weather_local.py
=======================
Fetches hourly weather data from Open-Meteo for Indian cities.
Saves raw CSV    → data/raw/weather/{City_Name}.csv       (Bronze)
Saves cleaned    → data/processed/weather/city={city}/    (Silver Parquet)
Writes audit     → data/raw/weather/run_summary.json


Usage
-----
  python src/1_ingestion/3_ingest_weather_local.py
  python src/1_ingestion/3_ingest_weather_local.py --days 30
  python src/1_ingestion/3_ingest_weather_local.py --start 2024-09-01 --end 2025-03-20

Local Bronze output  (raw CSV — one file per city, resume-safe)
---------------------------------------------------------------
  data/raw/weather/{City_Name}.csv

Local Silver output  (cleaned Parquet — one file per city)
----------------------------------------------------------
  data/processed/weather/city={City_Name}/part-0001.parquet

Columns (matches weather_suite.json)
  time, temperature, humidity, windspeed_100m, cloud_cover, rainfall,
  city_name, ingestion_date, source_file, pipeline_run_id

.env keys needed
-----------------
  None — no external service dependencies.

Dependencies
------------
  pip install pandas pyarrow requests python-dotenv
"""

import sys
import json
import time
import uuid
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
DEMO_DAYS = 7
OPEN_METEO = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_FIELDS = (
    "temperature_2m,relative_humidity_2m,wind_speed_100m,cloudcover,precipitation"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
LOCAL_BRONZE = PROJECT_ROOT / "data" / "raw" / "weather"
LOCAL_SILVER = PROJECT_ROOT / "data" / "processed" / "weather"

RUN_ID = str(uuid.uuid4())

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [WEATHER] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("weather_local")

# ─────────────────────────────────────────────────────────────
# CITIES
# ─────────────────────────────────────────────────────────────
CITIES = [
    # West Region
    {"name": "Durg_Chhattisgarh", "lat": 21.1904, "lon": 81.2849},
    {"name": "Raipur_Chhattisgarh", "lat": 21.2514, "lon": 81.6296},
    {"name": "Bilaspur_Chhattisgarh", "lat": 22.0797, "lon": 82.1396},
    {"name": "Korba_Chhattisgarh", "lat": 22.3450, "lon": 82.6820},
    {"name": "Gandhinagar_Gujarat", "lat": 23.2156, "lon": 72.6369},
    {"name": "Ahmedabad_Gujarat", "lat": 23.0225, "lon": 72.5714},
    {"name": "Surat_Gujarat", "lat": 21.1702, "lon": 72.8311},
    {"name": "Vadodara_Gujarat", "lat": 22.3072, "lon": 73.1812},
    {"name": "Indore_Madhya_Pradesh", "lat": 22.7196, "lon": 75.8577},
    {"name": "Bhopal_Madhya_Pradesh", "lat": 23.2599, "lon": 77.4126},
    {"name": "Jabalpur_Madhya_Pradesh", "lat": 23.1815, "lon": 79.9864},
    {"name": "Mumbai_Maharashtra", "lat": 18.9388, "lon": 72.8354},
    {"name": "Pune_Maharashtra", "lat": 18.5204, "lon": 73.8567},
    {"name": "Nagpur_Maharashtra", "lat": 21.1458, "lon": 79.0882},
    {"name": "Nashik_Maharashtra", "lat": 19.9975, "lon": 73.7898},
    {"name": "Aurangabad_Maharashtra", "lat": 19.8762, "lon": 75.3433},
    # South Region
    {"name": "Visakhapatnam_Andhra_Pradesh", "lat": 17.6868, "lon": 83.2185},
    {"name": "Hyderabad_Telangana", "lat": 17.3850, "lon": 78.4867},
    {"name": "Bangalore_Karnataka", "lat": 12.9716, "lon": 77.5946},
    {"name": "Chennai_Tamil_Nadu", "lat": 13.0827, "lon": 80.2707},
    {"name": "Coimbatore_Tamil_Nadu", "lat": 11.0168, "lon": 76.9558},
    {"name": "Madurai_Tamil_Nadu", "lat": 9.9252, "lon": 78.1198},
    {"name": "Thiruvananthapuram_Kerala", "lat": 8.5241, "lon": 76.9366},
    {"name": "Ernakulam_Kerala", "lat": 9.9816, "lon": 76.2999},
    {"name": "Mysore_Karnataka", "lat": 12.2958, "lon": 76.6394},
    {"name": "Bellary_Karnataka", "lat": 15.1394, "lon": 76.9214},
    # East Region
    {"name": "Patna_Bihar", "lat": 25.5941, "lon": 85.1376},
    {"name": "Ranchi_Jharkhand", "lat": 23.3441, "lon": 85.3096},
    {"name": "Dhanbad_Jharkhand", "lat": 23.7957, "lon": 86.4304},
    {"name": "Bhubaneswar_Odisha", "lat": 20.2961, "lon": 85.8245},
    {"name": "Rourkela_Odisha", "lat": 22.2604, "lon": 84.8536},
    {"name": "Kolkata_West_Bengal", "lat": 22.5726, "lon": 88.3639},
    {"name": "Durgapur_West_Bengal", "lat": 23.5204, "lon": 87.3119},
    # North Region
    {"name": "New_Delhi_Delhi", "lat": 28.6139, "lon": 77.2090},
    {"name": "Lucknow_Uttar_Pradesh", "lat": 26.8467, "lon": 80.9462},
    {"name": "Kanpur_Uttar_Pradesh", "lat": 26.4499, "lon": 80.3319},
    {"name": "Varanasi_Uttar_Pradesh", "lat": 25.3176, "lon": 82.9739},
    {"name": "Jaipur_Rajasthan", "lat": 26.9124, "lon": 75.7873},
    {"name": "Jodhpur_Rajasthan", "lat": 26.2389, "lon": 73.0243},
    {"name": "Ludhiana_Punjab", "lat": 30.9010, "lon": 75.8573},
    {"name": "Amritsar_Punjab", "lat": 31.6340, "lon": 74.8723},
    {"name": "Chandigarh", "lat": 30.7333, "lon": 76.7794},
    {"name": "Dehradun_Uttarakhand", "lat": 30.3165, "lon": 78.0322},
    {"name": "Shimla_Himachal_Pradesh", "lat": 31.1048, "lon": 77.1734},
    {"name": "Srinagar_Jammu_Kashmir", "lat": 34.0837, "lon": 74.7973},
    # North-East Region
    {"name": "Guwahati_Assam", "lat": 26.1445, "lon": 91.7362},
    {"name": "Shillong_Meghalaya", "lat": 25.5788, "lon": 91.8933},
    {"name": "Agartala_Tripura", "lat": 23.8315, "lon": 91.2868},
    {"name": "Imphal_Manipur", "lat": 24.8170, "lon": 93.9368},
    {"name": "Aizawl_Mizoram", "lat": 23.7271, "lon": 92.7176},
    {"name": "Kohima_Nagaland", "lat": 25.6701, "lon": 94.1077},
    {"name": "Itanagar_Arunachal_Pradesh", "lat": 27.0844, "lon": 93.6053},
    # Thermal plant sites
    {"name": "Sipat_Plant_Chhattisgarh", "lat": 22.1317, "lon": 82.2904},
    {"name": "Talcher_Plant_Odisha", "lat": 21.0883, "lon": 85.0806},
    {"name": "Chandrapur_Plant_Maharashtra", "lat": 20.0063, "lon": 79.2900},
    {"name": "Farakka_Plant_West_Bengal", "lat": 24.7727, "lon": 87.8937},
    # Solar sites
    {"name": "Bhadla_Solar_Rajasthan", "lat": 27.5397, "lon": 71.9153},
    {"name": "Pavagada_Solar_Karnataka", "lat": 14.2500, "lon": 77.4500},
    {"name": "Charanka_Solar_Gujarat", "lat": 23.8736, "lon": 71.2075},
    # Wind sites
    {"name": "Kutch_Wind_Gujarat", "lat": 22.8600, "lon": 69.3300},
    {"name": "Kanyakumari_Wind_Tamil_Nadu", "lat": 8.2600, "lon": 77.5600},
    {"name": "Jaisalmer_Wind_Rajasthan", "lat": 26.9120, "lon": 70.9120},
    # Hydro sites
    {"name": "Tehri_Dam_Uttarakhand", "lat": 30.2240, "lon": 78.2850},
    {"name": "Bhakra_Dam_Himachal_Pradesh", "lat": 31.3827, "lon": 76.4486},
    {"name": "Sardar_Sarovar_Gujarat", "lat": 21.8301, "lon": 73.7501},
]


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=DEMO_DAYS)
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# PATH HELPERS
# ─────────────────────────────────────────────────────────────
def bronze_path(city_name: str) -> Path:
    """data/raw/weather/{City_Name}.csv"""
    return LOCAL_BRONZE / f"{city_name}.csv"


def silver_path(city_name: str) -> Path:
    """data/processed/weather/city={City_Name}/part-0001.parquet"""
    return LOCAL_SILVER / f"city={city_name}" / "part-0001.parquet"


# ─────────────────────────────────────────────────────────────
# FETCH HELPERS
# ─────────────────────────────────────────────────────────────
def month_ranges(start: str, end: str) -> list:
    """Split date range into monthly chunks for Open-Meteo."""
    cur = datetime.strptime(start, "%Y-%m-%d")
    fin = datetime.strptime(end, "%Y-%m-%d")
    ranges = []
    while cur <= fin:
        nxt = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
        mth_end = min(nxt - timedelta(days=1), fin)
        ranges.append((cur.strftime("%Y-%m-%d"), mth_end.strftime("%Y-%m-%d")))
        cur = nxt
    return ranges


def fetch(params: dict, retries: int = 6, wait: int = 8) -> dict | None:
    """Call Open-Meteo API with retry logic."""
    for attempt in range(retries):
        try:
            r = requests.get(OPEN_METEO, params=params, timeout=60)
            if r.status_code != 200:
                time.sleep(wait)
                continue
            data = r.json()
            if data.get("hourly") and data["hourly"].get("time"):
                return data
            time.sleep(wait)
        except Exception as e:
            log.warning(f"  Fetch attempt {attempt+1} failed: {e}")
            time.sleep(wait)
    return None


# ─────────────────────────────────────────────────────────────
# LOAD EXISTING BRONZE
# ─────────────────────────────────────────────────────────────
def load_existing(city_name: str) -> pd.DataFrame:
    """Load existing Bronze CSV to find resume point."""
    path = bronze_path(city_name)
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, parse_dates=["time"])
        df.set_index("time", inplace=True)
        return df
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# SAVE BRONZE  (raw CSV)
# ─────────────────────────────────────────────────────────────
def save_bronze(df: pd.DataFrame, city_name: str):
    """Save raw CSV to Bronze — exactly as fetched from API."""
    path = bronze_path(city_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    log.info(f"  Bronze saved: {path.relative_to(PROJECT_ROOT)}")


# ─────────────────────────────────────────────────────────────
# SAVE SILVER  (cleaned Parquet with lineage columns)
# ─────────────────────────────────────────────────────────────
def save_silver(df: pd.DataFrame, city_name: str, date_str: str):
    """
    Clean and save Silver Parquet.
    Adds lineage columns: city_name, ingestion_date, source_file, pipeline_run_id.
    Resets index so 'time' becomes a regular column — matches weather_suite.json.
    """
    silver = df.copy()

    # Reset index so time is a column
    if silver.index.name == "time":
        silver = silver.reset_index()

    # Ensure correct column names (matches weather_suite.json)
    silver.rename(
        columns={
            "temperature_2m": "temperature",
            "relative_humidity_2m": "humidity",
            "wind_speed_100m": "windspeed_100m",
            "cloudcover": "cloud_cover",
            "precipitation": "rainfall",
        },
        inplace=True,
    )

    # Keep only suite columns + lineage
    keep = [
        "time",
        "temperature",
        "humidity",
        "windspeed_100m",
        "cloud_cover",
        "rainfall",
    ]
    silver = silver[[c for c in keep if c in silver.columns]]

    # Add lineage columns
    silver["city_name"] = city_name
    silver["ingestion_date"] = date_str
    silver["source_file"] = str(bronze_path(city_name))
    silver["pipeline_run_id"] = RUN_ID

    path = silver_path(city_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    silver.to_parquet(path, index=False, compression="snappy")
    log.info(f"  Silver saved: {path.relative_to(PROJECT_ROOT)} ({len(silver)} rows)")


# ─────────────────────────────────────────────────────────────
# SAVE RUN SUMMARY
# ─────────────────────────────────────────────────────────────
def save_summary(results: dict, date_str: str):
    """Save plain JSON summary alongside Bronze files."""
    path = LOCAL_BRONZE / "run_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run_id": RUN_ID,
                "date": date_str,
                "mode": "local",
                "run_at": datetime.utcnow().isoformat(),
                "cities": results,
            },
            indent=2,
        )
    )
    log.info(f"  Summary saved: {path.relative_to(PROJECT_ROOT)}")


# ─────────────────────────────────────────────────────────────
# PROCESS ONE CITY
# ─────────────────────────────────────────────────────────────
def process_city(city: dict, start_str: str, end_str: str) -> dict:
    """
    Full pipeline for one city:
    1. Load existing Bronze CSV (resume-safe)
    2. Fetch missing data from Open-Meteo
    3. Save updated Bronze CSV
    4. Save Silver Parquet with lineage columns
    """
    name = city["name"]
    log.info(f"Processing: {name}")

    # Load existing to find resume point
    existing = load_existing(name)
    if not existing.empty:
        last = existing.index.max()
        resume = (last + pd.Timedelta(hours=1)).strftime("%Y-%m-%d")
        log.info(f"  Resuming from {resume}")
    else:
        resume = start_str

    if resume > end_str:
        log.info("  Already up to date")
        return {"city": name, "status": "SKIPPED", "rows": 0}

    final_df = existing.copy()
    chunks_ok = 0

    for chunk_start, chunk_end in month_ranges(resume, end_str):
        data = fetch(
            {
                "latitude": city["lat"],
                "longitude": city["lon"],
                "timezone": "Asia/Kolkata",
                "start_date": chunk_start,
                "end_date": chunk_end,
                "hourly": HOURLY_FIELDS,
            }
        )

        if data is None:
            log.warning(f"  Chunk failed: {chunk_start} to {chunk_end}")
            continue

        chunk_df = pd.DataFrame(data["hourly"])
        chunk_df.rename(
            columns={
                "temperature_2m": "temperature",
                "relative_humidity_2m": "humidity",
                "wind_speed_100m": "windspeed_100m",
                "cloudcover": "cloud_cover",
                "precipitation": "rainfall",
            },
            inplace=True,
        )
        chunk_df["time"] = pd.to_datetime(chunk_df["time"])
        chunk_df.set_index("time", inplace=True)

        final_df = pd.concat([final_df, chunk_df])
        final_df = final_df[~final_df.index.duplicated(keep="last")]
        final_df.sort_index(inplace=True)

        chunks_ok += 1
        log.info(f"  Chunk done: {chunk_start} to {chunk_end}")
        time.sleep(1)

    if final_df.empty:
        log.warning(f"  No data for {name}")
        return {"city": name, "status": "NO_DATA", "rows": 0}

    # Save Bronze (raw)
    save_bronze(final_df, name)

    # Save Silver (cleaned + lineage)
    save_silver(final_df, name, end_str)

    return {"city": name, "status": "SUCCESS", "rows": len(final_df)}


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    from src.lineage import lineage_run, ds

    args = build_args()
    today = datetime.now().strftime("%Y-%m-%d")
    end_str = args.end or today
    start_str = args.start or (datetime.now() - timedelta(days=args.days)).strftime(
        "%Y-%m-%d"
    )

    log.info("=" * 55)
    log.info("WEATHER INGEST  —  LOCAL MODE")
    log.info(f"Range   : {start_str} to {end_str}  ({args.days} days)")
    log.info(f"Cities  : {len(CITIES)}")
    log.info(f"Run ID  : {RUN_ID[:8]}")
    log.info("Bronze  : data/raw/weather/")
    log.info("Silver  : data/processed/weather/")
    log.info("=" * 55)

    with lineage_run("ingest_weather",
        inputs=[ds("Open-Meteo API", namespace="https://api.open-meteo.com")],
        outputs=[ds("data/raw/weather"), ds("data/processed/weather")],
    ):
        results = {}
        success = 0
        skipped = 0
        failed = 0

        for city in CITIES:
            try:
                result = process_city(city, start_str, end_str)
                results[city["name"]] = result
                if result["status"] == "SUCCESS":
                    success += 1
                elif result["status"] == "SKIPPED":
                    skipped += 1
                else:
                    failed += 1
            except Exception as e:
                log.error(f"  {city['name']} crashed: {e}")
                results[city["name"]] = {"status": "CRASHED", "rows": 0}
                failed += 1

        save_summary(results, end_str)

    log.info("=" * 55)
    log.info(f"Done: {success} success, {skipped} skipped, {failed} failed")
    log.info("=== WEATHER INGEST COMPLETE ===")
    log.info("=" * 55)


if __name__ == "__main__":
    main()
