"""
4_query_online_store.py
=======================
Directly queries the SQLite online store to inspect materialized features.

Two modes:
  1. Raw SQLite queries — see exactly what's stored in the DB
  2. Feast API queries  — how your serving code would fetch features

Usage:
    python src/3_feature_engineering/4_query_online_store.py
"""

import sqlite3
from pathlib import Path

import pandas as pd
from feast import FeatureStore

# This script lives in src/3_feature_engineering/, but the Feast repo (feature_store.yaml,
# registry, and the online-store DB) lives in my_feature_store/feature_repo/. Resolve the
# store paths there so the script works from its new location.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEAST_REPO = PROJECT_ROOT / "my_feature_store" / "feature_repo"
DB_PATH = FEAST_REPO / "data" / "online_store.db"


# -------------------------------------------------------------
# 1. RAW SQLITE QUERIES
# -------------------------------------------------------------

def query_sqlite():
    print("=" * 70)
    print("1. RAW SQLITE — What's inside online_store.db")
    print("=" * 70)

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # List all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"\nTables ({len(tables)}):")
    for t in tables:
        cursor.execute(f"SELECT COUNT(*) FROM [{t}]")
        count = cursor.fetchone()[0]
        print(f"  {t:50s} -> {count:,} rows")

    # Schema of each table
    print("\nTable schemas:")
    for t in tables:
        cursor.execute(f"PRAGMA table_info([{t}])")
        cols = cursor.fetchall()
        print(f"\n  {t}:")
        for col in cols:
            print(f"    {col[1]:30s} {col[2]}")

    # Sample rows from each table
    print("\nSample rows (first 3 per table):")
    for t in tables:
        cursor.execute(f"SELECT * FROM [{t}] LIMIT 3")
        rows = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description]
        print(f"\n  {t}:")
        print(f"    Columns: {col_names}")
        for row in rows:
            # Truncate long binary values for display
            display = []
            for val in row:
                if isinstance(val, bytes) and len(val) > 50:
                    display.append(f"<bytes:{len(val)}B>")
                else:
                    display.append(val)
            print(f"    {display}")

    conn.close()


# -------------------------------------------------------------
# 2. FEAST API QUERIES
# -------------------------------------------------------------

def query_feast_api():
    print("\n" + "=" * 70)
    print("2. FEAST API — Online feature retrieval")
    print("=" * 70)

    store = FeatureStore(repo_path=str(FEAST_REPO))

    # All features across all views
    all_features = [
        "market_features:dam_mcp",
        "market_features:dam_purchase_bid",
        "market_features:dam_sell_bid",
        "market_features:dam_mcv",
        "market_features:dam_volume",
        "market_features:dam_bid_imbalance",
        "market_features:rtm_purchase_bid",
        "market_features:rtm_sell_bid",
        "market_features:rtm_mcv",
        "market_features:rtm_volume",
        "market_features:mcp_spread",
        "market_features:dam_mcp_lag_1d",
        "market_features:dam_mcp_lag_2d",
        "market_features:dam_mcp_lag_7d",
        "market_features:dam_mcp_roll_4h",
        "market_features:dam_mcp_roll_24h",
        "market_features:dam_mcp_roll_std_24h",
        "weather_features:avg_temp",
        "weather_features:avg_humidity",
        "weather_features:avg_windspeed",
        "weather_features:avg_cloud_cover",
        "weather_features:total_rainfall",
        "calendar_features:hour_sin",
        "calendar_features:hour_cos",
        "calendar_features:dow_sin",
        "calendar_features:dow_cos",
        "calendar_features:block_sin",
        "calendar_features:block_cos",
        "calendar_features:day_of_month",
        "calendar_features:is_weekend",
        "calendar_features:is_peak_hour",
        "calendar_features:is_morning_ramp",
        "calendar_features:is_off_peak",
        "calendar_features:is_ipl_match",
        "calendar_features:is_event",
        "calendar_features:is_festival",
        "calendar_features:is_wedding_season",
        "calendar_features:impact",
    ]

    # Test blocks — different times and scenarios
    test_blocks = [
        {"block_id": "2025-03-02T18:00", "label": "First block (Sun evening peak)"},
        {"block_id": "2025-03-10T08:00", "label": "Weekday morning ramp"},
        {"block_id": "2025-03-15T14:30", "label": "Weekend afternoon"},
        {"block_id": "2025-03-20T18:00", "label": "Weekday peak hour"},
        {"block_id": "2025-03-25T09:00", "label": "IPL day morning"},
        {"block_id": "2025-03-31T23:45", "label": "Last block (IPL night)"},
        {"block_id": "2025-01-01T00:00", "label": "Non-existent block (should be null)"},
    ]

    entity_rows = [{"block_id": b["block_id"]} for b in test_blocks]
    labels = [b["label"] for b in test_blocks]

    result = store.get_online_features(
        features=all_features,
        entity_rows=entity_rows,
    ).to_dict()

    # Display per-block results
    for i, label in enumerate(labels):
        print(f"\n  [{result['block_id'][i]}] — {label}")
        print(f"  {'-' * 60}")

        # Market
        dam_mcp = result["dam_mcp"][i]
        if dam_mcp is None:
            print("    WARNING: No features found (block not in store)")
            continue

        print("    Market:")
        print(f"      dam_mcp:         {result['dam_mcp'][i]:>10.2f} Rs/MWh")
        print(f"      dam_purchase_bid:{result['dam_purchase_bid'][i]:>10.2f} MW")
        print(f"      dam_sell_bid:    {result['dam_sell_bid'][i]:>10.2f} MW")
        print(f"      mcp_spread:      {result['mcp_spread'][i]:>10.2f}")
        print(f"      lag_1d:          {result['dam_mcp_lag_1d'][i]:>10.2f}")
        print(f"      lag_7d:          {result['dam_mcp_lag_7d'][i]:>10.2f}")
        print(f"      roll_24h:        {result['dam_mcp_roll_24h'][i]:>10.2f}")

        # Weather
        print("    Weather:")
        print(f"      avg_temp:        {result['avg_temp'][i]:>10.1f} C")
        print(f"      avg_humidity:    {result['avg_humidity'][i]:>10.1f} %")
        print(f"      avg_windspeed:   {result['avg_windspeed'][i]:>10.1f} km/h")
        print(f"      total_rainfall:  {result['total_rainfall'][i]:>10.1f} mm")

        # Calendar
        print("    Calendar:")
        print(f"      is_peak_hour:    {result['is_peak_hour'][i]:>10d}")
        print(f"      is_weekend:      {result['is_weekend'][i]:>10d}")
        print(f"      is_ipl_match:    {result['is_ipl_match'][i]:>10d}")
        print(f"      is_festival:     {result['is_festival'][i]:>10d}")
        print(f"      is_morning_ramp: {result['is_morning_ramp'][i]:>10d}")

    # Null check — count missing features across all blocks
    print(f"\n  Null check across {len(test_blocks)} blocks:")
    for key, vals in result.items():
        nulls = sum(1 for v in vals if v is None)
        if nulls > 0:
            print(f"    {key:30s} -> {nulls} nulls")


# -------------------------------------------------------------
# 3. COMPARE OFFLINE vs ONLINE
# -------------------------------------------------------------

def compare_offline_online():
    print("\n" + "=" * 70)
    print("3. OFFLINE vs ONLINE — Consistency check")
    print("=" * 70)

    store = FeatureStore(repo_path=str(FEAST_REPO))

    check_blocks = ["2025-03-10T08:00", "2025-03-20T18:00", "2025-03-31T23:45"]
    features = [
        "market_features:dam_mcp",
        "market_features:dam_mcp_lag_1d",
        "weather_features:avg_temp",
        "calendar_features:is_peak_hour",
    ]

    # Offline
    entity_df = pd.DataFrame({
        "block_id": check_blocks,
        "event_timestamp": pd.to_datetime([b.replace("T", " ") for b in check_blocks]),
    })
    offline_df = store.get_historical_features(
        entity_df=entity_df, features=features,
    ).to_df()

    # Online
    online_result = store.get_online_features(
        features=features,
        entity_rows=[{"block_id": b} for b in check_blocks],
    ).to_dict()

    print(f"\n  {'block_id':>20s} | {'source':>8s} | {'dam_mcp':>10s} | {'lag_1d':>10s} | {'temp':>8s} | {'peak':>6s}")
    print(f"  {'-' * 20}-+{'-' * 10}+{'-' * 12}+{'-' * 12}+{'-' * 10}+{'-' * 8}")

    for i, block in enumerate(check_blocks):
        off_row = offline_df[offline_df["block_id"] == block].iloc[0]
        print(f"  {block:>20s} | {'offline':>8s} | {off_row['dam_mcp']:>10.2f} | {off_row['dam_mcp_lag_1d']:>10.2f} | {off_row['avg_temp']:>8.1f} | {int(off_row['is_peak_hour']):>6d}")
        print(f"  {'':>20s} | {'online':>8s} | {online_result['dam_mcp'][i]:>10.2f} | {online_result['dam_mcp_lag_1d'][i]:>10.2f} | {online_result['avg_temp'][i]:>8.1f} | {online_result['is_peak_hour'][i]:>6d}")

        match = (
            abs(off_row["dam_mcp"] - online_result["dam_mcp"][i]) < 0.01
            and abs(off_row["avg_temp"] - online_result["avg_temp"][i]) < 0.01
        )
        print(f"  {'':>20s} | {'MATCH' if match else 'MISMATCH':>8s} |")

    print()


def main():
    print(f"Online store: {DB_PATH}")
    print(f"Size: {DB_PATH.stat().st_size / 1024:.0f} KB\n")

    query_sqlite()
    query_feast_api()
    compare_offline_online()

    print("=" * 70)
    print("All checks complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
