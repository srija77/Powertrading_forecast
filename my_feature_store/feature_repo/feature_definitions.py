"""
Feast feature definitions for March 2025 Energy Market features.

Entity:  block_id  — one 15-min time block (e.g. "2025-03-15T14:30")
Source:  data/march_2025_features.parquet (prepared by prepare_feast_data.py)

Three FeatureViews split by domain:
  1. market_features    — DAM/RTM bids, volumes, MCP, spreads, lags
  2. weather_features   — temperature, humidity, wind, cloud, rain
  3. calendar_features  — festivals, IPL, elections, time encodings
"""

from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field, FileSource
from feast.types import Float64, Int64, String

# ── Entity ──────────────────────────────────────────────────
block = Entity(
    name="block_id",
    join_keys=["block_id"],
    description="15-minute time block identifier (e.g. 2025-03-15T14:30)",
)

# ── Source ──────────────────────────────────────────────────
source = FileSource(
    path="data/march_2025_features.parquet",
    timestamp_field="event_timestamp",
)

# ── FeatureView 1: Market Features ─────────────────────────
market_features = FeatureView(
    name="market_features",
    entities=[block],
    ttl=timedelta(days=365),
    schema=[
        # DAM
        Field(name="dam_mcp", dtype=Float64),
        Field(name="dam_purchase_bid", dtype=Float64),
        Field(name="dam_sell_bid", dtype=Float64),
        Field(name="dam_mcv", dtype=Float64),
        Field(name="dam_volume", dtype=Float64),
        Field(name="dam_bid_imbalance", dtype=Float64),
        # RTM
        Field(name="rtm_purchase_bid", dtype=Float64),
        Field(name="rtm_sell_bid", dtype=Float64),
        Field(name="rtm_mcv", dtype=Float64),
        Field(name="rtm_volume", dtype=Float64),
        # Spread
        Field(name="mcp_spread", dtype=Float64),
        # Lag / rolling
        Field(name="dam_mcp_lag_1d", dtype=Float64),
        Field(name="dam_mcp_lag_2d", dtype=Float64),
        Field(name="dam_mcp_lag_7d", dtype=Float64),
        Field(name="dam_mcp_roll_4h", dtype=Float64),
        Field(name="dam_mcp_roll_24h", dtype=Float64),
        Field(name="dam_mcp_roll_std_24h", dtype=Float64),
    ],
    source=source,
)

# ── FeatureView 2: Weather Features ────────────────────────
weather_features = FeatureView(
    name="weather_features",
    entities=[block],
    ttl=timedelta(days=365),
    schema=[
        Field(name="avg_temp", dtype=Float64),
        Field(name="avg_humidity", dtype=Float64),
        Field(name="avg_windspeed", dtype=Float64),
        Field(name="avg_cloud_cover", dtype=Float64),
        Field(name="total_rainfall", dtype=Float64),
    ],
    source=source,
)

# ── FeatureView 3: Calendar & Time Features ─────────────────
calendar_features = FeatureView(
    name="calendar_features",
    entities=[block],
    ttl=timedelta(days=365),
    schema=[
        # Cyclical time encodings
        Field(name="hour_sin", dtype=Float64),
        Field(name="hour_cos", dtype=Float64),
        Field(name="dow_sin", dtype=Float64),
        Field(name="dow_cos", dtype=Float64),
        Field(name="block_sin", dtype=Float64),
        Field(name="block_cos", dtype=Float64),
        # Time flags
        Field(name="day_of_month", dtype=Int64),
        Field(name="is_weekend", dtype=Int64),
        Field(name="is_peak_hour", dtype=Int64),
        Field(name="is_morning_ramp", dtype=Int64),
        Field(name="is_off_peak", dtype=Int64),
        # Events (derived from calendar text columns)
        Field(name="is_ipl_match", dtype=Int64),
        Field(name="is_event", dtype=Int64),
        Field(name="is_festival", dtype=Int64),
        Field(name="is_wedding_season", dtype=Int64),
        Field(name="impact", dtype=Float64),
    ],
    source=source,
)

# ── FeatureService: the model's feature bundle ──────────────
# SINGLE SOURCE OF TRUTH for "which features the dam_mcp_forecast model uses".
# A FeatureService groups feature views into one named, versioned bundle. Every
# layer references this by name instead of re-listing the feature columns:
#   - train.py / evaluate.py / predict.py -> store.get_historical_features(features=fs)
#   - app.py / predict.py --online        -> store.get_online_features(features=fs)
# Add a feature to a view above and it flows to all layers automatically.
# Bump to dam_mcp_forecast_v2 when you want a new bundle without breaking v1.
dam_mcp_forecast_v1 = FeatureService(
    name="dam_mcp_forecast_v1",
    features=[market_features, weather_features, calendar_features],
)
