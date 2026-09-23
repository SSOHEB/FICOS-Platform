"""
SIH26006 — GDELT Geopolitical Event Feature Engineering

Transforms raw geopolitical_events (51.7k GDELT records) into daily
aggregated features for the forecasting pipeline.

Design principles:
- event_date is parsed from the actual event record, never inferred from filename
- GoldsteinScale is ONE input among several, not equated to "severity"
- Separate transparent features: conflict_intensity, event_volume,
  media_attention, tone, geographic_relevance, domain_relevance
- All rules are deterministic and documented
- No manual labeling of individual events
- All rolling features are backward-looking (no leakage)

The GDELT dataset is pre-filtered to conflict-related CAMEO codes (14-20),
QuadClass 3-4, GoldsteinScale -10 to -4. This is accounted for in the
feature design.
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import load_geopolitical_events, load_config


# ──────────────────────────────────────────────
# CAMEO EventRootCode mapping (codes present in data: 14-20)
# ──────────────────────────────────────────────
# Intensity tiers based on CAMEO ontology, not arbitrary assignment
INTENSITY_TIER = {
    14: 1,  # PROTEST — low physical intensity
    15: 2,  # EXHIBIT FORCE — medium
    16: 1,  # REDUCE RELATIONS — low physical, diplomatic
    17: 2,  # COERCE — medium
    18: 3,  # ASSAULT — high physical intensity
    19: 3,  # FIGHT — high physical intensity
    20: 3,  # USE UNCONVENTIONAL MASS VIOLENCE — highest
}

CAMEO_LABELS = {
    14: "protest",
    15: "exhibit_force",
    16: "reduce_relations",
    17: "coerce",
    18: "assault",
    19: "fight",
    20: "mass_violence",
}


def build_event_level_features(df):
    """
    Add transparent event-level features. Each feature is a separate
    dimension — they are NOT collapsed into a single score at this stage.

    Features:
    - conflict_intensity_tier: 1/2/3 from CAMEO code
    - media_attention: composite of NumMentions, NumSources, NumArticles
    - tone_negativity: inverse of AvgTone (more negative = higher)
    - geographic_relevance: whether event involves route-relevant countries
    - domain_relevance_shipping: proxy for shipping/trade relevance
    """
    cfg = load_config()

    # 1. Conflict intensity tier from CAMEO EventRootCode
    df["conflict_intensity_tier"] = df["EventRootCode"].map(INTENSITY_TIER).fillna(1).astype(int)

    # 2. Media attention — normalized composite
    # Each component is rank-normalized to [0,1] to avoid scale dominance
    for col in ["NumMentions", "NumSources", "NumArticles"]:
        rank_col = f"_{col}_rank"
        df[rank_col] = df[col].rank(pct=True)

    df["media_attention"] = (
        df["_NumMentions_rank"] * 0.4 +
        df["_NumSources_rank"] * 0.3 +
        df["_NumArticles_rank"] * 0.3
    )
    df.drop(columns=["_NumMentions_rank", "_NumSources_rank", "_NumArticles_rank"], inplace=True)

    # 3. Tone negativity — more negative AvgTone = higher negativity
    # AvgTone in this dataset is always negative (conflict events)
    # Normalize: shift so more negative = higher value
    df["tone_negativity"] = -df["AvgTone"]  # simple inversion; higher = more negative tone

    # 4. Geographic relevance
    route_countries = set(cfg.get("route_relevant_countries", []))
    maritime_countries = set(cfg.get("maritime_relevant_countries", []))
    all_relevant = route_countries | maritime_countries

    # Check ActionGeo, Actor1, Actor2 country codes
    df["geo_action_relevant"] = df["ActionGeo_CountryCode"].isin(all_relevant).astype(int)
    df["geo_actor1_relevant"] = df["Actor1CountryCode"].fillna("").isin(all_relevant).astype(int)
    df["geo_actor2_relevant"] = df["Actor2CountryCode"].fillna("").isin(all_relevant).astype(int)

    # Combined geographic relevance: any geo field matches
    df["geographic_relevance"] = (
        df["geo_action_relevant"] | df["geo_actor1_relevant"] | df["geo_actor2_relevant"]
    ).astype(int)

    # Route-specific: India-centric events
    df["india_relevant"] = (
        (df["ActionGeo_CountryCode"] == "IN") |
        (df["Actor1CountryCode"] == "IN") |
        (df["Actor2CountryCode"] == "IN")
    ).astype(int)

    # Maritime chokepoint relevance
    df["maritime_chokepoint_relevant"] = (
        df["ActionGeo_CountryCode"].isin(maritime_countries)
    ).astype(int)

    # 5. GoldsteinScale as a raw feature (not relabeled as "severity")
    # Already in [-10, -4] range for this dataset
    df["goldstein_magnitude"] = df["GoldsteinScale"].abs()

    # Clean up intermediate geo columns
    df.drop(columns=["geo_action_relevant", "geo_actor1_relevant", "geo_actor2_relevant"],
            inplace=True)

    return df


def aggregate_daily(df):
    """
    Aggregate event-level features to daily time series.
    Each feature family is aggregated separately and transparently.
    """
    daily = df.groupby("event_date").agg(
        # Volume features
        gdelt_event_count=("GLOBALEVENTID", "count"),
        gdelt_geo_relevant_count=("geographic_relevance", "sum"),
        gdelt_india_relevant_count=("india_relevant", "sum"),
        gdelt_maritime_relevant_count=("maritime_chokepoint_relevant", "sum"),

        # Conflict intensity features
        gdelt_high_intensity_count=("conflict_intensity_tier", lambda x: (x == 3).sum()),
        gdelt_med_intensity_count=("conflict_intensity_tier", lambda x: (x == 2).sum()),
        gdelt_low_intensity_count=("conflict_intensity_tier", lambda x: (x == 1).sum()),
        gdelt_mean_intensity=("conflict_intensity_tier", "mean"),

        # Media attention features
        gdelt_total_mentions=("NumMentions", "sum"),
        gdelt_total_sources=("NumSources", "sum"),
        gdelt_total_articles=("NumArticles", "sum"),
        gdelt_mean_media_attention=("media_attention", "mean"),
        gdelt_max_media_attention=("media_attention", "max"),

        # Tone features
        gdelt_mean_tone=("AvgTone", "mean"),
        gdelt_min_tone=("AvgTone", "min"),  # most negative
        gdelt_mean_tone_negativity=("tone_negativity", "mean"),
        gdelt_max_tone_negativity=("tone_negativity", "max"),

        # GoldsteinScale features (separate from intensity tier)
        gdelt_mean_goldstein=("GoldsteinScale", "mean"),
        gdelt_min_goldstein=("GoldsteinScale", "min"),  # most negative
        gdelt_mean_goldstein_magnitude=("goldstein_magnitude", "mean"),
        gdelt_max_goldstein_magnitude=("goldstein_magnitude", "max"),
    ).reset_index()

    daily.rename(columns={"event_date": "date"}, inplace=True)
    return daily


def add_persistence_features(daily, windows=None):
    """
    Add backward-looking rolling features for persistence analysis.
    For date t, rolling window uses [t-w+1, t] — no future information.
    """
    if windows is None:
        cfg = load_config()
        windows = cfg.get("gdelt_persistence_windows", [3, 7, 14, 30])

    daily = daily.sort_values("date").reset_index(drop=True)

    # Key features to create rolling versions of
    roll_cols = [
        "gdelt_event_count",
        "gdelt_geo_relevant_count",
        "gdelt_india_relevant_count",
        "gdelt_high_intensity_count",
        "gdelt_mean_intensity",
        "gdelt_total_mentions",
        "gdelt_mean_tone_negativity",
        "gdelt_mean_goldstein_magnitude",
    ]

    for w in windows:
        for col in roll_cols:
            if col in daily.columns:
                # Rolling sum for counts, rolling mean for averages
                if "count" in col or "total" in col:
                    daily[f"{col}_sum_{w}d"] = (
                        daily[col].rolling(window=w, min_periods=1).sum()
                    )
                else:
                    daily[f"{col}_ma_{w}d"] = (
                        daily[col].rolling(window=w, min_periods=1).mean()
                    )

    return daily


def build_gdelt_features(save=True):
    """
    Full GDELT feature pipeline:
    1. Load raw events
    2. Build event-level features
    3. Aggregate to daily
    4. Add persistence (rolling) features
    5. Save to outputs/
    """
    print("Loading geopolitical events...")
    df = load_geopolitical_events()
    print(f"  Loaded {len(df)} events")
    print(f"  ACTUAL date range: {df['event_date'].min()} -> {df['event_date'].max()}")
    print(f"  QuadClasses: {sorted(df['QuadClass'].unique())}")
    print(f"  EventRootCodes: {sorted(df['EventRootCode'].unique())}")
    print(f"  GoldsteinScale range: {df['GoldsteinScale'].min()} to {df['GoldsteinScale'].max()}")

    print("\nBuilding event-level features...")
    df = build_event_level_features(df)

    print("Aggregating to daily features...")
    daily = aggregate_daily(df)
    print(f"  Days with events: {len(daily)}")
    print(f"  Unique event dates: {daily['date'].nunique()}")

    # Expand to full date range — fill non-event days with 0
    # This is critical: GDELT events are clustered on ~26 dates.
    # Rolling windows need the full calendar to work correctly.
    full_dates = pd.DataFrame({
        "date": pd.date_range(start="2016-01-01", end="2026-09-07", freq="D")
    })
    daily = full_dates.merge(daily, on="date", how="left")
    fill_cols = [c for c in daily.columns if c != "date"]
    daily[fill_cols] = daily[fill_cols].fillna(0)
    print(f"  After calendar expansion: {daily.shape}")

    print("Adding persistence (rolling) features...")
    daily = add_persistence_features(daily)
    print(f"  Final daily features: {daily.shape}")

    if save:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_path = os.path.join(root, "outputs", "gdelt_daily_features.csv")
        daily.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")

    return daily


if __name__ == "__main__":
    build_gdelt_features()
