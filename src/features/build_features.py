import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_target(df: pd.DataFrame, delay_col: str = "arr_delay", threshold: int = 15) -> pd.DataFrame:
    df = df.copy()
    df["is_delayed"] = (df[delay_col] > threshold)
    logger.info(f"Target 'is_delayed' created — delay rate: {df['is_delayed'].mean():.1%}")
    return df


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Extract hour from scheduled departure/arrival (stored as integers like 1430 = 14:30)
    df["dep_hour"] = df["crs_dep_time"] // 100
    df["arr_hour"] = df["crs_arr_time"] // 100

    # Bin departure hour into time-of-day buckets
    # EDA showed evening flights have the highest delay rates
    def hour_to_bin(hour: int) -> str:
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"

    df["dep_hour_bin"] = df["dep_hour"].apply(hour_to_bin)

    # Operations peak — hours with above-average number of flights (airport congestion)
    df["is_ops_peak"] = df["dep_hour"].isin(range(6, 21))

    # Delay peak — hours with above-average delay rate per EDA
    df["is_delay_peak"] = df["dep_hour"].isin(range(13, 24))

    # Weekend flag — EDA showed Fridays and Sundays have higher delay rates
    df["is_weekend"] = df["day_of_week"].isin([5, 6, 7])

    # High delay months — Jan (winter storms), May (start of travel season),
    # Jun/Jul/Aug (peak summer travel), Dec (holiday travel)
    df["is_holiday_month"] = df["month"].isin([1, 5, 6, 7, 8, 12])

    logger.info("Temporal features created: dep_hour, arr_hour, dep_hour_bin, is_ops_peak, is_delay_peak, is_weekend, is_holiday_month")
    return df


def create_log_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Log-transform late departures to compress the right skew
    # clip(lower=0) sets early departures to 0 before log — their signal is captured separately below
    df["dep_delay_log"] = np.log1p(df["dep_delay"].clip(lower=0))

    # Capture early departure signal separately — information that dep_delay_log throws away
    # A flight departing 20 min early is very different from one departing exactly on time
    # (-dep_delay).clip(lower=0) turns negative delays into positive early-minutes, rest become 0
    df["dep_early"] = (-df["dep_delay"]).clip(lower=0)

    logger.info("Log features created: dep_delay_log, dep_early")
    return df


def create_route_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Combine origin and destination into a single route string
    # Some routes are consistently more delay-prone than others
    df["route"] = df["origin"] + "_" + df["dest"]

    logger.info("Route feature created: route")
    return df


def create_aggregation_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Average historical dep_delay per carrier — captures airline reliability
    carrier_avg = df.groupby("op_unique_carrier")["dep_delay"].transform("mean")
    df["carrier_avg_delay"] = carrier_avg

    # Average historical dep_delay per origin airport — captures airport congestion
    origin_avg = df.groupby("origin")["dep_delay"].transform("mean")
    df["origin_avg_delay"] = origin_avg

    # Average historical dep_delay per route
    route_avg = df.groupby("route")["dep_delay"].transform("mean")
    df["route_avg_delay"] = route_avg

    logger.info("Aggregation features created: carrier_avg_delay, origin_avg_delay, route_avg_delay")
    return df


def drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    cols_to_drop = [col for col in columns if col in df.columns]
    df = df.drop(columns=cols_to_drop)
    logger.info(f"Dropped {len(cols_to_drop)} columns: {cols_to_drop}")
    return df


def build_all_features(df: pd.DataFrame, cols_to_drop: list[str]) -> pd.DataFrame:
    logger.info("Starting feature engineering")
    df = create_target(df)
    df = create_temporal_features(df)
    df = create_log_features(df)
    df = create_route_feature(df)
    df = create_aggregation_features(df)
    df = drop_columns(df, cols_to_drop)
    logger.info(f"Feature engineering complete — final shape: {df.shape}")
    return df
