"""
Intraday time-of-day and day-of-week seasonality features.

Provides cyclical encoding of market session timing and expiry day detection
for NSE (National Stock Exchange) trading hours.
"""

import math
from typing import Dict


def time_of_day_features(hour: int, minute: int) -> Dict:
    """
    Compute cyclical time-of-day features for NSE market hours.

    NSE market operates 9:15 to 15:30 (375 minutes total). Encodes the time
    progression as cyclical features suitable for machine learning models.

    Args:
        hour: Hour in 24-hour format (0-23)
        minute: Minute (0-59)

    Returns:
        Dict with keys:
            - time_sin: Sine component of time progression (range [-1, 1])
            - time_cos: Cosine component of time progression (range [-1, 1])
            - is_opening_rush: 1 if 9:15-9:45, else 0
            - is_lunch_lull: 1 if 12:00-13:00, else 0
            - is_closing_auction: 1 if 15:00-15:30, else 0
            - market_fraction: Fraction of market day elapsed [0, 1]
    """
    # NSE market: 9:15 to 15:30 = 375 minutes
    market_open_hour, market_open_minute = 9, 15
    market_close_hour, market_close_minute = 15, 30

    minutes_since_open = (hour - market_open_hour) * 60 + (minute - market_open_minute)

    # Clamp to valid market hours
    minutes_since_open = max(0, min(375, minutes_since_open))

    # Normalize to [0, 1]
    market_fraction = minutes_since_open / 375.0

    # Cyclical encoding: 2π × fraction maps to a full circle
    angle = 2 * math.pi * market_fraction
    time_sin = math.sin(angle)
    time_cos = math.cos(angle)

    # Time zone detection
    is_opening_rush = 1 if (hour == 9 and minute >= 15 and minute < 45) or (
        hour == 9 and minute == 45
    ) else 0
    is_lunch_lull = 1 if hour == 12 or (hour == 13 and minute == 0) else 0
    is_closing_auction = 1 if hour == 15 and minute >= 0 and minute <= 30 else 0

    return {
        "time_sin": time_sin,
        "time_cos": time_cos,
        "is_opening_rush": is_opening_rush,
        "is_lunch_lull": is_lunch_lull,
        "is_closing_auction": is_closing_auction,
        "market_fraction": market_fraction,
    }


def day_of_week_feature(weekday: int) -> Dict:
    """
    Compute cyclical day-of-week features for NSE expiry tracking.

    Encodes day of week as cyclical features. NSE F&O expires on Thursdays.

    Args:
        weekday: Day of week (0=Monday, 1=Tuesday, ..., 4=Friday)

    Returns:
        Dict with keys:
            - dow_sin: Sine component of day progression (range [-1, 1])
            - dow_cos: Cosine component of day progression (range [-1, 1])
            - is_expiry_day: 1 if Thursday (contracts expire), else 0
    """
    # Clamp weekday to valid range
    weekday = max(0, min(4, weekday))

    # Cyclical encoding: 2π × (weekday / 5) maps a week to a circle
    angle = 2 * math.pi * (weekday / 5.0)
    dow_sin = math.sin(angle)
    dow_cos = math.cos(angle)

    # NSE F&O expiry is on Thursday (weekday == 3)
    is_expiry_day = 1 if weekday == 3 else 0

    return {
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "is_expiry_day": is_expiry_day,
    }


def is_no_trade_zone(hour: int, minute: int) -> bool:
    """
    Identify low-liquidity time windows where trading should be avoided.

    Returns True for lunch break (12:00-13:00) and market open gap (9:15-9:20)
    where liquidity is depressed and spreads widen.

    Args:
        hour: Hour in 24-hour format (0-23)
        minute: Minute (0-59)

    Returns:
        True if in a no-trade zone, False otherwise
    """
    # Lunch break: 12:00 to 13:00
    if hour == 12 or (hour == 13 and minute == 0):
        return True

    # Opening gap: first 5 minutes after market open (9:15 to 9:20)
    if hour == 9 and minute >= 15 and minute < 20:
        return True

    return False
