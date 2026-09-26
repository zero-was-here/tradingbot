"""
Economic Calendar Integration

Tracks scheduled economic events that cause market volatility.

The AI must know when NFP, CPI, FOMC, and other high-impact events are coming.
These events can cause 100-300 pip moves in minutes.

Data sources:
- ForexFactory.com
- Investing.com calendar
- Federal Reserve schedule
- Manual entry for known events
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EconomicCalendar:
    """
    Tracks scheduled economic events and their market impact

    Features:
    - High-impact event tracking (NFP, CPI, FOMC, etc.)
    - Time-until-event features
    - Expected volatility estimation
    - Event window detection
    """

    # High-impact USD events (most important for XAUUSD)
    HIGH_IMPACT_EVENTS = [
        'Non-Farm Payrolls',
        'NFP',
        'CPI',
        'Consumer Price Index',
        'Inflation',
        'FOMC',
        'Federal Reserve',
        'Fed Decision',
        'Interest Rate Decision',
        'GDP',
        'Unemployment',
        'Retail Sales',
        'Fed Chair',
        'Powell Speech',
        'Yellen Speech',
    ]

    # Expected volatility for each event type (multiplier of normal volatility)
    EVENT_VOLATILITY_MAP = {
        'NFP': 2.0,  # Non-Farm Payrolls: 100-200 pip moves
        'CPI': 1.8,  # Inflation: 80-150 pips
        'FOMC': 2.5,  # Fed decision: 150-300 pips
        'Fed Speech': 1.3,  # Fed Chair speech: 50-100 pips
        'GDP': 1.5,  # GDP release: 60-120 pips
        'Retail Sales': 1.2,  # 40-80 pips
        'Unemployment': 1.4,  # 50-100 pips
        'Interest Rate': 2.0,  # 100-200 pips
    }

    def __init__(self, calendar_file=None):
        """
        Initialize economic calendar

        Args:
            calendar_file: Path to JSON file with events (optional)
        """
        self.calendar_file = calendar_file or "data/economic_events.json"
        self.events = self.load_calendar()

        logger.info(f"📅 Economic Calendar initialized with {len(self.events)} events")

    def load_calendar(self):
        """
        Load economic events

        File format (JSON):
        [
            {
                "datetime": "2024-01-05 13:30:00",
                "event": "Non-Farm Payrolls",
                "currency": "USD",
                "impact": "HIGH",
                "forecast": "200K",
                "previous": "190K"
            },
            ...
        ]
        """

        # Check if file exists
        if Path(self.calendar_file).exists():
            with open(self.calendar_file, 'r') as f:
                events = json.load(f)

            # Convert datetime strings to datetime objects
            for event in events:
                event['datetime'] = datetime.fromisoformat(event['datetime'])

            logger.info(f"📅 Loaded {len(events)} events from {self.calendar_file}")
            return events
        else:
            logger.warning(f"⚠️ Calendar file not found: {self.calendar_file}")
            logger.info("⚠️ Using default calendar (2024 major events)")
            return self.get_default_2024_calendar()

    def get_default_2024_calendar(self):
        """
        Default calendar with major 2024 USD events

        This is a fallback - in production, scrape from ForexFactory or use API
        """

        events = []

        # NFP: First Friday of each month at 13:30 UTC
        nfp_months = range(1, 13)
        for month in nfp_months:
            # Find first Friday of the month
            first_day = datetime(2024, month, 1, 13, 30)
            # Find first Friday
            days_until_friday = (4 - first_day.weekday()) % 7
            if days_until_friday == 0 and first_day.day > 1:
                days_until_friday = 7
            nfp_date = first_day + timedelta(days=days_until_friday)

            events.append({
                'datetime': nfp_date,
                'event': 'Non-Farm Payrolls',
                'currency': 'USD',
                'impact': 'HIGH',
                'forecast': None,
                'previous': None,
            })

        # CPI: Mid-month (typically 13th-14th) at 13:30 UTC
        for month in range(1, 13):
            cpi_date = datetime(2024, month, 13, 13, 30)
            events.append({
                'datetime': cpi_date,
                'event': 'CPI',
                'currency': 'USD',
                'impact': 'HIGH',
                'forecast': None,
                'previous': None,
            })

        # FOMC: 8 meetings per year
        fomc_dates = [
            datetime(2024, 1, 31, 19, 0),
            datetime(2024, 3, 20, 19, 0),
            datetime(2024, 5, 1, 19, 0),
            datetime(2024, 6, 12, 19, 0),
            datetime(2024, 7, 31, 19, 0),
            datetime(2024, 9, 18, 19, 0),
            datetime(2024, 11, 7, 19, 0),
            datetime(2024, 12, 18, 19, 0),
        ]

        for fomc_date in fomc_dates:
            events.append({
                'datetime': fomc_date,
                'event': 'FOMC Rate Decision',
                'currency': 'USD',
                'impact': 'HIGH',
                'forecast': None,
                'previous': None,
            })

        logger.info(f"📅 Created default calendar with {len(events)} events")
        return events

    def get_features(self, current_time):
        """
        Get calendar features for a specific timestamp

        Returns dict with:
        - days_until_event: Days until next high-impact event
        - hours_until_event: Hours until next event
        - is_high_impact: 1 if next event is high-impact
        - is_event_window: 1 if within 1 hour of event
        - is_nfp: 1 if next event is NFP
        - is_cpi: 1 if next event is CPI
        - is_fomc: 1 if next event is FOMC
        - event_volatility_forecast: Expected volatility multiplier
        """

        # Convert to datetime if needed
        if isinstance(current_time, str):
            current_time = pd.to_datetime(current_time)
        elif isinstance(current_time, pd.Timestamp):
            current_time = current_time.to_pydatetime()

        # Find upcoming events
        upcoming_events = [e for e in self.events if e['datetime'] > current_time]

        if not upcoming_events:
            # No upcoming events - return neutral features
            return self._default_features()

        # Get next event
        next_event = min(upcoming_events, key=lambda e: e['datetime'])
        time_until = (next_event['datetime'] - current_time).total_seconds()

        # Build features
        features = {
            'days_until_event': time_until / 86400,  # Convert to days
            'hours_until_event': time_until / 3600,  # Convert to hours
            'is_high_impact': 1.0 if next_event.get('impact') == 'HIGH' else 0.0,
            'is_event_window': 1.0 if abs(time_until) < 3600 else 0.0,  # Within 1 hour

            # Event type (one-hot encoding)
            'is_nfp': 1.0 if self._is_event_type(next_event, 'NFP') else 0.0,
            'is_cpi': 1.0 if self._is_event_type(next_event, 'CPI') else 0.0,
            'is_fomc': 1.0 if self._is_event_type(next_event, 'FOMC') else 0.0,
            'is_fed_speech': 1.0 if self._is_event_type(next_event, 'Fed') else 0.0,

            # Expected volatility
            'event_volatility_forecast': self._estimate_volatility(next_event),

            # Time bucketing (useful for patterns)
            'event_in_24h': 1.0 if time_until < 86400 else 0.0,
            'event_in_1h': 1.0 if time_until < 3600 else 0.0,
        }

        return features

    def _default_features(self):
        """Return neutral features when no upcoming events"""
        return {
            'days_until_event': 30.0,  # Far in future
            'hours_until_event': 720.0,
            'is_high_impact': 0.0,
            'is_event_window': 0.0,
            'is_nfp': 0.0,
            'is_cpi': 0.0,
            'is_fomc': 0.0,
            'is_fed_speech': 0.0,
            'event_volatility_forecast': 1.0,  # Normal volatility
            'event_in_24h': 0.0,
            'event_in_1h': 0.0,
        }

    def _is_event_type(self, event, event_type):
        """Check if event matches type"""
        event_name = event['event'].upper()
        return event_type.upper() in event_name

    def _estimate_volatility(self, event):
        """
        Estimate expected volatility based on historical reactions

        Returns multiplier (1.0 = normal, 2.0 = double normal volatility)
        """

        event_name = event['event'].upper()

        for key, vol_multiplier in self.EVENT_VOLATILITY_MAP.items():
            if key.upper() in event_name:
                return vol_multiplier

        # Unknown event - assume moderate impact
        if event.get('impact') == 'HIGH':
            return 1.5
        else:
            return 1.0

    def add_event(self, datetime_str, event_name, currency='USD', impact='HIGH'):
        """
        Manually add an event

        Args:
            datetime_str: "2024-01-05 13:30:00"
            event_name: "Non-Farm Payrolls"
            currency: "USD"
            impact: "HIGH", "MEDIUM", or "LOW"
        """

        event = {
            'datetime': datetime.fromisoformat(datetime_str),
            'event': event_name,
            'currency': currency,
            'impact': impact,
            'forecast': None,
            'previous': None,
        }

        self.events.append(event)
        logger.info(f"➕ Added event: {event_name} on {datetime_str}")

    def save_calendar(self, filename=None):
        """Save calendar to JSON file"""

        filename = filename or self.calendar_file

        # Convert datetime to string for JSON
        events_serializable = []
        for event in self.events:
            event_copy = event.copy()
            event_copy['datetime'] = event_copy['datetime'].isoformat()
            events_serializable.append(event_copy)

        with open(filename, 'w') as f:
            json.dump(events_serializable, f, indent=2)

        logger.info(f"💾 Saved {len(self.events)} events to {filename}")

    def get_upcoming_events(self, current_time, days_ahead=7):
        """Get list of upcoming events in next N days"""

        if isinstance(current_time, str):
            current_time = pd.to_datetime(current_time)
        elif isinstance(current_time, pd.Timestamp):
            current_time = current_time.to_pydatetime()

        end_time = current_time + timedelta(days=days_ahead)

        upcoming = [
            e for e in self.events
            if current_time < e['datetime'] <= end_time
        ]

        # Sort by datetime
        upcoming.sort(key=lambda e: e['datetime'])

        return upcoming


def add_calendar_features_to_dataframe(df, calendar=None):
    """
    Add economic calendar features to a dataframe

    Args:
        df: DataFrame with 'time' column
        calendar: EconomicCalendar instance (will create if None)

    Returns:
        df with added calendar features
    """

    if calendar is None:
        calendar = EconomicCalendar()

    logger.info("📅 Adding economic calendar features to dataframe...")

    # Initialize feature columns
    feature_names = [
        'days_until_event', 'hours_until_event', 'is_high_impact',
        'is_event_window', 'is_nfp', 'is_cpi', 'is_fomc', 'is_fed_speech',
        'event_volatility_forecast', 'event_in_24h', 'event_in_1h'
    ]

    for feat in feature_names:
        df[feat] = 0.0

    # Compute features for each timestamp
    for idx, row in df.iterrows():
        timestamp = row['time']
        features = calendar.get_features(timestamp)

        for feat_name, feat_value in features.items():
            df.loc[idx, feat_name] = feat_value

    logger.info(f"✅ Added {len(feature_names)} calendar features")

    return df


# Example usage and testing
if __name__ == "__main__":
    print("📅 Economic Calendar Demo\n")

    # Create calendar
    calendar = EconomicCalendar()

    # Test: Get features for a specific date
    test_date = datetime(2024, 1, 5, 12, 0)  # Day before NFP
    features = calendar.get_features(test_date)

    print(f"Features for {test_date}:")
    for key, value in features.items():
        print(f"  {key}: {value}")

    # Show upcoming events
    print(f"\n📅 Upcoming events in next 30 days:")
    upcoming = calendar.get_upcoming_events(test_date, days_ahead=30)

    for event in upcoming:
        print(f"  {event['datetime'].strftime('%Y-%m-%d %H:%M')} - {event['event']} ({event['impact']})")

    # Save calendar
    calendar.save_calendar("data/economic_events.json")
    print(f"\n💾 Saved calendar to data/economic_events.json")

    print("\n✅ Economic Calendar working correctly!")
