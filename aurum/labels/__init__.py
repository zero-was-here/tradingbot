"""Supervised-learning targets (SPEC §6 ``aurum.labels``).

**Everything here looks forward by construction.** Labels are training targets for a
strategy's ``fit()`` on its TRAINING slice only; they must never feed features, forecasts,
sizing or risk. See :mod:`aurum.labels.triple_barrier` for the details.
"""

from aurum.labels.triple_barrier import (
    BARRIERS,
    LABEL_COLUMNS,
    apply_triple_barrier,
    average_uniqueness,
    cusum_filter,
    drop_label_tail,
    ewm_vol,
    fixed_horizon_labels,
    get_events,
    label_end_positions,
    meta_labels,
    num_concurrent_events,
    return_attribution_weights,
    time_decay_weights,
    triple_barrier_labels,
    uniqueness_weights,
)

__all__ = [
    "BARRIERS",
    "LABEL_COLUMNS",
    "apply_triple_barrier",
    "average_uniqueness",
    "cusum_filter",
    "drop_label_tail",
    "ewm_vol",
    "fixed_horizon_labels",
    "get_events",
    "label_end_positions",
    "meta_labels",
    "num_concurrent_events",
    "return_attribution_weights",
    "time_decay_weights",
    "triple_barrier_labels",
    "uniqueness_weights",
]
