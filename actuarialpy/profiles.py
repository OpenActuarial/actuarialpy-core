"""Light-touch domain profile defaults for actuarial summaries."""

from __future__ import annotations

import pandas as pd

# Profiles should not infer numerator/denominator names. A health numerator may
# include claims, rebates, capitation, and non-FFS expenses; revenue may include
# more than premium. Keep amount labels generic unless users override them.
PROFILE_DEFAULTS = {
    "health": {"ratio_col": "mlr"},
    "casualty": {"ratio_col": "loss_ratio"},
    "life": {"ratio_col": "benefit_ratio"},
}

PROFILE_LABELS: dict[str, dict[str, str]] = {name: {} for name in PROFILE_DEFAULTS}


def get_profile_defaults(profile: str | None = None) -> dict[str, str]:
    """Return non-label defaults for a supported profile."""
    if profile is None:
        return {}
    if profile not in PROFILE_DEFAULTS:
        raise ValueError(f"Unknown profile: {profile}. Valid profiles are: {sorted(PROFILE_DEFAULTS)}")
    return PROFILE_DEFAULTS[profile].copy()


def get_profile_labels(profile: str | None = None, labels: dict[str, str] | None = None) -> dict[str, str]:
    """Return user-requested output rename labels.

    Profiles no longer rename total_expense or total_revenue automatically.
    """
    if profile is not None and profile not in PROFILE_DEFAULTS:
        raise ValueError(f"Unknown profile: {profile}. Valid profiles are: {sorted(PROFILE_DEFAULTS)}")
    return dict(labels or {})


def apply_profile_labels(df: pd.DataFrame, *, profile: str | None = None, labels: dict[str, str] | None = None) -> pd.DataFrame:
    """Rename output columns using explicit user labels only."""
    return df.rename(columns=get_profile_labels(profile, labels))
