# Changelog

## 0.6.0

### Added

- **Credibility module** (`actuarialpy.credibility`), exposed at the top level.
  - `Buhlmann` and `BuhlmannStraub` greatest-accuracy credibility models —
    **moved from the `lossmodels` package** (their behavior is unchanged; the
    test suite is ported verbatim to confirm parity). This relocation puts
    credibility next to the experience and ratemaking workflows that consume it,
    and lets `lossmodels` step back from being a complete Klugman implementation
    toward being a Klugman-anchored loss-distribution toolkit.
  - `credibility_weighted_estimate(observed, complement, z)` — the type-stable
    credibility-weighting primitive (scalar → float, Series → Series, array →
    ndarray), for blending an experience estimate with its complement at a `Z`
    from any source.

## 0.5.0

Corrections and consistency pass. Two items change output and are called out as
behavior changes; the rest are additive, fixes, or tooling.

### Behavior changes

- **Metric primitives now preserve pandas types.** `safe_divide` (and every
  metric that builds on it, e.g. `loss_ratio`, `pmpm`, `ratio`,
  `actual_to_expected`) returns a `pandas.Series` with the original index and
  name when given a Series, instead of a bare `numpy.ndarray`. Scalar inputs
  return a native `float`. Series are still combined positionally and are
  assumed aligned. Code that relied on receiving an `ndarray` from Series input
  should call `.to_numpy()` explicitly.
- **`rolling_summary` is now calendar-aware.** The window is measured in
  calendar periods rather than rows. Each group is reindexed onto a dense,
  gap-free period grid (filled with zero amounts/exposure) before rolling, so a
  missing month no longer stretches the window across extra calendar time, and
  `period_start`/`period_end` are correct in the presence of gaps. A new `freq`
  argument (pandas offset alias, e.g. `"MS"`) controls the grid; it is inferred
  when omitted, and gapped data requires an explicit `freq`.

### Fixes

- `combined_ratio` no longer coerces scalar inputs to `numpy.float64`; scalar in
  → `float` out, matching the other primitives. Python lists are still summed
  element-wise rather than concatenated.
- `make_completion_triangle` had a docstring promising a "cumulative" triangle
  while only summing within cells. The docstring now states the snapshot
  assumption, and a new `cumulative=True` argument accumulates incremental
  amounts across lag.
- `as_list` now returns set inputs in a deterministic (string-sorted) order
  instead of arbitrary iteration order, since these become group-by keys and
  column orders.

### Additions

- Most workflow functions are re-exported from the top level, so
  `from actuarialpy import summarize_experience, rolling_summary, trend_summary`
  works without knowing the submodule layout.
- `relative_change` is the single canonical `a / b - 1` implementation;
  `percent_change`, `variance_pct`, and `period_change` delegate to it.
- A shared `EXPOSURE_SUFFIX` map and `per_exposure_name` helper in `columns`
  replace the per-exposure naming logic previously duplicated across
  `experience`, `rolling`, and `components` (output names unchanged).

### Tooling

- Added a `py.typed` marker so the existing type hints are visible to
  downstream type checkers.
- Added `ruff` and `mypy` configuration and a GitHub Actions CI workflow that
  runs the test suite, the linter, and the type checker across Python
  3.9–3.12.
- README package structure synced with the actual modules (it had listed a
  nonexistent `validation.py` and omitted several real modules).
