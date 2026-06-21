# ActuarialPy

**ActuarialPy** is a primitive-based Python toolkit for general actuarial analysis.

The package is organized bottom-up: small actuarial functions can be used directly, while larger summary, trend, rolling, cohort, completion, and reporting functions compose those primitives. It intentionally avoids wrapping ordinary pandas operations unless the wrapper adds actuarial logic or validation.

## Install locally

```bash
pip install -e .
```

## Modules

```text
actuarialpy/
├── metrics.py       # ratios, PMPM/PSPM, A/E, frequency, severity, adequacy
├── columns.py       # validation helpers, not general pandas wrappers
├── periods.py       # month/quarter/year periods and durations
├── completion.py    # completion math, completed claims, IBNR, simple triangles
├── profiles.py      # light profile defaults such as health ratio name = mlr
├── experience.py    # grouped summaries and multi-view summaries
├── components.py    # component/category summaries and driver analysis
├── compare.py       # variance, percent change, basis point change
├── trend.py         # trend factors, projection, current/prior trend summaries
├── rolling.py       # rolling 12 or other rolling-window summaries
├── cohorts.py       # first-year, cohort, and duration summaries
├── forecast.py      # simple expected values and actual-vs-expected comparisons
└── reporting.py     # Excel workbook output
```

## Core primitives

```python
from actuarialpy import loss_ratio, pmpm, actual_to_expected

loss_ratio(850_000, 1_000_000)
# 0.85

pmpm(1_000_000, 2_000)
# 500.0

actual_to_expected(1_100_000, 1_000_000)
# 1.10
```

## Completion factors and IBNR

Join factor tables using pandas directly, especially when the factor table has multiple factor columns.

```python
claims = claims.merge(
    factors,
    on=["line_of_business", "incurred_date"],
    how="left",
    validate="many_to_one",
)
```

Then use ActuarialPy for the actuarial completion math.

```python
from actuarialpy.completion import complete_claim_components

claims = complete_claim_components(
    claims,
    {
        "inpatient_claims": "inpatient_completion_factor",
        "outpatient_claims": "outpatient_completion_factor",
        "professional_claims": "professional_completion_factor",
        "pharmacy_claims": "pharmacy_completion_factor",
    },
)
```

This adds columns such as:

```text
inpatient_claims_completed
inpatient_claims_ibnr
outpatient_claims_completed
outpatient_claims_ibnr
```

## Experience summary

For member-level monthly data, create a true exposure column first.

```python
claims["member_months"] = 1
claims["total_expense"] = claims[
    [
        "inpatient_claims_completed",
        "outpatient_claims_completed",
        "professional_claims_completed",
        "pharmacy_claims_completed",
        "pharmacy_rebates",
        "non_ffs_expenses",
    ]
].sum(axis=1)
```

Then summarize.

```python
from actuarialpy.experience import summarize_experience

summary = summarize_experience(
    claims,
    groupby=["group_id", "product_code"],
    expense_cols=["total_expense"],
    revenue_cols=["premium"],
    exposure_cols=["member_months"],
    profile="health",  # sets ratio column to mlr
)
```

Default output labels remain generic:

```text
total_expense
total_revenue
mlr
expense_pmpm
revenue_pmpm
```

If you want claim/premium-specific labels, pass them explicitly:

```python
summary = summarize_experience(
    claims,
    groupby=["group_id"],
    expense_cols=["completed_claims"],
    revenue_cols=["premium"],
    total_expense_name="total_claims",
    total_revenue_name="total_premium",
    ratio_name="mlr",
)
```

## Component view

```python
from actuarialpy.components import summarize_components

components = summarize_components(
    claims,
    groupby=["incurred_date"],
    component_cols=[
        "inpatient_claims_completed",
        "outpatient_claims_completed",
        "professional_claims_completed",
        "pharmacy_claims_completed",
        "pharmacy_rebates",
        "non_ffs_expenses",
    ],
    exposure_col="member_months",
)
```

## Component driver analysis

```python
from actuarialpy.components import component_driver_analysis

claims["year"] = claims["incurred_date"].dt.year

drivers = component_driver_analysis(
    claims,
    period_col="year",
    prior_period=2025,
    current_period=2026,
    groupby=["product_code"],
    component_cols=[
        "inpatient_claims_completed",
        "outpatient_claims_completed",
        "professional_claims_completed",
        "pharmacy_claims_completed",
        "pharmacy_rebates",
        "non_ffs_expenses",
    ],
    exposure_col="member_months",
)
```

This explains which categories drove the total PMPM change.

## Trend summary

```python
from actuarialpy.trend import trend_summary

trend = trend_summary(
    claims,
    period_col="year",
    prior_period=2025,
    current_period=2026,
    groupby=["product_code"],
    amount_col="total_expense",
    exposure_col="member_months",
)
```

## Rolling view

```python
from actuarialpy.rolling import rolling_summary

rolling = rolling_summary(
    claims,
    date_col="incurred_date",
    window=12,
    expense_cols=["total_expense"],
    revenue_cols=["premium"],
    exposure_cols=["member_months"],
)
```

Rolling summaries include:

```text
period_start
period_end
total_expense
total_revenue
member_months
mlr
expense_pmpm
revenue_pmpm
```

Incomplete windows are omitted by default.

## Cohort / duration view

```python
from actuarialpy.cohorts import cohort_summary

first_year = cohort_summary(
    claims,
    entity_col="group_id",
    date_col="incurred_date",
    start_date_col="effective_date",
    duration_months=12,
    expense_cols=["total_expense"],
    revenue_cols=["premium"],
    exposure_cols=["member_months"],
)
```

## Forecast and actual-to-expected

```python
from actuarialpy.forecast import forecast_experience, compare_actual_to_expected

forecast = forecast_experience(
    claims,
    rate_col="base_claims_pmpm",
    exposure_col="member_months",
    annual_trend=0.07,
    months_forward=12,
)

comparison = compare_actual_to_expected(
    actual=actual_summary,
    expected=forecast,
    on=["group_id", "incurred_date"],
    actual_col="total_expense",
    expected_col="expected_expense",
)
```

## Excel report output

```python
from actuarialpy.reporting import to_excel_report

to_excel_report(
    {
        "group_product": group_product,
        "monthly": monthly,
        "rolling_12": rolling,
        "components": components,
    },
    "experience_report.xlsx",
)
```
