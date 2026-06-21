# ActuarialPy

**ActuarialPy** is a Python package for general actuarial analysis.

The initial version includes a basic loss ratio calculation. Future modules may include experience summaries, trend tools, actual-to-expected analysis, exposure-based metrics, validation, and reporting.

## Installation

```bash
pip install actuarialpy
```

## Usage

```python
from actuarialpy import loss_ratio

lr = loss_ratio(expenses=850_000, revenue=1_000_000)

print(lr)
# 0.85
```

## Loss ratio

```python
loss_ratio(expenses, revenue)
```

Calculates:

```text
loss ratio = expenses / revenue
```

Examples of use:

- health MLR: claims / premium
- P&C loss ratio: losses / earned premium
- combined-style ratios: losses plus expenses / revenue, if expenses are pre-summed before calling the function

## Development

Install locally in editable mode:

```bash
pip install -e .
```

Run tests:

```bash
pip install pytest
pytest
```

Build for PyPI:

```bash
python -m pip install --upgrade build twine
python -m build
twine check dist/*
twine upload dist/*
```
