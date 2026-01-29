# Causal Inference Tutorials

Hands-on Python tutorials for learning causal inference methods with longitudinal/time-varying data.

## Overview

These tutorials use `SEQdata.csv` - a simulated longitudinal dataset with 300 patients followed over up to 60 time points, featuring time-varying treatments and confounders.

## Tutorials

| # | Topic | Key Concepts | File |
|---|-------|--------------|------|
| 1 | Target Trial Emulation | Cloning, ITT effects, sequential trials | `01_target_trial_emulation.py` |
| 2 | Marginal Structural Models | Time-varying confounding, IPTW | `02_msm_iptw.py` (coming soon) |
| 3 | Parametric G-formula | Counterfactual simulation | `03_gformula.py` (coming soon) |
| 4 | Structural Nested Models | G-estimation | `04_snm.py` (coming soon) |

## Dataset Description

**SEQdata.csv** columns:
- `ID`: Patient identifier (1-300)
- `time`: Time point (0-59)
- `eligible`: Eligibility flag (0/1)
- `tx_init`: Treatment initiation at this time (0/1)
- `outcome`: Binary outcome event (0/1)
- `sex`: Sex (0/1)
- `N`, `L`, `P`: Time-varying confounders

## Requirements

```bash
pip install pandas numpy scikit-learn
```

## Usage

```bash
# Run Tutorial 1
python 01_target_trial_emulation.py
```

## Learning Path

1. **Start with Tutorial 1** - understand the target trial framework
2. **Tutorial 2** - learn how MSMs handle time-varying confounding
3. **Tutorial 3** - simulate counterfactual outcomes with g-formula
4. **Tutorial 4** - estimate time-varying treatment effects with SNMs

## References

- Hernán MA, Robins JM. Causal Inference: What If. Chapman & Hall/CRC, 2020.
- Hernán MA, Robins JM. Using Big Data to Emulate a Target Trial When a Randomized Trial Is Not Available. Am J Epidemiol. 2016.

## Author

Learning tutorials for causal inference methods.
