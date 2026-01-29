# Causal Inference Methods for Real-World Evidence

**Training Series for Epidemiologists and RWE Data Scientists**

A hands-on Python tutorial series for learning causal inference methods applicable to observational studies, pharmacoepidemiology, and real-world evidence (RWE) research.

## Overview

These tutorials are designed as training materials for teams working with longitudinal observational data. Each module builds on the previous, progressing from foundational concepts to advanced methods for handling time-varying confounding.

**Target Audience:**
- Epidemiologists and pharmacoepidemiologists
- RWE/HEOR scientists
- Biostatisticians working with observational data
- Researchers transitioning from clinical trials to RWE

## Training Modules

| Module | Topic | Key Concepts | Estimated Time |
|--------|-------|--------------|----------------|
| 1 | [Target Trial Emulation](01_target_trial_emulation.py) | Cloning, ITT effects, sequential trials | 45 min |
| 2 | [Marginal Structural Models](02_msm_iptw.py) | IPTW, time-varying confounding, weighted regression | 60 min |
| 3 | [Parametric G-formula](03_gformula.py) | Counterfactual simulation, treatment strategies | 60 min |
| 4 | [Structural Nested Models](04_snm_gestimation.py) | G-estimation, blip functions, effect heterogeneity | 75 min |

## Method Selection Guide

| Research Question | Recommended Method |
|-------------------|-------------------|
| "What is the effect of **initiating** treatment?" | Target Trial Emulation |
| "What is the marginal effect of treatment history?" | MSM with IPTW |
| "What would outcomes be under strategy X vs Y?" | Parametric G-formula |
| "What is the **direct effect** at each time point?" | Structural Nested Models |
| "Does the effect vary by subgroup?" | SNM with effect modification |

## Dataset

**SEQdata.csv** - Simulated longitudinal cohort data analogous to claims/EHR databases:

| Variable | Description | Type |
|----------|-------------|------|
| `ID` | Patient identifier | Integer (1-300) |
| `time` | Time point | Integer (0-59) |
| `eligible` | Eligibility flag | Binary |
| `tx_init` | Treatment at this time | Binary |
| `outcome` | Event indicator | Binary |
| `sex` | Sex | Binary |
| `N`, `L`, `P` | Time-varying confounders | Continuous |

## Prerequisites

### Knowledge
- Basic epidemiology (confounding, bias, study design)
- Regression modeling (linear, logistic)
- Familiarity with causal diagrams (DAGs) helpful but not required

### Technical
```bash
pip install pandas numpy scikit-learn scipy statsmodels
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/geethamalika/causal-inference-tutorials.git
cd causal-inference-tutorials

# Run tutorials in order
python 01_target_trial_emulation.py
python 02_msm_iptw.py
python 03_gformula.py
python 04_snm_gestimation.py
```

## Key Assumptions (All Methods)

1. **No unmeasured confounding** (Sequential exchangeability)
2. **Positivity** (All treatment levels possible for all covariate patterns)
3. **Consistency** (Well-defined interventions)
4. **Correct model specification**

## Regulatory Context

These methods align with:
- FDA guidance on RWE for regulatory decisions
- ICH E9(R1) addendum on estimands
- EMA guidance on non-interventional studies
- ISPE/ISPOR guidelines for comparative effectiveness

## References

### Foundational Texts
1. Hernán MA, Robins JM. *Causal Inference: What If.* Chapman & Hall/CRC, 2020. [Free online](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/)
2. Rothman KJ, Greenland S, Lash TL. *Modern Epidemiology.* 3rd ed. Lippincott Williams & Wilkins, 2008.

### Method-Specific Papers
- **Target Trials:** Hernán MA, Robins JM. Using Big Data to Emulate a Target Trial. *Am J Epidemiol.* 2016.
- **MSMs:** Robins JM, Hernán MA, Brumback B. Marginal Structural Models. *Epidemiology.* 2000.
- **G-formula:** Robins JM. A new approach to causal inference. *Computers and Mathematics.* 1986.
- **SNMs:** Vansteelandt S, Joffe M. Structural Nested Models. In: *Handbook of Causal Analysis.* 2014.

### Reporting Guidelines
- STROBE Statement for observational studies
- RECORD Extension for routinely collected data
- ISPE Guidelines for Good Pharmacoepidemiology Practices

## Learning Path Recommendations

**Week 1:** Complete Tutorials 1-2, focus on understanding confounding
**Week 2:** Complete Tutorials 3-4, compare methods on same data
**Week 3:** Apply to your own data, start with simplest applicable method
**Ongoing:** Sensitivity analyses, assumption checking, peer review

## Contributing

Contributions welcome! Please submit issues for:
- Bug reports
- Clarification requests
- Additional examples
- Method extensions

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

These tutorials draw on the causal inference curriculum developed at Harvard T.H. Chan School of Public Health and the extensive methodological work of James Robins, Miguel Hernán, and colleagues.

---

*Developed for training epidemiology and RWE teams in causal inference methods for observational studies.*
