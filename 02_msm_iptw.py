"""
=============================================================================
TUTORIAL 2: Marginal Structural Models (MSM) with IPTW
=============================================================================

LEARNING OBJECTIVES:
1. Understand time-varying confounding and why standard regression fails
2. Learn how to calculate time-varying propensity scores
3. Build stabilized inverse probability weights
4. Fit a marginal structural model using weighted regression
5. Interpret the causal effect estimate

DATASET: SEQdata.csv (same as Tutorial 1)

THE PROBLEM:
We want to estimate the causal effect of a TIME-VARYING treatment.
- Treatment can start, stop, and restart over time
- Confounders (L, N, P) change over time
- These confounders are affected by PAST treatment (treatment-confounder feedback)

WHY STANDARD REGRESSION FAILS:
If we adjust for L at time t:
  - We block the confounding path (good!)
  - But L_t may be affected by treatment at t-1
  - Adjusting for L_t blocks part of the treatment effect (bad!)

This is the "time-varying confounding affected by prior treatment" problem.

THE SOLUTION: MARGINAL STRUCTURAL MODELS
Instead of adjusting for confounders in the outcome model,
we WEIGHT observations to create a pseudo-population where
treatment is independent of confounders.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# STEP 1: UNDERSTAND THE DATA STRUCTURE
# =============================================================================

print("=" * 70)
print("STEP 1: Understanding Time-Varying Treatment")
print("=" * 70)

df = pd.read_csv('SEQdata.csv')

print("""
In Tutorial 1, we looked at treatment INITIATION (starting treatment).
Now we look at treatment STATUS at each time point.

Key difference:
- Tutorial 1: "Does STARTING treatment affect outcome?"
- Tutorial 2: "Does BEING ON treatment (over time) affect outcome?"
""")

# Look at treatment patterns
print("\n--- Treatment Patterns ---")
patient_5 = df[df['ID'] == 5][['ID', 'time', 'tx_init', 'outcome', 'L', 'N']]
print("Patient 5's trajectory:")
print(patient_5.head(15).to_string(index=False))

# Count treatment switches
def count_switches(patient_df):
    tx = patient_df['tx_init'].values
    return sum(tx[i] != tx[i-1] for i in range(1, len(tx)))

switches = [count_switches(df[df['ID']==pid]) for pid in df['ID'].unique()]
print(f"\nTreatment switches per patient: mean={np.mean(switches):.1f}, max={max(switches)}")

print("""
OBSERVATION:
- Treatment (tx_init) can switch on and off multiple times
- This is TIME-VARYING treatment
- We need to account for the HISTORY of treatment, not just current status
""")

# =============================================================================
# STEP 2: THE TIME-VARYING CONFOUNDING PROBLEM
# =============================================================================

print("\n" + "=" * 70)
print("STEP 2: The Time-Varying Confounding Problem")
print("=" * 70)

print("""
CAUSAL DIAGRAM (simplified):

    L(t-1) -----> L(t) -----> L(t+1)
      |            |            |
      v            v            v
    A(t-1) -----> A(t) -----> A(t+1)
      |            |            |
      +------------+------------+---> Y
      
Where:
- L(t) = time-varying confounder at time t
- A(t) = treatment at time t  
- Y = outcome

THE PROBLEM:
- L(t) confounds the A(t) -> Y relationship (we should adjust)
- But A(t-1) affects L(t) (treatment-confounder feedback)
- If we adjust for L(t), we block part of the A(t-1) -> Y effect!

SOLUTION: Weight by the inverse probability of treatment history.
""")

# Show evidence of treatment-confounder feedback
print("--- Evidence of Treatment-Confounder Feedback ---")

# Create lagged treatment
df_sorted = df.sort_values(['ID', 'time'])
df_sorted['tx_lag1'] = df_sorted.groupby('ID')['tx_init'].shift(1)
df_sorted = df_sorted.dropna(subset=['tx_lag1'])

# Check if past treatment predicts current L
mean_L_after_tx = df_sorted[df_sorted['tx_lag1'] == 1]['L'].mean()
mean_L_after_no_tx = df_sorted[df_sorted['tx_lag1'] == 0]['L'].mean()

print(f"Mean L after treatment at t-1:    {mean_L_after_tx:.4f}")
print(f"Mean L after no treatment at t-1: {mean_L_after_no_tx:.4f}")
print(f"Difference: {mean_L_after_tx - mean_L_after_no_tx:.4f}")

print("\nThis shows past treatment affects the confounder L!")

# =============================================================================
# STEP 3: CALCULATE TIME-VARYING PROPENSITY SCORES
# =============================================================================

print("\n" + "=" * 70)
print("STEP 3: Time-Varying Propensity Scores")
print("=" * 70)

print("""
For MSM, we need to model P(A(t) = a | past treatment, past confounders)

At each time t, we model:
  P(tx_init(t) = 1 | tx_init(t-1), L(t), N(t), P(t), ...)

This is the probability of receiving the treatment you actually received,
given your history.
""")

# Prepare data for propensity score modeling
df_ps = df.copy()
df_ps = df_ps.sort_values(['ID', 'time'])

# Create lagged variables
df_ps['tx_lag1'] = df_ps.groupby('ID')['tx_init'].shift(1)
df_ps['L_lag1'] = df_ps.groupby('ID')['L'].shift(1)
df_ps['N_lag1'] = df_ps.groupby('ID')['N'].shift(1)

# Drop first time point (no lag available)
df_ps = df_ps.dropna(subset=['tx_lag1'])

print(f"Analysis dataset: {len(df_ps):,} person-time observations")

# Fit propensity score model for treatment
# P(A(t) = 1 | history)
ps_covariates = ['tx_lag1', 'L', 'N', 'P', 'L_lag1', 'sex', 'time']
X_ps = df_ps[ps_covariates]
y_ps = df_ps['tx_init']

ps_model = LogisticRegression(max_iter=1000, random_state=42)
ps_model.fit(X_ps, y_ps)

# Predicted probability of treatment = 1
df_ps['ps_treated'] = ps_model.predict_proba(X_ps)[:, 1]

# Probability of receiving what you actually received
df_ps['ps_actual'] = np.where(
    df_ps['tx_init'] == 1,
    df_ps['ps_treated'],
    1 - df_ps['ps_treated']
)

print("\n--- Propensity Score Distribution ---")
print(f"P(A=1|history) for treated:   mean={df_ps.loc[df_ps['tx_init']==1, 'ps_treated'].mean():.3f}")
print(f"P(A=1|history) for untreated: mean={df_ps.loc[df_ps['tx_init']==0, 'ps_treated'].mean():.3f}")

# =============================================================================
# STEP 4: BUILD STABILIZED IPTW WEIGHTS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 4: Building Stabilized IPTW Weights")
print("=" * 70)

print("""
UNSTABILIZED WEIGHT at time t:
  w(t) = 1 / P(A(t) | history)

CUMULATIVE WEIGHT for person i up to time t:
  W_i(t) = product of w(s) for s = 0 to t

PROBLEM: Cumulative weights can get very large/unstable!

SOLUTION: STABILIZED WEIGHTS
  sw(t) = P(A(t) | baseline covariates) / P(A(t) | full history)

The numerator is the "marginal" probability (less conditioning).
This keeps weights closer to 1 while maintaining unbiasedness.
""")

# Fit numerator model (stabilization)
# P(A(t) = 1 | baseline covariates only)
num_covariates = ['tx_lag1', 'sex', 'time']  # Simpler model
X_num = df_ps[num_covariates]

num_model = LogisticRegression(max_iter=1000, random_state=42)
num_model.fit(X_num, y_ps)

df_ps['ps_num'] = num_model.predict_proba(X_num)[:, 1]
df_ps['ps_num_actual'] = np.where(
    df_ps['tx_init'] == 1,
    df_ps['ps_num'],
    1 - df_ps['ps_num']
)

# Calculate stabilized weight at each time point
df_ps['sw_t'] = df_ps['ps_num_actual'] / df_ps['ps_actual']

print("--- Weight at each time point ---")
print(f"Mean: {df_ps['sw_t'].mean():.3f}")
print(f"Min:  {df_ps['sw_t'].min():.3f}")
print(f"Max:  {df_ps['sw_t'].max():.3f}")

# Calculate CUMULATIVE stabilized weights
# W_i(t) = product of sw(s) for s = 1 to t
def calculate_cumulative_weights(group):
    """Calculate cumulative product of weights within each person."""
    group = group.sort_values('time')
    group['sw_cumulative'] = group['sw_t'].cumprod()
    return group

df_ps = df_ps.groupby('ID', group_keys=False).apply(calculate_cumulative_weights)

print("\n--- Cumulative Stabilized Weights ---")
print(f"Mean: {df_ps['sw_cumulative'].mean():.3f}")
print(f"Median: {df_ps['sw_cumulative'].median():.3f}")
print(f"Min:  {df_ps['sw_cumulative'].min():.3f}")
print(f"Max:  {df_ps['sw_cumulative'].max():.3f}")
print(f"99th percentile: {df_ps['sw_cumulative'].quantile(0.99):.3f}")

# Truncate extreme weights
weight_cap = df_ps['sw_cumulative'].quantile(0.99)
df_ps['sw_truncated'] = df_ps['sw_cumulative'].clip(upper=weight_cap)

print(f"\n--- After Truncation (cap at {weight_cap:.2f}) ---")
print(f"Mean: {df_ps['sw_truncated'].mean():.3f}")
print(f"Max:  {df_ps['sw_truncated'].max():.3f}")

# =============================================================================
# STEP 5: FIT THE MARGINAL STRUCTURAL MODEL
# =============================================================================

print("\n" + "=" * 70)
print("STEP 5: Fitting the Marginal Structural Model")
print("=" * 70)

print("""
THE MSM APPROACH:
1. Use the WEIGHTED data (pseudo-population)
2. Fit a simple outcome model: E[Y | A] (no confounders!)
3. In the pseudo-population, A is independent of L (confounders are "balanced")

We fit: logit(P(Y=1)) = β₀ + β₁*A

β₁ is interpreted as the causal log-odds ratio of treatment on outcome.
""")

# For MSM, we typically use the FINAL observation per person
# (when outcome is observed)
df_final = df_ps.groupby('ID').last().reset_index()

print(f"Final observations: {len(df_final)} patients")
print(f"Outcomes: {df_final['outcome'].sum()} events ({df_final['outcome'].mean()*100:.1f}%)")

# Method 1: Unweighted (naive) analysis
print("\n--- Method 1: Unweighted (Naive) Analysis ---")
X_naive = sm.add_constant(df_final['tx_init'])
y_out = df_final['outcome']

model_naive = sm.GLM(y_out, X_naive, family=sm.families.Binomial())
result_naive = model_naive.fit()

coef_naive = result_naive.params['tx_init']
or_naive = np.exp(coef_naive)
print(f"Log-odds ratio: {coef_naive:.4f}")
print(f"Odds Ratio:     {or_naive:.4f}")
print(f"95% CI:         ({np.exp(result_naive.conf_int().loc['tx_init', 0]):.4f}, "
      f"{np.exp(result_naive.conf_int().loc['tx_init', 1]):.4f})")

# Method 2: Weighted (MSM) analysis
print("\n--- Method 2: IPTW-Weighted (MSM) Analysis ---")
model_msm = sm.GLM(y_out, X_naive, family=sm.families.Binomial(),
                   freq_weights=df_final['sw_truncated'])
result_msm = model_msm.fit()

coef_msm = result_msm.params['tx_init']
or_msm = np.exp(coef_msm)
print(f"Log-odds ratio: {coef_msm:.4f}")
print(f"Odds Ratio:     {or_msm:.4f}")
print(f"95% CI:         ({np.exp(result_msm.conf_int().loc['tx_init', 0]):.4f}, "
      f"{np.exp(result_msm.conf_int().loc['tx_init', 1]):.4f})")

# Method 3: Adjusted regression (for comparison - this is BIASED!)
print("\n--- Method 3: Adjusted Regression (BIASED - for comparison) ---")
X_adj = sm.add_constant(df_final[['tx_init', 'L', 'N', 'P', 'sex']])
model_adj = sm.GLM(y_out, X_adj, family=sm.families.Binomial())
result_adj = model_adj.fit()

coef_adj = result_adj.params['tx_init']
or_adj = np.exp(coef_adj)
print(f"Log-odds ratio: {coef_adj:.4f}")
print(f"Odds Ratio:     {or_adj:.4f}")
print("(This is biased due to treatment-confounder feedback!)")

# =============================================================================
# STEP 6: UNDERSTANDING THE RESULTS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 6: Interpreting the Results")
print("=" * 70)

print("""
COMPARISON OF METHODS:
""")

print(f"{'Method':<35} {'Odds Ratio':<15} {'Interpretation'}")
print("-" * 70)
print(f"{'Naive (unweighted)':<35} {or_naive:<15.3f} Confounded")
print(f"{'Adjusted regression':<35} {or_adj:<15.3f} Biased (blocks causal path)")
print(f"{'MSM with IPTW':<35} {or_msm:<15.3f} Causal estimate")

print("""
INTERPRETATION:
- OR < 1: Treatment REDUCES the odds of outcome
- OR > 1: Treatment INCREASES the odds of outcome  
- OR = 1: No effect

The MSM estimate accounts for:
✓ Confounding by L, N, P
✓ Treatment-confounder feedback
✓ Time-varying nature of treatment

WHY MSM WORKS:
In the weighted pseudo-population:
- Treatment is independent of confounders
- We can estimate E[Y(a)] for each treatment level a
- The contrast E[Y(1)] - E[Y(0)] is causal
""")

# =============================================================================
# STEP 7: WEIGHT DIAGNOSTICS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 7: Weight Diagnostics")
print("=" * 70)

print("""
IMPORTANT: Always check your weights!
- Mean should be close to 1
- Extreme weights indicate positivity violations
- Large variance reduces precision
""")

print("\n--- Weight Distribution ---")
print(f"Mean:   {df_ps['sw_truncated'].mean():.3f} (should be ~1)")
print(f"Std:    {df_ps['sw_truncated'].std():.3f}")
print(f"Min:    {df_ps['sw_truncated'].min():.3f}")
print(f"Max:    {df_ps['sw_truncated'].max():.3f}")

# Check covariate balance after weighting
print("\n--- Covariate Balance (L) After Weighting ---")

# Unweighted difference
diff_unweighted = (df_ps.loc[df_ps['tx_init']==1, 'L'].mean() - 
                   df_ps.loc[df_ps['tx_init']==0, 'L'].mean())

# Weighted difference
def weighted_mean(values, weights):
    return np.average(values, weights=weights)

treated_mask = df_ps['tx_init'] == 1
control_mask = df_ps['tx_init'] == 0

L_treated_weighted = weighted_mean(df_ps.loc[treated_mask, 'L'], 
                                    df_ps.loc[treated_mask, 'sw_truncated'])
L_control_weighted = weighted_mean(df_ps.loc[control_mask, 'L'],
                                    df_ps.loc[control_mask, 'sw_truncated'])
diff_weighted = L_treated_weighted - L_control_weighted

print(f"Unweighted difference in L: {diff_unweighted:.4f}")
print(f"Weighted difference in L:   {diff_weighted:.4f}")
print(f"Reduction: {(1 - abs(diff_weighted)/abs(diff_unweighted))*100:.1f}%")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY: What We Learned")
print("=" * 70)

print("""
MARGINAL STRUCTURAL MODELS - KEY CONCEPTS:

1. TIME-VARYING CONFOUNDING:
   - Confounders change over time
   - Past treatment affects current confounders
   - Standard regression is BIASED

2. THE SOLUTION:
   - Don't adjust for confounders in outcome model
   - Instead, WEIGHT observations by inverse probability
   - This creates a pseudo-population where treatment is randomized

3. STABILIZED WEIGHTS:
   - Numerator: P(A | baseline)
   - Denominator: P(A | full history)
   - Keeps weights stable while maintaining causal interpretation

4. THE MSM:
   - Simple model: E[Y | A] with no confounders
   - Fit with weighted regression
   - Coefficient is the CAUSAL effect

5. DIAGNOSTICS:
   - Check weight distribution (mean ≈ 1, no extremes)
   - Verify covariate balance after weighting
   - Truncate extreme weights if needed

NEXT: Tutorial 3 - Parametric G-formula
The g-formula simulates counterfactual outcomes directly!
""")

print("\n" + "=" * 70)
print("Final Results")
print("=" * 70)
print(f"\nCAUSAL ODDS RATIO (MSM): {or_msm:.3f}")
print(f"95% CI: ({np.exp(result_msm.conf_int().loc['tx_init', 0]):.3f}, "
      f"{np.exp(result_msm.conf_int().loc['tx_init', 1]):.3f})")
if or_msm < 1:
    print(f"\nInterpretation: Treatment REDUCES odds of outcome by {(1-or_msm)*100:.1f}%")
elif or_msm > 1:
    print(f"\nInterpretation: Treatment INCREASES odds of outcome by {(or_msm-1)*100:.1f}%")
else:
    print("\nInterpretation: No effect of treatment on outcome")
