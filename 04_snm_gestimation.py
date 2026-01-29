"""
=============================================================================
TUTORIAL 4: Structural Nested Models and G-Estimation
=============================================================================
Real-World Evidence Training Series | Observational Study Methods

AUDIENCE: Epidemiologists, pharmacoepidemiologists, RWE scientists
PREREQUISITES: Tutorials 1-3 (Target Trials, MSM, G-formula)

=============================================================================
CLINICAL CONTEXT
=============================================================================

Consider this common scenario in pharmacoepidemiology:

    A patient with Type 2 Diabetes starts metformin. Their HbA1c improves,
    so they stay on metformin. Another patient's HbA1c doesn't improve,
    so their physician adds a second agent.
    
    Question: What is the causal effect of metformin on cardiovascular events?

THE CHALLENGE:
- Treatment decisions depend on intermediate outcomes (HbA1c)
- These intermediate outcomes are ALSO affected by prior treatment
- This is "time-varying confounding affected by prior treatment"
- Standard regression and even propensity scores can be biased

STRUCTURAL NESTED MODELS (SNMs) offer a solution by:
1. Modeling the TREATMENT EFFECT directly (not the outcome)
2. Using G-estimation to find effect parameters
3. Handling time-varying confounding without modeling confounder distributions

=============================================================================
LEARNING OBJECTIVES
=============================================================================

After this tutorial, you will be able to:
1. Explain when SNMs are preferred over MSMs or g-formula
2. Implement g-estimation for a structural nested mean model
3. Interpret the "blip function" - the causal effect of treatment
4. Understand the rank preservation assumption
5. Apply these methods to longitudinal RWE data

REFERENCE:
- Robins JM. Correcting for non-compliance in randomized trials using 
  structural nested mean models. Communications in Statistics. 1994.
- Vansteelandt S, Joffe M. Structural Nested Models and G-estimation. 
  Handbook of Causal Analysis for Social Research. 2014.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression, LinearRegression
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("=" * 75)
print("STRUCTURAL NESTED MODELS AND G-ESTIMATION")
print("Real-World Evidence Training Module 4")
print("=" * 75)

# =============================================================================
# SECTION 1: CONCEPTUAL FRAMEWORK
# =============================================================================

print("\n" + "=" * 75)
print("SECTION 1: Why Do We Need Another Method?")
print("=" * 75)

print("""
RECAP OF METHODS SO FAR:

┌─────────────────┬─────────────────────────────────────────────────────────┐
│ Method          │ What it estimates                                       │
├─────────────────┼─────────────────────────────────────────────────────────┤
│ Target Trial    │ Effect of INITIATING treatment at time t                │
│ MSM + IPTW      │ Marginal effect of treatment history on outcome         │
│ G-formula       │ E[Y] under specific treatment STRATEGIES                │
│ SNM             │ Direct effect of treatment at each time point           │
└─────────────────┴─────────────────────────────────────────────────────────┘

WHEN TO USE SNMs:

1. You want to estimate the DIRECT EFFECT of A(t) on Y
   - "How much does treatment at time t change the outcome?"
   - Not just "what happens if we follow strategy X"

2. You believe in EFFECT MODIFICATION
   - The treatment effect may vary by patient characteristics
   - E.g., "Metformin works better in patients with higher baseline HbA1c"

3. You want TREATMENT EFFECT HETEROGENEITY
   - Different effects at different times
   - Different effects for different patient subgroups

4. PRACTICAL ADVANTAGE:
   - SNMs only require modeling treatment (not confounders or outcome)
   - More robust when confounder models are hard to specify

REAL-WORLD EXAMPLE:
In a comparative effectiveness study of antihypertensives:
- Patients may switch drugs based on blood pressure response
- We want to know: "What is the BP-lowering effect of Drug A vs Drug B?"
- SNMs let us estimate this while handling treatment switching
""")

# =============================================================================
# SECTION 2: THE STRUCTURAL NESTED MEAN MODEL
# =============================================================================

print("\n" + "=" * 75)
print("SECTION 2: The Structural Nested Mean Model (SNMM)")
print("=" * 75)

print("""
THE KEY CONCEPT: THE "BLIP" FUNCTION

Definition: The blip function γ(t, H_t; ψ) represents the causal effect
            of receiving treatment A(t)=1 vs A(t)=0 at time t, given
            history H_t up to time t.

Mathematically:
    γ(t, H_t; ψ) = E[Y(ā_{t-1}, 1, 0̄) - Y(ā_{t-1}, 0, 0̄) | H_t]

In words:
    "The expected change in outcome if you receive treatment at time t
     (and no treatment thereafter) versus no treatment at time t
     (and no treatment thereafter), given your history up to t."

SIMPLE EXAMPLE:
    γ(t, H_t; ψ) = ψ₀ + ψ₁ × A(t)

    This says: Treatment at time t changes the outcome by ψ₀ + ψ₁
    
MORE COMPLEX:
    γ(t, H_t; ψ) = ψ₀ × A(t) + ψ₁ × A(t) × L(t) + ψ₂ × A(t) × t
    
    This allows:
    - ψ₀: Main treatment effect
    - ψ₁: Effect modification by confounder L
    - ψ₂: Effect modification by time
""")

# =============================================================================
# SECTION 3: LOAD AND EXPLORE DATA
# =============================================================================

print("\n" + "=" * 75)
print("SECTION 3: Data Preparation")
print("=" * 75)

df = pd.read_csv('SEQdata.csv')

print("""
STUDY DESIGN CONSIDERATIONS:

In this simulated cohort (analogous to a claims/EHR database):
- N = 300 patients followed longitudinally
- Time-varying treatment (tx_init): Binary, can switch on/off
- Time-varying confounders: L (continuous), N (continuous), P (continuous)
- Outcome: Binary event (could represent hospitalization, death, etc.)

In a real RWE study, you would document:
- Data source (e.g., Optum, MarketScan, CPRD)
- Study period and follow-up
- Inclusion/exclusion criteria
- Exposure definition
- Outcome ascertainment
- Covariate measurement windows
""")

# Prepare data
df = df.sort_values(['ID', 'time'])
df['tx_lag1'] = df.groupby('ID')['tx_init'].shift(1).fillna(0)
df['L_lag1'] = df.groupby('ID')['L'].shift(1)
df['cum_tx'] = df.groupby('ID')['tx_init'].cumsum()

# Keep only complete cases for modeling
df_analysis = df.dropna().copy()

print(f"Analysis cohort: {df_analysis['ID'].nunique()} patients")
print(f"Person-time observations: {len(df_analysis):,}")
print(f"Outcome events: {df_analysis['outcome'].sum()} ({df_analysis['outcome'].mean()*100:.1f}%)")
print(f"Treatment exposure: {df_analysis['tx_init'].mean()*100:.1f}% of person-time on treatment")

# Summary statistics
print("\n--- Baseline Characteristics (Time 0) ---")
baseline = df[df['time'] == 0]
print(f"Sex distribution: {baseline['sex'].value_counts().to_dict()}")
print(f"Mean L at baseline: {baseline['L'].mean():.3f} (SD: {baseline['L'].std():.3f})")

# =============================================================================
# SECTION 4: G-ESTIMATION ALGORITHM
# =============================================================================

print("\n" + "=" * 75)
print("SECTION 4: G-Estimation - The Core Algorithm")
print("=" * 75)

print("""
G-ESTIMATION: FINDING THE CAUSAL PARAMETER ψ

The key insight: If we knew the TRUE treatment effect ψ, we could
"remove" the effect of treatment from the outcome, creating a
"treatment-free" outcome that should be INDEPENDENT of treatment
(after adjusting for confounders).

ALGORITHM:

1. Propose a value for ψ (the treatment effect parameter)

2. Calculate the "treatment-free" potential outcome:
   Y*(ψ) = Y - Σ_t γ(t, H_t; ψ) × A(t)
   
   This "subtracts out" the treatment effect

3. The correct ψ is the one where Y*(ψ) is independent of A(t)
   conditional on confounders L(t)

4. Test this independence using a propensity score model:
   - Fit: P(A(t) | L(t), history)
   - Residual: A(t) - P(A(t) | ...)
   - The correct ψ makes: Cov(Y*(ψ), residual) = 0

This is solving an "estimating equation" - we find ψ that makes
the treatment-free outcome unrelated to treatment decisions.

INTUITION:
If we correctly remove the causal effect of treatment from Y,
what remains (Y*) should only depend on confounders, not on
whether someone happened to be treated.
""")

# =============================================================================
# SECTION 5: IMPLEMENTATION
# =============================================================================

print("\n" + "=" * 75)
print("SECTION 5: Implementation of G-Estimation")
print("=" * 75)

# Step 1: Fit the treatment (propensity) model
print("Step 1: Fitting propensity score model...")
print("        P(A(t) = 1 | L(t), A(t-1), sex, time)")

ps_features = ['L', 'tx_lag1', 'sex', 'time', 'N']
X_ps = df_analysis[ps_features]
y_tx = df_analysis['tx_init']

ps_model = LogisticRegression(max_iter=1000, random_state=42)
ps_model.fit(X_ps, y_tx)

# Get propensity scores and residuals
df_analysis['ps'] = ps_model.predict_proba(X_ps)[:, 1]
df_analysis['ps_residual'] = df_analysis['tx_init'] - df_analysis['ps']

print(f"        Propensity model accuracy: {ps_model.score(X_ps, y_tx):.3f}")
print(f"        Mean PS for treated: {df_analysis.loc[df_analysis['tx_init']==1, 'ps'].mean():.3f}")
print(f"        Mean PS for untreated: {df_analysis.loc[df_analysis['tx_init']==0, 'ps'].mean():.3f}")


# Step 2: Define the blip function and estimating equation
print("\nStep 2: Defining the blip function...")
print("""
        We'll use a simple SNMM:
        
        γ(t; ψ) = ψ × A(t)
        
        This assumes a CONSTANT treatment effect across time and patients.
        (We'll extend this later)
""")

def calculate_treatment_free_outcome(df, psi):
    """
    Calculate Y*(psi) = Y - sum of blip functions
    
    For a simple model: Y* = Y - psi * cumulative_treatment
    """
    # For each person, calculate their treatment-free outcome
    # Y* = Y - psi * (sum of treatments received)
    df_calc = df.copy()
    
    # Group by person and calculate
    def calc_y_star(group):
        group = group.sort_values('time')
        # The blip at each time is psi * A(t)
        # Total effect is sum of blips = psi * cumulative treatment
        group['y_star'] = group['outcome'] - psi * group['cum_tx']
        return group
    
    df_calc = df_calc.groupby('ID', group_keys=False).apply(calc_y_star)
    return df_calc['y_star']


def estimating_equation(psi, df):
    """
    The estimating equation for g-estimation.
    
    Returns the covariance between Y*(psi) and the PS residual.
    At the true psi, this should equal zero.
    """
    y_star = calculate_treatment_free_outcome(df, psi)
    
    # The estimating equation: sum of (Y* - E[Y*]) * (A - E[A|L])
    # Which simplifies to covariance of Y* and PS residual
    cov = np.cov(y_star, df['ps_residual'])[0, 1]
    
    return cov ** 2  # Square it for minimization


# Step 3: Solve for psi
print("\nStep 3: Solving the estimating equation...")
print("        Finding ψ such that Cov(Y*, A - E[A|L]) = 0")

# Grid search first to find approximate solution
psi_grid = np.linspace(-0.5, 0.5, 101)
ee_values = [estimating_equation(psi, df_analysis) for psi in psi_grid]

psi_init = psi_grid[np.argmin(ee_values)]
print(f"        Grid search estimate: ψ ≈ {psi_init:.4f}")

# Refine with optimization
result = minimize(lambda p: estimating_equation(p[0], df_analysis), 
                  x0=[psi_init], method='Nelder-Mead')
psi_hat = result.x[0]

print(f"        Optimized estimate: ψ = {psi_hat:.4f}")

# =============================================================================
# SECTION 6: INTERPRETATION
# =============================================================================

print("\n" + "=" * 75)
print("SECTION 6: Interpreting the Results")
print("=" * 75)

print(f"""
G-ESTIMATION RESULTS
--------------------
Estimated treatment effect (ψ): {psi_hat:.4f}

INTERPRETATION:
Each unit of treatment exposure {'increases' if psi_hat > 0 else 'decreases'} 
the probability of the outcome by {abs(psi_hat):.4f} (on the probability scale).

For a patient treated for 10 time periods:
  Expected change in outcome probability = 10 × {psi_hat:.4f} = {10*psi_hat:.3f}

CLINICAL INTERPRETATION:
If this were a study of, say, statin use and cardiovascular events:
  "Each additional month of statin exposure is associated with a 
   {abs(psi_hat)*100:.2f} percentage point {'increase' if psi_hat > 0 else 'decrease'} 
   in the probability of a cardiovascular event."
""")

# =============================================================================
# SECTION 7: CONFIDENCE INTERVALS (BOOTSTRAP)
# =============================================================================

print("\n" + "=" * 75)
print("SECTION 7: Statistical Inference")
print("=" * 75)

print("""
OBTAINING CONFIDENCE INTERVALS:

For g-estimation, we use the BOOTSTRAP:
1. Resample patients (with replacement)
2. Re-estimate ψ on each bootstrap sample
3. Calculate percentile-based CI
""")

print("Running bootstrap (this may take a moment)...")

def bootstrap_gestimation(df, n_bootstrap=200):
    """Bootstrap confidence interval for psi."""
    patient_ids = df['ID'].unique()
    psi_boots = []
    
    for b in range(n_bootstrap):
        # Resample patients
        boot_ids = np.random.choice(patient_ids, size=len(patient_ids), replace=True)
        
        # Create bootstrap dataset
        boot_dfs = []
        for i, pid in enumerate(boot_ids):
            pt_data = df[df['ID'] == pid].copy()
            pt_data['ID'] = i  # Renumber for uniqueness
            boot_dfs.append(pt_data)
        df_boot = pd.concat(boot_dfs, ignore_index=True)
        
        # Re-fit PS model
        ps_model_boot = LogisticRegression(max_iter=1000, random_state=b)
        ps_model_boot.fit(df_boot[ps_features], df_boot['tx_init'])
        df_boot['ps'] = ps_model_boot.predict_proba(df_boot[ps_features])[:, 1]
        df_boot['ps_residual'] = df_boot['tx_init'] - df_boot['ps']
        
        # Estimate psi
        result_boot = minimize(lambda p: estimating_equation(p[0], df_boot),
                               x0=[psi_hat], method='Nelder-Mead',
                               options={'maxiter': 100})
        psi_boots.append(result_boot.x[0])
        
        if (b + 1) % 50 == 0:
            print(f"        Completed {b + 1}/{n_bootstrap} bootstrap samples")
    
    return np.array(psi_boots)

psi_boots = bootstrap_gestimation(df_analysis, n_bootstrap=200)
ci_lower = np.percentile(psi_boots, 2.5)
ci_upper = np.percentile(psi_boots, 97.5)
se = np.std(psi_boots)

print(f"""
INFERENCE RESULTS:
  Point estimate: ψ = {psi_hat:.4f}
  Standard error: {se:.4f}
  95% CI: ({ci_lower:.4f}, {ci_upper:.4f})

Statistical significance:
  {'The effect is statistically significant (CI excludes 0)' 
   if (ci_lower > 0 or ci_upper < 0) 
   else 'The effect is NOT statistically significant (CI includes 0)'}
""")

# =============================================================================
# SECTION 8: EFFECT MODIFICATION
# =============================================================================

print("\n" + "=" * 75)
print("SECTION 8: Exploring Effect Modification")
print("=" * 75)

print("""
EXTENDING THE MODEL:

In practice, treatment effects often vary by:
- Baseline characteristics (age, sex, comorbidities)
- Time-varying factors (current disease severity)
- Time itself (early vs late treatment effects)

EXTENDED SNMM:
    γ(t, H_t; ψ) = ψ₁ × A(t) + ψ₂ × A(t) × L(t) + ψ₃ × A(t) × sex

This allows us to ask:
- Does treatment work differently for high-L vs low-L patients?
- Is there a sex difference in treatment effect?
""")

# Implement extended model with effect modification by L
def estimating_equation_extended(params, df):
    """
    Extended model: blip = psi1*A + psi2*A*L
    """
    psi1, psi2 = params
    
    df_calc = df.copy()
    
    def calc_y_star_extended(group):
        group = group.sort_values('time')
        # Blip at each time: psi1*A + psi2*A*L
        blip_sum = (psi1 * group['tx_init'] + psi2 * group['tx_init'] * group['L']).cumsum()
        group['y_star'] = group['outcome'] - blip_sum
        return group
    
    df_calc = df_calc.groupby('ID', group_keys=False).apply(calc_y_star_extended)
    
    # Two estimating equations (one for each parameter)
    ee1 = np.cov(df_calc['y_star'], df_calc['ps_residual'])[0, 1]
    ee2 = np.cov(df_calc['y_star'], df_calc['ps_residual'] * df_calc['L'])[0, 1]
    
    return ee1**2 + ee2**2

print("Fitting extended model with effect modification by L...")
result_ext = minimize(lambda p: estimating_equation_extended(p, df_analysis),
                      x0=[psi_hat, 0],
                      method='Nelder-Mead')
psi1_hat, psi2_hat = result_ext.x

print(f"""
EXTENDED MODEL RESULTS:
  ψ₁ (main effect): {psi1_hat:.4f}
  ψ₂ (effect modification by L): {psi2_hat:.4f}

INTERPRETATION:
  For a patient with L = 0:
    Treatment effect = {psi1_hat:.4f}
  
  For a patient with L = 2:
    Treatment effect = {psi1_hat:.4f} + {psi2_hat:.4f} × 2 = {psi1_hat + 2*psi2_hat:.4f}

  {'Treatment is more effective for patients with higher L' 
   if psi2_hat < 0 
   else 'Treatment is less effective (or more harmful) for patients with higher L'}
""")

# =============================================================================
# SECTION 9: COMPARISON WITH OTHER METHODS
# =============================================================================

print("\n" + "=" * 75)
print("SECTION 9: When to Use Each Method")
print("=" * 75)

print("""
DECISION FRAMEWORK FOR METHOD SELECTION:

┌─────────────────────────────────────────────────────────────────────────┐
│ QUESTION                                           │ RECOMMENDED METHOD │
├─────────────────────────────────────────────────────────────────────────┤
│ "What happens if we initiate treatment now?"       │ Target Trial       │
│ "What is the effect of sustained treatment?"       │ MSM or G-formula   │
│ "What would outcomes be under strategy X vs Y?"    │ G-formula          │
│ "What is the direct effect of treatment at time t?"│ SNM (g-estimation) │
│ "Does the effect vary by patient characteristics?" │ SNM (g-estimation) │
└─────────────────────────────────────────────────────────────────────────┘

PRACTICAL CONSIDERATIONS:

SNMs are PREFERRED when:
✓ Treatment effects may be heterogeneous
✓ Confounder models are difficult to specify correctly
✓ You want direct effect estimates, not just policy comparisons
✓ Rank preservation assumption is plausible

SNMs are CHALLENGING when:
✗ Many time points (computational complexity)
✗ Complex blip function specification needed
✗ Binary outcomes with rare events
✗ Need to communicate results to non-technical audiences

REGULATORY CONTEXT:
- FDA guidance increasingly accepts RWE for regulatory decisions
- ICH E9(R1) emphasizes clear estimand definition
- SNMs provide precise causal estimands suitable for regulatory submissions
""")

# =============================================================================
# SECTION 10: ASSUMPTIONS AND SENSITIVITY
# =============================================================================

print("\n" + "=" * 75)
print("SECTION 10: Critical Assumptions")
print("=" * 75)

print("""
ASSUMPTIONS FOR VALID CAUSAL INFERENCE:

1. NO UNMEASURED CONFOUNDING (Sequential Exchangeability)
   ───────────────────────────────────────────────────────
   Y(ā) ⊥ A(t) | L̄(t), Ā(t-1)
   
   "Given measured confounders, treatment at time t is as good as random"
   
   IN PRACTICE: 
   - List ALL potential confounders
   - Use DAGs to identify adjustment sets
   - Consider negative control exposures/outcomes
   - Conduct sensitivity analyses for unmeasured confounding

2. POSITIVITY (Experimental Treatment Assignment)
   ───────────────────────────────────────────────
   0 < P(A(t) = a | H_t) < 1 for all a, t, H_t
   
   "Everyone has some chance of receiving each treatment level"
   
   IN PRACTICE:
   - Check propensity score distributions
   - Trim or truncate extreme weights
   - Restrict to populations with clinical equipoise

3. CONSISTENCY (Well-Defined Interventions)
   ──────────────────────────────────────────
   Y = Y(ā) when A = ā
   
   "The potential outcome under treatment a equals the observed 
    outcome when treatment a is actually received"
   
   IN PRACTICE:
   - Clearly define the treatment (dose, duration, formulation)
   - Consider treatment versions (generic vs brand, adherence levels)

4. CORRECT MODEL SPECIFICATION
   ────────────────────────────
   - Propensity score model is correctly specified
   - Blip function form is correct
   
   IN PRACTICE:
   - Use flexible models (splines, interactions)
   - Conduct sensitivity analyses with different specifications

5. RANK PRESERVATION (for some SNM estimands)
   ──────────────────────────────────────────
   "Treatment doesn't change the ranking of outcomes across individuals"
   
   This is a strong assumption that may not hold in practice.
""")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 75)
print("SUMMARY: Key Takeaways for RWE Practice")
print("=" * 75)

print(f"""
STRUCTURAL NESTED MODELS - EXECUTIVE SUMMARY

METHOD:
- Models the TREATMENT EFFECT directly (the "blip function")
- Uses g-estimation to find effect parameters
- Handles time-varying confounding affected by prior treatment

OUR RESULTS:
- Point estimate: ψ = {psi_hat:.4f}
- 95% CI: ({ci_lower:.4f}, {ci_upper:.4f})
- {'Statistically significant' if (ci_lower > 0 or ci_upper < 0) else 'Not statistically significant'}

WHEN TO USE:
- Heterogeneous treatment effects
- Direct effect estimation
- Robust to confounder model misspecification

KEY ASSUMPTIONS:
- No unmeasured confounding
- Positivity
- Consistency
- Correct propensity score model

REPORTING CHECKLIST (per STROBE/RECORD guidelines):
□ Clearly state the causal question and estimand
□ Describe the blip function specification
□ Report propensity score model diagnostics
□ Provide confidence intervals (bootstrap recommended)
□ Discuss assumptions and limitations
□ Conduct sensitivity analyses

FURTHER READING:
1. Hernán MA, Robins JM. Causal Inference: What If. 2020.
2. Vansteelandt S, Joffe M. Structural Nested Models. 2014.
3. Daniel RM, et al. Methods for dealing with time-varying 
   confounding. Statistics in Medicine. 2013.
""")

print("\n" + "=" * 75)
print("END OF TUTORIAL 4")
print("You have completed the Causal Inference Methods Training Series!")
print("=" * 75)

print("""
SERIES SUMMARY:

Tutorial 1: Target Trial Emulation
             → Cloning, ITT effects, point treatment initiation
             
Tutorial 2: Marginal Structural Models
             → IPTW, time-varying confounding, weighted regression
             
Tutorial 3: Parametric G-formula  
             → Counterfactual simulation, treatment strategies
             
Tutorial 4: Structural Nested Models
             → G-estimation, blip functions, effect heterogeneity

NEXT STEPS FOR YOUR TEAM:
1. Apply these methods to your own RWE data
2. Practice on publicly available datasets (NHANES, MIMIC)
3. Implement sensitivity analyses for unmeasured confounding
4. Stay current with methodological developments
5. Engage with regulatory guidance on RWE
""")
