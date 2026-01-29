"""
=============================================================================
TUTORIAL 3: Parametric G-Formula
=============================================================================

LEARNING OBJECTIVES:
1. Understand the g-formula as a method for causal inference
2. Learn how to model the data-generating process
3. Simulate counterfactual outcomes under different treatment strategies
4. Estimate causal effects by comparing simulated outcomes

DATASET: SEQdata.csv

THE QUESTION:
"What would the outcome risk be if EVERYONE followed treatment strategy A
 vs. if EVERYONE followed treatment strategy B?"

Examples of treatment strategies:
- "Always treat" (A=1 at all times)
- "Never treat" (A=0 at all times)
- "Treat only if L > threshold"

THE G-FORMULA APPROACH:
Instead of weighting (like MSM), we:
1. Model the entire data-generating process
2. Simulate what would happen under each treatment strategy
3. Compare the simulated outcomes

This is like building a "digital twin" of each patient and asking
"what if they had received different treatment?"
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# =============================================================================
# STEP 1: UNDERSTAND THE G-FORMULA CONCEPT
# =============================================================================

print("=" * 70)
print("STEP 1: The G-Formula Concept")
print("=" * 70)

print("""
THE G-FORMULA (Robins, 1986):

For a time-varying treatment A and confounders L, the counterfactual mean is:

E[Y(ā)] = Σ_l E[Y|Ā=ā, L̄=l] × P(L̄=l|Ā=ā)

In words:
1. For each possible confounder history l
2. Find the probability of that history under treatment strategy ā  
3. Find the expected outcome given that history and treatment
4. Average over all possible histories

PRACTICAL APPROACH (Monte Carlo simulation):
1. Model P(L(t) | past L, past A) - how confounders evolve
2. Model P(Y(t) | past L, past A) - how outcome depends on history
3. Model P(A(t) | past L, past A) - natural treatment (for comparison)
4. Simulate many patients forward under each strategy
5. Compare average outcomes
""")

# =============================================================================
# STEP 2: LOAD AND PREPARE DATA
# =============================================================================

print("\n" + "=" * 70)
print("STEP 2: Prepare the Data")
print("=" * 70)

df = pd.read_csv('SEQdata.csv')

# Create lagged variables for modeling
df = df.sort_values(['ID', 'time'])
df['tx_lag1'] = df.groupby('ID')['tx_init'].shift(1)
df['L_lag1'] = df.groupby('ID')['L'].shift(1)
df['N_lag1'] = df.groupby('ID')['N'].shift(1)
df['outcome_lag1'] = df.groupby('ID')['outcome'].shift(1)

# For cumulative treatment exposure
df['cum_tx'] = df.groupby('ID')['tx_init'].cumsum()

# Drop first time point (no lags)
df_model = df.dropna(subset=['tx_lag1']).copy()

# Only keep observations before outcome (for modeling)
# We need to identify when outcome first occurs
df_model['outcome_occurred'] = df_model.groupby('ID')['outcome'].cumsum()
df_model = df_model[df_model['outcome_occurred'] <= df_model['outcome']]

print(f"Modeling dataset: {len(df_model):,} person-time observations")
print(f"Unique patients: {df_model['ID'].nunique()}")

# Look at data structure
print("\n--- Sample Patient Trajectory ---")
sample = df_model[df_model['ID'] == 10][['ID', 'time', 'tx_init', 'tx_lag1', 'L', 'L_lag1', 'outcome']]
print(sample.head(10).to_string(index=False))

# =============================================================================
# STEP 3: FIT THE COMPONENT MODELS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 3: Fit the Component Models")
print("=" * 70)

print("""
We need to model three things:
1. CONFOUNDER MODEL: P(L(t) | L(t-1), A(t-1), ...)
2. TREATMENT MODEL: P(A(t) | L(t), A(t-1), ...) [natural course]
3. OUTCOME MODEL: P(Y(t) | L(t), A(t), history, ...)

These models capture the data-generating process.
""")

# ----- MODEL 1: Confounder (L) Evolution -----
print("\n--- Model 1: Confounder Evolution ---")
print("Modeling: L(t) ~ L(t-1) + A(t-1) + N(t-1) + time")

L_features = ['L_lag1', 'tx_lag1', 'N_lag1', 'time', 'sex']
X_L = df_model[L_features]
y_L = df_model['L']

model_L = LinearRegression()
model_L.fit(X_L, y_L)

print(f"R² = {model_L.score(X_L, y_L):.3f}")
print("Coefficients:")
for feat, coef in zip(L_features, model_L.coef_):
    print(f"  {feat}: {coef:.4f}")

# ----- MODEL 2: Treatment (natural course) -----
print("\n--- Model 2: Treatment Model (Natural Course) ---")
print("Modeling: P(A(t)=1 | L(t), A(t-1), ...)")

A_features = ['L', 'tx_lag1', 'N', 'time', 'sex']
X_A = df_model[A_features]
y_A = df_model['tx_init']

model_A = LogisticRegression(max_iter=1000)
model_A.fit(X_A, y_A)

print(f"Accuracy = {model_A.score(X_A, y_A):.3f}")

# ----- MODEL 3: Outcome -----
print("\n--- Model 3: Outcome Model ---")
print("Modeling: P(Y(t)=1 | L(t), A(t), cum_tx, ...)")

Y_features = ['L', 'tx_init', 'cum_tx', 'N', 'time', 'sex']
X_Y = df_model[Y_features]
y_Y = df_model['outcome']

model_Y = LogisticRegression(max_iter=1000)
model_Y.fit(X_Y, y_Y)

print(f"Accuracy = {model_Y.score(X_Y, y_Y):.3f}")

# =============================================================================
# STEP 4: DEFINE TREATMENT STRATEGIES
# =============================================================================

print("\n" + "=" * 70)
print("STEP 4: Define Treatment Strategies")
print("=" * 70)

print("""
We'll compare three treatment strategies:

1. ALWAYS TREAT: A(t) = 1 for all t
   "What if everyone received treatment at every time point?"

2. NEVER TREAT: A(t) = 0 for all t
   "What if no one ever received treatment?"

3. NATURAL COURSE: A(t) follows the observed/modeled pattern
   "What if treatment followed the natural decision process?"
""")

def strategy_always_treat(L, tx_lag, N, time, sex):
    """Always return 1 (always treat)"""
    return 1

def strategy_never_treat(L, tx_lag, N, time, sex):
    """Always return 0 (never treat)"""
    return 0

def strategy_natural(L, tx_lag, N, time, sex, model):
    """Use the fitted treatment model"""
    X = np.array([[L, tx_lag, N, time, sex]])
    prob = model.predict_proba(X)[0, 1]
    return np.random.binomial(1, prob)

# =============================================================================
# STEP 5: SIMULATE COUNTERFACTUAL TRAJECTORIES
# =============================================================================

print("\n" + "=" * 70)
print("STEP 5: Simulate Counterfactual Trajectories")
print("=" * 70)

print("""
THE SIMULATION ALGORITHM:

For each patient i and treatment strategy:
1. Start with their baseline values (time=0)
2. For t = 1, 2, 3, ..., T:
   a. Simulate L(t) from L model given history
   b. Assign A(t) according to the strategy
   c. Simulate Y(t) from outcome model
   d. If Y(t) = 1, stop (outcome occurred)
3. Record final outcome status

Repeat for many simulated "clones" to get stable estimates.
""")

def simulate_trajectory(baseline, strategy_fn, model_L, model_Y, model_A, 
                        max_time=30, strategy_name=''):
    """
    Simulate a single patient trajectory under a given treatment strategy.
    
    Parameters:
    - baseline: dict with initial values (L, N, sex, etc.)
    - strategy_fn: function that returns treatment at each time
    - model_L, model_Y: fitted models
    - max_time: maximum follow-up time
    
    Returns:
    - outcome: 1 if outcome occurred, 0 otherwise
    - outcome_time: time of outcome (or max_time if censored)
    """
    
    # Initialize
    L = baseline['L']
    N = baseline['N']
    sex = baseline['sex']
    tx_lag = 0  # No prior treatment at baseline
    cum_tx = 0
    
    for t in range(1, max_time + 1):
        # Step 1: Simulate confounder L(t)
        X_L = np.array([[L, tx_lag, N, t, sex]])
        L_new = model_L.predict(X_L)[0]
        # Add some noise
        L_new += np.random.normal(0, 0.1)
        
        # Step 2: Assign treatment according to strategy
        if strategy_name == 'natural':
            tx = strategy_fn(L_new, tx_lag, N, t, sex, model_A)
        else:
            tx = strategy_fn(L_new, tx_lag, N, t, sex)
        
        cum_tx += tx
        
        # Step 3: Simulate outcome
        X_Y = np.array([[L_new, tx, cum_tx, N, t, sex]])
        prob_Y = model_Y.predict_proba(X_Y)[0, 1]
        outcome = np.random.binomial(1, prob_Y)
        
        if outcome == 1:
            return 1, t
        
        # Update for next iteration
        L = L_new
        tx_lag = tx
        # N could also evolve, but we'll keep it simple
    
    return 0, max_time


def run_gformula_simulation(df, strategy_fn, strategy_name, model_L, model_Y, model_A,
                            n_simulations=1000, max_time=30):
    """
    Run g-formula simulation for a given treatment strategy.
    
    For each of n_simulations:
    - Sample a patient (with replacement)
    - Simulate their trajectory under the strategy
    - Record the outcome
    """
    
    # Get baseline characteristics for each patient
    baselines = df.groupby('ID').first()[['L', 'N', 'sex']].reset_index()
    
    outcomes = []
    outcome_times = []
    
    for i in range(n_simulations):
        # Sample a patient
        baseline = baselines.sample(1).iloc[0].to_dict()
        
        # Simulate trajectory
        outcome, outcome_time = simulate_trajectory(
            baseline, strategy_fn, model_L, model_Y, model_A,
            max_time=max_time, strategy_name=strategy_name
        )
        
        outcomes.append(outcome)
        outcome_times.append(outcome_time)
    
    return np.array(outcomes), np.array(outcome_times)

# Run simulations for each strategy
print("\nRunning simulations (this may take a moment)...")

n_sims = 2000
max_t = 30

print(f"\nSimulating {n_sims} trajectories per strategy, up to time {max_t}...")

# Strategy 1: Always treat
print("\n  Strategy 1: Always Treat...", end=" ")
outcomes_always, times_always = run_gformula_simulation(
    df_model, strategy_always_treat, 'always',
    model_L, model_Y, model_A, n_sims, max_t
)
print(f"Done. Outcome rate: {outcomes_always.mean():.3f}")

# Strategy 2: Never treat
print("  Strategy 2: Never Treat...", end=" ")
outcomes_never, times_never = run_gformula_simulation(
    df_model, strategy_never_treat, 'never',
    model_L, model_Y, model_A, n_sims, max_t
)
print(f"Done. Outcome rate: {outcomes_never.mean():.3f}")

# Strategy 3: Natural course
print("  Strategy 3: Natural Course...", end=" ")
outcomes_natural, times_natural = run_gformula_simulation(
    df_model, strategy_natural, 'natural',
    model_L, model_Y, model_A, n_sims, max_t
)
print(f"Done. Outcome rate: {outcomes_natural.mean():.3f}")

# =============================================================================
# STEP 6: ESTIMATE CAUSAL EFFECTS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 6: Estimate Causal Effects")
print("=" * 70)

print("""
CAUSAL CONTRASTS:

The g-formula gives us E[Y(strategy)] for each strategy.
We can compute:

1. Risk Difference: E[Y(always treat)] - E[Y(never treat)]
2. Risk Ratio: E[Y(always treat)] / E[Y(never treat)]
3. Number Needed to Treat: 1 / |Risk Difference|
""")

risk_always = outcomes_always.mean()
risk_never = outcomes_never.mean()
risk_natural = outcomes_natural.mean()

risk_difference = risk_always - risk_never
risk_ratio = risk_always / risk_never if risk_never > 0 else np.inf

print("\n--- Estimated Outcome Risks ---")
print(f"{'Strategy':<20} {'Risk':<10} {'95% CI'}")
print("-" * 50)

# Bootstrap confidence intervals
def bootstrap_ci(outcomes, n_boot=1000):
    means = []
    for _ in range(n_boot):
        sample = np.random.choice(outcomes, size=len(outcomes), replace=True)
        means.append(sample.mean())
    return np.percentile(means, [2.5, 97.5])

ci_always = bootstrap_ci(outcomes_always)
ci_never = bootstrap_ci(outcomes_never)
ci_natural = bootstrap_ci(outcomes_natural)

print(f"{'Always Treat':<20} {risk_always:<10.3f} ({ci_always[0]:.3f}, {ci_always[1]:.3f})")
print(f"{'Never Treat':<20} {risk_never:<10.3f} ({ci_never[0]:.3f}, {ci_never[1]:.3f})")
print(f"{'Natural Course':<20} {risk_natural:<10.3f} ({ci_natural[0]:.3f}, {ci_natural[1]:.3f})")

print("\n--- Causal Effect Estimates ---")
print(f"Risk Difference (Always vs Never): {risk_difference:+.3f}")
print(f"  Interpretation: Treatment {'increases' if risk_difference > 0 else 'decreases'} "
      f"risk by {abs(risk_difference)*100:.1f} percentage points")

print(f"\nRisk Ratio (Always vs Never): {risk_ratio:.3f}")
print(f"  Interpretation: Treatment {'increases' if risk_ratio > 1 else 'decreases'} "
      f"risk by a factor of {risk_ratio:.2f}")

if risk_difference != 0:
    nnt = abs(1 / risk_difference)
    print(f"\nNumber Needed to Treat: {nnt:.1f}")
    print(f"  Interpretation: Treat {nnt:.0f} patients to {'cause' if risk_difference > 0 else 'prevent'} 1 outcome")

# =============================================================================
# STEP 7: VISUALIZE SURVIVAL CURVES
# =============================================================================

print("\n" + "=" * 70)
print("STEP 7: Time-to-Event Analysis")
print("=" * 70)

print("""
We can also look at WHEN outcomes occur, not just whether they occur.
This gives us survival curves under each strategy.
""")

# Calculate survival at each time point
def calculate_survival_curve(outcome_times, outcomes, max_time):
    """Calculate Kaplan-Meier-like survival curve."""
    survival = []
    for t in range(max_time + 1):
        # Proportion who have NOT had outcome by time t
        survived = np.mean((outcome_times > t) | (outcomes == 0))
        survival.append(survived)
    return survival

survival_always = calculate_survival_curve(times_always, outcomes_always, max_t)
survival_never = calculate_survival_curve(times_never, outcomes_never, max_t)
survival_natural = calculate_survival_curve(times_natural, outcomes_natural, max_t)

print("\n--- Survival Probability Over Time ---")
print(f"{'Time':<8} {'Always Treat':<15} {'Never Treat':<15} {'Natural':<15}")
print("-" * 55)
for t in [0, 5, 10, 15, 20, 25, 30]:
    if t <= max_t:
        print(f"{t:<8} {survival_always[t]:<15.3f} {survival_never[t]:<15.3f} {survival_natural[t]:<15.3f}")

# Mean survival time
mean_time_always = times_always[outcomes_always == 1].mean() if outcomes_always.sum() > 0 else max_t
mean_time_never = times_never[outcomes_never == 1].mean() if outcomes_never.sum() > 0 else max_t

print(f"\nMean time to outcome (among those with outcome):")
print(f"  Always Treat: {mean_time_always:.1f}")
print(f"  Never Treat:  {mean_time_never:.1f}")

# =============================================================================
# STEP 8: COMPARISON WITH OTHER METHODS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 8: G-Formula vs Other Methods")
print("=" * 70)

print("""
METHOD COMPARISON:

┌─────────────────────┬────────────────────────────────────────────┐
│ Method              │ Approach                                   │
├─────────────────────┼────────────────────────────────────────────┤
│ Target Trial        │ Clone patients at each eligible time       │
│ (Tutorial 1)        │ Compare initiators vs non-initiators       │
├─────────────────────┼────────────────────────────────────────────┤
│ MSM with IPTW       │ Weight observations to balance confounders │
│ (Tutorial 2)        │ Fit simple outcome model on weighted data  │
├─────────────────────┼────────────────────────────────────────────┤
│ G-Formula           │ Model entire data-generating process       │
│ (This tutorial)     │ Simulate counterfactual trajectories       │
└─────────────────────┴────────────────────────────────────────────┘

WHEN TO USE EACH:

G-FORMULA is best when:
✓ You want to compare SUSTAINED treatment strategies
✓ You want to estimate E[Y(strategy)] directly
✓ You're comfortable with parametric modeling assumptions
✓ You want survival curves under different strategies

MSM/IPTW is best when:
✓ You want a single summary measure (OR, HR)
✓ You're worried about model misspecification
✓ Treatment decisions are complex (hard to model)

TARGET TRIAL is best when:
✓ You're interested in treatment INITIATION effects
✓ You want to mimic a specific trial design
✓ You need clear eligibility criteria
""")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY: What We Learned")
print("=" * 70)

print("""
PARAMETRIC G-FORMULA - KEY CONCEPTS:

1. THE IDEA:
   - Model the data-generating process completely
   - Simulate "what if" scenarios (counterfactuals)
   - Compare outcomes under different treatment strategies

2. THE COMPONENTS:
   - Confounder model: How L evolves over time
   - Treatment model: Natural treatment patterns
   - Outcome model: How Y depends on history

3. THE SIMULATION:
   - Start with baseline characteristics
   - At each time: simulate L → assign A → check Y
   - Repeat for many "Monte Carlo" samples

4. TREATMENT STRATEGIES:
   - "Always treat" - everyone gets treatment
   - "Never treat" - no one gets treatment
   - "Treat if L > threshold" - dynamic strategies
   - "Natural course" - follow observed patterns

5. CAUSAL EFFECTS:
   - Risk Difference: E[Y(a=1)] - E[Y(a=0)]
   - Risk Ratio: E[Y(a=1)] / E[Y(a=0)]
   - Survival curves under each strategy

6. ASSUMPTIONS:
   - Correct model specification (parametric)
   - No unmeasured confounding
   - Positivity (all strategies possible)
   - Consistency (well-defined interventions)

NEXT: Tutorial 4 - Structural Nested Models (G-estimation)
""")

print("\n" + "=" * 70)
print("Final Results: G-Formula Estimates")
print("=" * 70)
print(f"""
Treatment Strategy Comparison (at {max_t} time units):

  ALWAYS TREAT:  {risk_always*100:.1f}% outcome risk
  NEVER TREAT:   {risk_never*100:.1f}% outcome risk
  NATURAL:       {risk_natural*100:.1f}% outcome risk

Causal Effect (Always vs Never):
  Risk Difference: {risk_difference*100:+.1f} percentage points
  Risk Ratio:      {risk_ratio:.2f}
""")
