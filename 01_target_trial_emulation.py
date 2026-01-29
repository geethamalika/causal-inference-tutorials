"""
=============================================================================
TUTORIAL 1: Target Trial Emulation
=============================================================================

LEARNING OBJECTIVES:
1. Understand the target trial framework
2. Learn how to "clone" patients at each eligible time point
3. Estimate intention-to-treat (ITT) and per-protocol (PP) effects
4. Understand why this handles time-varying confounding

DATASET: SEQdata.csv
- 300 patients followed over up to 60 time points
- Treatment can be initiated (tx_init=1) at any time
- Outcome is a binary event (outcome=1)
- Time-varying confounders: N, L, P

THE PROBLEM:
We want to estimate: "What is the causal effect of initiating treatment 
vs. not initiating treatment on the outcome?"

WHY IS THIS HARD?
- Treatment timing varies (some start early, some late, some never)
- Confounders change over time and affect both treatment and outcome
- Simple comparisons (treated vs untreated) are biased

THE SOLUTION: TARGET TRIAL EMULATION
- At each time t, among those eligible and not yet treated:
  - Compare those who START treatment at t vs those who DON'T start at t
  - This mimics a randomized trial starting at time t
- Combine evidence across all these "mini-trials"
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# STEP 1: LOAD AND EXPLORE THE DATA
# =============================================================================

print("=" * 70)
print("STEP 1: Understanding the Data Structure")
print("=" * 70)

df = pd.read_csv('../SEQdata.csv')

print(f"\nDataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Look at one patient's trajectory
print("\n--- Example: Patient 1's trajectory ---")
patient_1 = df[df['ID'] == 1][['ID', 'time', 'eligible', 'tx_init', 'outcome', 'N', 'L']]
print(patient_1.to_string(index=False))

print("""
KEY OBSERVATIONS:
- Patient 1 is followed from time 0 to 18
- They are eligible (eligible=1) throughout
- tx_init switches on/off (treatment decisions at each time)
- Outcome=1 at time 18 (event occurred, follow-up ends)
- N, L change over time (time-varying confounders)
""")

# =============================================================================
# STEP 2: THE CORE IDEA - "CLONING" AT EACH TIME POINT
# =============================================================================

print("=" * 70)
print("STEP 2: The Cloning Approach")
print("=" * 70)

print("""
THE KEY INSIGHT:
At each time t, among people who:
  (1) Are still eligible
  (2) Have NOT yet started treatment
  (3) Have NOT yet had the outcome

We create a "clone" who enters a hypothetical trial:
  - Treatment group: those who START treatment at time t
  - Control group: those who DON'T start treatment at time t

This is called "sequential trial emulation" or "cloning".
""")

# Let's implement this step by step

def find_treatment_initiation_time(patient_df):
    """Find the first time a patient initiates treatment."""
    treated_times = patient_df[patient_df['tx_init'] == 1]['time']
    if len(treated_times) > 0:
        return treated_times.min()
    return None

def find_outcome_time(patient_df):
    """Find the time of outcome event."""
    outcome_times = patient_df[patient_df['outcome'] == 1]['time']
    if len(outcome_times) > 0:
        return outcome_times.min()
    return None

# Analyze treatment initiation patterns
print("\n--- Treatment Initiation Patterns ---")
initiation_times = []
for pid in df['ID'].unique():
    pt = df[df['ID'] == pid]
    init_time = find_treatment_initiation_time(pt)
    if init_time is not None:
        initiation_times.append(init_time)

print(f"Patients who ever initiated treatment: {len(initiation_times)}/{df['ID'].nunique()}")
print(f"Mean time to initiation: {np.mean(initiation_times):.1f}")
print(f"Range: {np.min(initiation_times)} to {np.max(initiation_times)}")

# =============================================================================
# STEP 3: BUILD THE EXPANDED (CLONED) DATASET
# =============================================================================

print("\n" + "=" * 70)
print("STEP 3: Building the Expanded Dataset")
print("=" * 70)

def expand_data_for_trial_emulation(df):
    """
    Create the expanded dataset for sequential trial emulation.
    
    For each patient, at each time they are:
    - Eligible
    - Not yet treated (prior to this time)
    - Not yet had outcome
    
    We create a "trial entry" with:
    - trial_time: the time of entry into this mini-trial
    - assigned_treatment: 1 if they START treatment at this time, 0 otherwise
    """
    
    expanded_rows = []
    
    for pid in df['ID'].unique():
        pt = df[df['ID'] == pid].sort_values('time')
        
        outcome_time = find_outcome_time(pt)
        first_treatment_time = find_treatment_initiation_time(pt)
        
        for _, row in pt.iterrows():
            t = row['time']
            
            # Check eligibility criteria for trial entry at time t
            # 1. Must be eligible
            if row['eligible'] != 1:
                continue
            
            # 2. Must not have had outcome yet
            if outcome_time is not None and t > outcome_time:
                continue
            
            # 3. For simplicity in this tutorial: must not have initiated treatment before
            #    (In full implementation, you might allow re-entry)
            if first_treatment_time is not None and t > first_treatment_time:
                continue
            
            # This person enters a "trial" at time t
            # Their "assigned treatment" is whether they START treatment at t
            assigned_trt = 1 if (first_treatment_time is not None and t == first_treatment_time) else 0
            
            # Calculate follow-up time and outcome for this trial entry
            # Follow-up starts at trial_time t
            max_followup = pt['time'].max()
            
            # Determine outcome status during follow-up
            if outcome_time is not None and outcome_time >= t:
                followup_outcome = 1
                followup_time = outcome_time - t
            else:
                followup_outcome = 0
                followup_time = max_followup - t
            
            expanded_rows.append({
                'ID': pid,
                'trial_time': t,
                'assigned_treatment': assigned_trt,
                'followup_time': followup_time,
                'outcome': followup_outcome,
                'sex': row['sex'],
                'N': row['N'],
                'L': row['L'],
                'P': row['P']
            })
    
    return pd.DataFrame(expanded_rows)

# Build expanded dataset
print("Building expanded dataset (this mimics the 'cloning' step)...")
expanded_df = expand_data_for_trial_emulation(df)

print(f"\nOriginal data: {len(df):,} rows")
print(f"Expanded data: {len(expanded_df):,} rows")
print(f"Expansion factor: {len(expanded_df)/len(df):.1f}x")

print("\n--- Sample of Expanded Data ---")
print(expanded_df.head(20).to_string(index=False))

print("""
WHAT HAPPENED:
- Each patient can contribute multiple "trial entries" (one per eligible time)
- assigned_treatment=1 only at the FIRST time they start treatment
- The same patient appears multiple times with different trial_time values
- This is the "cloning" - we're asking "what if this person entered a trial at time t?"
""")

# =============================================================================
# STEP 4: ESTIMATE THE INTENTION-TO-TREAT (ITT) EFFECT
# =============================================================================

print("=" * 70)
print("STEP 4: Intention-to-Treat (ITT) Analysis")
print("=" * 70)

print("""
ITT ESTIMAND:
"What is the effect of being ASSIGNED to start treatment at time t
 vs. being ASSIGNED to not start treatment at time t?"

This compares outcomes regardless of what happens after assignment.
(Even if someone assigned to treatment stops, they stay in treatment group)
""")

# Simple ITT analysis: compare outcomes by assigned treatment
itt_treated = expanded_df[expanded_df['assigned_treatment'] == 1]
itt_control = expanded_df[expanded_df['assigned_treatment'] == 0]

risk_treated = itt_treated['outcome'].mean()
risk_control = itt_control['outcome'].mean()
risk_difference = risk_treated - risk_control
risk_ratio = risk_treated / risk_control

print(f"\n--- Unadjusted ITT Results ---")
print(f"N assigned to treatment: {len(itt_treated):,}")
print(f"N assigned to control:   {len(itt_control):,}")
print(f"Risk in treated:   {risk_treated:.4f} ({risk_treated*100:.2f}%)")
print(f"Risk in control:   {risk_control:.4f} ({risk_control*100:.2f}%)")
print(f"Risk Difference:   {risk_difference:.4f} ({risk_difference*100:.2f} percentage points)")
print(f"Risk Ratio:        {risk_ratio:.3f}")

# =============================================================================
# STEP 5: ADJUST FOR CONFOUNDERS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 5: Adjusted Analysis (Inverse Probability Weighting)")
print("=" * 70)

print("""
WHY ADJUST?
Even though we're comparing at the same time point, there may be 
confounders that predict BOTH treatment initiation AND the outcome.

We use Inverse Probability of Treatment Weighting (IPTW):
1. Model P(assigned_treatment=1 | confounders)
2. Weight each observation by 1/P (for treated) or 1/(1-P) (for control)
3. This creates a "pseudo-population" where confounders are balanced
""")

# Fit propensity score model
covariates = ['trial_time', 'sex', 'N', 'L', 'P']
X = expanded_df[covariates]
y = expanded_df['assigned_treatment']

ps_model = LogisticRegression(max_iter=1000)
ps_model.fit(X, y)

# Get propensity scores
expanded_df['ps'] = ps_model.predict_proba(X)[:, 1]

print("\n--- Propensity Score Distribution ---")
print(f"Treated mean PS:   {expanded_df.loc[expanded_df['assigned_treatment']==1, 'ps'].mean():.4f}")
print(f"Control mean PS:   {expanded_df.loc[expanded_df['assigned_treatment']==0, 'ps'].mean():.4f}")

# Calculate stabilized IPTW weights
p_trt = expanded_df['assigned_treatment'].mean()
expanded_df['iptw'] = np.where(
    expanded_df['assigned_treatment'] == 1,
    p_trt / expanded_df['ps'],
    (1 - p_trt) / (1 - expanded_df['ps'])
)

# Trim extreme weights
weight_cap = expanded_df['iptw'].quantile(0.99)
expanded_df['iptw_trimmed'] = expanded_df['iptw'].clip(upper=weight_cap)

print(f"\nIPTW weights: mean={expanded_df['iptw_trimmed'].mean():.2f}, "
      f"range=[{expanded_df['iptw_trimmed'].min():.2f}, {expanded_df['iptw_trimmed'].max():.2f}]")

# Weighted ITT analysis
def weighted_mean(values, weights):
    return np.average(values, weights=weights)

treated_mask = expanded_df['assigned_treatment'] == 1
control_mask = expanded_df['assigned_treatment'] == 0

risk_treated_adj = weighted_mean(
    expanded_df.loc[treated_mask, 'outcome'],
    expanded_df.loc[treated_mask, 'iptw_trimmed']
)
risk_control_adj = weighted_mean(
    expanded_df.loc[control_mask, 'outcome'],
    expanded_df.loc[control_mask, 'iptw_trimmed']
)

risk_diff_adj = risk_treated_adj - risk_control_adj
risk_ratio_adj = risk_treated_adj / risk_control_adj

print(f"\n--- IPTW-Adjusted ITT Results ---")
print(f"Risk in treated (weighted):   {risk_treated_adj:.4f} ({risk_treated_adj*100:.2f}%)")
print(f"Risk in control (weighted):   {risk_control_adj:.4f} ({risk_control_adj*100:.2f}%)")
print(f"Risk Difference (adjusted):   {risk_diff_adj:.4f} ({risk_diff_adj*100:.2f} pp)")
print(f"Risk Ratio (adjusted):        {risk_ratio_adj:.3f}")

# =============================================================================
# STEP 6: SUMMARY AND KEY TAKEAWAYS
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY: What We Learned")
print("=" * 70)

print("""
TARGET TRIAL EMULATION - KEY CONCEPTS:

1. THE PROBLEM: 
   - In observational data, treatment timing varies
   - Simple treated/untreated comparisons are confounded
   - We need to compare "apples to apples"

2. THE SOLUTION (CLONING):
   - At each time t, among eligible untreated people:
   - Compare those who START treatment vs those who DON'T
   - This mimics a sequence of randomized trials

3. ITT vs PER-PROTOCOL:
   - ITT: Analyze by initial assignment (ignores adherence)
   - Per-Protocol: Analyze by actual treatment received (requires more assumptions)

4. WEIGHTING:
   - IPTW balances confounders between groups
   - Creates a "pseudo-population" where treatment is independent of confounders

5. INTERPRETATION:
   - Risk Ratio < 1: Treatment reduces outcome risk
   - Risk Ratio > 1: Treatment increases outcome risk
   - Risk Ratio = 1: No effect

NEXT STEPS:
- Tutorial 2: Build MSM with IPTW from scratch
- Tutorial 3: Parametric G-formula for simulating counterfactuals
""")

print("\n" + "=" * 70)
print("Results Comparison")
print("=" * 70)
print(f"{'Metric':<30} {'Unadjusted':<15} {'IPTW-Adjusted':<15}")
print("-" * 60)
print(f"{'Risk Difference':<30} {risk_difference:>+.4f}       {risk_diff_adj:>+.4f}")
print(f"{'Risk Ratio':<30} {risk_ratio:>.3f}          {risk_ratio_adj:>.3f}")
