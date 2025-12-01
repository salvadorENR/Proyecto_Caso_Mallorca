import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pulp import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
import sys
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

# ==========================================
# 0. ROBUST LOGGING
# ==========================================
class Logger(object):
    def __init__(self, filename='optimization_results.txt'):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8', buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger()

print("=====================================================================")
print("                OPTIMIZATION & ANALYSIS REPORT - MALLORCA             ")
print("=====================================================================\n")

# ==========================================
# 1. LOAD AND PREPARE DATA (WITH ROBUST CLEANING)
# ==========================================
if not os.path.exists("Mallorca.csv"):
    print("Error: 'Mallorca.csv' not found.")
    exit()

df = pd.read_csv("Mallorca.csv")

if 'Year' in df.columns:
    df.set_index('Year', inplace=True)
df.sort_index(inplace=True)

print("Dataset shape:", df.shape)
print(f"Years: {df.index.min()} to {df.index.max()}")

# COMPREHENSIVE DATA CLEANING
print("\n--- Comprehensive Data Cleaning ---")

# 1. Check for and remove any completely empty columns
empty_cols = df.columns[df.isnull().all()].tolist()
if empty_cols:
    print(f"Removing empty columns: {empty_cols}")
    df = df.drop(columns=empty_cols)

# 2. Check for infinite values
inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
if inf_count > 0:
    print(f"Found {inf_count} infinite values. Replacing with NaN...")
    df = df.replace([np.inf, -np.inf], np.nan)

# 3. Handle missing values
missing_before = df.isnull().sum().sum()
print(f"Missing values before cleaning: {missing_before}")

if missing_before > 0:
    print("Column-wise missing values:")
    print(df.isnull().sum())
    
    # Fill missing values with column means
    df_clean = df.fillna(df.mean())
    
    # For any remaining NaN (if whole column was NaN), fill with 0
    df_clean = df_clean.fillna(0)
    
    missing_after = df_clean.isnull().sum().sum()
    print(f"Missing values after cleaning: {missing_after}")
else:
    df_clean = df.copy()
    print("No missing values found.")

# 4. Final sanity check
if df_clean.isnull().sum().sum() > 0 or np.isinf(df_clean.select_dtypes(include=[np.number])).sum().sum() > 0:
    print("WARNING: Data still contains invalid values after cleaning!")
    # Emergency cleanup - drop any remaining problematic rows
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna()
    print(f"Final dataset shape: {df_clean.shape}")

print("Data cleaning complete.")

# Prepare features and target
if 'SustainableTourismIndex' in df_clean.columns:
    y = df_clean['SustainableTourismIndex']
    X = df_clean.drop('SustainableTourismIndex', axis=1)
    print("Using existing SustainableTourismIndex as target")
else:
    X = df_clean.copy()
    # Normalize all variables to [0,1] range for meaningful weighted average
    X_normalized = (X - X.min()) / (X.max() - X.min())
    weights = [1/len(X.columns)] * len(X.columns)
    y = X_normalized.dot(weights)
    df_clean['y'] = y
    print("Created normalized target variable 'y'")

print(f"Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
print("Features:", X.columns.tolist())

# ==============================================================================
# PART 1: EXPLORATORY DATA ANALYSIS
# ==============================================================================
def exploratory_data_analysis(df):
    print("\n" + "="*60)
    print("PART 1: EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    print("\nDataset Overview:")
    print(df.describe())
    
    print("\nCorrelation with Target:")
    correlations = df.corr()['y'].sort_values(ascending=False)
    for var, corr in correlations.items():
        print(f"  {var}: {corr:.4f}")
    
    # Enhanced visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Time series
    axes[0,0].plot(df.index, df['y'], marker='o', linewidth=2, color='purple')
    axes[0,0].set_title('Sustainable Tourism Index Over Time', fontweight='bold')
    axes[0,0].set_ylabel('Index Value')
    axes[0,0].grid(True, alpha=0.3)
    
    # Correlation heatmap
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=axes[0,1])
    axes[0,1].set_title('Correlation Matrix', fontweight='bold')
    
    # Distribution
    axes[1,0].hist(df['y'], bins=8, alpha=0.7, color='green', edgecolor='black')
    axes[1,0].set_title('Distribution of Target Variable', fontweight='bold')
    axes[1,0].set_xlabel('Index Value')
    
    # Top correlated variable
    top_var = correlations.index[1]
    axes[1,1].scatter(df[top_var], df['y'], alpha=0.7)
    axes[1,1].set_xlabel(top_var)
    axes[1,1].set_ylabel('y')
    axes[1,1].set_title(f'{top_var} vs Target (r={correlations[top_var]:.3f})', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('EDA_Plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return correlations

correlations = exploratory_data_analysis(df_clean.copy())

# ==============================================================================
# PART 2: REGRESSION ANALYSIS WITH MATHEMATICAL FORMULATIONS
# ==============================================================================
def print_regression_equation(coefficients, feature_names, intercept, method_name):
    """Print readable regression equation"""
    print(f"\n{method_name} Equation:")
    equation = f"y = {intercept:.6f}"
    for i, coef in enumerate(coefficients):
        if abs(coef) > 1e-10:  # Avoid printing near-zero coefficients
            sign = " + " if coef >= 0 else " - "
            equation += f"{sign}{abs(coef):.6f}*{feature_names[i]}"
    print(equation)

def compare_regression_methods(X_data, y_data):
    print("\n" + "="*60)
    print("PART 2: REGRESSION ANALYSIS")
    print("="*60)
    
    # DATA VALIDATION CHECK
    print("\n--- Data Validation ---")
    if X_data.isnull().sum().sum() > 0 or y_data.isnull().sum() > 0:
        print("WARNING: Data contains NaN values. Cleaning...")
        X_data = X_data.fillna(X_data.mean())
        X_data = X_data.fillna(0)
        y_data = y_data.fillna(y_data.mean())
        y_data = y_data.fillna(0)
    
    if np.isinf(X_data.select_dtypes(include=[np.number])).sum().sum() > 0:
        print("WARNING: Data contains infinite values. Cleaning...")
        X_data = X_data.replace([np.inf, -np.inf], np.nan)
        X_data = X_data.fillna(X_data.mean())
        X_data = X_data.fillna(0)
    
    if np.isinf(y_data).sum() > 0:
        print("WARNING: Target contains infinite values. Cleaning...")
        y_data = y_data.replace([np.inf, -np.inf], np.nan)
        y_data = y_data.fillna(y_data.mean())
        y_data = y_data.fillna(0)
    
    print("Data validation complete.")
    
    n_samples, n_features = X_data.shape
    feature_names = X_data.columns.tolist()
    results_list = []

    # =================================================================
    # QUESTION 2: L1 NORM FORMULATION
    # =================================================================
    print("\n--- QUESTION 2: L1 Norm (Least Absolute Deviations) ---")
    print("Mathematical Formulation:")
    print("Minimize: Σ|y_i - (β₀ + Σβ_j*x_ij)|")
    print("Subject to: y_i - (β₀ + Σβ_j*x_ij) = u_i - v_i, u_i, v_i ≥ 0")
    print("Equivalent LP: Minimize Σ(u_i + v_i)")
    
    prob_l1 = LpProblem("L1_Regression", LpMinimize)
    beta_l1 = [LpVariable(f"b_{j}", cat='Continuous') for j in range(n_features)]
    beta0_l1 = LpVariable("b0", cat='Continuous')
    u = [LpVariable(f"u_{i}", lowBound=0) for i in range(n_samples)]
    v = [LpVariable(f"v_{i}", lowBound=0) for i in range(n_samples)]
    
    prob_l1 += lpSum(u) + lpSum(v)
    
    for i in range(n_samples):
        prediction = beta0_l1 + lpSum([beta_l1[j] * X_data.iloc[i,j] for j in range(n_features)])
        prob_l1 += (y_data.iloc[i] - prediction == u[i] - v[i])
    
    prob_l1.solve(PULP_CBC_CMD(msg=0))
    
    if prob_l1.status == 1:
        l1_intercept = value(beta0_l1)
        l1_coeffs = [value(beta_l1[j]) for j in range(n_features)]
        l1_pred = [l1_intercept + sum(l1_coeffs[j] * X_data.iloc[i,j] for j in range(n_features)) 
                  for i in range(n_samples)]
        
        print_regression_equation(l1_coeffs, feature_names, l1_intercept, "L1 Norm")
        l1_mae = value(prob_l1.objective) / n_samples
        print(f"MAE (L1): {l1_mae:.6f}")
        results_list.append({
            "Model": "L1 (Least Absolute)", 
            "MAE": l1_mae,
            "MSE": mean_squared_error(y_data, l1_pred),
            "Max_Error": max_error(y_data, l1_pred)
        })
    else:
        print("L1 regression failed")

    # =================================================================
    # QUESTION 3: L∞ NORM FORMULATION  
    # =================================================================
    print("\n--- QUESTION 3: L∞ Norm (Minimax) ---")
    print("Mathematical Formulation:")
    print("Minimize: max|y_i - (β₀ + Σβ_j*x_ij)|")
    print("Subject to: |y_i - (β₀ + Σβ_j*x_ij)| ≤ M for all i")
    print("Equivalent LP: Minimize M subject to -M ≤ residual_i ≤ M")
    
    prob_inf = LpProblem("L_inf_Regression", LpMinimize)
    beta_inf = [LpVariable(f"bi_{j}", cat='Continuous') for j in range(n_features)]
    beta0_inf = LpVariable("bi0", cat='Continuous')
    max_dev = LpVariable("max_dev", lowBound=0)
    
    prob_inf += max_dev
    
    for i in range(n_samples):
        prediction = beta0_inf + lpSum([beta_inf[j] * X_data.iloc[i,j] for j in range(n_features)])
        prob_inf += (y_data.iloc[i] - prediction <= max_dev)
        prob_inf += (prediction - y_data.iloc[i] <= max_dev)
    
    prob_inf.solve(PULP_CBC_CMD(msg=0))
    
    if prob_inf.status == 1:
        linf_intercept = value(beta0_inf)
        linf_coeffs = [value(beta_inf[j]) for j in range(n_features)]
        linf_pred = [linf_intercept + sum(linf_coeffs[j] * X_data.iloc[i,j] for j in range(n_features)) 
                    for i in range(n_samples)]
        
        print_regression_equation(linf_coeffs, feature_names, linf_intercept, "L∞ Norm")
        linf_max_error = value(max_dev)
        print(f"Maximum Absolute Error (L∞): {linf_max_error:.6f}")
        results_list.append({
            "Model": "L∞ (Minimax)", 
            "MAE": mean_absolute_error(y_data, linf_pred),
            "MSE": mean_squared_error(y_data, linf_pred),
            "Max_Error": linf_max_error
        })
    else:
        print("L∞ regression failed")

    # =================================================================
    # QUESTION 4: L2 NORM (LEAST SQUARES)
    # =================================================================
    print("\n--- QUESTION 4: L2 Norm (Least Squares) ---")
    l2_model = LinearRegression()
    l2_model.fit(X_data, y_data)
    l2_pred = l2_model.predict(X_data)
    
    print_regression_equation(l2_model.coef_, feature_names, l2_model.intercept_, "L2 Norm")
    print(f"R² Score: {l2_model.score(X_data, y_data):.4f}")
    print(f"MSE: {mean_squared_error(y_data, l2_pred):.6f}")
    
    results_list.append({
        "Model": "L2 (Least Squares)", 
        "MAE": mean_absolute_error(y_data, l2_pred),
        "MSE": mean_squared_error(y_data, l2_pred),
        "Max_Error": max_error(y_data, l2_pred),
        "R²": l2_model.score(X_data, y_data)
    })

    # =================================================================
    # QUESTION 5: COMPARISON PLOT
    # =================================================================
    plt.figure(figsize=(12, 6))
    plt.plot(df_clean.index, y_data, 'ko-', label='Actual', linewidth=2, markersize=4)
    
    if 'l1_pred' in locals():
        plt.plot(df_clean.index, l1_pred, 'ro--', label='L1 Prediction', alpha=0.8)
    if 'linf_pred' in locals():  
        plt.plot(df_clean.index, linf_pred, 'bs-.', label='L∞ Prediction', alpha=0.8)
    
    plt.plot(df_clean.index, l2_pred, 'g^:', label='L2 Prediction', alpha=0.8)
    
    plt.title('Comparison of Regression Methods (Question 5)', fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel('Sustainable Tourism Index')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('Regression_Comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Performance summary
    results_df = pd.DataFrame(results_list)
    print("\n" + "="*50)
    print("REGRESSION PERFORMANCE SUMMARY")
    print("="*50)
    print(results_df.to_string(index=False))

compare_regression_methods(X, y)

# ==============================================================================
# PART 3: COUNTERFACTUAL OPTIMIZATION (FIXED)
# ==============================================================================
def solve_counterfactual_robust(target_increase_pct, max_vars=None, logical_constraints=False, description=""):
    print(f"\n{description}")
    
    # Use normalized data for counterfactual analysis
    X_normalized = (X - X.min()) / (X.max() - X.min())
    x_current = X_normalized.iloc[-1].values
    std_devs = X_normalized.std().values
    weights = [1/len(X.columns)] * len(X.columns)
    y_current = np.dot(x_current, weights)
    
    p = len(X.columns)
    feature_names = X.columns.tolist()
    
    # =================================================================
    # QUESTION 6: COUNTERFACTUAL FORMULATION
    # =================================================================
    if "Q7" in description or "Q8" in description:
        print("\n--- QUESTION 6: Counterfactual Optimization Formulation ---")
        print("Objective: Minimize Σβ_j + k⋅Σδ_j")
        print("Subject to:")
        print("  Σw_j⋅β_j ≥ ε (Target improvement)")
        print("  β_j ≤ a⋅σ_j (Bound by standard deviation)") 
        print("  m⋅δ_j ≤ β_j ≤ M⋅δ_j (Linking constraints)")
        print("  μ_l ≤ Σδ_j ≤ μ_u (Variable selection bounds)")
        print("  0 ≤ x_j + β_j ≤ 1 (Range preservation)")
    
    prob = LpProblem("Counterfactual_Optimization", LpMinimize)
    
    # Decision variables
    beta = [LpVariable(f"beta_{i}", lowBound=0, upBound=1-x_current[i]) for i in range(p)]
    delta = [LpVariable(f"delta_{i}", cat='Binary') for i in range(p)]
    
    # Objective function (minimize changes + small penalty for number of changes)
    prob += lpSum(beta) + 0.001 * lpSum(delta)
    
    # Target constraint
    target_gain = y_current * target_increase_pct
    prob += lpSum([beta[i] * weights[i] for i in range(p)]) >= target_gain
    
    # Bound constraints (more conservative)
    alpha = 0.5  # More conservative bound
    M = 10       # Reasonable Big-M value
    
    for i in range(p):
        # Change bounded by standard deviation
        prob += beta[i] <= alpha * (std_devs[i] if std_devs[i] > 0 else 0.1)
        # Linking constraint
        prob += beta[i] <= M * delta[i]
        # Small activation threshold
        prob += 0.001 * delta[i] <= beta[i]
    
    # Variable selection constraints
    if max_vars is not None:
        prob += lpSum(delta) <= max_vars
    else:
        prob += lpSum(delta) >= 1  # At least one variable must change
    
    # Logical constraints (Question 10)
    if logical_constraints:
        idx = {name: i for i, name in enumerate(feature_names)}
        
        # a) MaritimeTraffic → PassengersArriving
        if 'MaritimeTraffic' in idx and 'PassengersArriving' in idx:
            prob += delta[idx['MaritimeTraffic']] <= delta[idx['PassengersArriving']]
        
        # b) Exactly one of SchoolingRate OR RenewableResources
        if 'SchoolingRate' in idx and 'RenewableResources' in idx:
            prob += delta[idx['SchoolingRate']] + delta[idx['RenewableResources']] == 1
        
        # c) At least one of Poverty OR VehicleRegistration
        if 'Poverty' in idx and 'VehicleRegistration' in idx:
            prob += delta[idx['Poverty']] + delta[idx['VehicleRegistration']] >= 1
    
    # Solve
    prob.solve(PULP_CBC_CMD(msg=0))
    
    print(f"Status: {LpStatus[prob.status]}")
    
    if prob.status == LpStatusOptimal:
        # Calculate results
        beta_values = [beta[i].varValue for i in range(p)]
        new_values = [x_current[i] + beta_values[i] for i in range(p)]
        new_y = np.dot(new_values, weights)
        improvement = (new_y - y_current) / y_current
        
        print("Recommended Changes:")
        changes_made = False
        for i in range(p):
            if delta[i].varValue > 0.5 and beta_values[i] > 1e-6:
                original_scale = X.iloc[-1, i]
                change_scale = beta_values[i] * (X.iloc[:, i].max() - X.iloc[:, i].min())
                print(f"  {feature_names[i]}: +{beta_values[i]:.4f} "
                      f"(Original: {original_scale:.2f} → New: {original_scale + change_scale:.2f})")
                changes_made = True
        
        if not changes_made:
            print("  No significant changes needed")
        
        print(f"Target improvement: {target_increase_pct*100:.1f}%")
        print(f"Achieved improvement: {improvement*100:.2f}%")
        print(f"Variables changed: {int(sum(delta[i].varValue for i in range(p)))}")
        
        return True
    else:
        print("No feasible solution found with current constraints.")
        print("Try relaxing constraints or reducing target improvement.")
        return False

print("\n" + "="*60)
print("PART 3: COUNTERFACTUAL SCENARIOS")
print("="*60)

# Execute scenarios with better parameters
solve_counterfactual_robust(0.01, description="Q7a: 1% Increase (Minimal changes)")
solve_counterfactual_robust(0.05, max_vars=4, description="Q7b: 5% Increase (Max 4 variables)")
solve_counterfactual_robust(0.25, max_vars=1, description="Q8: 25% Increase (Single variable investment)")
solve_counterfactual_robust(0.01, logical_constraints=True, description="Q10: Logical Constraints Test")

print("\n=====================================================================")
print("                       ANALYSIS COMPLETE                             ")
print("All results saved to: 'optimization_results.txt'")
print("Visualizations saved to: 'EDA_Plots.png', 'Regression_Comparison.png'")
print("=====================================================================")