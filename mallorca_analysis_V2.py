import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pulp import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, r2_score
import warnings
import sys
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

# ==========================================
# 0. LOGGING SETUP
# ==========================================
class Logger(object):
    def __init__(self, filename='optimization_results_dropNA_normalized.txt'):
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
print("      OPTIMIZATION REPORT - STRATEGY B: LISTWISE DELETION            ")
print("      (Prioritizing Data Integrity over Historical Length)           ")
print("      With Normalization for Raw Data Compatibility                  ")
print("=====================================================================\n")

# ==========================================
# 1. LOAD DATA
# ==========================================
if not os.path.exists("Mallorca.csv"):
    print("Error: 'Mallorca.csv' not found.")
    exit()

df = pd.read_csv("Mallorca.csv")

if 'Year' in df.columns:
    df.set_index('Year', inplace=True)
df.sort_index(inplace=True)

print(f"Original Dataset shape: {df.shape}")
print(f"Original Years: {df.index.min()} to {df.index.max()}")

# ==========================================
# 2. DATA CLEANING (STRATEGY B: DROP NA)
# ==========================================
print("\n--- Cleaning Strategy B: Dropping Rows with Missing Values ---")

# Define Target Column Name
target_col = 'SustainableTourismIndex'

# 1. INITIAL CLEANING: Drop rows with missing values first
# We do this first to ensure X and y are aligned
df_clean = df.dropna()

# 2. SEPARATE TARGET (y) AND FEATURES (X)
# This logic is fixed to prevent 'y' from appearing inside 'X'

if target_col in df_clean.columns:
    # Case A: Target exists in CSV
    y = df_clean[target_col]
    # Drop the target from the dataframe to create X
    X = df_clean.drop(columns=[target_col])
    
    # SAFETY CHECK: If there is also a column named 'y' from a previous run, drop it too
    if 'y' in X.columns:
        X = X.drop(columns=['y'])
        
else:
    # Case B: Target does not exist, we must calculate it
    print(f"Target column '{target_col}' not found. Calculating index from features...")
    
    # Take all columns as features initially
    X = df_clean.copy()
    
    # SAFETY CHECK: Drop 'y' if it exists from a previous run
    if 'y' in X.columns:
        X = X.drop(columns=['y'])

    # Normalize features to [0,1] just for the index calculation
    X_norm_calc = (X - X.min()) / (X.max() - X.min())
    weights = [1/len(X.columns)] * len(X.columns)
    y = X_norm_calc.dot(weights)

# 3. Finalize df_clean for display/plots
df_clean['y'] = y

rows_before = len(df)
rows_after = len(df_clean)
dropped_count = rows_before - rows_after

print(f"Dropped {dropped_count} rows containing missing values.")
print(f"Final Clean Dataset shape: {df_clean.shape}")
print(f"Remaining Years: {df_clean.index.tolist()}")

# Print clean data sample (User Request)
print("\n--- Clean Data (All Rows) ---")
print(df_clean)

# FINAL SAFETY CHECK
# This guarantees X (Features) and y (Target) are distinct
col_intersection = set(X.columns).intersection(set(['y', target_col]))
if col_intersection:
    print(f"\nCRITICAL WARNING: X still contains target columns: {col_intersection}")
    X = X.drop(columns=list(col_intersection))
    print("Fixed: Target columns removed from X.")

if len(df_clean) < 5:
    print("\nWARNING: Dataset is very small (< 5 observations). Regression results might be overfitting.")

# ==============================================================================
# PART 1: EXPLORATORY DATA ANALYSIS
# ==============================================================================
def exploratory_data_analysis(df):
    print("\n" + "="*60)
    print("PART 1: EXPLORATORY DATA ANALYSIS (CLEAN DATA)")
    print("="*60)
    
    print("\nCorrelation with Target:")
    correlations = df.corr()['y'].sort_values(ascending=False)
    print(correlations.head())
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0,0].plot(df.index, df['y'], marker='o', color='purple')
    axes[0,0].set_title('Sustainable Tourism Index (Clean Data)')
    axes[0,0].grid(True)
    
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=axes[0,1])
    axes[0,1].set_title('Correlation Matrix')
    
    sns.histplot(df['y'], kde=True, ax=axes[1,0], color='green')
    axes[1,0].set_title('Distribution of Index (y)')
    
    if len(correlations) > 1:
        top_var = correlations.index[1]
        axes[1,1].scatter(df[top_var], df['y'], alpha=0.7)
        axes[1,1].set_xlabel(top_var)
        axes[1,1].set_ylabel('y')
        axes[1,1].set_title(f'{top_var} vs y')
    
    plt.tight_layout()
    plt.savefig('EDA_Plots_DropNA.png')
    # plt.show() # Commented out to prevent blocking in some environments

exploratory_data_analysis(df_clean.copy())

# ==============================================================================
# PART 2: REGRESSION ANALYSIS
# ==============================================================================
def compare_regression_methods(X_data, y_data):
    print("\n" + "="*60)
    print("PART 2: REGRESSION ANALYSIS")
    print("="*60)
    
    n_samples, n_features = X_data.shape
    feature_names = X_data.columns.tolist()
    results_list = []

    # --- 1. L2 Norm (Least Squares) ---
    l2_model = LinearRegression()
    l2_model.fit(X_data, y_data)
    l2_pred = l2_model.predict(X_data)
    
    mae_l2 = mean_absolute_error(y_data, l2_pred)
    mse_l2 = mean_squared_error(y_data, l2_pred)
    rmse_l2 = np.sqrt(mse_l2)
    max_l2 = max_error(y_data, l2_pred)
    r2_l2 = r2_score(y_data, l2_pred)
    
    results_list.append({"Model": "L2 (Least Squares)", "MAE": mae_l2, "RMSE": rmse_l2, "Max_Error": max_l2, "R2": r2_l2})

    # --- 2. L1 Norm (Least Absolute Deviations) ---
    prob_l1 = LpProblem("L1_Regression", LpMinimize)
    beta = [LpVariable(f"b_{j}", cat='Continuous') for j in range(n_features)]
    beta0 = LpVariable("b0", cat='Continuous')
    u = [LpVariable(f"u_{i}", lowBound=0) for i in range(n_samples)]
    
    prob_l1 += lpSum(u)
    
    for i in range(n_samples):
        # Calculate predicted value using list comprehension
        y_hat = beta0 + lpSum([beta[j] * X_data.iloc[i,j] for j in range(n_features)])
        prob_l1 += y_data.iloc[i] - y_hat <= u[i]
        prob_l1 += y_hat - y_data.iloc[i] <= u[i]
        
    prob_l1.solve(PULP_CBC_CMD(msg=0))
    
    l1_pred = [value(beta0) + sum(value(beta[j]) * X_data.iloc[i,j] for j in range(n_features)) for i in range(n_samples)]
    mae_l1 = mean_absolute_error(y_data, l1_pred)
    results_list.append({"Model": "L1 (Least Abs Dev)", "MAE": mae_l1, "RMSE": np.sqrt(mean_squared_error(y_data, l1_pred)), "Max_Error": max_error(y_data, l1_pred), "R2": r2_score(y_data, l1_pred)})

    # --- 3. L_inf Norm (Minimax) ---
    prob_inf = LpProblem("L_inf_Regression", LpMinimize)
    beta_inf = [LpVariable(f"bi_{j}", cat='Continuous') for j in range(n_features)]
    beta0_inf = LpVariable("bi0", cat='Continuous')
    max_dev = LpVariable("max_dev", lowBound=0)
    
    prob_inf += max_dev
    
    for i in range(n_samples):
        y_hat = beta0_inf + lpSum([beta_inf[j] * X_data.iloc[i,j] for j in range(n_features)])
        prob_inf += y_data.iloc[i] - y_hat <= max_dev
        prob_inf += y_hat - y_data.iloc[i] <= max_dev
        
    prob_inf.solve(PULP_CBC_CMD(msg=0))
    linf_pred = [value(beta0_inf) + sum(value(beta_inf[j]) * X_data.iloc[i,j] for j in range(n_features)) for i in range(n_samples)]
    results_list.append({"Model": "L_inf (Minimax)", "MAE": mean_absolute_error(y_data, linf_pred), "RMSE": np.sqrt(mean_squared_error(y_data, linf_pred)), "Max_Error": value(max_dev), "R2": r2_score(y_data, linf_pred)})
    
    # --- PRINT EQUATIONS (FIXED) ---
    print("\n" + "-"*30)
    print("MODEL EQUATIONS (Coefficients)")
    print("-"*30)
    
    # L2 Equation
    l2_coeffs = dict(zip(feature_names, l2_model.coef_))
    print(f"\n[L2 OLS] y = {l2_model.intercept_:.4f} + " + " + ".join([f"{coef:.4f}*{name}" for name, coef in l2_coeffs.items()]))
    
    # L1 Equation
    l1_coeffs = {name: value(beta[i]) for i, name in enumerate(feature_names)}
    print(f"\n[L1 LAD] y = {value(beta0):.4f} + " + " + ".join([f"{coef:.4f}*{name}" for name, coef in l1_coeffs.items()]))
    
    # L_inf Equation
    linf_coeffs = {name: value(beta_inf[i]) for i, name in enumerate(feature_names)}
    print(f"\n[L_inf Minimax] y = {value(beta0_inf):.4f} + " + " + ".join([f"{coef:.4f}*{name}" for name, coef in linf_coeffs.items()]))
    print("-" * 60)

    # Print Metrics Table
    res_df = pd.DataFrame(results_list)
    print("\n" + res_df.to_string(index=False))
    
    # Plot Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(y_data.index, y_data, 'ko-', label='Actual')
    plt.plot(y_data.index, l2_pred, 'g--', label='L2 Pred')
    plt.title("Regression Comparison (Dropped NA Strategy)")
    plt.legend()
    plt.savefig('Regression_Comparison_DropNA.png')
    # plt.show()

compare_regression_methods(X, y)

# ==============================================================================
# PART 3: COUNTERFACTUAL OPTIMIZATION (NORMALIZED)
# ==============================================================================
def solve_counterfactual_robust(target_increase_pct, max_vars=None, logical_constraints=False, description=""):
    print(f"\n{description}")
    
    # IMPORTANT: Use Cleaned X from global scope (Strategy B)
    # Normalize X to [0,1] for optimization stability
    X_normalized = (X - X.min()) / (X.max() - X.min())
    
    # Check for NaN created by normalization (e.g. if max=min in a column)
    if X_normalized.isnull().sum().sum() > 0:
        X_normalized = X_normalized.fillna(0)

    # Get latest year from the CLEANED dataset
    x_current = X_normalized.iloc[-1].values
    std_devs = X_normalized.std().values
    
    # Weights
    weights = [1/len(X.columns)] * len(X.columns)
    
    # Calculate current y based on normalized inputs
    y_current = np.dot(x_current, weights)
    
    p = len(X.columns)
    feature_names = X.columns.tolist()
    
    prob = LpProblem("Counterfactual_Optimization", LpMinimize)
    
    # Variables: beta (change) and delta (binary decision)
    beta = [LpVariable(f"beta_{i}", lowBound=0, upBound=1-x_current[i]) for i in range(p)]
    delta = [LpVariable(f"delta_{i}", cat='Binary') for i in range(p)]
    
    # Objective: Minimize sum of changes + penalty for number of changes
    prob += lpSum(beta) + 0.001 * lpSum(delta)
    
    # Constraint: Target Increase
    target_gain = y_current * target_increase_pct
    prob += lpSum([beta[i] * weights[i] for i in range(p)]) >= target_gain
    
    alpha = 0.5  
    M = 10       
    
    # Constraints: Bounds and Linking
    for i in range(p):
        prob += beta[i] <= alpha * (std_devs[i] if std_devs[i] > 0 else 0.1)
        prob += beta[i] <= M * delta[i]
        prob += 0.001 * delta[i] <= beta[i]
    
    # Cardinality Constraints
    if max_vars is not None:
        prob += lpSum(delta) <= max_vars
    else:
        prob += lpSum(delta) >= 1
    
    # Logical Constraints (Q10)
    if logical_constraints:
        idx = {name: i for i, name in enumerate(feature_names)}
        
        # a) Maritime -> Passengers
        if 'MaritimeTraffic' in idx and 'PassengersArriving' in idx:
            prob += delta[idx['MaritimeTraffic']] <= delta[idx['PassengersArriving']]
            
        # b) XOR Schooling / Renewable
        if 'SchoolingRate' in idx and 'RenewableResources' in idx:
            prob += delta[idx['SchoolingRate']] + delta[idx['RenewableResources']] == 1
            
        # c) OR Poverty / Vehicle
        if 'Poverty' in idx and 'VehicleRegistration' in idx:
            prob += delta[idx['Poverty']] + delta[idx['VehicleRegistration']] >= 1
    
    prob.solve(PULP_CBC_CMD(msg=0))
    
    print(f"Status: {LpStatus[prob.status]}")
    
    if prob.status == LpStatusOptimal:
        beta_values = [beta[i].varValue for i in range(p)]
        new_y = y_current + np.dot(beta_values, weights) 
        improvement = (new_y - y_current) / y_current
        
        print("Recommended Changes:")
        changes_made = False
        for i in range(p):
            if delta[i].varValue > 0.5 and beta_values[i] > 1e-6:
                # Denormalize to show real scale
                original_scale = X.iloc[-1, i]
                range_scale = X.iloc[:, i].max() - X.iloc[:, i].min()
                # If range is 0 (constant column), avoid error
                if range_scale == 0: range_scale = 1
                
                change_scale = beta_values[i] * range_scale
                
                print(f"  {feature_names[i]}: +{beta_values[i]:.4f} (Norm) | +{change_scale:,.2f} (Real)")
                changes_made = True
        
        if not changes_made:
            print("  No significant changes needed")
        
        print(f"Target: {target_increase_pct*100:.1f}% -> Achieved: {improvement*100:.2f}%")
        return True
    else:
        print("No feasible solution.")
        return False

print("\n" + "="*60)
print("PART 3: COUNTERFACTUAL SCENARIOS")
print("="*60)

solve_counterfactual_robust(0.01, description="Q7a: 1% Increase")
solve_counterfactual_robust(0.05, max_vars=4, description="Q7b: 5% Increase (Max 4 vars)")
solve_counterfactual_robust(0.25, max_vars=1, description="Q8: 25% Increase (1 var)")
solve_counterfactual_robust(0.01, logical_constraints=True, description="Q10: Logic Test")

print("\nAnalysis Complete. Results saved to 'optimization_results_dropNA_normalized.txt'")