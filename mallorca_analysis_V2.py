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
print("      (With Separate Windows for Every Plot)                         ")
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

# ==========================================
# 2. DATA CLEANING (ROBUST FIX)
# ==========================================
print("\n--- Cleaning Strategy B: Dropping Rows with Missing Values ---")

target_col = 'SustainableTourismIndex'

# 1. Drop NA rows first
df_clean = df.dropna()

# 2. SEPARATE TARGET (y) AND FEATURES (X)
if target_col in df_clean.columns:
    y = df_clean[target_col]
    X = df_clean.drop(columns=[target_col])
    # Safety: Drop 'y' if it exists in X from a previous run
    if 'y' in X.columns: X = X.drop(columns=['y'])
else:
    # If target missing, calculate it
    print(f"Target column '{target_col}' not found. Calculating index...")
    X = df_clean.copy()
    if 'y' in X.columns: X = X.drop(columns=['y'])
    
    X_norm_calc = (X - X.min()) / (X.max() - X.min())
    weights = [1/len(X.columns)] * len(X.columns)
    y = X_norm_calc.dot(weights)

# Add y back to df_clean for EDA purposes (but keep X pure for regression)
df_clean['y'] = y

# Final Safety Check for Data Leakage
common_cols = set(X.columns).intersection(set(['y', target_col]))
if common_cols:
    X = X.drop(columns=list(common_cols))

print(f"Final Clean Dataset shape: {df_clean.shape}")

# ==============================================================================
# PART 1: EXPLORATORY DATA ANALYSIS (SEPARATE WINDOWS)
# ==============================================================================
def exploratory_data_analysis(df):
    print("\n" + "="*60)
    print("PART 1: EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # --- Plot 1: Time Series ---
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['y'], marker='o', color='purple', linewidth=2)
    plt.title('Sustainable Tourism Index over Time')
    plt.ylabel('Index Value')
    plt.xlabel('Year')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('EDA_TimeSeries.png')
    print("Displaying Time Series Plot... (Close window to continue)")
    plt.show()
    
    # --- Plot 2: Correlation Matrix ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('EDA_Correlation.png')
    print("Displaying Correlation Matrix... (Close window to continue)")
    plt.show()
    
    # --- B. Box Plots for Key Predictors ---
    print("\nGenerating Box Plots for Top Predictors...")
    key_vars = ['MaritimeTraffic', 'PassengersArriving', 'Poverty']
    valid_vars = [v for v in key_vars if v in df.columns]
    
    if valid_vars:
        for col in valid_vars:
            # Create a separate figure for EACH variable
            plt.figure(figsize=(8, 6))
            
            # Create bins for the continuous variable
            plot_df = df.copy()
            try:
                plot_df['Category'] = pd.qcut(plot_df[col], q=3, labels=["Low", "Medium", "High"])
            except ValueError:
                plot_df['Category'] = pd.cut(plot_df[col], bins=3, labels=["Low", "Medium", "High"])
            
            sns.boxplot(x='Category', y='y', data=plot_df, palette="Set2")
            sns.stripplot(x='Category', y='y', data=plot_df, color='black', alpha=0.5)
            
            plt.title(f'Impact of {col}\non Tourism Index')
            plt.ylabel('Sustainable Tourism Index (y)')
            plt.xlabel(f'{col} Level')
            
            plt.tight_layout()
            plt.savefig(f'EDA_BoxPlot_{col}.png')
            print(f"Displaying Box Plot for {col}... (Close window to continue)")
            plt.show()
    else:
        print("Warning: Requested variables for box plots not found in dataset.")

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

    # --- 1. L2 Norm (OLS) ---
    l2_model = LinearRegression()
    l2_model.fit(X_data, y_data)
    l2_pred = l2_model.predict(X_data)
    
    results_list.append({
        "Model": "L2 (Least Squares)", 
        "MAE": mean_absolute_error(y_data, l2_pred), 
        "R2": r2_score(y_data, l2_pred)
    })

    # --- 2. L1 Norm (LAD) ---
    prob_l1 = LpProblem("L1_Regression", LpMinimize)
    beta = [LpVariable(f"b_{j}", cat='Continuous') for j in range(n_features)]
    beta0 = LpVariable("b0", cat='Continuous')
    u = [LpVariable(f"u_{i}", lowBound=0) for i in range(n_samples)]
    
    prob_l1 += lpSum(u)
    for i in range(n_samples):
        y_hat = beta0 + lpSum([beta[j] * X_data.iloc[i,j] for j in range(n_features)])
        prob_l1 += y_data.iloc[i] - y_hat <= u[i]
        prob_l1 += y_hat - y_data.iloc[i] <= u[i]
    prob_l1.solve(PULP_CBC_CMD(msg=0))
    
    l1_pred = [value(beta0) + sum(value(beta[j]) * X_data.iloc[i,j] for j in range(n_features)) for i in range(n_samples)]
    results_list.append({
        "Model": "L1 (Least Abs Dev)", 
        "MAE": mean_absolute_error(y_data, l1_pred), 
        "R2": r2_score(y_data, l1_pred)
    })

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
    results_list.append({
        "Model": "L_inf (Minimax)", 
        "MAE": mean_absolute_error(y_data, linf_pred), 
        "R2": r2_score(y_data, linf_pred)
    })
    
    # --- PRINT EQUATIONS ---
    print("\n" + "-"*30)
    print("MODEL EQUATIONS (Coefficients)")
    print("-"*30)
    
    l2_coeffs = dict(zip(feature_names, l2_model.coef_))
    print(f"\n[L2 OLS] y = {l2_model.intercept_:.4f} + " + " + ".join([f"{coef:.4f}*{name}" for name, coef in l2_coeffs.items()]))
    
    l1_coeffs = {name: value(beta[i]) for i, name in enumerate(feature_names)}
    print(f"\n[L1 LAD] y = {value(beta0):.4f} + " + " + ".join([f"{coef:.4f}*{name}" for name, coef in l1_coeffs.items()]))
    
    linf_coeffs = {name: value(beta_inf[i]) for i, name in enumerate(feature_names)}
    print(f"\n[L_inf Minimax] y = {value(beta0_inf):.4f} + " + " + ".join([f"{coef:.4f}*{name}" for name, coef in linf_coeffs.items()]))
    print("-" * 60)

    # Print Metrics
    res_df = pd.DataFrame(results_list)
    print("\n" + res_df.to_string(index=False))
    
    # Plot Comparison (Separate Window)
    plt.figure(figsize=(10, 6))
    plt.plot(y_data.index, y_data, 'ko-', label='Actual')
    plt.plot(y_data.index, l2_pred, 'g--', label='L2 Pred')
    plt.title("Regression Comparison")
    plt.legend()
    plt.savefig('Regression_Comparison.png')
    print("Displaying Regression Comparison... (Close window to continue)")
    plt.show()

compare_regression_methods(X, y)

# ==============================================================================
# PART 3: COUNTERFACTUAL OPTIMIZATION
# ==============================================================================
def solve_counterfactual_robust(target_increase_pct, max_vars=None, logical_constraints=False, description=""):
    print(f"\n{description}")
    
    X_normalized = (X - X.min()) / (X.max() - X.min())
    if X_normalized.isnull().sum().sum() > 0: X_normalized = X_normalized.fillna(0)

    x_current = X_normalized.iloc[-1].values
    std_devs = X_normalized.std().values
    weights = [1/len(X.columns)] * len(X.columns)
    y_current = np.dot(x_current, weights)
    
    p = len(X.columns)
    feature_names = X.columns.tolist()
    
    prob = LpProblem("Counterfactual_Optimization", LpMinimize)
    
    beta = [LpVariable(f"beta_{i}", lowBound=0, upBound=1-x_current[i]) for i in range(p)]
    delta = [LpVariable(f"delta_{i}", cat='Binary') for i in range(p)]
    
    prob += lpSum(beta) + 0.001 * lpSum(delta)
    
    target_gain = y_current * target_increase_pct
    prob += lpSum([beta[i] * weights[i] for i in range(p)]) >= target_gain
    
    for i in range(p):
        prob += beta[i] <= 0.5 * (std_devs[i] if std_devs[i] > 0 else 0.1)
        prob += beta[i] <= 10 * delta[i]
        prob += 0.001 * delta[i] <= beta[i]
    
    if max_vars: prob += lpSum(delta) <= max_vars
    else: prob += lpSum(delta) >= 1
    
    if logical_constraints:
        idx = {name: i for i, name in enumerate(feature_names)}
        if 'MaritimeTraffic' in idx and 'PassengersArriving' in idx:
            prob += delta[idx['MaritimeTraffic']] <= delta[idx['PassengersArriving']]
        if 'SchoolingRate' in idx and 'RenewableResources' in idx:
            prob += delta[idx['SchoolingRate']] + delta[idx['RenewableResources']] == 1
        if 'Poverty' in idx and 'VehicleRegistration' in idx:
            prob += delta[idx['Poverty']] + delta[idx['VehicleRegistration']] >= 1
    
    prob.solve(PULP_CBC_CMD(msg=0))
    
    if prob.status == LpStatusOptimal:
        beta_values = [beta[i].varValue for i in range(p)]
        new_y = y_current + np.dot(beta_values, weights)
        print("Recommended Changes:")
        for i in range(p):
            if delta[i].varValue > 0.5:
                # Denormalize
                scale = X.iloc[:, i].max() - X.iloc[:, i].min()
                if scale == 0: scale = 1
                real_change = beta_values[i] * scale
                print(f"  {feature_names[i]}: +{real_change:,.2f} (Real)")
        print(f"Target: {target_increase_pct*100:.1f}% -> Achieved: {(new_y - y_current)/y_current*100:.2f}%")
    else:
        print("No feasible solution.")

print("\n" + "="*60)
print("PART 3: COUNTERFACTUAL SCENARIOS")
print("="*60)
solve_counterfactual_robust(0.01, description="Q7a: 1% Increase")
solve_counterfactual_robust(0.05, max_vars=4, description="Q7b: 5% Increase (Max 4 vars)")
solve_counterfactual_robust(0.25, max_vars=1, description="Q8: 25% Increase (1 var)")
solve_counterfactual_robust(0.01, logical_constraints=True, description="Q10: Logic Test")