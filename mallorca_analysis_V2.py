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
# 0. CONFIGURACIÓN DE REGISTRO (LOGGING)
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
print("      INFORME DE OPTIMIZACIÓN - ESTRATEGIA B: ELIMINACIÓN POR LISTA  ")
print("      (Priorizando Integridad de Datos sobre Longitud Histórica)     ")
print("      (Con Ventanas Separadas para cada Gráfico)                     ")
print("=====================================================================\n")

# ==========================================
# 1. CARGA DE DATOS
# ==========================================
if not os.path.exists("Mallorca.csv"):
    print("Error: No se encontró el archivo 'Mallorca.csv'.")
    exit()

df = pd.read_csv("Mallorca.csv")

if 'Year' in df.columns:
    df.set_index('Year', inplace=True)
df.sort_index(inplace=True)

# ==========================================
# 2. LIMPIEZA DE DATOS (ARREGLO ROBUSTO)
# ==========================================
print("\n--- Estrategia de Limpieza B: Eliminación de filas con valores faltantes ---")

target_col = 'SustainableTourismIndex'

# 1. Eliminar filas con NA primero
df_clean = df.dropna()

# 2. SEPARAR OBJETIVO (y) Y VARIABLES (X)
if target_col in df_clean.columns:
    y = df_clean[target_col]
    X = df_clean.drop(columns=[target_col])
    # Seguridad: Eliminar 'y' si existe en X de una ejecución anterior
    if 'y' in X.columns: X = X.drop(columns=['y'])
else:
    # Si falta el objetivo, calcularlo
    print(f"Columna objetivo '{target_col}' no encontrada. Calculando índice...")
    X = df_clean.copy()
    if 'y' in X.columns: X = X.drop(columns=['y'])
    
    X_norm_calc = (X - X.min()) / (X.max() - X.min())
    weights = [1/len(X.columns)] * len(X.columns)
    y = X_norm_calc.dot(weights)

# Añadir y de nuevo a df_clean para propósitos de EDA (pero mantener X puro para regresión)
df_clean['y'] = y

# Chequeo Final de Seguridad para Fuga de Datos
common_cols = set(X.columns).intersection(set(['y', target_col]))
if common_cols:
    X = X.drop(columns=list(common_cols))

print(f"Dimensiones del Dataset Final Limpio: {df_clean.shape}")

# ==============================================================================
# PARTE 1: ANÁLISIS EXPLORATORIO DE DATOS (VENTANAS SEPARADAS)
# ==============================================================================
def exploratory_data_analysis(df):
    print("\n" + "="*60)
    print("PARTE 1: ANÁLISIS EXPLORATORIO DE DATOS")
    print("="*60)
    
    # --- Gráfico 1: Serie Temporal ---
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['y'], marker='o', color='purple', linewidth=2)
    plt.title('Índice de Turismo Sostenible a lo largo del tiempo')
    plt.ylabel('Valor del Índice')
    plt.xlabel('Año')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('EDA_SerieTemporal.png')
    print("Mostrando Gráfico de Serie Temporal... (Cierre la ventana para continuar)")
    plt.show()
    
    # --- Gráfico 2: Matriz de Correlación ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Matriz de Correlación')
    plt.tight_layout()
    plt.savefig('EDA_Correlacion.png')
    print("Mostrando Matriz de Correlación... (Cierre la ventana para continuar)")
    plt.show()
    
    # --- B. Diagramas de Caja para Predictores Clave ---
    print("\nGenerando Diagramas de Caja para Predictores Principales...")
    key_vars = ['MaritimeTraffic', 'PassengersArriving', 'Poverty']
    valid_vars = [v for v in key_vars if v in df.columns]
    
    if valid_vars:
        for col in valid_vars:
            # Crear una figura separada para CADA variable
            plt.figure(figsize=(8, 6))
            
            # Crear bins para la variable continua
            plot_df = df.copy()
            try:
                plot_df['Category'] = pd.qcut(plot_df[col], q=3, labels=["Bajo", "Medio", "Alto"])
            except ValueError:
                plot_df['Category'] = pd.cut(plot_df[col], bins=3, labels=["Bajo", "Medio", "Alto"])
            
            sns.boxplot(x='Category', y='y', data=plot_df, palette="Set2")
            sns.stripplot(x='Category', y='y', data=plot_df, color='black', alpha=0.5)
            
            plt.title(f'Impacto de {col}\nen el Índice de Turismo')
            plt.ylabel('Índice de Turismo Sostenible (y)')
            plt.xlabel(f'Nivel de {col}')
            
            plt.tight_layout()
            plt.savefig(f'EDA_BoxPlot_{col}.png')
            print(f"Mostrando Diagrama de Caja para {col}... (Cierre la ventana para continuar)")
            plt.show()
    else:
        print("Advertencia: Las variables solicitadas para los diagramas de caja no se encontraron en el dataset.")

exploratory_data_analysis(df_clean.copy())

# ==============================================================================
# PARTE 2: ANÁLISIS DE REGRESIÓN
# ==============================================================================
def compare_regression_methods(X_data, y_data):
    print("\n" + "="*60)
    print("PARTE 2: ANÁLISIS DE REGRESIÓN")
    print("="*60)
    
    n_samples, n_features = X_data.shape
    feature_names = X_data.columns.tolist()
    results_list = []

    # --- 1. Norma L2 (Mínimos Cuadrados Ordinarios - OLS) ---
    l2_model = LinearRegression()
    l2_model.fit(X_data, y_data)
    l2_pred = l2_model.predict(X_data)
    
    results_list.append({
        "Modelo": "L2 (Mínimos Cuadrados)", 
        "MAE": mean_absolute_error(y_data, l2_pred), 
        "R2": r2_score(y_data, l2_pred)
    })

    # --- 2. Norma L1 (Mínimas Desviaciones Absolutas - LAD) ---
    prob_l1 = LpProblem("Regresion_L1", LpMinimize)
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
        "Modelo": "L1 (Mínimas Desv. Abs.)", 
        "MAE": mean_absolute_error(y_data, l1_pred), 
        "R2": r2_score(y_data, l1_pred)
    })

    # --- 3. Norma L-inf (Minimax) ---
    prob_inf = LpProblem("Regresion_L_inf", LpMinimize)
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
        "Modelo": "L_inf (Minimax)", 
        "MAE": mean_absolute_error(y_data, linf_pred), 
        "R2": r2_score(y_data, linf_pred)
    })
    
    # --- IMPRIMIR ECUACIONES ---
    print("\n" + "-"*30)
    print("ECUACIONES DEL MODELO (Coeficientes)")
    print("-"*30)
    
    l2_coeffs = dict(zip(feature_names, l2_model.coef_))
    print(f"\n[L2 OLS] y = {l2_model.intercept_:.4f} + " + " + ".join([f"{coef:.4f}*{name}" for name, coef in l2_coeffs.items()]))
    
    l1_coeffs = {name: value(beta[i]) for i, name in enumerate(feature_names)}
    print(f"\n[L1 LAD] y = {value(beta0):.4f} + " + " + ".join([f"{coef:.4f}*{name}" for name, coef in l1_coeffs.items()]))
    
    linf_coeffs = {name: value(beta_inf[i]) for i, name in enumerate(feature_names)}
    print(f"\n[L_inf Minimax] y = {value(beta0_inf):.4f} + " + " + ".join([f"{coef:.4f}*{name}" for name, coef in linf_coeffs.items()]))
    print("-" * 60)

    # Imprimir Métricas
    res_df = pd.DataFrame(results_list)
    print("\n" + res_df.to_string(index=False))
    
    # Gráfico de Comparación (Ventana Separada)
    plt.figure(figsize=(12, 7))
    
    # 1. Datos Reales (Puntos Grandes)
    plt.plot(y_data.index, y_data, 'ko', label='Real', markersize=10, zorder=5)
    
    # 2. L2 (Linea Gruesa, Verde)
    plt.plot(y_data.index, l2_pred, 'g-', label='L2 (Mín Cuadrados)', linewidth=4, alpha=0.5)
    
    # 3. L1 (Linea Media, Roja Punteada)
    plt.plot(y_data.index, l1_pred, 'r--', label='L1 (Mín Desv Abs)', linewidth=2.5)
    
    # 4. L-inf (Linea Fina, Azul)
    plt.plot(y_data.index, linf_pred, 'b:', label='L-inf (Minimax)', linewidth=2)
    
    plt.title("Comparación de Modelos de Regresión (Superpuestos por Ajuste Perfecto)")
    plt.legend()
    plt.ylabel('Valor del Índice')
    plt.xlabel('Año')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('Regression_Comparison.png')
    print("Mostrando Comparación de Regresión... (Cierre la ventana para continuar)")
    plt.show()

compare_regression_methods(X, y)

# ==============================================================================
# PARTE 3: OPTIMIZACIÓN CONTRAFACTUAL
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
    
    prob = LpProblem("Optimizacion_Contrafactual", LpMinimize)
    
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
        print("Cambios Recomendados:")
        for i in range(p):
            if delta[i].varValue > 0.5:
                # Desnormalizar
                scale = X.iloc[:, i].max() - X.iloc[:, i].min()
                if scale == 0: scale = 1
                real_change = beta_values[i] * scale
                print(f"  {feature_names[i]}: +{real_change:,.2f} (Real)")
        print(f"Objetivo: {target_increase_pct*100:.1f}% -> Logrado: {(new_y - y_current)/y_current*100:.2f}%")
    else:
        print("No se encontró solución factible.")

print("\n" + "="*60)
print("PARTE 3: ESCENARIOS CONTRAFACTUALES")
print("="*60)
solve_counterfactual_robust(0.01, description="Q7a: Incremento del 1%")
solve_counterfactual_robust(0.05, max_vars=4, description="Q7b: Incremento del 5% (Máx 4 vars)")
solve_counterfactual_robust(0.25, max_vars=1, description="Q8: Incremento del 25% (1 var)")
solve_counterfactual_robust(0.01, logical_constraints=True, description="Q10: Prueba Lógica")

print("\n=====================================================================")
print("                       ANÁLISIS COMPLETO                             ")
print("Todos los resultados guardados en: 'optimization_results_dropNA_normalized.txt'")
print("Visualizaciones guardadas como: 'EDA_*.png', 'Regression_Comparison.png'")
print("=====================================================================")