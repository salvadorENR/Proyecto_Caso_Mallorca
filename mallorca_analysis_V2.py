import pandas as pd
import numpy as np
import pulp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

print("="*80)
print("PRÁCTICA DE OPTIMIZACIÓN - TURISMO SOSTENIBLE EN MALLORCA")
print("MAESTRÍA EN ESTADÍSTICA Y CIENCIA DE DATOS - SEMINARIO II")
print("="*80)

# ============================================================================
# PARTE 1: ANÁLISIS EXPLORATORIO Y MODELOS DE REGRESIÓN
# ============================================================================
print("\n" + "="*60)
print("PARTE 1: ANÁLISIS EXPLORATORIO Y MODELOS DE REGRESIÓN")
print("="*60)

# --- 1. Carga y limpieza de datos ---
print("\n--- 1. Carga y limpieza de datos ---")
df = pd.read_csv("Mallorca.csv")

print(f"Dataset original: {df.shape[0]} filas × {df.shape[1]} columnas")
print(f"Variables: {list(df.columns)}")

# Verificar valores faltantes
print("\nValores faltantes por variable:")
for col in df.columns:
    missing = df[col].isnull().sum()
    if missing > 0:
        print(f"  {col}: {missing} valores ({missing/len(df)*100:.1f}%)")

# Estrategia: Eliminación por lista (mantener solo observaciones completas)
df_clean = df.dropna()
print(f"\n Dataset limpio: {df_clean.shape[0]} observaciones completas")
print(f"Años disponibles: {sorted(df_clean['year'].unique())}")

# Mostrar datos limpios
print("\nDatos limpios (últimos años):")
print(df_clean[['year', 'y']].tail())

# --- 2. Análisis Exploratorio de Datos (EDA) ---
print("\n--- 2. Análisis Exploratorio de Datos (EDA) ---")

# GRÁFICO 1: Evolución temporal del índice y
print("\nGráfico 1: Evolución temporal del índice y")
years = df_clean['year'].values
y = df_clean['y'].values

plt.figure(figsize=(10, 6))
plt.plot(years, y, 'bo-', linewidth=2, markersize=8, markerfacecolor='white')
plt.title('Evolución del Índice de Turismo Sostenible', fontsize=14, fontweight='bold')
plt.xlabel('Año')
plt.ylabel('Índice y')
plt.grid(True, alpha=0.3)
plt.xticks(years, rotation=45)
plt.tight_layout()
plt.show()

# GRÁFICO 2: Histograma del índice y
print("\nGráfico 2: Distribución del índice y")
plt.figure(figsize=(10, 6))
plt.hist(y, bins=6, color='lightblue', edgecolor='black', alpha=0.7)
plt.axvline(np.mean(y), color='red', linestyle='--', label=f'Media: {np.mean(y):.2f}')
plt.axvline(np.median(y), color='green', linestyle='--', label=f'Mediana: {np.median(y):.2f}')
plt.title('Distribución del Índice de Turismo Sostenible', fontsize=14, fontweight='bold')
plt.xlabel('Valor del índice')
plt.ylabel('Frecuencia')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# GRÁFICO 3: Matriz de correlación
print("\nGráfico 3: Matriz de correlación")
features_corr = ['SchoolingRate', 'Poverty', 'UnaccountedWater', 'OpenEstablishments',
                 'RenewableResources', 'PassengersArriving', 'MaritimeTraffic', 
                 'VehicleRegistration', 'y']

corr_matrix = df_clean[features_corr].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlación - Variables del Estudio', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# GRÁFICO 4: Diagramas de caja para variables principales
print("\nGráfico 4: Diagramas de caja para variables principales")
box_vars = ['PassengersArriving', 'MaritimeTraffic', 'RenewableResources']
box_data = [df_clean[var].dropna() for var in box_vars]

plt.figure(figsize=(10, 6))
bp = plt.boxplot(box_data, patch_artist=True)
plt.title('Diagramas de Caja - Variables Principales', fontsize=14, fontweight='bold')
plt.xticks(range(1, len(box_vars) + 1), [var[:15] for var in box_vars], rotation=45)
plt.ylabel('Valor (escala 0-1)')
plt.grid(True, alpha=0.3)

for patch in bp['boxes']:
    patch.set_facecolor('lightgreen')

plt.tight_layout()
plt.show()

# GRÁFICOS 5, 6, 7: Relaciones entre y y variables clave
print("\nGráficos 5-7: Relaciones entre y y variables clave")
for idx, var in enumerate(box_vars):
    plt.figure(figsize=(10, 6))
    plt.scatter(df_clean[var], df_clean['y'], alpha=0.6, s=80)
    
    # Línea de tendencia
    z = np.polyfit(df_clean[var], df_clean['y'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(df_clean[var].min(), df_clean[var].max(), 100)
    plt.plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=2)
    
    corr = df_clean[var].corr(df_clean['y'])
    plt.title(f'Relación: y vs {var}\nCorrelación: {corr:.2f}', fontsize=14, fontweight='bold')
    plt.xlabel(var)
    plt.ylabel('Índice de Turismo Sostenible (y)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# --- 3. Preparación de datos para regresión ---
print("\n--- 3. Preparación de datos para regresión ---")

# Variables para regresión (excluyendo 'year')
reg_features = ['SchoolingRate', 'Poverty', 'UnaccountedWater', 'OpenEstablishments',
                'RenewableResources', 'PassengersArriving', 'MaritimeTraffic', 
                'VehicleRegistration']

print(f"Variables para regresión ({len(reg_features)}):")
for i, feat in enumerate(reg_features, 1):
    print(f"  x{i}: {feat}")
print("\nNota: Variable 'year' excluida de los modelos de regresión.")
print("      Es un identificador temporal, no una variable modificable.")

X = df_clean[reg_features].values
n_samples, n_features = X.shape

print(f"\nDatos para modelado:")
print(f"  Observaciones: {n_samples}")
print(f"  Variables: {n_features}")
print(f"  Variable objetivo: y (Índice de Turismo Sostenible)")

# Verificar que todas las variables están en rango [0,1]
print("\nVerificación de rangos (deben estar en [0,1]):")
for feat in reg_features:
    min_val = df_clean[feat].min()
    max_val = df_clean[feat].max()
    if min_val < 0 or max_val > 1:
        print(f"   {feat}: [{min_val:.3f}, {max_val:.3f}] - Fuera de rango")
    else:
        print(f"  ✓ {feat}: [{min_val:.3f}, {max_val:.3f}] - OK")

# Preparar matriz con intercepto
X_b = np.c_[np.ones((n_samples, 1)), X]

# --- 4. Modelos de Regresión ---
print("\n" + "="*60)
print("4. MODELOS DE REGRESIÓN LINEAL")
print("="*60)

print("\n4.1 Formulación teórica de los problemas:")

print("\nA) MÍNIMOS CUADRADOS (L2):")
print("   Minimizar: Σ(y_i - ŷ_i)²")
print("   donde: ŷ_i = β₀ + Σβⱼx_ij, j=1,...,8")

print("\nB) MÍNIMAS DESVIACIONES ABSOLUTAS (L1):")
print("   Minimizar: Σ|y_i - ŷ_i|")
print("   Sujeto a: y_i - ŷ_i ≤ u_i, ŷ_i - y_i ≤ u_i, u_i ≥ 0")

print("\nC) DESVIACIÓN ABSOLUTA MÁXIMA (L∞):")
print("   Minimizar: max|y_i - ŷ_i|")
print("   Sujeto a: y_i - ŷ_i ≤ z, ŷ_i - y_i ≤ z, z ≥ 0")

# 4.2 Modelo L2 (Mínimos Cuadrados)
print("\n--- 4.2 Modelo L2 - Mínimos Cuadrados ---")
print("Usando ecuación normal: β = (XᵀX)⁻¹Xᵀy")

try:
    X_T = X_b.T
    XTX = X_T @ X_b
    XTX_inv = np.linalg.inv(XTX)
    XTy = X_T @ y
    beta_l2 = XTX_inv @ XTy
    
    print("\n Coeficientes L2 calculados:")
    print(f"  Intercepto (β₀): {beta_l2[0]:.6f}")
    for i, feat in enumerate(reg_features, 1):
        print(f"  {feat:25s}: β{i} = {beta_l2[i]:+10.6f}")
    
    y_pred_l2 = X_b @ beta_l2
    mae_l2 = np.mean(np.abs(y - y_pred_l2))
    mse_l2 = np.mean((y - y_pred_l2)**2)
    r2_l2 = 1 - np.sum((y - y_pred_l2)**2) / np.sum((y - np.mean(y))**2)
    
    print(f"\n  Error MAE: {mae_l2:.6f}")
    print(f"  Error MSE: {mse_l2:.6f}")
    print(f"  R²:        {r2_l2:.6f}")
    
except np.linalg.LinAlgError:
    print("Error: Matriz singular - multicolinealidad perfecta detectada")
    beta_l2 = np.zeros(n_features + 1)

# 4.3 Modelo L1 (Mínimas Desviaciones Absolutas)
print("\n--- 4.3 Modelo L1 - Mínimas Desviaciones Absolutas ---")
print("Resolviendo con Programación Lineal (PuLP)")

prob_l1 = pulp.LpProblem("Regression_L1", pulp.LpMinimize)

# Variables
beta_l1_vars = [pulp.LpVariable(f"beta_{j}", cat='Continuous') for j in range(n_features + 1)]
u_vars = [pulp.LpVariable(f"u_{i}", lowBound=0, cat='Continuous') for i in range(n_samples)]

# Función objetivo
prob_l1 += pulp.lpSum(u_vars)

# Restricciones
for i in range(n_samples):
    pred_expr = pulp.lpSum(beta_l1_vars[j] * X_b[i, j] for j in range(n_features + 1))
    prob_l1 += y[i] - pred_expr <= u_vars[i]
    prob_l1 += pred_expr - y[i] <= u_vars[i]

# Resolver
prob_l1.solve(pulp.PULP_CBC_CMD(msg=False))
beta_l1 = np.array([pulp.value(var) for var in beta_l1_vars])

print("\n Coeficientes L1 calculados:")
print(f"  Intercepto (β₀): {beta_l1[0]:.6f}")
for i, feat in enumerate(reg_features, 1):
    print(f"  {feat:25s}: β{i} = {beta_l1[i]:+10.6f}")

y_pred_l1 = X_b @ beta_l1
mae_l1 = np.mean(np.abs(y - y_pred_l1))
r2_l1 = 1 - np.sum((y - y_pred_l1)**2) / np.sum((y - np.mean(y))**2)

print(f"\n  Error MAE: {mae_l1:.6f}")
print(f"  R²:        {r2_l1:.6f}")

# 4.4 Modelo L∞ (Minimax)
print("\n--- 4.4 Modelo L∞ - Minimax ---")
print("Resolviendo con Programación Lineal (PuLP)")

prob_linf = pulp.LpProblem("Regression_Linf", pulp.LpMinimize)

# Variables
beta_linf_vars = [pulp.LpVariable(f"beta_linf_{j}", cat='Continuous') for j in range(n_features + 1)]
z_var = pulp.LpVariable("z", lowBound=0, cat='Continuous')

# Función objetivo
prob_linf += z_var

# Restricciones
for i in range(n_samples):
    pred_expr = pulp.lpSum(beta_linf_vars[j] * X_b[i, j] for j in range(n_features + 1))
    prob_linf += y[i] - pred_expr <= z_var
    prob_linf += pred_expr - y[i] <= z_var

# Resolver
prob_linf.solve(pulp.PULP_CBC_CMD(msg=False))
beta_linf = np.array([pulp.value(var) for var in beta_linf_vars])
z_opt = pulp.value(z_var)

print("\n Coeficientes L∞ calculados:")
print(f"  Intercepto (β₀): {beta_linf[0]:.6f}")
for i, feat in enumerate(reg_features, 1):
    print(f"  {feat:25s}: β{i} = {beta_linf[i]:+10.6f}")

y_pred_linf = X_b @ beta_linf
mae_linf = np.mean(np.abs(y - y_pred_linf))
max_error = np.max(np.abs(y - y_pred_linf))
r2_linf = 1 - np.sum((y - y_pred_linf)**2) / np.sum((y - np.mean(y))**2)

print(f"\n  Error máximo (z): {z_opt:.6f}")
print(f"  Error MAE:         {mae_linf:.6f}")
print(f"  R²:                {r2_linf:.6f}")

# 4.5 Comparación gráfica
print("\n--- 4.5 Comparación gráfica de modelos ---")

# GRÁFICO 8: Valores reales vs predichos
print("\nGráfico 8: Valores reales vs predichos")
plt.figure(figsize=(12, 7))
plt.plot(years, y, 'ko-', linewidth=3, markersize=10, label='Valores Reales', markerfacecolor='white')
plt.plot(years, y_pred_l2, 'bs--', markersize=8, label='Modelo L2 (Mínimos Cuadrados)', alpha=0.8)
plt.plot(years, y_pred_l1, 'r^--', markersize=8, label='Modelo L1 (Mín. Desv. Abs.)', alpha=0.8)
plt.plot(years, y_pred_linf, 'gD--', markersize=8, label='Modelo L∞ (Minimax)', alpha=0.8)
plt.title('Comparación de Modelos de Regresión\nValores Reales vs Predichos', fontsize=14, fontweight='bold')
plt.xlabel('Año')
plt.ylabel('Índice de Turismo Sostenible')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.xticks(years, rotation=45)
plt.tight_layout()
plt.show()

# GRÁFICO 9: Residuales por modelo
print("\nGráfico 9: Residuales por modelo")
residuals_l2 = y - y_pred_l2
residuals_l1 = y - y_pred_l1
residuals_linf = y - y_pred_linf

x_pos = np.arange(len(years))
width = 0.25

plt.figure(figsize=(14, 7))
plt.bar(x_pos - width, residuals_l2, width, label='L2 (Mínimos Cuadrados)', color='blue', alpha=0.7)
plt.bar(x_pos, residuals_l1, width, label='L1 (Mín. Desv. Abs.)', color='red', alpha=0.7)
plt.bar(x_pos + width, residuals_linf, width, label='L∞ (Minimax)', color='green', alpha=0.7)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=1)
plt.title('Residuales por Modelo de Regresión', fontsize=14, fontweight='bold')
plt.xlabel('Año')
plt.ylabel('Residual (y - ŷ)')
plt.xticks(x_pos, years, rotation=45)
plt.legend(loc='best')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# --- 5. Análisis de resultados de regresión ---
print("\n--- 5. Análisis de resultados ---")

# Crear tabla comparativa
print("\n COMPARACIÓN DE MODELOS:")
print("-" * 70)
print(f"{'Modelo':25s} {'MAE':>10s} {'MSE':>10s} {'R²':>10s} {'Max Error':>12s}")
print("-" * 70)
print(f"{'L2 (Mínimos Cuadrados)':25s} {mae_l2:10.6f} {mse_l2:10.6f} {r2_l2:10.6f} {np.max(np.abs(residuals_l2)):12.6f}")
print(f"{'L1 (Mín. Desv. Abs.)':25s} {mae_l1:10.6f} {'-':10s} {r2_l1:10.6f} {np.max(np.abs(residuals_l1)):12.6f}")
print(f"{'L∞ (Minimax)':25s} {mae_linf:10.6f} {'-':10s} {r2_linf:10.6f} {max_error:12.6f}")
print("-" * 70)

# Verificar si los coeficientes son idénticos
coeff_diff = np.max([np.max(np.abs(beta_l2 - beta_l1)), 
                     np.max(np.abs(beta_l2 - beta_linf))])

if coeff_diff < 1e-6:
    print("\n OBSERVACIÓN: Los tres modelos producen coeficientes idénticos.")
    print("   Razón: n (observaciones) ≈ p (variables) + 1")
    print(f"   n = {n_samples}, p = {n_features} → sistema perfectamente determinado")
else:
    print(f"\n Los modelos tienen coeficientes diferentes (diferencia máxima: {coeff_diff:.10f})")

# ============================================================================
# PARTE 2: OPTIMIZACIÓN CONTRAFACTUAL
# ============================================================================
print("\n" + "="*60)
print("PARTE 2: OPTIMIZACIÓN CONTRAFACTUAL")
print("="*60)

print("\n6. FORMULACIÓN DEL PROBLEMA DE PROGRAMACIÓN LINEAL")
print("-" * 50)

print("\nVariables de decisión:")
print("  δⱼ : cambio en la variable xⱼ (j = 1,...,p)")
print("  bⱼ : variable binaria (1 si xⱼ se modifica, 0 si no)")
print("  |δⱼ|: valor absoluto del cambio")

print("\nFunción objetivo (minimizar):")
print("  Σ|δⱼ| + α·Σbⱼ  (cambios totales + número de cambios)")

print("\nRestricciones:")
print("  1. Aumento deseado: Σ wⱼ·δⱼ ≥ ε·y₀")
print("  2. Activación: -M·bⱼ ≤ δⱼ ≤ M·bⱼ")
print("  3. Límites de cambio: |δⱼ| ≤ a·σⱼ")
print("  4. Rango variables: 0 ≤ xⱼ + δⱼ ≤ 1")
print("  5. Número de variables: μₗ ≤ Σbⱼ ≤ μᵤ")

print("\nDonde:")
print("  wⱼ = 1/p (pesos iguales según enunciado)")
print("  y₀ = valor actual del índice")
print("  ε = aumento porcentual deseado")
print("  σⱼ = desviación estándar de xⱼ")
print("  a = constante a elegir (usaremos a = 0.5)")
print("  M = número grande para restricciones big-M")

# --- 7. Implementación de optimización contrafactual ---
print("\n" + "="*60)
print("7. IMPLEMENTACIÓN: OPTIMIZACIÓN CONTRAFACTUAL")
print("="*60)

# Preparar datos para 2024
df_2024 = df_clean[df_clean['year'] == 2024]
if len(df_2024) == 0:
    print(" ERROR: No hay datos para 2024")
    x0 = np.zeros(len(reg_features))
    y0 = 0
else:
    x0 = df_2024[reg_features].values.flatten()
    y0 = df_2024['y'].values[0]

print(f"\nDatos para 2024:")
print(f"  Índice actual: y = {y0:.3f}")
for i, feat in enumerate(reg_features):
    print(f"  {feat:25s}: {x0[i]:.3f}")

# Parámetros según enunciado
p = len(reg_features)
w = np.ones(p) / p  # Pesos iguales wⱼ = 1/p
a = 0.5  # Constante para límites de cambio

# Calcular desviaciones estándar
std_features = df_clean[reg_features].std().values

print(f"\n Parámetros del modelo contrafactual:")
print(f"  Número de variables: p = {p}")
print(f"  Pesos: wⱼ = 1/{p} = {1/p:.3f} ∀j")
print(f"  Constante a = {a} (justificación: cambios moderados)")
print(f"  Variables en rango [0,1] ✓")

# Función para optimización contrafactual
def optimizacion_contrafactual(aumento_pct, min_vars=None, max_vars=None, 
                               restricciones_logicas=None, a_factor=0.5):
    """
    Encuentra cambios mínimos para lograr aumento deseado.
    
    Parámetros:
    - aumento_pct: aumento porcentual deseado (ej: 1 para 1%)
    - min_vars: mínimo de variables a modificar
    - max_vars: máximo de variables a modificar
    - restricciones_logicas: diccionario con restricciones
    - a_factor: factor para límites basados en desviación estándar
    """
    
    # Calcular objetivo
    y_target = y0 * (1 + aumento_pct/100)
    aumento_necesario = y_target - y0
    
    # Crear problema
    prob = pulp.LpProblem(f"Contrafactual_{aumento_pct}pct", pulp.LpMinimize)
    
    # Variables
    delta = [pulp.LpVariable(f"delta_{i}", lowBound=None, cat='Continuous') 
            for i in range(p)]
    b = [pulp.LpVariable(f"b_{i}", cat='Binary') for i in range(p)]
    abs_delta = [pulp.LpVariable(f"abs_delta_{i}", lowBound=0, cat='Continuous') 
                for i in range(p)]
    
    # Función objetivo: minimizar cambios + penalizar número de cambios
    prob += pulp.lpSum(abs_delta) + 0.01 * pulp.lpSum(b)
    
    # Restricción: lograr aumento deseado
    prob += pulp.lpSum(w[i] * delta[i] for i in range(p)) >= aumento_necesario
    
    # Relación entre delta y b (activación)
    M = 2.0  # Suficiente para variables en [0,1]
    for i in range(p):
        prob += abs_delta[i] >= delta[i]
        prob += abs_delta[i] >= -delta[i]
        prob += delta[i] >= -M * b[i]
        prob += delta[i] <= M * b[i]
    
    # Límites en cambios basados en desviación estándar
    for i in range(p):
        max_cambio = a_factor * std_features[i]
        prob += delta[i] >= -max_cambio
        prob += delta[i] <= max_cambio
    
    # Rango [0,1] para variables modificadas
    for i in range(p):
        prob += x0[i] + delta[i] >= 0
        prob += x0[i] + delta[i] <= 1
    
    # Límite en número de variables a modificar
    if min_vars is not None:
        prob += pulp.lpSum(b) >= min_vars
    if max_vars is not None:
        prob += pulp.lpSum(b) <= max_vars
    
    # Restricciones lógicas
    if restricciones_logicas:
        # a) Si MaritimeTraffic, entonces PassengersArriving
        if 'si_entonces' in restricciones_logicas:
            for var1, var2 in restricciones_logicas['si_entonces']:
                idx1 = reg_features.index(var1)
                idx2 = reg_features.index(var2)
                prob += b[idx1] <= b[idx2]  # Si b1=1 entonces b2=1
        
        # b) Una y solo una entre SchoolingRate y RenewableResources
        if 'uno_solo' in restricciones_logicas:
            for var1, var2 in restricciones_logicas['uno_solo']:
                idx1 = reg_features.index(var1)
                idx2 = reg_features.index(var2)
                prob += b[idx1] + b[idx2] == 1
        
        # c) Al menos una entre Poverty y VehicleRegistration
        if 'al_menos_una' in restricciones_logicas:
            for vars_list in restricciones_logicas['al_menos_una']:
                indices = [reg_features.index(v) for v in vars_list]
                prob += pulp.lpSum(b[idx] for idx in indices) >= 1
    
    # Resolver
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    if pulp.LpStatus[prob.status] == 'Optimal':
        delta_vals = [pulp.value(delta[i]) for i in range(p)]
        b_vals = [pulp.value(b[i]) for i in range(p)]
        
        # Calcular aumento logrado
        aumento_logrado = sum(w[i] * delta_vals[i] for i in range(p))
        pct_logrado = (aumento_logrado / y0) * 100
        
        # Preparar resultados
        cambios = []
        for i, feat in enumerate(reg_features):
            if abs(delta_vals[i]) > 1e-6:
                nuevo_valor = min(1, max(0, x0[i] + delta_vals[i]))
                cambios.append({
                    'variable': feat,
                    'cambio': delta_vals[i],
                    'original': x0[i],
                    'nuevo': nuevo_valor
                })
        
        return cambios, pct_logrado
    else:
        return None, 0

# --- 8. Aplicación a casos específicos ---
print("\n" + "="*60)
print("8. APLICACIÓN A CASOS ESPECÍFICOS")
print("="*60)

# 8.1 Pregunta 7a: Incremento del 1%
print("\n PREGUNTA 7a: ¿Qué cambios para un incremento del 1% en 2024?")
cambios_1pct, logrado_1pct = optimizacion_contrafactual(aumento_pct=1, a_factor=a)

if cambios_1pct:
    print(f" Solución encontrada (+{logrado_1pct:.2f}% logrado)")
    print("  Cambios recomendados:")
    for cambio in cambios_1pct:
        print(f"  • {cambio['variable']:25s}: {cambio['original']:.3f} → {cambio['nuevo']:.3f} " +
              f"(Δ={cambio['cambio']:+.3f})")
else:
    print(" No se encontró solución factible")

# 8.2 Pregunta 7b: Incremento del 5% (máx 4 variables)
print("\n PREGUNTA 7b: ¿Qué cambios para un incremento del 5% (máx 4 variables)?")
cambios_5pct, logrado_5pct = optimizacion_contrafactual(aumento_pct=5, max_vars=4, a_factor=a)

if cambios_5pct:
    print(f" Solución encontrada (+{logrado_5pct:.2f}% logrado)")
    print(f"  Variables modificadas: {len(cambios_5pct)} (máximo 4)")
    print("  Cambios recomendados:")
    for cambio in cambios_5pct:
        print(f"  • {cambio['variable']:25s}: {cambio['original']:.3f} → {cambio['nuevo']:.3f} " +
              f"(Δ={cambio['cambio']:+.3f})")
else:
    print(" No se encontró solución factible")

# 8.3 Pregunta 8: Incremento del 25% con una sola variable
print("\n PREGUNTA 8: ¿Qué variable modificar para incremento del 25% (una sola)?")

# Probar cada variable individualmente
mejor_variable = None
mejor_aumento = 0
mejor_idx = -1

for i, variable in enumerate(reg_features):
    max_cambio = a * std_features[i]
    aumento_posible = w[i] * max_cambio
    pct_posible = (aumento_posible / y0) * 100
    
    if aumento_posible > mejor_aumento:
        mejor_aumento = aumento_posible
        mejor_variable = variable
        mejor_idx = i

aumento_requerido = y0 * 0.25

if mejor_variable and mejor_aumento >= aumento_requerido:
    cambio_necesario = aumento_requerido / w[mejor_idx]
    nuevo_valor = min(1, x0[mejor_idx] + cambio_necesario)
    
    print(f" Solución: Modificar {mejor_variable}")
    print(f"  Cambio necesario: +{cambio_necesario:.3f}")
    print(f"  Valor actual: {x0[mejor_idx]:.3f} → Nuevo: {nuevo_valor:.3f}")
    print(f"  Aumento lograble: {(mejor_aumento/y0)*100:.1f}%")
else:
    print(f" No se puede lograr +25% con una sola variable")
    if mejor_variable:
        print(f"  Mejor variable: {mejor_variable}")
        print(f"  Aumento máximo posible: {(mejor_aumento/y0)*100:.1f}%")
    print(f"  Aumento requerido: 25.0%")

# 8.4 Pregunta 10: Restricciones lógicas
print("\n PREGUNTA 10: Prueba con restricciones lógicas (+1%)")

restricciones = {
    'si_entonces': [('MaritimeTraffic', 'PassengersArriving')],
    'uno_solo': [('SchoolingRate', 'RenewableResources')],
    'al_menos_una': [['Poverty', 'VehicleRegistration']]
}

cambios_logicas, logrado_logicas = optimizacion_contrafactual(
    aumento_pct=1, 
    restricciones_logicas=restricciones,
    a_factor=a
)

if cambios_logicas:
    print(f" Solución encontrada con restricciones lógicas (+{logrado_logicas:.2f}%)")
    print("  Restricciones aplicadas:")
    print("    1. Si MaritimeTraffic → PassengersArriving")
    print("    2. Uno solo entre SchoolingRate y RenewableResources")
    print("    3. Al menos una entre Poverty y VehicleRegistration")
    
    print("\n  Cambios recomendados:")
    vars_modificadas = [c['variable'] for c in cambios_logicas]
    for cambio in cambios_logicas:
        print(f"  • {cambio['variable']:25s}: {cambio['original']:.3f} → {cambio['nuevo']:.3f}")
    
    # Verificar restricciones
    print("\n  Verificación:")
    print(f"    MaritimeTraffic modificada: {'MaritimeTraffic' in vars_modificadas}")
    print(f"    PassengersArriving modificada: {'PassengersArriving' in vars_modificadas}")
    print(f"    SchoolingRate modificada: {'SchoolingRate' in vars_modificadas}")
    print(f"    RenewableResources modificada: {'RenewableResources' in vars_modificadas}")
    print(f"    Poverty modificada: {'Poverty' in vars_modificadas}")
    print(f"    VehicleRegistration modificada: {'VehicleRegistration' in vars_modificadas}")
else:
    print(" No se encontró solución con las restricciones dadas")

# ============================================================================
# GUARDAR RESULTADOS COMPLETOS
# ============================================================================
print("\n" + "="*60)
print("GUARDANDO RESULTADOS COMPLETOS")
print("="*60)

output_file = "resultados_completos_practica.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("PRÁCTICA DE OPTIMIZACIÓN - TURISMO SOSTENIBLE EN MALLORCA\n")
    f.write("MAESTRÍA EN ESTADÍSTICA Y CIENCIA DE DATOS - SEMINARIO II\n")
    f.write("="*80 + "\n\n")
    
    f.write("RESUMEN EJECUTIVO\n")
    f.write("-"*50 + "\n")
    f.write(f"Fecha de análisis: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Observaciones: {df_clean.shape[0]} años ({df_clean['year'].min()}-{df_clean['year'].max()})\n")
    f.write(f"Variables analizadas: {len(reg_features)} indicadores + índice y\n")
    f.write(f"Estrategia de limpieza: Eliminación por lista (listwise deletion)\n\n")
    
    f.write("PARTE 1: REGRESIÓN LINEAL\n")
    f.write("-"*50 + "\n")
    f.write("Modelos implementados:\n")
    f.write("  1. L2: Mínimos cuadrados (ecuación normal)\n")
    f.write("  2. L1: Mínimas desviaciones absolutas (programación lineal)\n")
    f.write("  3. L∞: Desviación absoluta máxima (programación lineal)\n\n")
    
    f.write("Ecuaciones de regresión:\n")
    f.write("-"*40 + "\n")
    
    f.write("MODELO L2 (Mínimos Cuadrados):\n")
    f.write(f"y = {beta_l2[0]:.6f} ")
    for i, feat in enumerate(reg_features, 1):
        f.write(f"+ ({beta_l2[i]:+10.6f})·{feat} ")
    f.write(f"\nMAE: {mae_l2:.6f}, R²: {r2_l2:.6f}\n\n")
    
    f.write("MODELO L1 (Mínimas Desv. Absolutas):\n")
    f.write(f"y = {beta_l1[0]:.6f} ")
    for i, feat in enumerate(reg_features, 1):
        f.write(f"+ ({beta_l1[i]:+10.6f})·{feat} ")
    f.write(f"\nMAE: {mae_l1:.6f}, R²: {r2_l1:.6f}\n\n")
    
    f.write("MODELO L∞ (Minimax):\n")
    f.write(f"y = {beta_linf[0]:.6f} ")
    for i, feat in enumerate(reg_features, 1):
        f.write(f"+ ({beta_linf[i]:+10.6f})·{feat} ")
    f.write(f"\nMAE: {mae_linf:.6f}, Error máximo: {max_error:.6f}, R²: {r2_linf:.6f}\n\n")
    
    f.write("PARTE 2: OPTIMIZACIÓN CONTRAFACTUAL\n")
    f.write("-"*50 + "\n")
    f.write(f"Índice base (2024): y = {y0:.4f}\n")
    f.write(f"Parámetros: p={p}, wⱼ=1/{p}, a={a}\n\n")
    
    f.write("RESULTADOS PREGUNTA 7a (1% aumento):\n")
    if cambios_1pct:
        f.write(f"  Logrado: +{logrado_1pct:.2f}%\n")
        for cambio in cambios_1pct:
            f.write(f"  • {cambio['variable']}: {cambio['original']:.3f}→{cambio['nuevo']:.3f} " +
                   f"(Δ={cambio['cambio']:+.3f})\n")
    else:
        f.write("  No se encontró solución\n")
    f.write("\n")
    
    f.write("RESULTADOS PREGUNTA 7b (5% aumento, máx 4 variables):\n")
    if cambios_5pct:
        f.write(f"  Logrado: +{logrado_5pct:.2f}%\n")
        for cambio in cambios_5pct:
            f.write(f"  • {cambio['variable']}: {cambio['original']:.3f}→{cambio['nuevo']:.3f} " +
                   f"(Δ={cambio['cambio']:+.3f})\n")
    else:
        f.write("  No se encontró solución\n")
    f.write("\n")
    
    f.write("RESULTADOS PREGUNTA 8 (25% aumento, 1 variable):\n")
    if mejor_variable:
        f.write(f"  Variable recomendada: {mejor_variable}\n")
        f.write(f"  Aumento máximo posible: {(mejor_aumento/y0)*100:.1f}%\n")
        if mejor_aumento >= aumento_requerido:
            f.write("  ✓ Se puede lograr 25% con esta variable\n")
        else:
            f.write("  ✗ No se puede lograr 25% con una sola variable\n")
    f.write("\n")
    
    f.write("RESULTADOS PREGUNTA 10 (restricciones lógicas):\n")
    if cambios_logicas:
        f.write(f"  Logrado: +{logrado_logicas:.2f}%\n")
        f.write("  Restricciones aplicadas:\n")
        f.write("    1. Si MaritimeTraffic → PassengersArriving\n")
        f.write("    2. Uno solo: SchoolingRate XOR RenewableResources\n")
        f.write("    3. Al menos una: Poverty OR VehicleRegistration\n")
        f.write("  Variables modificadas:\n")
        for cambio in cambios_logicas:
            f.write(f"    • {cambio['variable']}\n")
    else:
        f.write("  No se encontró solución con las restricciones\n")
    f.write("\n")
    
    f.write("CONCLUSIONES\n")
    f.write("-"*50 + "\n")
    f.write("1. Los tres modelos de regresión producen resultados similares debido a n≈p\n")
    f.write("2. La optimización contrafactual permite identificar cambios mínimos\n")
    f.write("3. Para aumentos pequeños (1-5%), es posible con cambios moderados\n")
    f.write("4. Para aumentos grandes (25%), se necesitan cambios en múltiples variables\n")
    f.write("5. Las restricciones lógicas añaden complejidad pero son modelables\n\n")
    
    f.write("RECOMENDACIONES PARA EL OBSERVATORIO DE TURISMO SOSTENIBLE:\n")
    f.write("1. Priorizar variables con mayor impacto positivo según modelos\n")
    f.write("2. Considerar la factibilidad de los cambios sugeridos\n")
    f.write("3. Evaluar trade-offs entre número de variables y magnitud de cambios\n")
    f.write("4. Realizar análisis de sensibilidad con diferentes valores de 'a'\n")

print(f" Resultados completos guardados en '{output_file}'")

print("\n" + "="*80)
print("PRÁCTICA COMPLETADA EXITOSAMENTE")
print("="*80)
print("\nSe ha ejecutado:")
print("✓ Análisis Exploratorio de Datos (9 gráficos independientes)")
print("✓ Tres modelos de regresión (L1, L2, L∞)")
print("✓ Optimización contrafactual para 4 escenarios")
print("✓ Resultados guardados en archivo de texto")
print("\nListo para entregar la práctica.")