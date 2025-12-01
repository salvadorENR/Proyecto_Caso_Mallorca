import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pulp import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CARGA DE DATOS COMPLETOS (CON NAs)
# ============================================================

data_csv = """SchoolingRate,Poverty,UnaccountedWater,OpenEstablishments,RenewableResources,PassengersArriving,MaritimeTraffic,VehicleRegistration,year,y
NA,NA,0.268937154553063,NA,NA,NA,NA,NA,2000,
NA,NA,0.289370484678842,NA,NA,NA,NA,NA,2001,
NA,NA,0.345454393242882,NA,NA,NA,NA,NA,2002,
NA,NA,0.135696481452749,NA,NA,NA,NA,NA,2003,
NA,NA,0.350212837835947,NA,NA,NA,NA,NA,2004,
NA,NA,0.300015457248727,NA,NA,NA,NA,NA,2005,
NA,NA,0.740252465347468,NA,NA,0.539634441051694,0.197809866555226,NA,2006,
NA,NA,0.6260940640793,NA,NA,0.575474860133046,0.231801463875782,NA,2007,
NA,NA,0,NA,NA,0.546400677791924,0.114771025970812,NA,2008,
NA,NA,0.30826520885199,0.201057377585995,NA,0.491625196366845,0.0492724237125156,NA,2009,
0.0857142857142849,NA,0.538327492432989,0.178654472218289,0,0.505482246989817,0.0668395416867841,NA,2010,
0.228571428571428,NA,0.702875669559377,0.0944464279468287,0.202126684389037,0.567975934044519,0.10624054140683,NA,2011,
0.457142857142856,NA,0.815516928215165,0.0919788926264519,0.264552204601705,0.559442072620711,0.0566252753896027,NA,2012,
0.114285714285712,NA,0.766418935060871,0.0964773414400024,0.269777248143735,0.577535064308042,0.164322896591458,NA,2013,
0.114285714285712,0.243323442136498,0.779518382898881,0.0541695459526252,0.391053258777157,0.608175529085555,0.245635860899342,NA,2014,
0,0,0.77326949345204,0.0372835672366548,0.491979099825832,0.650146160514485,0.373200077900263,0.578006821889367,2015,0.47
0.428571428571429,0.504451038575668,0.786306743047201,0.0612050627802821,0.299477495645797,0.769808587880075,0.590339454454718,0.879504671511197,2016,0.35
0.399999999999998,0.138476755687438,1,0.0748470735938813,0.362453020441837,0.840152865030393,0.657620735264881,0.965408571852291,2017,0.68
0.685714285714283,0.925816023738872,0.933966707700023,0,0.0704005866715556,0.881970124842532,0.733972629417898,1,2018,0.8
0.914285714285711,1,0.718189616415493,0.00709380425217276,0.297277477312311,0.909555615174227,0.806930587619538,0.794861337683524,2019,0.75
1,0.657764589515331,0.153190021137617,1,0.478228985241544,0,0,0,2020,0.49
0.457142857142856,0.421364985163205,0.484186876382556,0.390883173964627,0.746356219635164,0.371114391861121,0.512337251337896,0.128244104997775,2021,0.41
0.599999999999998,0.504451038575668,0.648452665370229,0.0461838884489565,1,0.898964961279741,0.889613399656752,0.379875426368085,2022,0.55
0.714285714285714,0.527860204418068,0.428609854296801,0.00744222140455053,0.741528401625569,1,1,0.485874239952543,2023,0.62
0.590476190476189,0.48455874271898,0.520416465349862,0.148169761272711,0.829294873753578,0.756693117713621,0.800650216998216,0.331331257106134,2024,0.49"""

# Cargar datos y reemplazar 'NA' con np.nan
df_original = pd.read_csv(StringIO(data_csv.replace('NA', '')))

print("="*80)
print("ANÁLISIS DE DATOS FALTANTES (NAs)")
print("="*80)

# Contar NAs por columna
na_counts = df_original.isna().sum()
na_percentages = (na_counts / len(df_original) * 100).round(2)

na_summary = pd.DataFrame({
    'Variable': na_counts.index,
    'NAs': na_counts.values,
    'Total': len(df_original),
    'Porcentaje_NA': na_percentages.values
})

print("\n" + na_summary.to_string(index=False))

print(f"\n📊 Total de observaciones: {len(df_original)}")
print(f"📅 Período completo: {df_original['year'].min():.0f} - {df_original['year'].max():.0f}")

# ============================================================
# ESTRATEGIAS PARA MANEJAR NAs
# ============================================================

print("\n" + "="*80)
print("ESTRATEGIAS PARA MANEJAR DATOS FALTANTES")
print("="*80)

print("""
OPCIÓN 1: Eliminar filas con NAs (Listwise deletion)
  ✓ Ventaja: Solo usa datos completos
  ✗ Desventaja: Reduce mucho el tamaño (de 25 a 10 observaciones)

OPCIÓN 2: Imputación con Media/Mediana
  ✓ Ventaja: Mantiene todas las observaciones
  ✗ Desventaja: Introduce sesgo si NAs no son aleatorios

OPCIÓN 3: Usar solo variables completas
  ✓ Ventaja: Más observaciones disponibles
  ✗ Desventaja: Pierde información de variables importantes
""")

# ============================================================
# ESTRATEGIA 1: ELIMINAR FILAS CON NAs (RECOMENDADA)
# ============================================================

print("\n" + "="*80)
print("ESTRATEGIA SELECCIONADA: Eliminar filas con NAs en y")
print("="*80)

# Primero, calculemos 'y' para las filas que tienen todas las variables X
df = df_original.copy()

# Identificar filas con 'y' disponible
df_complete_y = df[df['y'].notna()].copy()

print(f"\n✓ Observaciones con 'y' disponible: {len(df_complete_y)}")
print(f"✓ Años disponibles: {df_complete_y['year'].min():.0f} - {df_complete_y['year'].max():.0f}")

# Para las filas sin 'y', calcularla si tienen todas las X
df_missing_y = df[df['y'].isna()].copy()

# Verificar qué filas tienen todas las variables X disponibles
feature_cols = ['SchoolingRate', 'Poverty', 'UnaccountedWater', 'OpenEstablishments', 
                'RenewableResources', 'PassengersArriving', 'MaritimeTraffic', 'VehicleRegistration']

df_missing_y['complete_X'] = df_missing_y[feature_cols].notna().all(axis=1)
print(f"\n📋 Observaciones sin 'y' pero con todas las X completas: {df_missing_y['complete_X'].sum()}")

# Calcular 'y' para las filas que tienen todas las X
if df_missing_y['complete_X'].sum() > 0:
    p = len(feature_cols)
    weight_v = [1/p] * p
    for idx in df_missing_y[df_missing_y['complete_X']].index:
        df.loc[idx, 'y'] = df.loc[idx, feature_cols].dot(weight_v)

# Dataset final: solo filas con 'y' disponible
df_final = df[df['y'].notna()].copy()

print(f"\n✓ Dataset final: {len(df_final)} observaciones con 'y' calculado")
print(f"✓ Años finales: {sorted(df_final['year'].astype(int).unique())}")

# ============================================================
# OPCIÓN ALTERNATIVA: IMPUTACIÓN
# ============================================================

print("\n" + "="*80)
print("COMPARACIÓN: Dataset con Imputación de NAs (Media)")
print("="*80)

df_imputed = df_original.copy()

# Imputar NAs con la media de cada columna (excepto year)
imputer = SimpleImputer(strategy='mean')
cols_to_impute = [col for col in df_imputed.columns if col not in ['year', 'y']]

df_imputed[cols_to_impute] = imputer.fit_transform(df_imputed[cols_to_impute])

# Calcular 'y' para todos si está faltante
p = len(feature_cols)
weight_v = [1/p] * p
for idx in df_imputed[df_imputed['y'].isna()].index:
    df_imputed.loc[idx, 'y'] = df_imputed.loc[idx, feature_cols].dot(weight_v)

print(f"✓ Dataset imputado: {len(df_imputed)} observaciones")
print(f"✓ Todas las variables completas: {df_imputed.notna().all().all()}")

# ============================================================
# ANÁLISIS EXPLORATORIO (DATASET FINAL)
# ============================================================

print("\n" + "="*80)
print("1. ANÁLISIS EXPLORATORIO DE DATOS (DATASET FINAL)")
print("="*80)

df = df_final.copy()  # Usar dataset sin NAs

print(f"\n📊 Dimensiones del dataset: {df.shape[0]} filas × {df.shape[1]} columnas")
print(f"📅 Período: {df['year'].min():.0f} - {df['year'].max():.0f}")

# Estadísticas descriptivas
print("\n" + "="*80)
print("ESTADÍSTICAS DESCRIPTIVAS")
print("="*80)
print(df.describe().round(4))

# Correlaciones con la variable objetivo
print("\n" + "="*80)
print("CORRELACIÓN CON LA VARIABLE OBJETIVO (y)")
print("="*80)
correlations = df.drop(columns=['year']).corr()['y'].sort_values(ascending=False)
print(correlations)

# Preparar datos para regresión
X = df[feature_cols].values
y_true = df['y'].values
n_samples, n_features = X.shape

print(f"\n✓ Variables explicativas: {n_features}")
print(f"✓ Observaciones: {n_samples}")
print(f"✓ Variables: {feature_cols}")

# ============================================================
# 2. REGRESIÓN CON NORMA ℓ1
# ============================================================

print("\n" + "="*80)
print("2. REGRESIÓN CON NORMA ℓ1 - MINIMIZAR SUMA DE DESVIACIONES ABSOLUTAS")
print("="*80)

print("""
FORMULACIÓN TEÓRICA:
-------------------
Minimizar: Σ|yᵢ - (β₀ + β₁x₁ᵢ + β₂x₂ᵢ + ... + βₚxₚᵢ)|

Variables de decisión:
- βⱼ: coeficientes de regresión (j = 0, 1, ..., p)
- dᵢ⁺, dᵢ⁻: desviaciones positiva y negativa para cada observación i

Restricciones:
- yᵢ = β₀ + Σ(βⱼ·xⱼᵢ) + dᵢ⁺ - dᵢ⁻  ∀i
- dᵢ⁺, dᵢ⁻ ≥ 0  ∀i

Función objetivo:
- Minimizar: Σ(dᵢ⁺ + dᵢ⁻)
""")

model_l1 = LpProblem("Regression_L1", LpMinimize)
beta = [LpVariable(f"beta{j}", cat='Continuous') for j in range(n_features + 1)]
d_plus = [LpVariable(f"d_plus_{i}", lowBound=0) for i in range(n_samples)]
d_minus = [LpVariable(f"d_minus_{i}", lowBound=0) for i in range(n_samples)]

for i in range(n_samples):
    model_l1 += (
        y_true[i] == beta[0] + lpSum([beta[j+1] * X[i, j] for j in range(n_features)]) 
        + d_plus[i] - d_minus[i]
    )

model_l1 += lpSum(d_plus) + lpSum(d_minus)

print("⏳ Resolviendo modelo ℓ1...")
model_l1.solve(PULP_CBC_CMD(msg=0))

beta_l1 = np.array([value(beta[j]) for j in range(n_features + 1)])
intercept_l1 = beta_l1[0]
coef_l1 = beta_l1[1:]
y_pred_l1 = intercept_l1 + X @ coef_l1

mae_l1 = mean_absolute_error(y_true, y_pred_l1)
mse_l1 = mean_squared_error(y_true, y_pred_l1)
rmse_l1 = np.sqrt(mse_l1)
r2_l1 = r2_score(y_true, y_pred_l1)

print(f"\n✓ Estado: {LpStatus[model_l1.status]}")
print(f"\nECUACIÓN DE LA RECTA (ℓ1):")
print(f"y = {intercept_l1:.6f}", end="")
for i, name in enumerate(feature_cols):
    sign = "+" if coef_l1[i] >= 0 else ""
    print(f" {sign}{coef_l1[i]:.6f}·{name}", end="")
print()

print(f"\nMÉTRICAS DE ERROR:")
print(f"  MAE:  {mae_l1:.6f}")
print(f"  RMSE: {rmse_l1:.6f}")
print(f"  R²:   {r2_l1:.6f}")

# ============================================================
# 3. REGRESIÓN CON NORMA ℓ∞
# ============================================================

print("\n" + "="*80)
print("3. REGRESIÓN CON NORMA ℓ∞ - MINIMIZAR DESVIACIÓN ABSOLUTA MÁXIMA")
print("="*80)

print("""
FORMULACIÓN TEÓRICA:
-------------------
Minimizar: max|yᵢ - (β₀ + β₁x₁ᵢ + ... + βₚxₚᵢ)|

Variables de decisión:
- βⱼ: coeficientes de regresión
- z: desviación máxima

Restricciones:
- |yᵢ - predicciónᵢ| ≤ z  ∀i

Función objetivo:
- Minimizar: z
""")

model_linf = LpProblem("Regression_Linf", LpMinimize)
beta_inf = [LpVariable(f"beta_inf{j}", cat='Continuous') for j in range(n_features + 1)]
z = LpVariable("z", lowBound=0)

for i in range(n_samples):
    prediction = beta_inf[0] + lpSum([beta_inf[j+1] * X[i, j] for j in range(n_features)])
    model_linf += (y_true[i] - prediction <= z)
    model_linf += (y_true[i] - prediction >= -z)

model_linf += z

print("⏳ Resolviendo modelo ℓ∞...")
model_linf.solve(PULP_CBC_CMD(msg=0))

beta_linf = np.array([value(beta_inf[j]) for j in range(n_features + 1)])
intercept_linf = beta_linf[0]
coef_linf = beta_linf[1:]
y_pred_linf = intercept_linf + X @ coef_linf

mae_linf = mean_absolute_error(y_true, y_pred_linf)
mse_linf = mean_squared_error(y_true, y_pred_linf)
rmse_linf = np.sqrt(mse_linf)
r2_linf = r2_score(y_true, y_pred_linf)
max_error_linf = np.max(np.abs(y_true - y_pred_linf))

print(f"\n✓ Estado: {LpStatus[model_linf.status]}")
print(f"\nECUACIÓN DE LA RECTA (ℓ∞):")
print(f"y = {intercept_linf:.6f}", end="")
for i, name in enumerate(feature_cols):
    sign = "+" if coef_linf[i] >= 0 else ""
    print(f" {sign}{coef_linf[i]:.6f}·{name}", end="")
print()

print(f"\nMÉTRICAS DE ERROR:")
print(f"  Error Máximo: {max_error_linf:.6f}")
print(f"  MAE:  {mae_linf:.6f}")
print(f"  RMSE: {rmse_linf:.6f}")
print(f"  R²:   {r2_linf:.6f}")

# ============================================================
# 4. REGRESIÓN CON NORMA ℓ2 (SCIKIT-LEARN)
# ============================================================

print("\n" + "="*80)
print("4. REGRESIÓN CON NORMA ℓ2 - MÍNIMOS CUADRADOS (SCIKIT-LEARN)")
print("="*80)

model_l2 = LinearRegression()
model_l2.fit(X, y_true)

intercept_l2 = model_l2.intercept_
coef_l2 = model_l2.coef_
y_pred_l2 = model_l2.predict(X)

mae_l2 = mean_absolute_error(y_true, y_pred_l2)
mse_l2 = mean_squared_error(y_true, y_pred_l2)
rmse_l2 = np.sqrt(mse_l2)
r2_l2 = r2_score(y_true, y_pred_l2)

print(f"\nECUACIÓN DE LA RECTA (ℓ2):")
print(f"y = {intercept_l2:.6f}", end="")
for i, name in enumerate(feature_cols):
    sign = "+" if coef_l2[i] >= 0 else ""
    print(f" {sign}{coef_l2[i]:.6f}·{name}", end="")
print()

print(f"\nMÉTRICAS DE ERROR:")
print(f"  MAE:  {mae_l2:.6f}")
print(f"  MSE:  {mse_l2:.6f}")
print(f"  RMSE: {rmse_l2:.6f}")
print(f"  R²:   {r2_l2:.6f}")

# ============================================================
# 5. COMPARACIÓN GRÁFICA
# ============================================================

print("\n" + "="*80)
print("5. COMPARACIÓN DE RESULTADOS")
print("="*80)

comparison = pd.DataFrame({
    'Método': ['ℓ1 (MAD)', 'ℓ∞ (Chebyshev)', 'ℓ2 (MCO)'],
    'MAE': [mae_l1, mae_linf, mae_l2],
    'RMSE': [rmse_l1, rmse_linf, rmse_l2],
    'R²': [r2_l1, r2_linf, r2_l2],
    'Error_Máximo': [
        np.max(np.abs(y_true - y_pred_l1)),
        max_error_linf,
        np.max(np.abs(y_true - y_pred_l2))
    ]
})

print("\n" + comparison.to_string(index=False))

# Visualizaciones
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle(f'Comparación de Métodos de Regresión (n={n_samples} observaciones)', 
             fontsize=16, fontweight='bold')

years = df['year'].values

# Gráfico 1: Valores reales vs predicciones
ax1 = axes[0, 0]
ax1.plot(years, y_true, 'ko-', linewidth=2, markersize=8, label='Valores Reales', zorder=5)
ax1.plot(years, y_pred_l1, 's--', linewidth=1.5, markersize=6, label='ℓ1 (MAD)', alpha=0.8)
ax1.plot(years, y_pred_linf, '^--', linewidth=1.5, markersize=6, label='ℓ∞ (Chebyshev)', alpha=0.8)
ax1.plot(years, y_pred_l2, 'd--', linewidth=1.5, markersize=6, label='ℓ2 (MCO)', alpha=0.8)
ax1.set_xlabel('Año', fontsize=11)
ax1.set_ylabel('Valor de y', fontsize=11)
ax1.set_title('Valores Reales vs Predicciones', fontsize=12, fontweight='bold')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# Gráfico 2: Residuos
ax2 = axes[0, 1]
residuals_l1 = y_true - y_pred_l1
residuals_linf = y_true - y_pred_linf
residuals_l2 = y_true - y_pred_l2

x_pos = np.arange(len(years))
width = 0.25
ax2.bar(x_pos - width, residuals_l1, width, label='ℓ1', alpha=0.8)
ax2.bar(x_pos, residuals_linf, width, label='ℓ∞', alpha=0.8)
ax2.bar(x_pos + width, residuals_l2, width, label='ℓ2', alpha=0.8)
ax2.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax2.set_xlabel('Observación', fontsize=11)
ax2.set_ylabel('Residuo', fontsize=11)
ax2.set_title('Residuos por Método', fontsize=12, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(years.astype(int), rotation=45)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Gráfico 3: Scatter plot
ax3 = axes[1, 0]
ax3.scatter(y_true, y_pred_l1, s=100, alpha=0.6, label='ℓ1', marker='s')
ax3.scatter(y_true, y_pred_linf, s=100, alpha=0.6, label='ℓ∞', marker='^')
ax3.scatter(y_true, y_pred_l2, s=100, alpha=0.6, label='ℓ2', marker='d')
lims = [min(y_true.min(), y_pred_l1.min(), y_pred_l2.min()) - 0.05,
        max(y_true.max(), y_pred_l1.max(), y_pred_l2.max()) + 0.05]
ax3.plot(lims, lims, 'k--', alpha=0.5, linewidth=2, label='Predicción Perfecta')
ax3.set_xlim(lims)
ax3.set_ylim(lims)
ax3.set_xlabel('Valores Reales', fontsize=11)
ax3.set_ylabel('Valores Predichos', fontsize=11)
ax3.set_title('Valores Reales vs Predichos', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Gráfico 4: Comparación de métricas
ax4 = axes[1, 1]
metrics = ['MAE', 'RMSE', 'R²']
l1_metrics = [mae_l1, rmse_l1, r2_l1]
linf_metrics = [mae_linf, rmse_linf, r2_linf]
l2_metrics = [mae_l2, rmse_l2, r2_l2]

x_pos = np.arange(len(metrics))
width = 0.25
ax4.bar(x_pos - width, l1_metrics, width, label='ℓ1', alpha=0.8)
ax4.bar(x_pos, linf_metrics, width, label='ℓ∞', alpha=0.8)
ax4.bar(x_pos + width, l2_metrics, width, label='ℓ2', alpha=0.8)
ax4.set_xlabel('Métrica', fontsize=11)
ax4.set_ylabel('Valor', fontsize=11)
ax4.set_title('Comparación de Métricas', fontsize=12, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(metrics)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("ANÁLISIS FINAL")
print("="*80)
print(f"""
📊 DATASET UTILIZADO: {n_samples} observaciones (años {df['year'].min():.0f}-{df['year'].max():.0f})

⚠️ NOTA IMPORTANTE SOBRE NAs:
- Total de datos originales: 25 años (2000-2024)
- Datos con 'y' disponible: {n_samples} años
- Datos descartados por NAs: {25 - n_samples} años

🎯 MEJOR MODELO POR MÉTRICA:
- Menor MAE: {comparison.loc[comparison['MAE'].idxmin(), 'Método']}
- Mayor R²: {comparison.loc[comparison['R²'].idxmax(), 'Método']}
- Menor Error Máximo: {comparison.loc[comparison['Error_Máximo'].idxmin(), 'Método']}
""")