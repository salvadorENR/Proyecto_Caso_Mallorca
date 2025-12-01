import pandas as pd
import numpy as np
from pulp import *
import matplotlib.pyplot as plt
from io import StringIO

# ============================================================
# CARGA DE DATOS
# ============================================================

data_csv = """SchoolingRate,Poverty,UnaccountedWater,OpenEstablishments,RenewableResources,PassengersArriving,MaritimeTraffic,VehicleRegistration,year,y
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

df = pd.read_csv(StringIO(data_csv))

feature_cols = ['SchoolingRate', 'Poverty', 'UnaccountedWater', 'OpenEstablishments', 
                'RenewableResources', 'PassengersArriving', 'MaritimeTraffic', 'VehicleRegistration']

# Calcular estadísticas
p = len(feature_cols)
weight_v = [1/p] * p  # Pesos iguales
std_vector = df[feature_cols].std()

print("="*80)
print("OPTIMIZACIÓN CONTRAFACTUAL - TURISMO SOSTENIBLE MALLORCA")
print("="*80)
print(f"\n📊 Variables explicativas (p={p}): {feature_cols}")
print(f"⚖️  Pesos (wⱼ = 1/p): {weight_v[0]:.4f} para cada variable")
print(f"\n📈 Desviación estándar de cada variable:")
for i, col in enumerate(feature_cols):
    print(f"   {col:25s}: σ = {std_vector[i]:.6f}")

# ============================================================
# 6. FORMULACIÓN DEL PROBLEMA DE PROGRAMACIÓN LINEAL
# ============================================================

print("\n" + "="*80)
print("6. FORMULACIÓN TEÓRICA DEL PROBLEMA DE OPTIMIZACIÓN CONTRAFACTUAL")
print("="*80)

print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                    MODELO DE OPTIMIZACIÓN CONTRAFACTUAL                   ║
╚═══════════════════════════════════════════════════════════════════════════╝

OBJETIVO:
---------
Encontrar los cambios mínimos en las variables explicativas para lograr un
incremento deseado ε en el índice de turismo sostenible y.

VARIABLES DE DECISIÓN:
---------------------
• βⱼ ∈ ℝ  : Cambio propuesto para la variable xⱼ (j = 1, ..., p)
• δⱼ ∈ {0,1}: Indicador binario que vale 1 si se modifica xⱼ, 0 en caso contrario

FUNCIÓN OBJETIVO:
----------------
Minimizar:  Σⱼ₌₁ᵖ βⱼ + k·Σⱼ₌₁ᵖ δⱼ

Donde:
- El primer término minimiza la magnitud total de los cambios
- El segundo término penaliza el número de variables modificadas
- k es un parámetro de penalización (k > 0)

RESTRICCIONES:
-------------

1) RESTRICCIÓN DE INCREMENTO DESEADO:
   Σⱼ₌₁ᵖ wⱼ(xⱼ + βⱼ) ≥ y_actual + ε
   
   Equivalentemente:
   Σⱼ₌₁ᵖ wⱼ·βⱼ ≥ ε

2) RESTRICCIÓN DE SELECCIÓN DE VARIABLES:
   μₗ ≤ Σⱼ₌₁ᵖ δⱼ ≤ μᵤ
   
   Donde:
   - μₗ: número mínimo de variables a modificar
   - μᵤ: número máximo de variables a modificar

3) RESTRICCIÓN DE COTA SUPERIOR DE CAMBIOS:
   βⱼ ≤ a·f_std(xⱼ) = a·σⱼ    ∀j
   
   Donde:
   - a: constante definida por el usuario
   - σⱼ: desviación estándar de la variable xⱼ
   
   JUSTIFICACIÓN DE 'a':
   Usaremos a = 0.3 (30% de la desviación estándar) porque:
   • Permite cambios significativos pero realistas
   • Evita modificaciones extremas que sean poco factibles
   • Es consistente con variaciones naturales en los datos históricos

4) RESTRICCIÓN DE RANGO [0,1]:
   0 ≤ xⱼ + βⱼ ≤ 1    ∀j
   
   Equivalentemente:
   -xⱼ ≤ βⱼ ≤ 1 - xⱼ    ∀j

5) RESTRICCIÓN DE ACOPLAMIENTO (Big-M):
   Si δⱼ = 0, entonces βⱼ = 0
   Si δⱼ = 1, entonces βⱼ puede ser cualquier valor válido
   
   Implementación:
   m·δⱼ ≤ βⱼ ≤ M·δⱼ    ∀j
   
   Donde:
   - m: número negativo grande (ej: -50000)
   - M: número positivo grande (ej: 50000)

6) RESTRICCIÓN DE NO NEGATIVIDAD DE CAMBIOS (opcional):
   βⱼ ≥ 0    ∀j
   
   Esto asegura que solo se incrementen variables, nunca se disminuyan.
   (Se puede omitir si se permiten decrementos)

FORMULACIÓN COMPLETA:
--------------------
min  Σⱼ₌₁ᵖ βⱼ + k·Σⱼ₌₁ᵖ δⱼ

s.a.:
    Σⱼ₌₁ᵖ wⱼ·βⱼ ≥ ε                          [Incremento deseado]
    μₗ ≤ Σⱼ₌₁ᵖ δⱼ ≤ μᵤ                        [Número de modificaciones]
    βⱼ ≤ a·σⱼ                  ∀j            [Cota superior]
    m·δⱼ ≤ βⱼ ≤ M·δⱼ           ∀j            [Acoplamiento]
    βⱼ ≤ 1 - xⱼ                ∀j            [Rango superior]
    βⱼ ≥ 0                     ∀j            [No negatividad]
    δⱼ ∈ {0,1}                 ∀j            [Variable binaria]
    βⱼ ∈ ℝ                     ∀j            [Variable continua]
""")

# ============================================================
# FUNCIÓN GENÉRICA PARA RESOLVER OPTIMIZACIÓN CONTRAFACTUAL
# ============================================================

def solve_counterfactual(year_idx, epsilon_pct, a=0.3, mu_l=1, mu_u=8, k=10, 
                         beta_non_negative=True, additional_constraints=None,
                         verbose=True):
    """
    Resuelve el problema de optimización contrafactual.
    
    Parámetros:
    -----------
    year_idx : int
        Índice del año en el DataFrame
    epsilon_pct : float
        Porcentaje de incremento deseado en y (ej: 0.01 = 1%)
    a : float
        Constante para la cota superior de cambios (ej: 0.3 = 30% de σ)
    mu_l : int
        Número mínimo de variables a modificar
    mu_u : int
        Número máximo de variables a modificar
    k : float
        Penalización por número de variables modificadas
    beta_non_negative : bool
        Si True, solo se permiten incrementos (βⱼ ≥ 0)
    additional_constraints : function
        Función que recibe (model, beta, delta) y agrega restricciones adicionales
    verbose : bool
        Si True, imprime resultados detallados
    
    Retorna:
    --------
    dict con resultados de la optimización
    """
    
    # Datos del año seleccionado
    X_year = df.iloc[year_idx][feature_cols].values
    y_current = df.iloc[year_idx]['y']
    year = int(df.iloc[year_idx]['year'])
    
    # Parámetros
    epsilon = epsilon_pct * y_current
    max_incr = a * std_vector.values
    m, M = -50000, 50000
    
    # Crear modelo
    model = LpProblem(f"Counterfactual_{year}", LpMinimize)
    
    # Variables de decisión
    beta = [LpVariable(f"beta_{j}", lowBound=0 if beta_non_negative else None) 
            for j in range(p)]
    delta = [LpVariable(f"delta_{j}", cat='Binary') for j in range(p)]
    
    # Restricciones
    
    # 1. Incremento deseado en y
    model += (lpSum([weight_v[j] * beta[j] for j in range(p)]) >= epsilon,
              "Incremento_minimo")
    
    # 2. Número de variables modificadas
    model += (lpSum(delta) >= mu_l, "Min_variables")
    model += (lpSum(delta) <= mu_u, "Max_variables")
    
    # 3. Cota superior de cambios
    for j in range(p):
        model += (beta[j] <= max_incr[j], f"Cota_superior_{j}")
    
    # 4. Acoplamiento beta-delta (Big-M)
    for j in range(p):
        model += (m * delta[j] <= beta[j], f"Acoplamiento_inferior_{j}")
        model += (beta[j] <= M * delta[j], f"Acoplamiento_superior_{j}")
    
    # 5. Rango [0,1] para las variables
    for j in range(p):
        model += (beta[j] <= 1 - X_year[j], f"Rango_superior_{j}")
    
    # 6. Restricciones adicionales (si existen)
    if additional_constraints:
        additional_constraints(model, beta, delta)
    
    # Función objetivo
    model += lpSum(beta) + k * lpSum(delta)
    
    # Resolver
    status = model.solve(PULP_CBC_CMD(msg=0))
    
    # Extraer resultados
    beta_values = np.array([value(beta[j]) for j in range(p)])
    delta_values = np.array([value(delta[j]) for j in range(p)])
    
    X_new = X_year + beta_values
    y_new = np.dot(X_new, weight_v)
    y_increase = y_new - y_current
    y_increase_pct = (y_increase / y_current) * 100
    
    # Crear DataFrame de resultados
    results_df = pd.DataFrame({
        'Variable': feature_cols,
        'Valor_Actual': X_year,
        'Cambio_β': beta_values,
        'Modificar_δ': delta_values.astype(bool),
        'Valor_Nuevo': X_new,
        'Cambio_%': np.where(X_year > 0, (beta_values / X_year * 100), 0)
    })
    
    # Filtrar solo variables modificadas
    modified_df = results_df[results_df['Modificar_δ']].copy()
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"RESULTADOS - AÑO {year}")
        print(f"{'='*80}")
        print(f"Estado: {LpStatus[status]}")
        print(f"\n📊 INDICADOR DE SOSTENIBILIDAD:")
        print(f"   y actual:     {y_current:.6f}")
        print(f"   y nuevo:      {y_new:.6f}")
        print(f"   Incremento:   {y_increase:.6f} ({y_increase_pct:.2f}%)")
        print(f"   Objetivo:     {epsilon:.6f} ({epsilon_pct*100:.1f}%)")
        print(f"\n🔧 VARIABLES MODIFICADAS ({int(delta_values.sum())}/{p}):")
        print(modified_df.to_string(index=False))
        print(f"\n📈 MAGNITUD TOTAL DE CAMBIOS: {beta_values.sum():.6f}")
    
    return {
        'status': status,
        'year': year,
        'y_current': y_current,
        'y_new': y_new,
        'y_increase': y_increase,
        'y_increase_pct': y_increase_pct,
        'epsilon': epsilon,
        'epsilon_pct': epsilon_pct,
        'beta_values': beta_values,
        'delta_values': delta_values,
        'results_df': results_df,
        'modified_df': modified_df,
        'n_modified': int(delta_values.sum()),
        'total_change': beta_values.sum()
    }

# ============================================================
# 7. PREGUNTA 7: INCREMENTOS DEL 1% Y 5% EN 2024
# ============================================================

print("\n" + "="*80)
print("7. CAMBIOS PARA INCREMENTOS EN EL AÑO 2024")
print("="*80)

# Año 2024 es el índice 9 en el DataFrame
year_2024_idx = 9

print("\n" + "-"*80)
print("7.a) Incremento del 1% en 2024")
print("-"*80)

result_7a = solve_counterfactual(
    year_idx=year_2024_idx,
    epsilon_pct=0.01,  # 1%
    a=0.3,
    mu_l=1,
    mu_u=8,
    k=10,
    beta_non_negative=True
)

print("\n" + "-"*80)
print("7.b) Incremento del 5% en 2024 (máximo 4 variables)")
print("-"*80)

result_7b = solve_counterfactual(
    year_idx=year_2024_idx,
    epsilon_pct=0.05,  # 5%
    a=0.3,
    mu_l=1,
    mu_u=4,  # Máximo 4 variables
    k=10,
    beta_non_negative=True
)

# ============================================================
# 8. PREGUNTA 8: MODIFICAR UNA SOLA VARIABLE (25% INCREMENTO)
# ============================================================

print("\n" + "="*80)
print("8. RECOMENDACIÓN: MODIFICAR UNA SOLA VARIABLE PARA INCREMENTO DEL 25%")
print("="*80)

print("""
ESTRATEGIA:
-----------
Para identificar la mejor variable a modificar, probaremos cada una
individualmente y compararemos:
1. ¿Es factible lograr 25% con una sola variable?
2. ¿Cuánto cambio se requiere?
3. ¿Es realista ese cambio?
""")

best_result = None
best_var = None
feasible_vars = []

for j, var_name in enumerate(feature_cols):
    print(f"\n{'-'*80}")
    print(f"Probando: {var_name}")
    print(f"{'-'*80}")
    
    # Crear restricciones para forzar solo esta variable
    def force_single_variable(model, beta, delta):
        for i in range(p):
            if i == j:
                model += (delta[i] == 1, f"Force_{var_name}")
            else:
                model += (delta[i] == 0, f"Block_{i}")
    
    try:
        result = solve_counterfactual(
            year_idx=year_2024_idx,
            epsilon_pct=0.25,  # 25%
            a=0.3,
            mu_l=1,
            mu_u=1,  # Solo 1 variable
            k=0,
            beta_non_negative=True,
            additional_constraints=force_single_variable,
            verbose=False
        )
        
        if LpStatus[result['status']] == 'Optimal':
            feasible_vars.append((var_name, result))
            print(f"✓ FACTIBLE")
            print(f"  Cambio requerido: {result['beta_values'][j]:.6f}")
            print(f"  Valor actual: {result['results_df'].iloc[j]['Valor_Actual']:.6f}")
            print(f"  Valor nuevo: {result['results_df'].iloc[j]['Valor_Nuevo']:.6f}")
            print(f"  Incremento real: {result['y_increase_pct']:.2f}%")
            
            # Guardar el mejor (menor cambio requerido)
            if best_result is None or result['beta_values'][j] < best_result['beta_values'][j]:
                best_result = result
                best_var = var_name
        else:
            print(f"✗ NO FACTIBLE - No se puede lograr 25% con esta variable")
    except Exception as e:
        print(f"✗ ERROR: {e}")

print(f"\n{'='*80}")
print("RECOMENDACIÓN FINAL")
print(f"{'='*80}")

if best_var:
    j_best = feature_cols.index(best_var)
    print(f"\n🏆 VARIABLE RECOMENDADA: {best_var}")
    print(f"\n   Razón: Requiere el menor cambio absoluto para lograr 25% de incremento")
    print(f"\n   📊 Detalles:")
    print(f"      • Cambio requerido (β): {best_result['beta_values'][j_best]:.6f}")
    print(f"      • Valor actual:         {best_result['results_df'].iloc[j_best]['Valor_Actual']:.6f}")
    print(f"      • Valor nuevo:          {best_result['results_df'].iloc[j_best]['Valor_Nuevo']:.6f}")
    print(f"      • Cambio relativo:      {best_result['results_df'].iloc[j_best]['Cambio_%']:.2f}%")
    print(f"      • Incremento en y:      {best_result['y_increase_pct']:.2f}%")
    
    print(f"\n   💡 INTERPRETACIÓN:")
    if 'Poverty' in best_var or 'Unaccounted' in best_var or 'Vehicle' in best_var:
        print(f"      Esta variable tiene impacto NEGATIVO en sostenibilidad.")
        print(f"      Aumentarla mejora el índice (menos pobreza, menos agua perdida, etc.)")
    else:
        print(f"      Esta variable tiene impacto POSITIVO en sostenibilidad.")
        print(f"      Aumentarla directamente mejora el índice turístico.")
    
    # Mostrar todas las opciones factibles
    if len(feasible_vars) > 1:
        print(f"\n   📋 OTRAS OPCIONES FACTIBLES:")
        for var, res in feasible_vars:
            if var != best_var:
                j_var = feature_cols.index(var)
                print(f"      • {var:25s}: Cambio = {res['beta_values'][j_var]:.6f}")
else:
    print("\n❌ NINGUNA VARIABLE INDIVIDUAL puede lograr un incremento del 25%")
    print("   Se requiere modificar múltiples variables simultáneamente.")

# ============================================================
# 9. PREGUNTA 10: RESTRICCIONES ADICIONALES
# ============================================================

print("\n" + "="*80)
print("10. FORMULACIÓN DE RESTRICCIONES ADICIONALES")
print("="*80)

print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║              RESTRICCIONES LÓGICAS ADICIONALES                            ║
╚═══════════════════════════════════════════════════════════════════════════╝

a) Si se modifica MaritimeTraffic, también debe modificarse PassengersArriving
   ---------------------------------------------------------------------------
   
   FORMULACIÓN:
   δ_MaritimeTraffic ≤ δ_PassengersArriving
   
   EXPLICACIÓN:
   • Si δ_MaritimeTraffic = 1 → δ_PassengersArriving debe ser 1
   • Si δ_MaritimeTraffic = 0 → δ_PassengersArriving puede ser 0 o 1
   
   CÓDIGO PYTHON:
   idx_maritime = feature_cols.index('MaritimeTraffic')
   idx_passengers = feature_cols.index('PassengersArriving')
   model += (delta[idx_maritime] <= delta[idx_passengers], 
             "Implicacion_Maritime_Passengers")


b) Se debe modificar una y solo una entre SchoolRate y RenewableResources
   -----------------------------------------------------------------------
   
   FORMULACIÓN:
   δ_SchoolingRate + δ_RenewableResources = 1
   
   EXPLICACIÓN:
   • Exactamente una de las dos variables debe valer 1
   • Esto garantiza que se modifica una y solo una
   
   CÓDIGO PYTHON:
   idx_school = feature_cols.index('SchoolingRate')
   idx_renewable = feature_cols.index('RenewableResources')
   model += (delta[idx_school] + delta[idx_renewable] == 1,
             "XOR_School_Renewable")


c) Al menos una de las variables Poverty o VehicleRegistration debe modificarse
   -----------------------------------------------------------------------------
   
   FORMULACIÓN:
   δ_Poverty + δ_VehicleRegistration ≥ 1
   
   EXPLICACIÓN:
   • La suma debe ser al menos 1
   • Permite que ambas sean modificadas (suma = 2)
   • Pero al menos una debe ser modificada
   
   CÓDIGO PYTHON:
   idx_poverty = feature_cols.index('Poverty')
   idx_vehicle = feature_cols.index('VehicleRegistration')
   model += (delta[idx_poverty] + delta[idx_vehicle] >= 1,
             "AtLeastOne_Poverty_Vehicle")


╔═══════════════════════════════════════════════════════════════════════════╗
║                    IMPLEMENTACIÓN CONJUNTA                                ║
╚═══════════════════════════════════════════════════════════════════════════╝

def add_all_restrictions(model, beta, delta):
    '''Agrega todas las restricciones adicionales a)-c)'''
    
    # Obtener índices de variables
    idx_maritime = feature_cols.index('MaritimeTraffic')
    idx_passengers = feature_cols.index('PassengersArriving')
    idx_school = feature_cols.index('SchoolingRate')
    idx_renewable = feature_cols.index('RenewableResources')
    idx_poverty = feature_cols.index('Poverty')
    idx_vehicle = feature_cols.index('VehicleRegistration')
    
    # a) Maritime → Passengers
    model += (delta[idx_maritime] <= delta[idx_passengers], 
              "Restriction_a_Maritime_implies_Passengers")
    
    # b) Exactamente uno: School XOR Renewable
    model += (delta[idx_school] + delta[idx_renewable] == 1,
              "Restriction_b_XOR_School_Renewable")
    
    # c) Al menos uno: Poverty OR Vehicle
    model += (delta[idx_poverty] + delta[idx_vehicle] >= 1,
              "Restriction_c_AtLeastOne_Poverty_Vehicle")
""")

# Demostración práctica
print("\n" + "="*80)
print("DEMOSTRACIÓN: Optimización con restricciones a), b) y c)")
print("="*80)

def add_all_restrictions(model, beta, delta):
    """Implementa las tres restricciones adicionales"""
    idx_maritime = feature_cols.index('MaritimeTraffic')
    idx_passengers = feature_cols.index('PassengersArriving')
    idx_school = feature_cols.index('SchoolingRate')
    idx_renewable = feature_cols.index('RenewableResources')
    idx_poverty = feature_cols.index('Poverty')
    idx_vehicle = feature_cols.index('VehicleRegistration')
    
    # a) Maritime → Passengers
    model += (delta[idx_maritime] <= delta[idx_passengers])
    
    # b) School XOR Renewable (exactamente uno)
    model += (delta[idx_school] + delta[idx_renewable] == 1)
    
    # c) Poverty OR Vehicle (al menos uno)
    model += (delta[idx_poverty] + delta[idx_vehicle] >= 1)

# Resolver con las restricciones adicionales
result_10 = solve_counterfactual(
    year_idx=year_2024_idx,
    epsilon_pct=0.05,  # 5% de incremento
    a=0.3,
    mu_l=1,
    mu_u=6,
    k=5,
    beta_non_negative=True,
    additional_constraints=add_all_restrictions
)

print("\n✓ VERIFICACIÓN DE RESTRICCIONES:")
idx_maritime = feature_cols.index('MaritimeTraffic')
idx_passengers = feature_cols.index('PassengersArriving')
idx_school = feature_cols.index('SchoolingRate')
idx_renewable = feature_cols.index('RenewableResources')
idx_poverty = feature_cols.index('Poverty')
idx_vehicle = feature_cols.index('VehicleRegistration')

delta_vals = result_10['delta_values']

print(f"\n  a) Maritime → Passengers:")
print(f"     Maritime modificado: {bool(delta_vals[idx_maritime])}")
print(f"     Passengers modificado: {bool(delta_vals[idx_passengers])}")
print(f"     ✓ Cumple: {not delta_vals[idx_maritime] or delta_vals[idx_passengers]}")

print(f"\n  b) School XOR Renewable:")
print(f"     School modificado: {bool(delta_vals[idx_school])}")
print(f"     Renewable modificado: {bool(delta_vals[idx_renewable])}")
print(f"     Suma: {int(delta_vals[idx_school] + delta_vals[idx_renewable])}")
print(f"     ✓ Cumple: {delta_vals[idx_school] + delta_vals[idx_renewable] == 1}")

print(f"\n  c) Poverty OR Vehicle (al menos uno):")
print(f"     Poverty modificado: {bool(delta_vals[idx_poverty])}")
print(f"     Vehicle modificado: {bool(delta_vals[idx_vehicle])}")
print(f"     Suma: {int(delta_vals[idx_poverty] + delta_vals[idx_vehicle])}")
print(f"     ✓ Cumple: {delta_vals[idx_poverty] + delta_vals[idx_vehicle] >= 1}")

print("\n" + "="*80)
print("FIN DEL ANÁLISIS")
print("="*80)