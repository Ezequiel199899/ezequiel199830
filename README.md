# ===============================
# Librerías necesarias
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ===============================
# Simulación de inflación anual por país (%)
# ===============================
inflacion_dict = {
    'Argentina': 120.0,
    'Italia': 5.2,
    'Paraguay': 4.5,
    'España': 3.9,
    'Uruguay': 5.1,
    'Alemania': 4.0
}

# ===============================
# Cargar datos de ejemplo
# ===============================
# Simulamos un DataFrame de clientes
np.random.seed(42)
n = 500
df = pd.DataFrame({
    'Edad': np.random.randint(18, 65, size=n),
    'Gasto_Mes_Anterior': np.random.uniform(50, 500, size=n),
    'Fecha_Ingreso': pd.date_range(start='2010-01-01', periods=n, freq='D'),
    'Pais': np.random.choice(list(inflacion_dict.keys()), size=n),
    'Precio_Pagado': np.random.uniform(20, 200, size=n),
    'Afectado_Aumento': np.random.choice([0, 1], size=n),
    'Churn': np.random.choice([0, 1], size=n),
    'ID_Servicio_Comprado': np.random.choice(['A', 'B', 'C'], size=n)
})

# ===============================
# Preprocesamiento de Datos
# ===============================
df['Antiguedad_Dias'] = (pd.to_datetime('today') - pd.to_datetime(df['Fecha_Ingreso'])).dt.days
df['Inflacion_Pais'] = df['Pais'].map(inflacion_dict)
df['Inflacion_Ajustada'] = df['Inflacion_Pais'] * (1 - (df['Antiguedad_Dias'] / df['Antiguedad_Dias'].max()))
df['Categoria_Edad'] = pd.cut(df['Edad'], bins=[18, 30, 50, 100], labels=['18-30', '31-50', '51+'])

# Variables dummies
df_final_transformado = pd.get_dummies(df, columns=['Pais', 'Categoria_Edad', 'ID_Servicio_Comprado'], drop_first=True)

# ===============================
# Variables de interés para churn
# ===============================
df_ml = df_final_transformado[[
    'Edad', 'Gasto_Mes_Anterior', 'Antiguedad_Dias',
    'Inflacion_Ajustada', 'Precio_Pagado', 'Afectado_Aumento', 'Churn'
]]

# ===============================
# Entrenar modelo de Churn
# ===============================
X = df_ml.drop('Churn', axis=1)
y = df_ml['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy del modelo de Churn: {accuracy * 100:.2f}%')
print('\nReporte de clasificación:')
print(classification_report(y_test, y_pred))

# ===============================
# Proyección de inflación a 20 años
# ===============================
# Promedio de inflación base
inflacion_media = np.mean(list(inflacion_dict.values()))
print(f'\nInflación promedio base: {inflacion_media:.2f}%')

# Simulación de inflación acumulada año a año (simple)
años = np.arange(2025, 2025 + 20)
inflacion_anual = []
valor = 100  # Base 100

for _ in años:
    valor = valor * (1 + inflacion_media / 100)
    inflacion_anual.append(valor)

# ===============================
# Regresión lineal para proyectar inflación
# ===============================
X_years = años.reshape(-1, 1)
y_inflacion = inflacion_anual

reg = LinearRegression()
reg.fit(X_years, y_inflacion)

pred_inflacion = reg.predict(X_years)

# Métricas de regresión
rmse = mean_squared_error(y_inflacion, pred_inflacion, squared=False)
r2 = r2_score(y_inflacion, pred_inflacion)

print(f'\nProyección de inflación: RMSE={rmse:.2f}, R2={r2:.4f}')
print(f'Pendiente: {reg.coef_[0]:.2f}, Intercepto: {reg.intercept_:.2f}')

# ===============================
# Gráfico de inflación proyectada
# ===============================
plt.figure(figsize=(12, 6))
plt.plot(años, y_inflacion, label='Inflación simulada', marker='o')
plt.plot(años, pred_inflacion, label='Regresión lineal', linestyle='--')
plt.title('Proyección de Inflación a 20 años')
plt.xlabel('Año')
plt.ylabel('Índice de inflación acumulada')
plt.legend()
plt.show()

# ===============================
# Guardar coeficientes para uso futuro
# ===============================
coef_df = pd.DataFrame({
    'Característica': X.columns,
    'Coeficiente': model.coef_[0]
}).sort_values(by='Coeficiente', ascending=False)

print('\nImportancia de las características para Churn:')
print(coef_df)

plt.figure(figsize=(10, 6))
sns.barplot(x='Coeficiente', y='Característica', data=coef_df)
plt.title('Importancia de las características en la predicción de Churn')
plt.show()

