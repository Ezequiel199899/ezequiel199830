# --- Librerías necesarias ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# --- Simulación de inflación anual por país (%) --- 

inflacion_dict = {
    'Argentina': 120.0,  # Inflación anual estimada ) 
    'Italia': 5.2,
    'Paraguay': 4.5,
    'España': 3.9,
    'Uruguay': 5.1,
    'Alemania': 4.0
}

# --- Cargar datos ---
# Aquí deberías cargar tus datos (csv, base de datos, etc.)
# Ejemplo:
df = pd.read_csv('tus_datos.csv')

# --- Preprocesamiento de Datos ---
# Supongamos que ya tienes algunas columnas relevantes en tu dataframe
df['Antiguedad_Dias'] = (pd.to_datetime('today') - pd.to_datetime(df['Fecha_Ingreso'])).dt.days

# --- Asignar inflación según el país ---
df['Inflacion_Pais'] = df['Pais'].map(inflacion_dict)

# Ajuste de inflación en función de la antigüedad (clientes más recientes perciben más inflación actual)
df['Inflacion_Ajustada'] = df['Inflacion_Pais'] * (
    1 - (df['Antiguedad_Dias'] / df['Antiguedad_Dias'].max())
)

# --- Feature Engineering --- 
# Algunas características adicionales, por ejemplo, categorización por edad, etc.
df['Categoria_Edad'] = pd.cut(df['Edad'], bins=[18, 30, 50, 100], labels=['18-30', '31-50', '51+'])

# --- Transformación de Datos ---
# Convertir variables categóricas en variables dummies
df_final_transformado = pd.get_dummies(df, columns=['Pais', 'Categoria_Edad', 'ID_Servicio_Comprado'], drop_first=True)

# --- Variables de interés para el modelo ---
df_ml = df_final_transformado[[
    'Edad', 'Gasto_Mes_Anterior', 'Antiguedad_Dias', 'Inflacion_Ajustada', 'Precio_Pagado', 'Afectado_Aumento', 'Churn'
]]

# --- Dividir en datos de entrenamiento y prueba ---
X = df_ml.drop('Churn', axis=1)
y = df_ml['Churn']

# División en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Escalado de los datos para mejorar el rendimiento de los modelos ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Modelo de predicción de churn ---
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predicción en el conjunto de prueba
y_pred = model.predict(X_test_scaled)

# --- Evaluación del modelo ---
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy del modelo: {accuracy * 100:.2f}%')
print('\nReporte de clasificación:')
print(classification_report(y_test, y_pred))

# --- Importancia de las características ---
# Mostramos la importancia de cada característica (coeficientes del modelo)
coef_df = pd.DataFrame({
    'Característica': X.columns,
    'Coeficiente': model.coef_[0]
})
coef_df = coef_df.sort_values(by='Coeficiente', ascending=False)

print('\nImportancia de las características:')
print(coef_df)

# --- Visualización de la importancia de las características ---
plt.figure(figsize=(10, 6))
sns.barplot(x='Coeficiente', y='Característica', data=coef_df)
plt.title('Importancia de las características en la predicción de Churn')
plt.show()

# --- Ajustar el modelo para predicciones adicionales ---
# Si tienes nuevos clientes con datos de inflación, podrías predecir el churn de la siguiente manera:
# Ejemplo con nuevos datos:
nuevos_clientes = pd.DataFrame({
    'Edad': [25, 40, 30],
    'Gasto_Mes_Anterior': [100, 200, 150],
    'Antiguedad_Dias': [150, 200, 180],
    'Inflacion_Ajustada': [120.0, 5.2, 4.5],  # Inflación ajustada para cada cliente
    'Precio_Pagado': [50, 60, 55],
    'Afectado_Aumento': [1, 0, 1]
})

# Predecir el churn para nuevos clientes
nuevos_clientes_scaled = scaler.transform(nuevos_clientes)
predicciones = model.predict(nuevos_clientes_scaled)
print('\nPredicciones de Churn para nuevos clientes:')
print(predicciones)
