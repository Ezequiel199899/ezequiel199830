import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===================================
# Parámetros base
# ===================================
np.random.seed(42)

años = np.arange(2025, 2025 + 20)
inflacion_media = 5.0  # 5% anual
salario_base = 200
gasto_cliente_base = 10  # gasto anual por cliente

# Ajustes:
# - Salario ajustado: 2× inflación
# - Factura ajustada: +1/3 de inflación
# - Plan mínimo: +1% anual

salarios = []
facturas = []
gasto_cliente = []

salario_actual = salario_base
factura_actual = gasto_cliente_base
gasto_actual = gasto_cliente_base

for año in años:
    # Simular inflación del año
    inflacion = inflacion_media
    
    # Ajustar salario para no perder empleados
    ajuste_salario = salario_actual * (1 + 2 * inflacion / 100)
    salarios.append(ajuste_salario)
    salario_actual = ajuste_salario
    
    # Ajustar factura para no perder clientes
    ajuste_factura = factura_actual * (1 + (inflacion / 3) / 100)
    
    # Plan mínimo: subir 1% anual
    ajuste_factura = max(ajuste_factura, factura_actual * 1.01)
    facturas.append(ajuste_factura)
    factura_actual = ajuste_factura
    
    # Gasto del cliente: base + probabilidad de aumento
    gasto_probabilidad = np.random.rand()
    if gasto_probabilidad < 0.5:  # 50% chance de aumento
        gasto_actual = gasto_actual * (1 + inflacion / 100)
    else:
        # Si no se ajusta bien, alta probabilidad de baja: el gasto se mantiene
        gasto_actual = gasto_actual
    gasto_cliente.append(gasto_actual)

# ===================================
# DataFrame de resultados
# ===================================
df = pd.DataFrame({
    'Año': años,
    'Salario_Ajustado': salarios,
    'Factura_Ajustada': facturas,
    'Gasto_Cliente': gasto_cliente
})

# ===================================
# Recomendación general
# ===================================
print("\n📊 RECOMENDACIÓN EMPRESARIAL:")
print("- Mantener ajuste salarial al doble de inflación para no perder empleados.")
print("- Subir facturas al menos 1% anual, ideal 1/3 de la inflación para sostener margen.")
print("- Mejorar precio al cliente y bajar costos internos para expandirse.")
print("- Gasto del cliente promedio parte de $10 anuales y puede crecer hasta 20% en 20 años.")

# ===================================
# Visualizaciones
# ===================================
plt.figure(figsize=(12, 6))
plt.plot(df['Año'], df['Salario_Ajustado'], label='Salario Ajustado (2× Inflación)')
plt.plot(df['Año'], df['Factura_Ajustada'], label='Factura Ajustada (1/3 Inflación, 1% min)')
plt.plot(df['Año'], df['Gasto_Cliente'], label='Gasto Cliente')
plt.title('Proyección Salarial, Factura y Gasto Cliente (20 años)')
plt.xlabel('Año')
plt.ylabel('USD')
plt.legend()
plt.grid(True)
plt.show()

# ===================================
# Facturación total por cliente acumulada
# ===================================
facturacion_total = df['Factura_Ajustada'].sum()
print(f'\n💵 Facturación total estimada por cliente durante 20 años: USD {facturacion_total:.2f}')

# ===================================
# Guardar a CSV (opcional)
# ===================================
df.to_csv('proyeccion_facturacion_clientes_20_anos.csv', index=False)
print("Archivo guardado: proyeccion_facturacion_clientes_20_anos.csv")


