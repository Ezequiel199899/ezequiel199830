import pandas as pd
import numpy as np

# Parámetros iniciales
salario_base = 1000
gasto_cliente_base = 10
años = range(2025, 2045)
inflacion_media = 3

# Listas para guardar resultados
salarios = []
facturas = []
gasto_cliente = []

# Valores iniciales
salario_actual = salario_base
factura_actual = gasto_cliente_base
gasto_actual = gasto_cliente_base

# Simulación año por año
for año in años:
    inflacion = inflacion_media

    ajuste_salario = salario_actual * (1 + 2 * inflacion / 100)
    salarios.append(ajuste_salario)
    salario_actual = ajuste_salario

    ajuste_factura = factura_actual * (1 + inflacion / 3 / 100)
    ajuste_factura = max(ajuste_factura, factura_actual * 1.01)
    facturas.append(ajuste_factura)
    factura_actual = ajuste_factura

    if np.random.rand() < 0.5:
        gasto_actual *= (1 + inflacion / 100)
    gasto_cliente.append(gasto_actual)

# DataFrame de resultados
df = pd.DataFrame({
    'Año': años,
    'Salario_Ajustado': salarios,
    'Factura_Ajustada': facturas,
    'Gasto_Cliente': gasto_cliente
})

# Recomendación empresarial
print("\n📊 Recomendación empresarial basada en la simulación:")
print("- Mantener el ajuste salarial al doble de la inflación es clave para retener talento.")
print("- Aumentar facturas al menos un 1% anual y considerar el impacto inflacionario mejora el margen.")
print("- Optimizar costos internos y mejorar el valor ofrecido al cliente impulsa la rentabilidad.")
print(f"- El gasto promedio del cliente, desde ${gasto_cliente_base} anuales, proyecta un crecimiento significativo en 20 años.")

# Facturación total acumulada
facturacion_total = df['Factura_Ajustada'].sum()
print(f"\n💵 Facturación total estimada por cliente durante 20 años: USD {facturacion_total:.2f}")
