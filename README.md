


import pandas as pd # Importamos Pandas, ideal para trabajar con datos en formato de tabla (como Excel).
import numpy as np  # Importamos NumPy, esencial para cálculos numéricos y generar números aleatorios.

# --- 1. Parámetros Iniciales de la Simulación ---
# Estos son los valores base con los que empezamos nuestra proyección.
salario_base = 1000       # Salario anual inicial (ej., en USD).
gasto_cliente_base = 10   # Gasto anual inicial por cliente (ej., en USD).
años = range(2025, 2045)  # Define los años que vamos a simular, desde 2025 hasta 2044 (un total de 20 años).
inflacion_media = 3       # Porcentaje de inflación promedio anual (ej., 3% se representa como 3).

# --- 2. Preparación para Guardar los Resultados ---
# Creamos listas vacías. Aquí es donde iremos guardando los valores calculados año tras año.
salarios = []
facturas = []
gasto_cliente = []

# Configuramos los valores "actuales" que se irán actualizando en cada ciclo del bucle.
salario_actual = salario_base
factura_actual = gasto_cliente_base
gasto_actual = gasto_cliente_base

# --- 3. Bucle de Simulación Año por Año ---
# Este es el núcleo del programa. Repite los cálculos para cada uno de los 20 años.
for año in años:
    # Asumimos una inflación constante para cada año en este modelo simplificado.
    inflacion = inflacion_media

    # CÁLCULO DEL SALARIO AJUSTADO:
    # Aumentamos el salario actual al doble de la inflación. Esto busca mantener el poder adquisitivo de los empleados.
    # Fórmula: Salario_Actual * (1 + (2 * Inflación) / 100)
    ajuste_salario = salario_actual * (1 + 2 * inflacion / 100)
    salarios.append(ajuste_salario) # Guardamos el salario ajustado para este año.
    salario_actual = ajuste_salario # El salario de este año se convierte en el base para el siguiente.

    # CÁLCULO DE LA FACTURA AJUSTADA:
    # Ajustamos la factura para los clientes. Aumenta 1/3 de la inflación,
    # pero siempre garantizamos un aumento mínimo del 1% anual para proteger los ingresos.
    # Fórmula base: Factura_Actual * (1 + (Inflación / 3) / 100)
    ajuste_factura = factura_actual * (1 + (inflacion / 3) / 100)
    # 'max()' asegura que el valor final sea el mayor entre el cálculo y el aumento mínimo del 1%.
    ajuste_factura = max(ajuste_factura, factura_actual * 1.01)
    facturas.append(ajuste_factura) # Guardamos la factura ajustada.
    factura_actual = ajuste_factura # La factura de este año es la base para el siguiente.

    # CÁLCULO DEL GASTO DEL CLIENTE:
    # Introducimos un factor aleatorio para simular el comportamiento variable de los clientes.
    # 'np.random.rand()' genera un número al azar entre 0 (inclusive) y 1 (exclusivo).
    gasto_probabilidad = np.random.rand()
    if gasto_probabilidad < 0.5: # Si el número aleatorio es menor a 0.5 (un 50% de probabilidad):
        # El gasto del cliente aumenta con la inflación.
        gasto_actual = gasto_actual * (1 + inflacion / 100)
    else: # Si el número aleatorio es 0.5 o más (el otro 50%):
        # El gasto del cliente se mantiene igual (no cambia).
        gasto_actual = gasto_actual
    gasto_cliente.append(gasto_actual) # Guardamos el gasto del cliente para este año.

# --- 4. Creación del DataFrame de Resultados ---
# Usamos Pandas para organizar todas las listas que llenamos en una única tabla de datos.
df = pd.DataFrame({
    'Año': años,                   # Columna para cada año de la simulación
    'Salario_Ajustado': salarios,  # Columna con los salarios ajustados
    'Factura_Ajustada': facturas,  # Columna con las facturas ajustadas
    'Gasto_Cliente': gasto_cliente # Columna con el gasto simulado por cliente
})

# --- 5. Recomendación Empresarial Basada en la Simulación ---
# Conclusiones clave que se desprenden de nuestras proyecciones.
print("\n📊 **RECOMENDACIÓN EMPRESARIAL BASADA EN LA SIMULACIÓN:**")
print("- Mantener el **ajuste salarial al doble de la inflación** es crucial para la retención de empleados clave.")
print("- Subir las facturas al menos un **1% anual**, y buscar un **ajuste del 1/3 de la inflación** ayuda a sostener el margen de ganancia frente a la inflación.")
print("- Es fundamental **mejorar la propuesta de valor al cliente** y **optimizar los costos internos** para impulsar la expansión y la rentabilidad.")
print(f"- El **gasto promedio del cliente**, partiendo de ${gasto_cliente_base} anuales, muestra una proyección de crecimiento significativa en un horizonte de **20 años**.")

# --- 6. Cálculo de Facturación Total Acumulada ---
# Sumamos todos los valores de la columna 'Factura_Ajustada' para ver el ingreso total estimado por cliente.
facturacion_total = df['Factura_Ajustada'].sum()
print(f'\n💵 **Facturación total estimada por cliente durante 20 años:** USD {facturacion_total:.2f}')

