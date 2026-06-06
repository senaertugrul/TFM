# Guía de Estilos — TFM Predicción de Tráfico M30
> Documento de referencia para homogeneizar gráficas y tablas entre todos los modelos del proyecto.
> Aplica a: VAR, XGBoost, LSTM y cualquier modelo adicional.

---

## 1. Paleta de Colores

| Elemento                        | Color HEX   | Descripción              |
|---------------------------------|-------------|--------------------------|
| Real (168h)                     | `#1f77b4`   | Azul estándar matplotlib |
| Predicción del modelo           | `#ff7f0e`   | Naranja, línea discontinua |
| Línea vertical separadora       | `grey`      | Gris, línea punteada     |
| Historia real (si se incluye)   | `#4d4d4d`   | Gris oscuro              |

---

## 2. Especificaciones de la Gráfica de Predicción

### 2.1 Estructura general
- Grid de **2×2 subplots**, uno por sensor (3820, 6642, 6676, 6782)
- Tamaño de figura: `figsize=(18, 10)`
- Título principal dinámico con el nombre del modelo y el horizonte predicho
- `plt.tight_layout(rect=[0, 0, 1, 0.96])`

### 2.2 Semana de referencia para la gráfica
- **Período:** 2 de febrero de 2026 (lunes) → 8 de febrero de 2026 (domingo)
- **Duración:** 168 horas exactas
- **Motivo:** Semana laboral ordinaria, sin festivos en Madrid

### 2.3 Líneas a representar
| Línea               | Color      | Estilo         | Grosor | Etiqueta                      |
|---------------------|------------|----------------|--------|-------------------------------|
| Real (168h)         | `#1f77b4`  | Sólida         | 2      | `'Real (168h)'`               |
| Predicción modelo   | `#ff7f0e`  | Discontinua `--` | 2    | `'Predicción MODELO(lag)'`    |
| Inicio predicción   | `grey`     | Punteada `:`   | 1.5    | `'Inicio predicción'`         |

> ⚠️ **No incluir** la línea de historia real (gris oscuro) en la versión final de la gráfica.
> Se eliminó para no restar detalle visual a la zona de predicción.

### 2.4 Formato de ejes y grid
```python
ax.set_title(f'Sensor {sensor_id}', fontsize=13, fontweight='bold')
ax.set_ylabel('Intensidad (veh/h)', fontsize=11)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b\n%Hh'))
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper right', fontsize=9)
ax.tick_params(axis='x', rotation=0)
```

### 2.5 Plantilla de código reutilizable

```python
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# ---------------------------------------------------------------
# PARÁMETROS A AJUSTAR POR CADA MODELO:
# - nombre_modelo   : str  → ej. 'VAR(48)', 'XGBoost', 'LSTM'
# - lag_modelo      : int  → ventana de entrada del modelo en horas
# - df_pred_semana  : DataFrame con índice temporal y columnas de intensidad por sensor
# - real_semana     : DataFrame con el real del 2-8 feb 2026
# - cols_intensidad : lista de columnas de intensidad de los 4 sensores
# ---------------------------------------------------------------

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 10))
fig.suptitle(
    f'{nombre_modelo} — Predicción: 2 al 8 de febrero 2026 (168h)',
    fontsize=16, fontweight='bold', y=0.98
)
axes = axes.flatten()

for i, col in enumerate(cols_intensidad):
    ax = axes[i]
    sensor_id = col.split('_')[1]

    # Predicción del modelo
    ax.plot(
        df_pred_semana.index, df_pred_semana[col],
        color='#ff7f0e', linewidth=2, linestyle='--',
        label=f'Predicción {nombre_modelo}'
    )
    # Real de la semana
    ax.plot(
        real_semana.index, real_semana[col],
        color='#1f77b4', linewidth=2, label='Real (168h)'
    )
    # Línea vertical separadora
    ax.axvline(
        x=pd.Timestamp("2026-02-02 00:00:00"),
        color='grey', linestyle=':', linewidth=1.5, label='Inicio predicción'
    )

    ax.set_title(f'Sensor {sensor_id}', fontsize=13, fontweight='bold')
    ax.set_ylabel('Intensidad (veh/h)', fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b\n%Hh'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=9)
    ax.tick_params(axis='x', rotation=0)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
```

---

## 3. Especificaciones de la Tabla Resumen de Métricas

### 3.1 Estructura
- Una tabla por horizonte: **1 HORA**, **24 HORAS**, **168 HORAS**
- Filas: 4 sensores + 1 fila de **PROMEDIO** al final
- Columnas: Sensor | RMSE | MAE | R²

### 3.2 Casos especiales
| Caso                          | Tratamiento         |
|-------------------------------|---------------------|
| R² con n=1 (horizonte 1h)     | Mostrar `N/A`       |
| R² negativo                   | Mostrar el valor tal cual (es un resultado válido) |
| Promedio de R² con N/A        | Excluir los N/A del cálculo del promedio, mostrar `N/A` si todos son N/A |

### 3.3 Plantilla de código reutilizable

```python
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sensores      = ["Sensor 3820", "Sensor 6642", "Sensor 6676", "Sensor 6782"]
horizontes    = ["1 HORA", "24 HORAS", "168 HORAS"]
cols_intensidad = [col for col in cols_endogenas if 'intensidad' in col]

# resultados = diccionario con estructura:
# { "1 HORA": { "Sensor XXXX": {"RMSE": x, "MAE": x, "R²": x o None} }, ... }

for horizonte in horizontes:
    print(f"{'='*58}")
    print(f"  HORIZONTE: {horizonte}")
    print(f"{'='*58}")
    print(f"  {'Sensor':<14} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
    print(f"  {'-'*44}")

    rmse_vals, mae_vals, r2_vals = [], [], []

    for sensor in sensores:
        m = resultados[horizonte][sensor]
        r2_str = f"{m['R²']:>10.4f}" if m['R²'] is not None else f"{'N/A':>10}"
        print(f"  {sensor:<14} {m['RMSE']:>10.2f} {m['MAE']:>10.2f} {r2_str}")
        rmse_vals.append(m['RMSE'])
        mae_vals.append(m['MAE'])
        if m['R²'] is not None:
            r2_vals.append(m['R²'])

    print(f"  {'-'*44}")
    r2_prom_str = f"{sum(r2_vals)/len(r2_vals):>10.4f}" if r2_vals else f"{'N/A':>10}"
    print(f"  {'PROMEDIO':<14} "
          f"{sum(rmse_vals)/len(rmse_vals):>10.2f} "
          f"{sum(mae_vals)/len(mae_vals):>10.2f} "
          f"{r2_prom_str}")
    print()
```

---

## 4. Especificaciones del Excel Comparativo

### 4.1 Estructura del archivo
- Una **hoja por horizonte**: `1_HORA`, `24_HORAS`, `168_HORAS`
- Columnas agrupadas por modelo, con RMSE, MAE y R² para cada uno
- Fila de promedios al final con fondo azul claro

### 4.2 Estilos de celdas

| Elemento              | Color fondo HEX | Fuente              |
|-----------------------|-----------------|---------------------|
| Título horizonte      | `2F4F8F`        | Blanca, negrita, 12pt |
| Cabecera modelo       | Según modelo*   | Blanca, negrita, 10pt |
| Subcabecera métricas  | `4472C4`        | Blanca, negrita, 10pt |
| Datos modelo 1        | `FCE4D6`        | Normal, 10pt        |
| Datos modelo 2        | `E2EFDA`        | Normal, 10pt        |
| Fila PROMEDIO         | `D9E1F2`        | Negrita, 10pt       |

> *Colores sugeridos para cabeceras de modelos:
> - VAR-48: `C55A11` (naranja oscuro)
> - VAR-168: `375623` (verde oscuro)
> - XGBoost: `7B2D8B` (morado)
> - LSTM: `1F4E79` (azul oscuro)

### 4.3 Anchos de columna recomendados
```python
anchos = [16, 12, 12, 12, 12, 12, 12]  # Sensor + 3 métricas × 2 modelos
```

### 4.4 Ruta de exportación
```python
ruta_excel = "../../data/processed/comparativa_MODELOS.xlsx"
```

---

## 5. Métricas de Evaluación

| Métrica | Función                          | Librería   |
|---------|----------------------------------|------------|
| RMSE    | `np.sqrt(mean_squared_error())` | sklearn    |
| MAE     | `mean_absolute_error()`         | sklearn    |
| R²      | `r2_score()`                    | sklearn    |

### Conjuntos de evaluación
- **Validación:** Durante el desarrollo del modelo (nov-dic 2025)
- **Test:** Evaluación final, nunca tocar hasta tener el modelo listo (ene-feb 2026)

### Horizontes de predicción
| Horizonte | Pasos | Descripción      |
|-----------|-------|------------------|
| 1h        | 1     | Corto plazo      |
| 24h       | 24    | Medio plazo      |
| 168h      | 168   | Largo plazo (1 semana) |

---

## 6. Sensores del Proyecto

| ID Sensor | Descripción |
|-----------|-------------|
| 3820      | Sensor 1    |
| 6642      | Sensor 2    |
| 6676      | Sensor 3    |
| 6782      | Sensor 4    |

---

## 7. Splits Temporales del Dataset

| Split | Inicio           | Fin              | Filas  |
|-------|------------------|------------------|--------|
| Train | 2024-01-01 00:00 | 2025-10-31 23:00 | 16.080 |
| Val   | 2025-11-01 00:00 | 2025-12-31 23:00 | 1.464  |
| Test  | 2026-01-01 00:00 | 2026-02-28 23:00 | 1.416  |

> ⚠️ Los tres splits no tienen solapamiento y fueron generados en el notebook de VAR
> como archivos `*_clean.csv`. Todos los modelos deben partir de estos mismos archivos.

---

*Documento generado para el TFM de predicción de tráfico M30 — Versión 1.0*