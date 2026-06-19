# Predicción del Tráfico Vehicular en la M-30 mediante Modelos Estadísticos, Machine Learning y Deep Learning

Repositorio correspondiente al Trabajo Fin de Máster (TFM) desarrollado en el Máster en Big Data & Analytics.

El proyecto implementa y compara distintos enfoques de predicción de series temporales para estimar el flujo vehicular en la autopista urbana M-30 (Madrid), evaluando modelos estadísticos clásicos, algoritmos de Machine Learning y arquitecturas de Deep Learning bajo un mismo marco metodológico.

---

# Objetivos

El objetivo principal consiste en comparar el rendimiento predictivo de diferentes familias de modelos para la predicción del tráfico vehicular en múltiples horizontes temporales.

En concreto, el proyecto evalúa:

- Modelos estadísticos clásicos.
- Modelos de Machine Learning.
- Modelos de Deep Learning.

Todos los modelos fueron entrenados y evaluados utilizando exactamente la misma estrategia de partición temporal, ventanas históricas, horizontes de predicción y métricas de evaluación.

---

# Modelos implementados

## Modelos de referencia

- Persistence

## Modelos estadísticos

- SARIMAX
- VARX (Vector Autoregression with Exogenous Variables)
- VECMX (Vector Error Correction Model with Exogenous Variables)

## Machine Learning

- Support Vector Regression (SVR)
- Random Forest Regressor
- XGBoost Regressor

## Deep Learning

- Long Short-Term Memory (LSTM)
- Temporal Convolutional Network (TCN)

---

# Estructura del proyecto

```text
tfm-prediccion-trafico/
├── .gitignore
├── requirements.txt
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── Split_Datasets/
│   │   └── models/
│   ├── results/
│   ├── test/
│   └── extra/
├── documents/
├── notebooks/
│   └── Modelos/
└── src/
```

---

# Tecnologías utilizadas

- Python 3.11.8
- Pandas
- NumPy
- Statsmodels
- Scikit-learn
- XGBoost
- PyTorch
- Matplotlib
- Seaborn
- Jupyter Notebook

---

# Instalación

Clonar el repositorio:

```bash
git clone https://github.com/USUARIO/tfm-prediccion-trafico.git
```

Acceder al directorio:

```bash
cd tfm-prediccion-trafico
```

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

---

# Flujo de trabajo

El proyecto sigue una metodología basada en CRISP-DM:

1. Comprensión del negocio.
2. Comprensión de los datos.
3. Preparación de los datos.
4. Modelado.
5. Evaluación comparativa de modelos.
6. Análisis de resultados.

---

# Estrategia experimental

Todos los modelos fueron evaluados bajo un diseño experimental homogéneo:

- División cronológica Train – Validation – Test.
- Prevención de Data Leakage.
- Ventanas históricas de 48 h y 168 h.
- Horizontes de predicción de 1 h, 24 h y 168 h.
- Evaluación mediante RMSE, MAE y R².

Los modelos estadísticos fueron evaluados mediante **Rolling Forecast**, mientras que los modelos de Machine Learning y Deep Learning utilizaron una estrategia de **Direct Forecasting**, manteniendo la comparabilidad mediante el uso de los mismos conjuntos de datos y métricas de evaluación.

---

# Reproducibilidad

Para garantizar la reproducibilidad del experimento:

- Se fijó una versión específica de Python.
- Se utilizaron versiones controladas de todas las librerías.
- La estructura del repositorio mantiene separadas las fases de preparación, entrenamiento y evaluación.
- Los resultados fueron obtenidos a partir de conjuntos temporales independientes de entrenamiento, validación y prueba.

---

# Autores

Trabajo Fin de Máster desarrollado por:

- Sena Ertugrul
- Luis Fernando Puentes
- Ronaldo Barrios

Máster en Big Data & Analytics

---

# Licencia

Este repositorio tiene fines exclusivamente académicos y de investigación.
