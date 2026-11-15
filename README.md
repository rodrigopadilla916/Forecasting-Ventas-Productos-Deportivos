# Forecasting de ventas de productos deportivos

Dashboard de Streamlit para simular y visualizar las ventas diarias previstas
durante noviembre de 2025. Utiliza un modelo
`HistGradientBoostingRegressor` y permite explorar escenarios de descuento y
precios de la competencia.

## Contenido

- `app/`: aplicación de Streamlit.
- `notebooks/`: notebooks de exploración, preparación de variables y
  entrenamiento.
- `data/`: ubicación local para datos originales y procesados.
- `models/`: ubicación local para el modelo entrenado.

## Instalación

Requiere Python y PowerShell en Windows:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Archivos locales necesarios

Los datos procesados y el modelo no se incluyen en este repositorio. Para
ejecutar la aplicación, coloca los archivos en estas rutas:

- `data/processed/df_inferencia_transformado.csv`
- `models/modelo_final.joblib`

Consulta los notebooks para revisar el flujo de preparación y entrenamiento.
Los archivos de datos y el modelo están excluidos de Git mediante `.gitignore`.

## Ejecutar la aplicación

Con los archivos locales disponibles y el entorno activado:

```powershell
streamlit run app/app.py
```
