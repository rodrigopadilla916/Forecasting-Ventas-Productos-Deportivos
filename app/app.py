"""
Dashboard de simulación de ventas - Noviembre 2025
===================================================
Aplicación Streamlit para simular y visualizar predicciones de ventas diarias
de noviembre de 2025 usando un modelo HistGradientBoostingRegressor
previamente entrenado.

El usuario puede ajustar el descuento aplicado a un producto y el escenario
de precios de la competencia; la app recalcula las predicciones día a día de
forma recursiva (actualizando los lags y la media móvil con cada nueva
predicción) y muestra un dashboard con KPIs, gráfico diario, tabla detallada
y comparativa de escenarios de competencia.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# ---------------------------------------------------------------------------
# Configuración general de la página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Simulador de Ventas · Noviembre 2025",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

MORADO = "#667eea"
MORADO_OSCURO = "#764ba2"
ROJO_BF = "#e63946"

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "modelo_final.joblib"
DATA_PATH = BASE_DIR / "data" / "processed" / "df_inferencia_transformado.csv"

LAG_COLS = [
    "lag_1_unidades_vendidas",
    "lag_2_unidades_vendidas",
    "lag_3_unidades_vendidas",
    "lag_4_unidades_vendidas",
    "lag_5_unidades_vendidas",
    "lag_6_unidades_vendidas",
    "lag_7_unidades_vendidas",
]
MA_COL = "media_movil_7_dias"

sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "sans-serif"

# ---------------------------------------------------------------------------
# Estilos CSS personalizados (paleta morada/azul)
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    .main-header {{
        background: linear-gradient(90deg, {MORADO} 0%, {MORADO_OSCURO} 100%);
        padding: 1.6rem 2rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.35);
    }}
    .main-header h1 {{
        margin: 0;
        font-size: 1.9rem;
    }}
    .main-header p {{
        margin: 0.3rem 0 0 0;
        opacity: 0.92;
        font-size: 1.05rem;
    }}
    div[data-testid="stMetric"] {{
        background: linear-gradient(135deg, rgba(102,126,234,0.10), rgba(118,75,162,0.10));
        border: 1px solid rgba(102,126,234,0.25);
        padding: 1rem 1rem 0.6rem 1rem;
        border-radius: 12px;
    }}
    .seccion-titulo {{
        color: {MORADO_OSCURO};
        border-left: 6px solid {MORADO};
        padding-left: 0.6rem;
        margin-top: 1.2rem;
    }}
    .stButton > button {{
        background: linear-gradient(90deg, {MORADO} 0%, {MORADO_OSCURO} 100%);
        color: white;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 0.7rem 0;
        border-radius: 10px;
        border: none;
        width: 100%;
    }}
    .stButton > button:hover {{
        opacity: 0.9;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Carga de recursos (modelo y datos) - cacheados
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def cargar_modelo():
    modelo = joblib.load(MODEL_PATH)
    return modelo, list(modelo.feature_names_in_)


@st.cache_data(show_spinner=False)
def cargar_datos():
    df = pd.read_csv(DATA_PATH, encoding="utf-8", parse_dates=["fecha"])
    df = df.sort_values(["producto_id", "fecha"]).reset_index(drop=True)
    return df


try:
    modelo, feature_cols = cargar_modelo()
    df_inferencia = cargar_datos()
    error_carga = None
except Exception as exc:  # noqa: BLE001 - mostramos cualquier error de carga al usuario
    modelo, feature_cols, df_inferencia = None, None, None
    error_carga = str(exc)

DIAS_SEMANA_ES = {
    0: "Lunes",
    1: "Martes",
    2: "Miércoles",
    3: "Jueves",
    4: "Viernes",
    5: "Sábado",
    6: "Domingo",
}


# ---------------------------------------------------------------------------
# Lógica de simulación recursiva
# ---------------------------------------------------------------------------
def simular_escenarios_producto(
    df_producto: pd.DataFrame,
    ajuste_descuento_pct: float,
    escenarios_competencia: dict,
    modelo,
    feature_cols: list,
) -> dict:
    """Simula varios escenarios recursivos, agrupando las predicciones por día.

    - Recalcula precio_venta, descuento_porcentaje, precio_Competencia y
      ratio_precio según los controles del usuario.
    - Para el día 1 utiliza los lags ya presentes en el dataset (calculados
      desde octubre).
    - En cada día predice todos los escenarios en una sola llamada al modelo.
    """
    dias_base = df_producto.sort_values("fecha").reset_index(drop=True)
    dias_por_escenario = {}
    lags_por_escenario = {}
    predicciones_por_escenario = {}

    for nombre_escenario, ajuste_competencia_pct in escenarios_competencia.items():
        dias = dias_base.copy()
        dias["precio_venta"] = dias["precio_base"] * (
            1 - ajuste_descuento_pct / 100.0
        )
        dias["descuento_porcentaje"] = (
            (dias["precio_base"] - dias["precio_venta"]) / dias["precio_base"] * 100.0
        )
        dias["precio_Competencia"] = dias["precio_Competencia"] * (
            1 + ajuste_competencia_pct / 100.0
        )
        dias["ratio_precio"] = dias["precio_venta"] / dias["precio_Competencia"]

        dias_por_escenario[nombre_escenario] = dias
        lags_por_escenario[nombre_escenario] = (
            dias.loc[0, LAG_COLS].astype(float).tolist()
        )
        predicciones_por_escenario[nombre_escenario] = []

    nombres_escenarios = list(dias_por_escenario)
    for i in range(len(dias_base)):
        filas = []
        for nombre_escenario in nombres_escenarios:
            dias = dias_por_escenario[nombre_escenario]
            lags_actuales = lags_por_escenario[nombre_escenario]

            for col, val in zip(LAG_COLS, lags_actuales):
                dias.loc[i, col] = val
            dias.loc[i, MA_COL] = float(np.mean(lags_actuales))
            filas.append(dias.loc[[i], feature_cols])

        predicciones_dia = modelo.predict(pd.concat(filas, ignore_index=True))
        for nombre_escenario, pred in zip(nombres_escenarios, predicciones_dia):
            pred = max(float(pred), 0.0)
            predicciones_por_escenario[nombre_escenario].append(pred)
            lags_por_escenario[nombre_escenario] = (
                [pred] + lags_por_escenario[nombre_escenario][:-1]
            )

    for nombre_escenario, dias in dias_por_escenario.items():
        dias["unidades_predichas"] = predicciones_por_escenario[nombre_escenario]
        dias["ingresos_predichos"] = dias["unidades_predichas"] * dias["precio_venta"]
        dias["dia_semana_nombre"] = dias["fecha"].dt.dayofweek.map(DIAS_SEMANA_ES)

    return dias_por_escenario


def formato_eur(valor: float, decimales: int = 2) -> str:
    texto = f"{valor:,.{decimales}f} €"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


def formato_unidades(valor: float) -> str:
    return f"{valor:,.0f}".replace(",", ".")


# ---------------------------------------------------------------------------
# Encabezado principal
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>📈 Simulador de Ventas — Noviembre 2025</h1>
        <p>Predicciones recursivas día a día con ajuste de descuento y escenarios de competencia</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if error_carga is not None:
    st.error(
        "❌ No se pudieron cargar el modelo o los datos necesarios para la simulación.\n\n"
        f"Detalle técnico: {error_carga}\n\n"
        f"Verifica que existan los archivos:\n- `{MODEL_PATH}`\n- `{DATA_PATH}`"
    )
    st.stop()

productos = (
    df_inferencia[["producto_id", "nombre"]]
    .drop_duplicates()
    .sort_values("nombre")
    .reset_index(drop=True)
)

# ---------------------------------------------------------------------------
# Sidebar - Controles de simulación
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎛️ Controles de Simulación")
    st.markdown("---")

    nombre_producto = st.selectbox(
        "🛍️ Producto",
        options=productos["nombre"].tolist(),
        help="Elige el producto sobre el que quieres simular ventas para noviembre 2025.",
    )
    producto_id = productos.loc[
        productos["nombre"] == nombre_producto, "producto_id"
    ].iloc[0]

    st.markdown("### 💸 Ajuste de descuento")
    ajuste_descuento = st.slider(
        "Variación sobre el precio base",
        min_value=-50,
        max_value=50,
        value=0,
        step=5,
        format="%d%%",
        help="Un valor positivo aplica un descuento sobre el precio base; un valor negativo lo encarece.",
    )

    st.markdown("### 🏷️ Escenario de competencia")
    escenario_competencia = st.radio(
        "Variación del precio de la competencia",
        options=["Actual (0%)", "Competencia -5%", "Competencia +5%"],
        help="Simula cómo reaccionarían las ventas si la competencia bajara o subiera sus precios.",
    )
    mapa_escenarios = {
        "Actual (0%)": 0,
        "Competencia -5%": -5,
        "Competencia +5%": 5,
    }
    st.markdown("---")
    simular = st.button("🚀 Simular Ventas", width="stretch")

    st.markdown("---")
    st.caption(
        "ℹ️ El modelo predice día a día de forma recursiva: cada predicción "
        "actualiza los lags y la media móvil del día siguiente."
    )

# ---------------------------------------------------------------------------
# Ejecución de la simulación
# ---------------------------------------------------------------------------
if simular:
    with st.spinner("🔄 Calculando predicciones recursivas para los 30 días de noviembre..."):
        df_producto = df_inferencia[df_inferencia["producto_id"] == producto_id]

        resultados_escenarios = simular_escenarios_producto(
            df_producto, ajuste_descuento, mapa_escenarios, modelo, feature_cols
        )
        resultado_principal = resultados_escenarios[escenario_competencia]

    st.session_state["resultado_principal"] = resultado_principal
    st.session_state["resultados_escenarios"] = resultados_escenarios
    st.session_state["producto_simulado"] = nombre_producto
    st.session_state["escenario_simulado"] = escenario_competencia

# ---------------------------------------------------------------------------
# Dashboard de resultados
# ---------------------------------------------------------------------------
if "resultado_principal" not in st.session_state:
    st.info(
        "👈 Configura los controles en el panel lateral y pulsa **Simular Ventas** "
        "para generar el dashboard de predicciones de noviembre 2025."
    )
    st.stop()

resultado = st.session_state["resultado_principal"]
resultados_escenarios = st.session_state["resultados_escenarios"]
producto_simulado = st.session_state["producto_simulado"]
escenario_simulado = st.session_state["escenario_simulado"]

st.markdown(
    f"### 🎯 Resultados de la simulación — **{producto_simulado}**  "
    f"<span style='color:{MORADO_OSCURO}; font-weight:600;'>({escenario_simulado})</span>",
    unsafe_allow_html=True,
)
st.markdown("---")

# --- KPIs ---
st.markdown('<h4 class="seccion-titulo">📊 KPIs del mes</h4>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

unidades_totales = resultado["unidades_predichas"].sum()
ingresos_totales = resultado["ingresos_predichos"].sum()
precio_promedio = resultado["precio_venta"].mean()
descuento_promedio = resultado["descuento_porcentaje"].mean()

col1.metric("📦 Unidades totales proyectadas", formato_unidades(unidades_totales))
col2.metric("💰 Ingresos proyectados", formato_eur(ingresos_totales))
col3.metric("🏷️ Precio promedio de venta", formato_eur(precio_promedio))
col4.metric("💸 Descuento promedio", f"{descuento_promedio:.1f}%")

st.markdown("---")

# --- Gráfico de predicción diaria ---
st.markdown(
    '<h4 class="seccion-titulo">📈 Predicción diaria de unidades vendidas</h4>',
    unsafe_allow_html=True,
)

fig, ax = plt.subplots(figsize=(12, 5))
sns.lineplot(
    data=resultado,
    x="dia_mes",
    y="unidades_predichas",
    ax=ax,
    color=MORADO_OSCURO,
    linewidth=2.5,
    marker="o",
    markersize=5,
)

fila_bf = resultado.loc[resultado["es_black_friday"], "unidades_predichas"]
if not fila_bf.empty:
    dia_bf = resultado.loc[resultado["es_black_friday"], "dia_mes"].iloc[0]
    valor_bf = fila_bf.iloc[0]
    ax.axvline(dia_bf, color=ROJO_BF, linestyle="--", linewidth=1.8, alpha=0.8)
    ax.scatter([dia_bf], [valor_bf], color=ROJO_BF, s=140, zorder=5, edgecolor="white", linewidth=1.5)
    ax.annotate(
        f"BLACK FRIDAY\n{valor_bf:.0f} uds.",
        xy=(dia_bf, valor_bf),
        xytext=(dia_bf - 5.5, valor_bf + (resultado["unidades_predichas"].max() * 0.12)),
        fontsize=10,
        fontweight="bold",
        color=ROJO_BF,
        arrowprops=dict(arrowstyle="->", color=ROJO_BF, lw=1.5),
    )

ax.set_xlabel("Día de noviembre 2025", fontsize=11)
ax.set_ylabel("Unidades vendidas predichas", fontsize=11)
ax.set_title(f"Predicción de ventas diarias — {producto_simulado}", fontsize=13, fontweight="bold")
ax.set_xticks(range(1, 31))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

st.markdown("---")

# --- Tabla detallada ---
st.markdown('<h4 class="seccion-titulo">📋 Detalle diario de noviembre</h4>', unsafe_allow_html=True)

tabla = resultado[
    [
        "fecha",
        "dia_semana_nombre",
        "precio_venta",
        "precio_Competencia",
        "descuento_porcentaje",
        "unidades_predichas",
        "ingresos_predichos",
        "es_black_friday",
    ]
].copy()
tabla["fecha"] = tabla["fecha"].dt.strftime("%d/%m/%Y")
tabla["destacado"] = tabla["es_black_friday"].apply(lambda x: "🔥 Black Friday" if x else "")
tabla = tabla.rename(
    columns={
        "fecha": "Fecha",
        "dia_semana_nombre": "Día",
        "precio_venta": "Precio venta (€)",
        "precio_Competencia": "Precio competencia (€)",
        "descuento_porcentaje": "Descuento (%)",
        "unidades_predichas": "Unidades predichas",
        "ingresos_predichos": "Ingresos (€)",
        "destacado": "Evento",
    }
).drop(columns=["es_black_friday"])


def resaltar_black_friday(fila):
    if fila["Evento"]:
        return ["background-color: rgba(230, 57, 70, 0.18); font-weight: 700;"] * len(fila)
    return [""] * len(fila)


st.dataframe(
    tabla.style.apply(resaltar_black_friday, axis=1).format(
        {
            "Precio venta (€)": "{:.2f}",
            "Precio competencia (€)": "{:.2f}",
            "Descuento (%)": "{:.1f}",
            "Unidades predichas": "{:.0f}",
            "Ingresos (€)": "{:.2f}",
        }
    ),
    use_container_width=True,
    height=420,
)

st.markdown("---")

# --- Comparativa de escenarios de competencia ---
st.markdown(
    '<h4 class="seccion-titulo">🏁 Comparativa de escenarios de competencia</h4>',
    unsafe_allow_html=True,
)
st.caption(
    "Se mantiene el descuento elegido por el usuario y solo varía el precio de la competencia."
)

cols_escenarios = st.columns(3)
for col, (nombre_esc, df_esc) in zip(cols_escenarios, resultados_escenarios.items()):
    uds = df_esc["unidades_predichas"].sum()
    ingresos = df_esc["ingresos_predichos"].sum()
    es_actual = nombre_esc == escenario_simulado
    icono = "👉 " if es_actual else ""
    with col:
        st.markdown(f"**{icono}{nombre_esc}**")
        st.metric("Unidades totales", formato_unidades(uds))
        st.metric("Ingresos totales", formato_eur(ingresos, decimales=0))

st.markdown("---")
st.caption(
    "Simulación generada con un modelo HistGradientBoostingRegressor y predicciones "
    "recursivas (lags y media móvil actualizados día a día)."
)
