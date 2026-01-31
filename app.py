import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Configuración inicial
st.set_page_config(page_title="Análisis ENOE", layout="wide")

# --- 1. CARGA DE DATOS ---
@st.cache_data
def load_data():
    df = pd.read_csv('enoe_light.csv', encoding="latin1", low_memory=False)
    # Limpieza inmediata de columnas críticas
    cols = ['eda', 'sex', 'clase1', 'clase2', 'fac_tri', 'anios_esc', 'n_hij', 'e_con', 't_loc_tri', 'ingocup', 'ing7c', 'seg_soc']
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

with st.spinner('Cargando base de datos pesada...'):
    df = load_data()

# --- 2. FUNCIÓN MAESTRA DEL MODELO (CACHEADA) ---
@st.cache_data
def procesar_modelo_completo(_df):
    columnas_pnea = ['eda', 'sex', 'anios_esc', 'n_hij', 'e_con', 't_loc_tri', 'clase1']
    df_p = _df[columnas_pnea].copy()
    df_p = df_p[df_p['eda'] >= 15].dropna(subset=['clase1'])

    # Ingeniería de variables
    df_p['target_participacion'] = (df_p['clase1'] == 1).astype(int)
    df_p['es_mujer'] = (df_p['sex'] == 2).astype(int)
    df_p['mujer_con_hijos'] = df_p['es_mujer'] * df_p['n_hij'].fillna(0)
    df_p['eda_2'] = df_p['eda'] ** 2
    df_p['es_urbano'] = df_p['t_loc_tri'].isin([1, 2]).astype(int)
    df_p['tiene_pareja'] = df_p['e_con'].isin([1, 5]).astype(int)
    df_p['anios_esc'] = df_p['anios_esc'].fillna(df_p['anios_esc'].median())

    features = ['eda', 'eda_2', 'es_mujer', 'mujer_con_hijos', 'anios_esc', 'es_urbano', 'tiene_pareja']
    X = df_p[features]
    y = df_p['target_participacion']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    
    return modelo, features, X, y, X_test, y_test, y_pred, df_p

# Ejecutar entrenamiento
modelo_pnea, features, X, y, X_test, y_test, y_pred, df_pnea = procesar_modelo_completo(df)

# --- INICIO DE INTERFAZ ---
st.title("📊 Análisis de la Viabilidad Económica Por Género")
st.markdown("Estudio basado en microdatos de la ENOE (Q1-2024)")

# --- GRÁFICA 1: DINÁMICA LABORAL ---
st.header("Dinámica Laboral por Edad y Género")
df_ocupados = df[(df['eda'] >= 15) & (df['eda'] < 99) & (df['clase2'] == 1)].copy()
resumen = df_ocupados.groupby(['eda', 'sex'])['fac_tri'].sum().unstack()

fig_din = go.Figure()
if 1 in resumen.columns:
    fig_din.add_trace(go.Scatter(x=resumen.index, y=resumen[1], name='Hombres', line=dict(color='#1f77b4', width=3)))
if 2 in resumen.columns:
    fig_din.add_trace(go.Scatter(x=resumen.index, y=resumen[2], name='Mujeres', line=dict(color='#d62728', width=3)))

fig_din.update_layout(xaxis_title="Edad", yaxis_title="Personas", template="plotly_white", hovermode="x unified")
st.plotly_chart(fig_din, use_container_width=True)

# --- GRÁFICA 2: COEFICIENTES ---
st.divider()
st.header("🧠 Modelo Predictivo e Impacto")
with st.expander("Ver detalles técnicos del modelo"):
    st.write("✅ Regresión Logística entrenada con factor de estratificación.")

coef_df = pd.DataFrame({'Variable': features, 'Coeficiente': modelo_pnea.coef_[0]}).sort_values('Coeficiente')
fig_coef = go.Figure(go.Bar(
    x=coef_df['Coeficiente'], y=coef_df['Variable'], orientation='h',
    marker_color=['#d62728' if x < 0 else '#2ca02c' for x in coef_df['Coeficiente']]
))
fig_coef.update_layout(title="Impacto Relativo de las Variables", template="plotly_white")
st.plotly_chart(fig_coef, use_container_width=True)

# --- GRÁFICA 3: SIMULADOR ---
st.divider()
st.header("📉 Simulador: El 'Impuesto' a la Maternidad")
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1: edad_i = st.number_input("Edad", 15, 80, 30)
with col_s2: esc_i = st.number_input("Escolaridad", 0, 24, 12)
with col_s3: urb_i = st.toggle("¿Urbano?", True)

def calc_p(es_m):
    c = dict(zip(features, modelo_pnea.coef_[0]))
    h_r = np.arange(0, 7)
    z = (modelo_pnea.intercept_[0] + c['eda']*edad_i + c['eda_2']*(edad_i**2) + 
         c['es_mujer']*es_m + (c['mujer_con_hijos']*es_m*h_r) + c['anios_esc']*esc_i + 
         c['es_urbano']*(1 if urb_i else 0) + c['tiene_pareja']*1)
    return 1 / (1 + np.exp(-z))

fig_sim = go.Figure()
fig_sim.add_trace(go.Scatter(x=np.arange(0,7), y=calc_p(0), name="Hombres", line=dict(color='#1f77b4')))
fig_sim.add_trace(go.Scatter(x=np.arange(0,7), y=calc_p(1), name="Mujeres", line=dict(color='#d62728')))
fig_sim.update_layout(title="Probabilidad vs Número de Hijos", yaxis_range=[0,1.1], template="plotly_white")
st.plotly_chart(fig_sim, use_container_width=True)

# --- GRÁFICA 4: EFECTOS MARGINALES ---
st.divider()
st.header("🎯 Efectos Marginales (Impacto en %)")
@st.cache_data
def get_ame(X, y):
    X_c = sm.add_constant(X)
    res = sm.Logit(y, X_c).fit(disp=0)
    ame = res.get_margeff(at='overall', method='dydx').summary_frame()
    ame.index = features
    return ame

ame_df = get_ame(X, y)

# Accedemos por posición para evitar el KeyError:
# Columna 0: dy/dx | Columna 4: Conf. Int. Lower | Columna 5: Conf. Int. Upper
val_ame = ame_df.iloc[:, 0]
conf_inf = ame_df.iloc[:, 4]
conf_sup = ame_df.iloc[:, 5]

fig_ame = go.Figure()

fig_ame.add_trace(go.Scatter(
    x=val_ame, 
    y=ame_df.index, 
    mode='markers+text',
    text=[f"{v*100:+.1f}%" for v in val_ame], # Etiquetas de texto automáticas
    textposition="top center",
    error_x=dict(
        type='data', 
        symmetric=False, 
        array=conf_sup - val_ame, 
        arrayminus=val_ame - conf_inf,
        color='#d62728'
    ),
    marker=dict(color='#2ca02c', size=12)
))

fig_ame.add_vline(x=0, line_dash="dash", line_color="black")
fig_ame.update_layout(
    title="Efectos Marginales (Impacto Directo en Probabilidad %)",
    xaxis_title="Cambio en Probabilidad",
    template="plotly_white",
    height=500,
    xaxis=dict(tickformat=".1%") # Muestra el eje X como porcentaje
)

st.plotly_chart(fig_ame, use_container_width=True)

# --- GRÁFICA 5: MATRIZ DE CONFUSIÓN ---
st.divider()
st.header("🎯 Validación del Modelo")
reporte = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred)
fig_cm = ff.create_annotated_heatmap(cm[::-1], x=['Pred: No', 'Pred: Sí'], y=['Real: Sí', 'Real: No'], colorscale='Blues')
st.plotly_chart(fig_cm, use_container_width=True)

# --- SECCIÓN: FALSOS POSITIVOS ---
st.divider()
st.header("🔍 Análisis de Potencial")
df_res = X_test.copy()
df_res['real'], df_res['pred'] = y_test, y_pred
indices_fp = df_res[(df_res['real'] == 0) & (df_res['pred'] == 1)].index
fp_completos = df_pnea.loc[indices_fp]

col_f1, col_f2 = st.columns(2)
fp_h = fp_completos[fp_completos['es_mujer'] == 0]
fig_fp = go.Figure(data=[go.Pie(labels=['Hombres', 'Mujeres'], values=[len(fp_h), len(fp_completos)-len(fp_h)], hole=.4)])
col_f1.plotly_chart(fig_fp, use_container_width=True)
col_f2.table(pd.DataFrame({
    "Métrica": ["Total FP", "Escolaridad Prom.", "Hijos Prom."],
    "Valor": [len(fp_completos), f"{fp_completos['anios_esc'].mean():.1f}", f"{fp_completos['n_hij'].mean():.1f}"]
}))

# --- GRÁFICA 6: BRECHA SALARIAL ---
st.divider()
st.header("⚖️ Brecha de Ingresos")
pob_o = df[df['clase2'] == 1].copy()
dist_g = pob_o.groupby(['sex', 'ing7c'])['fac_tri'].sum().reset_index()
n_ing = {1:'<1 SM', 2:'1-2 SM', 3:'2-3 SM', 4:'3-5 SM', 5:'>5 SM', 6:'Sin Ing', 7:'N/E'}
dist_g['Rango'] = dist_g['ing7c'].map(n_ing)
dist_g['Sexo'] = dist_g['sex'].map({1:'Hombres', 2:'Mujeres'})

fig_sal = px.bar(dist_g, x='Rango', y='fac_tri', color='Sexo', barmode='group',
                 color_discrete_map={'Hombres':'#3498db', 'Mujeres':'#e74c3c'})
fig_sal.update_layout(template="plotly_white", yaxis_title="Personas")
st.plotly_chart(fig_sal, use_container_width=True)

# --- RATIO FINAL ---
st.divider()
st.header("🏁 Veredicto Final")
muj_t = df[(df['sex'] == 2) & (df['eda'] >= 25) & (df['eda'] <= 40)]['fac_tri'].sum()
hom_e = df[(df['sex'] == 1) & (df['ing7c'].isin([4, 5])) & (df['clase2'] == 1)]['fac_tri'].sum()
ratio = muj_t / hom_e

st.metric("Ratio de Competencia", f"{ratio:.1f} a 1", "Mujeres por cada Hombre de Élite")
fig_r = go.Figure(go.Bar(x=[1, ratio], y=['Hombres Élite', 'Mujeres Target'], orientation='h', marker_color=['#3498db', '#e74c3c']))
st.plotly_chart(fig_r, use_container_width=True)