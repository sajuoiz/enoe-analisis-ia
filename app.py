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
#-------------------DISTRIBUCION SALARIAL POR SEXO (TECHO DE CRISTAL)---------------------
st.divider()
st.header("⚖️ La Brecha de Ingresos: Hombres vs. Mujeres")

# 1. PRIMERO: Definir la población ocupada (Esto evita el NameError)
poblacion_ocupada = df[df['clase2'] == 1].copy()

# 2. SEGUNDO: Calcular los promedios reales para las etiquetas
stats_ingresos = poblacion_ocupada[poblacion_ocupada['ingocup'] < 999999].groupby('ing7c', observed=False).agg({
    'fac_tri': 'sum',
    'ingocup': 'mean'
}).reset_index()

# Diccionario base para los nombres
nombres_base = {
    1: 'Hasta 1 SM', 2: '1 a 2 SM', 3: '2 a 3 SM', 
    4: '3 a 5 SM', 5: 'Más de 5 SM', 6: 'Sin ingresos', 7: 'No especificado'
}

# Crear etiquetas dinámicas con los montos reales
etiquetas_dinamicas = {}
for _, fila in stats_ingresos.iterrows():
    rango_id = fila['ing7c']
    nombre = nombres_base.get(rango_id, "Otro")
    monto = fila['ingocup']
    
    if monto > 0:
        etiquetas_dinamicas[rango_id] = f"{nombre}<br>(Avg: ${monto/1000:.1f}k)"
    else:
        etiquetas_dinamicas[rango_id] = nombre

# 3. TERCERO: Agrupar por sexo para la gráfica
dist_genero = poblacion_ocupada.groupby(['sex', 'ing7c'], observed=False)['fac_tri'].sum().reset_index()

# Aplicar las etiquetas y nombres de sexo
dist_genero['Rango'] = dist_genero['ing7c'].map(etiquetas_dinamicas)
dist_genero['Sexo'] = dist_genero['sex'].map({1: 'Hombres', 2: 'Mujeres'})

# 4. CUARTO: Crear la gráfica de Plotly
fig_gen = go.Figure()

for genero, color in zip(['Hombres', 'Mujeres'], ['#3498db', '#e74c3c']):
    datos_subset = dist_genero[dist_genero['Sexo'] == genero]
    fig_gen.add_trace(go.Bar(
        x=datos_subset['Rango'],
        y=datos_subset['fac_tri'],
        name=genero,
        marker_color=color,
        text=[f"{v/1e6:.1f}M" for v in datos_subset['fac_tri']],
        textposition='auto',
        hovertemplate="<b>%{x}</b><br>Personas: %{y:,.0f}<extra></extra>"
    ))

fig_gen.update_layout(
    title='Brecha Salarial con Ingresos Promedio Reales (ENOE 2024)',
    xaxis_title='Rango de SM e Ingreso Promedio Detectado',
    yaxis_title='Millones de Personas',
    barmode='group',
    template="plotly_white",
    height=600,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig_gen, use_container_width=True)

#-------------FORMALES VS INFORMALES---------------------
#----------------🛡️ ESTABILIDAD VS RIESGO: FORMALIDAD EN LA ÉLITE ----------------
#----------------🛡️ ESTABILIDAD VS RIESGO: FORMALIDAD EN LA ÉLITE ----------------
st.divider()
st.header("🛡️ Estabilidad vs. Riesgo: Formalidad en la Élite")

# 1. Filtrar hombres de la élite
hombres_elite = df[
    (df['sex'] == 1) & 
    (df['ing7c'].isin([4, 5])) &
    (df['clase2'] == 1)
].copy()

# 2. Procesar formalidad
resumen_social = hombres_elite[hombres_elite['seg_soc'].isin([1, 2])].copy()
dist_formalidad = resumen_social.groupby('seg_soc', observed=False)['fac_tri'].sum().reset_index()

dist_formalidad['Estatus'] = dist_formalidad['seg_soc'].map({
    1: 'Formal (Con Seg. Social)', 
    2: 'Informal (Sin Seg. Social)'
})

# 3. Métricas rápidas
total_v = dist_formalidad['fac_tri'].sum()
formal_v = dist_formalidad[dist_formalidad['seg_soc'] == 1]['fac_tri'].sum()
informal_v = dist_formalidad[dist_formalidad['seg_soc'] == 2]['fac_tri'].sum()

col_f1, col_f2 = st.columns([2, 1])

with col_f1:
    fig_formal = go.Figure(go.Bar(
        x=dist_formalidad['Estatus'],
        y=dist_formalidad['fac_tri'],
        marker_color=['#2ecc71', '#e74c3c'], 
        text=[f"{(v/total_v)*100:.1f}%" for v in dist_formalidad['fac_tri']],
        textposition='auto',
    ))

    fig_formal.update_layout(
        title='Seguridad Social en Hombres de Altos Ingresos',
        yaxis_title='Personas',
        template='plotly_white',
        height=400
    )
    st.plotly_chart(fig_formal, use_container_width=True)

with col_f2:
    st.write("### Resumen")
    # AQUÍ ESTABA EL ERROR, YA CORREGIDO:
    st.metric("Total Élite Masculina", f"{total_v/1e6:.2f}M")
    st.metric("Tasa de Formalidad", f"{(formal_v/total_v)*100:.1f}%")
    
    st.info(f"Existen **{informal_v:,.0f}** hombres en este rango sin red de protección social.")


#---------- HOMBRES MAYORES A 3SM SEGMENTADOS POR EDAD ----------
#-------------------- 🏆 LA ÉLITE MASCULINA POR EDAD -----------------------
st.divider()
st.header("🏆 La Élite Masculina: ¿A qué edad alcanzan el éxito?")
st.markdown("""
Analizamos la distribución por edad de los hombres que ganan **más de 3 Salarios Mínimos**. 
Este gráfico muestra cuándo ocurre el pico de ingresos en la vida del hombre mexicano.
""")

# 1. Filtrar hombres de la élite
hombres_elite_edad = df[
    (df['sex'] == 1) & 
    (df['ing7c'].isin([4, 5])) &
    (df['clase2'] == 1)
].copy()

# 2. Definir rangos de edad exactos
bins = [20, 30, 40, 50, 100]
labels = ['20-29 años', '30-39 años', '40-49 años', '50+ años']
hombres_elite_edad['rango_edad'] = pd.cut(hombres_elite_edad['eda'], bins=bins, labels=labels, right=False)

# 3. Agrupar por población expandida
dist_edad_pnea = hombres_elite_edad.groupby('rango_edad', observed=False)['fac_tri'].sum().reset_index()

# 4. Crear gráfica con Plotly
fig_edad = go.Figure(go.Bar(
    x=dist_edad_pnea['rango_edad'],
    y=dist_edad_pnea['fac_tri'],
    # Usamos un degradado de azul para denotar madurez profesional
    marker_color=['#AED6F1', '#5DADE2', '#2E86C1', '#1B4F72'], 
    text=[f"{v/1e3:.0f}k" if v < 1e6 else f"{v/1e6:.2f}M" for v in dist_edad_pnea['fac_tri']],
    textposition='auto',
))

fig_edad.update_layout(
    title='Hombres con Ingresos > 3 SM por Rango de Edad',
    xaxis_title="Rango de Edad",
    yaxis_title="Estimación Nacional (Personas)",
    template='plotly_white',
    height=500
)

st.plotly_chart(fig_edad, use_container_width=True)

# 5. Análisis dinámico para el video
max_rango = dist_edad_pnea.loc[dist_edad_pnea['fac_tri'].idxmax(), 'rango_edad']

# --- RATIO FINAL ---
st.divider()
st.header("🏁 Veredicto Final")
muj_t = df[(df['sex'] == 2) & (df['eda'] >= 25) & (df['eda'] <= 40)]['fac_tri'].sum()
hom_e = df[(df['sex'] == 1) & (df['ing7c'].isin([4, 5])) & (df['clase2'] == 1)]['fac_tri'].sum()
ratio = muj_t / hom_e

st.metric("Ratio de Competencia", f"{ratio:.1f} a 1", "Mujeres por cada Hombre VIABLE ECONÓMICAMENTE")
fig_r = go.Figure(go.Bar(x=[1, ratio], y=['Hombres Élite', 'Mujeres Target'], orientation='h', marker_color=['#3498db', '#e74c3c']))
st.plotly_chart(fig_r, use_container_width=True)