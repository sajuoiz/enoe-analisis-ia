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
from sklearn.metrics import classification_report, accuracy_score

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
st.markdown("Estudio basado en microdatos de la ENOE (T1-2024-SDEMT)")

# --- GRÁFICA 1: DINÁMICA LABORAL (ACTUALIZADA CON TOTALES) ---
st.header("Dinámica Laboral por Edad y Género (Población Empleada)")

# 1. Filtro y copia
df_ocupados = df[(df['eda'] >= 15) & (df['eda'] < 99) & (df['clase2'] == 1)].copy()

# 2. Agrupación y suma ponderada
resumen = df_ocupados.groupby(['eda', 'sex'])['fac_tri'].sum().unstack()

# 3. Cálculo de Totales Nacionales (Sumando los factores de expansión)
total_h_emp = resumen[1].sum() if 1 in resumen.columns else 0
total_m_emp = resumen[2].sum() if 2 in resumen.columns else 0
suma_total_emp = total_h_emp + total_m_emp

# 4. Viñetas informativas superiores (Opcional, ayuda mucho a la lectura)
c1, c2, c3 = st.columns(3)
c1.metric("Total Empleados", f"{int(suma_total_emp):,}")
c2.metric("Hombres 👨", f"{int(total_h_emp):,}")
c3.metric("Mujeres 👩", f"{int(total_m_emp):,}")

fig_din = go.Figure()

# 5. Agregar trazos con totales en el nombre (Leyenda)
if 1 in resumen.columns:
    fig_din.add_trace(go.Scatter(
        x=resumen.index, 
        y=resumen[1], 
        name=f'Hombres (Total: {int(total_h_emp):,})', 
        line=dict(color='#1f77b4', width=3),
        fill='tozeroy', # Relleno opcional para ver volumen
        opacity=0.1
    ))

if 2 in resumen.columns:
    fig_din.add_trace(go.Scatter(
        x=resumen.index, 
        y=resumen[2], 
        name=f'Mujeres (Total: {int(total_m_emp):,})', 
        line=dict(color='#d62728', width=3),
        fill='tozeroy',
        opacity=0.1
    ))

fig_din.update_layout(
    xaxis_title="Edad (Años)", 
    yaxis_title="Cantidad de Personas (Expandido)", 
    template="plotly_white", 
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig_din, use_container_width=True)

st.info("""
**Dato:** Las mujeres siempre se encuentran subcontratadas sin importar su edad en comparaciòn con los hombres.
        Lo cual rompe la idea de que las mujeres solo enfrentan dificultades laborales en edades de maternidad.
""")

# --- SECCIÓN: DEMOGRAFÍA TOTAL (Nacional) ---
st.divider()
st.subheader("🌐 Universo de Análisis: Población 15+")

@st.cache_data
def calcular_poblacion_total(_df):
    # Filtrar población de 15 años y más (excluyendo no especificados 99)
    df_15 = _df[(_df['eda'] >= 15) & (_df['eda'] < 99)].copy()
    
    # Calcular totales ponderados por factor de expansión
    pob_sexo = df_15.groupby('sex')['fac_tri'].sum()
    
    hombres = pob_sexo.get(1, 0)
    mujeres = pob_sexo.get(2, 0)
    total = hombres + mujeres
    
    return hombres, mujeres, total

# Obtener los valores
h_total, m_total, nacional_total = calcular_poblacion_total(df)

# Mostrar en columnas con estilo de Dashboard
c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Población Total (15+)", f"{int(nacional_total):,}")
    st.caption("Universo total de ciudadanos en edad laboral.")

with c2:
    st.metric("Hombres 👨", f"{int(h_total):,}")
    st.write(f"({(h_total/nacional_total*100):.1f}% del total)")

with c3:
    st.metric("Mujeres 👩", f"{int(m_total):,}")
    st.write(f"({(m_total/nacional_total*100):.1f}% del total)")

# Mensaje para tu video de TikTok
st.info(f"""
💡 **Dato clave para el cierre:** Este análisis representa a un universo de **{int(nacional_total):,}** mexicanos. 
Para que una opinión personal sea estadísticamente relevante frente a este número, debería estar respaldada por una muestra técnica, no por anécdotas.
""")

# --- GRÁFICA 2: COEFICIENTES ---
st.divider()
st.header("🧠 Modelo de ML: Regresión Logística")

with st.expander("🔍 ¿Qué es este modelo y cómo funciona?"):
    st.markdown("""
    ### ¿Qué es la Regresión Logística?
    A diferencia de una regresión lineal (que predice números como el precio de una casa), la **Regresión Logística** es un algoritmo de **clasificación**. Se utiliza para predecir la probabilidad de que una observación pertenezca a una de dos categorías (en este caso: *¿Debería estar empleado/a o no?*).

    ### ¿Qué mide?
    El modelo mide la relación entre una variable dependiente binaria (Empleabilidad) y múltiples variables independientes (Edad, Educación, Horas trabajadas, etc.).

    ### ¿Cómo lo mide?
    1. **Probabilidad Logística:** El modelo calcula una puntuación basada en los datos de entrada y la pasa por una función llamada **Sigmoide**.
    2. **Función Sigmoide:** Esta función "aplasta" cualquier número para que el resultado siempre esté entre **0 y 1**, lo que interpretamos como una probabilidad.
    3. **Límite de Decisión:** Por defecto, si la probabilidad es mayor a **0.5 (50%)**, el modelo clasifica a la persona como parte de la "Poblacion Empleada".

    ### ¿Por qué es pertinente en este análisis?
    La Regresión Logística no solo nos da una predicción, sino que nos permite calcular los Efectos Marginales Promedio. Esto nos dice, por ejemplo: *"Por cada año adicional de estudio, las probabilidades de pertenecer a la Población Empleada aumentan un X%"*.

    > **Nota Estadística:** Es el modelo ideal cuando buscamos entender el impacto específico de factores sociales y demográficos en una oportunidad de vida.
    """)
with st.expander("Ver detalles técnicos del modelo"):
    st.write("✅ Regresión Logística entrenada con factor de estratificación.")

coef_df = pd.DataFrame({'Variable': features, 'Coeficiente': modelo_pnea.coef_[0]}).sort_values('Coeficiente')
fig_coef = go.Figure(go.Bar(
    x=coef_df['Coeficiente'], y=coef_df['Variable'], orientation='h',
    marker_color=['#d62728' if x < 0 else '#2ca02c' for x in coef_df['Coeficiente']]
))
fig_coef.update_layout(title="Impacto Relativo de las Variables", template="plotly_white", yaxis_title="Variables", xaxis_title="Dirección y Peso de cada variable en el modelo")
st.plotly_chart(fig_coef, use_container_width=True)

st.info("""
**Dato:** Esta gráfca solo refleja la dirección del impacto de cada variable en la probabilidad de empleo, no la probabilidad real.
        Permite observar que variables tienen mayor impacto positivo o negativo en la probabilidad de empleo.
""")

# --- GRÁFICA 3: SIMULADOR ---
st.divider()
st.header("📉 Simulador: El 'Impuesto' a la Maternidad")
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1: edad_i = st.number_input("Edad", 15, 80, 30)
with col_s2: esc_i = st.number_input("Escolaridad (Años cursados)", 0, 24, 12)
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
fig_sim.update_layout(title="Probabilidad vs Número de Hijos", yaxis_range=[0,1.1], template="plotly_white", yaxis_title="Probabilidad de Empleo", xaxis_title="Número de Hijos")
st.plotly_chart(fig_sim, use_container_width=True)

st.info("""
**Dato:** Hay que tomar en cuenta que aquí el INEGI por metodología de la encuesta en cuanto a natalidad solamente pone o anota el número de hijos
        para la mujer (madre), la encuesta como tal no tiene información sobre el número de hijos de los hombres, de ahí que la gráfica de hombres
        permanezca horizontal. Aún así la grafica de hombres como punto de referencia permite ver como cae la probabilidad de empleo en mujeres 
        conforme aumenta el número de hijos.
""")

# --- GRÁFICA 4: EFECTOS MARGINALES ---
st.divider()
st.header("🎯 Efectos Marginales (Impacto en %)")
st.subheader("🔬 Interpretación del Impacto de las Variables")

with st.expander("🔍 ¿Qué son los Efectos Marginales y en qué se diferencian de los Odds Ratio?"):
    st.markdown("""
    Al analizar un modelo de regresión logística, la mejor forma de hacerlo es con los EMP: **Efectos Marginales Promedio (AME)**.
                
    #### Efectos Marginales Promedio (AME)
    Es una medida de **probabilidad absoluta** (puntos porcentuales). Nos dice cuánto cambia la probabilidad real (de 0 a 100%) cuando una variable aumenta en una unidad.
    * **Ejemplo:** Si el Efecto Marginal de 'Educación' es **0.006**, significa que, en promedio, un año más de estudio aumenta la probabilidad de ser Empleado en **0.6 puntos porcentuales**.
    * **Ventaja:** Es mucho más intuitivo. Nos permite decir: *"Si una persona estudia la universidad, su probabilidad de ganar más sube un X%"*.
   
    """)
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
#------------------- 📈 VALIDACIÓN DEL MODELO ---------------------
st.divider()
st.header("📈 Validación del Modelo: ¿Qué tan confiable es?")


with st.expander("🔍 Interpretación de la Matriz de Confusión"):
    st.markdown("""
    ### ¿Qué es la Matriz de Confusión?
    Es la herramienta definitiva para evaluar un modelo de clasificación. No solo nos dice si el modelo acertó, sino que nos revela **en qué dirección se equivoca**.

    ### Los 4 Cuadrantes de la Verdad:
    
    1. **Verdaderos Positivos (Top-Right):** Personas que el modelo predijo correctamente como parte de la "Población Empleada".
    2. **Verdaderos Negativos (Bottom-Left):** Personas que el modelo identificó correctamente como "no empleadas".
    3. **Falsos Positivos (Error Tipo I):** El modelo predijo que alguien tendría empleo, pero en la realidad no es así.
    4. **Falsos Negativos (Error Tipo II):** El modelo dijo que alguien no sería parte de la Población Empleada, cuando en realidad sí lo es.

    ### ¿Qué significa el Accuracy?
    La **Precisión Global** es el porcentaje total de aciertos (tanto positivos como negativos) sobre el total de casos. 
    
    > **Dato para el análisis:** En problemas de "élite económica", donde los casos de éxito son pocos, una matriz de confusión equilibrada es más importante que un accuracy alto, ya que nos asegura que el modelo no está simplemente "adivinando" que nadie tendrá éxito.
    """)

st.subheader("📊 Métricas de Rendimiento del Modelo")

# 1. Calculamos el reporte de clasificación como diccionario
# Asegúrate de haber definido y_test y y_pred anteriormente
reporte = classification_report(y_test, y_pred, output_dict=True)

# 2. Mostramos las métricas en columnas para un look profesional
c1, c2, c3, c4 = st.columns(4)

c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.1%}")
# Usamos '1' porque es la etiqueta de nuestra clase objetivo (élite)
c2.metric("Precision", f"{reporte['1']['precision']:.1%}")
c3.metric("Recall", f"{reporte['1']['recall']:.1%}")
c4.metric("F1-Score", f"{reporte['1']['f1-score']:.1%}")

# 3. Explicación técnica para el usuario
with st.expander("🔍 ¿Qué significan estos números?"):
    st.markdown(f"""
    Para evaluar la confiabilidad de la predicción, analizamos cuatro dimensiones:

    * **Accuracy (Exactitud):** Es el porcentaje total de aciertos. Aunque es alto, en poblaciones desbalanceadas puede ser engañoso.
    * **Precision (Calidad):** Responde a: *De todos los que el modelo predijo como Empleados, ¿cuántos realmente lo son?* (Evita falsas alarmas).
    * **Recall (Alcance):** Responde a: *De todos los que son Empleados en la vida real, ¿a cuántos logró detectar el modelo?* (Evita ignorar casos de empeabilidad).
    * **F1-Score (Equilibrio):** Es la métrica más robusta. Combina Precision y Recall en un solo número. Si este número es alto, el modelo es confiable para predecir la clase "Empleados".

    > **Interpretación:** Un F1-Score por encima del **0.70** se considera un modelo sólido para análisis sociales con microdatos de la ENOE.
    """)
reporte = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred)
fig_cm = ff.create_annotated_heatmap(cm[::-1], x=['Pred: No', 'Pred: Sí'], y=['Real: Sí', 'Real: No'], colorscale='Blues')
st.plotly_chart(fig_cm, use_container_width=True)

# --- SECCIÓN: FALSOS POSITIVOS ---
# --- SECCIÓN: FALSOS POSITIVOS (CON EDAD Y SIN HORAS) ---
st.divider()
st.header("🔍 Análisis de Falsos Positivos (Personas que el modelo predice deberían estar empleadas, pero no lo están en la realidad)")

df_res = X_test.copy()
df_res['real'], df_res['pred'] = y_test, y_pred

# Identificar índices de Falsos Positivos
indices_fp = df_res[(df_res['real'] == 0) & (df_res['pred'] == 1)].index
fp_completos = df_pnea.loc[indices_fp].copy()

# Separar por género
fp_h = fp_completos[fp_completos['es_mujer'] == 0]
fp_m = fp_completos[fp_completos['es_mujer'] == 1]

col_graf, col_tablas = st.columns([1, 1.2])

with col_graf:
    st.subheader("Distribución por Género")
    fig_fp = go.Figure(data=[go.Pie(
        labels=['Hombres', 'Mujeres'], 
        values=[len(fp_h), len(fp_m)], 
        hole=.4,
        marker_colors=['#3498db', '#e74c3c']
    )])
    fig_fp.update_layout(showlegend=True, height=400)
    st.plotly_chart(fig_fp, use_container_width=True)

with col_tablas:
    st.subheader("Características Promedio")
    
    # --- Lógica para calcular cantidad y porcentaje con pareja ---
    def info_pareja(df_filtrado):
        if not df_filtrado.empty:
            # Si el nombre es 'tiene_pareja' o 'edo_civil', lo buscamos
            col = 'tiene_pareja' if 'tiene_pareja' in df_filtrado.columns else 'edo_civil'
            if col in df_filtrado.columns:
                cantidad = int(df_filtrado[col].sum())
                porcentaje = (df_filtrado[col].mean() * 100)
                return f"{cantidad} ({porcentaje:.1f}%)"
        return "0 (0.0%)"

    # Tabla para Hombres
    st.write("**👨 Hombres (Falsos Positivos)**")
    st.table(pd.DataFrame({
        "Métrica": ["Cantidad Total", "Edad Promedio", "Escolaridad", "Hijos Prom.", "Con Pareja (Cant. y %)"],
        "Valor": [
            f"{len(fp_h)}", 
            f"{fp_h['eda'].mean():.1f} años" if not fp_h.empty else "0.0",
            f"{fp_h['anios_esc'].mean():.1f} años" if not fp_h.empty else "0.0", 
            f"{fp_h['n_hij'].mean():.1f}" if not fp_h.empty else "0.0",
            info_pareja(fp_h)
        ]
    }))

    # Tabla para Mujeres
    st.write("**👩 Mujeres (Falsos Positivos)**")
    st.table(pd.DataFrame({
        "Métrica": ["Cantidad Total", "Edad Promedio", "Escolaridad", "Hijos Prom.", "Con Pareja (Cant. y %)"],
        "Valor": [
            f"{len(fp_m)}", 
            f"{fp_m['eda'].mean():.1f} años" if not fp_m.empty else "0.0",
            f"{fp_m['anios_esc'].mean():.1f} años" if not fp_m.empty else "0.0", 
            f"{fp_m['n_hij'].mean():.1f}" if not fp_m.empty else "0.0",
            info_pareja(fp_m)
        ]
    }))

st.info("""
**Dato:** Ambos tienen caracteristicas muy parecidas, ya no serían rentables para el mercado laboral dado su nivel de escolaridad y edad.
        Tambièn observamos que es más probale que las mujeres tengan pareja, como una posibilidad para buscar apoyo ente la imposibilidad de acerder
        al mercado laboral. Lo cual para los hombres no es una opcion, el porcentaje de ellos con pareja es muchísimo menor.
        (no hay que pasar por alto la gran posibilidad de que las mujeres quedan en una situación de vulnerabilidad.)
        ¿Qué se debería hacer con ellos, abandonarles?.
""")

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


# --- CÁLCULO DE DENSIDAD POR RANGOS --------------------------------------
def generar_grafico_densidad(_df):
    # 1. Crear los rangos (bins)
    bins = [15, 25, 35, 45, 55, 65, 75, 85, 95]
    labels = ['15-25', '25-35', '35-45', '45-55', '55-65', '65-75', '75-85', '85-95']
    
    df_bins = _df[(_df['eda'] >= 15) & (_df['eda'] < 95)].copy()
    df_bins['rango_edad'] = pd.cut(df_bins['eda'], bins=bins, labels=labels, right=False)
    
    # 2. Sumar población expandida por rango y sexo
    densidad = df_bins.groupby(['rango_edad', 'sex'])['fac_tri'].sum().reset_index()
    densidad['sex'] = densidad['sex'].map({1: 'Hombres', 2: 'Mujeres'})
    
    # --- OPCIÓN A: HEATMAP (ESTILO DENSIDAD) ---
    fig = px.density_heatmap(
        densidad, 
        x="rango_edad", 
        y="sex", 
        z="fac_tri",
        color_continuous_scale="Viridis",
        labels={'fac_tri': 'Población', 'rango_edad': 'Rango de Edad', 'sex': 'Género'},
        title="Concentración de Población en México"
    )
    
    return fig

# En Streamlit:
st.header("🍯 Densidad Poblacional por Rangos")
fig_densidad = generar_grafico_densidad(df)
st.plotly_chart(fig_densidad, use_container_width=True)

# --- SECCIÓN: EL PARADOJA DE LA PREPARACIÓN (15-25 años) ---
# --- SECCIÓN: EL SEGMENTO PROFESIONAL JOVEN (23-28 años) ---
st.divider()
st.subheader("🎓 Título Universitario: El Segmento de 23 a 28 años")

@st.cache_data
def analizar_profesionales_jovenes(_df):
    # 1. Filtramos el rango específico de 23 a 28 años
    df_prof = _df[(_df['eda'] >= 23) & (_df['eda'] <= 28)].copy()
    
    # 2. Definimos nivel universitario (16+ años de escolaridad)
    df_prof['es_universitario'] = (df_prof['anios_esc'] >= 16).astype(int)
    
    def obtener_metricas(sub_df):
        if sub_df.empty: return 0, 0
        pob_total = sub_df['fac_tri'].sum()
        pob_univ = (sub_df['es_universitario'] * sub_df['fac_tri']).sum()
        porcentaje = (pob_univ / pob_total) * 100
        return pob_univ, porcentaje

    # Cálculos por sexo
    h_univ, h_pct = obtener_metricas(df_prof[df_prof['sex'] == 1])
    m_univ, m_pct = obtener_metricas(df_prof[df_prof['sex'] == 2])
    
    return h_univ, h_pct, m_univ, m_pct, (h_univ + m_univ)

h_u, h_p, m_u, m_p, total_u = analizar_profesionales_jovenes(df)

# Visualización en Streamlit
c1, c2 = st.columns(2)

with c1:
    st.metric("Mujeres (23-28) con Universidad", f"{m_p:.1f}%")
    st.write(f"Cifra expandida: **{int(m_u):,}** mujeres")

with c2:
    st.metric("Hombres (23-28) con Universidad", f"{h_p:.1f}%")
    st.write(f"Cifra expandida: **{int(h_u):,}** hombres")

st.markdown(f"""
> **Conclusión del Segmento:** En este rango de edad, hay un total de **{int(total_u):,}** personas con formación universitaria. 
> Si tu gráfica de ingresos muestra que este grupo sigue ganando cerca del salario mínimo, estamos ante la prueba estadística de que **el título ya no es garantía de movilidad social inmediata** en México.
""")

#------------------- 📦 DIAGRAMA DE CAJAS: DISPERSIÓN SALARIAL SEPARADA ---------------------
st.divider()
st.header("📦 Dispersión Salarial: Grupos Oficiales del INEGI")
st.markdown("""
Distribución interna de los ingresos basada en las categorías estandarizadas de la variable **ing7c**.
Esta visualización permite comparar la estructura salarial base frente a la élite.
""")

# 1. Asegurar variable y cálculo de SM dinámico
poblacion_ocupada = df[df['clase2'] == 1].copy()
salario_minimo_base = poblacion_ocupada[poblacion_ocupada['ing7c'] == 1]['ingocup'].max()
if not salario_minimo_base or salario_minimo_base == 0:
    salario_minimo_base = 7468

# 2. Creación de columnas
col_caja1, col_caja2 = st.columns([3, 2])

with col_caja1:
    st.subheader("Estructura Salarial (Rangos 1-4)")
    # Filtramos grupos 1, 2, 3 y 4 (Metodología completa)
    df_base = poblacion_ocupada[
        (poblacion_ocupada['ing7c'].isin([1, 2, 3, 4])) & 
        (poblacion_ocupada['ingocup'] > 0)
    ].copy()
    
    mapeo_inegi = {
        1: f'Hasta 1 SM\n(<${salario_minimo_base:,.0f})',
        2: f'1-2 SM\n(${salario_minimo_base:,.0f}-${salario_minimo_base*2:,.0f})',
        3: f'2-3 SM\n(${salario_minimo_base*2:,.0f}-${salario_minimo_base*3:,.0f})',
        4: f'3-5 SM\n(${salario_minimo_base*3:,.0f}-${salario_minimo_base*5:,.0f})'
    }
    
    df_base['Rango'] = df_base['ing7c'].map(mapeo_inegi)
    df_base['Sexo'] = df_base['sex'].map({1: 'Hombres', 2: 'Mujeres'})

    fig_base = go.Figure()
    for gen, col in zip(['Hombres', 'Mujeres'], ['#3498db', '#e74c3c']):
        sub = df_base[df_base['Sexo'] == gen]
        fig_base.add_trace(go.Box(y=sub['ingocup'], x=sub['Rango'], name=gen, marker_color=col))
    
    fig_base.update_layout(title="Distribución: Salarios Base y Medios", yaxis_title="MXN", boxmode='group', template='plotly_white')
    st.plotly_chart(fig_base, use_container_width=True)

with col_caja2:
    st.subheader("La Élite (Rango 5)")
    # Solo rango 5
    df_elite = poblacion_ocupada[
        (poblacion_ocupada['ing7c'] == 5) & 
        (poblacion_ocupada['ingocup'] > 0) & 
        (poblacion_ocupada['ingocup'] < 250000)
    ].copy()
    
    etiqueta_elite = f'5+ SM\n(> ${salario_minimo_base*5:,.0f})'
    df_elite['Sexo'] = df_elite['sex'].map({1: 'Hombres', 2: 'Mujeres'})

    fig_elite = go.Figure()
    for gen, col in zip(['Hombres', 'Mujeres'], ['#3498db', '#e74c3c']):
        sub = df_elite[df_elite['Sexo'] == gen]
        fig_elite.add_trace(go.Box(y=sub['ingocup'], x=[etiqueta_elite] * len(sub), name=gen, marker_color=col))
    
    fig_elite.update_layout(title="Distribución: Salarios Altos", yaxis_title="MXN", boxmode='group', template='plotly_white')
    st.plotly_chart(fig_elite, use_container_width=True)

# 3. Nota técnica
st.info(f"""
📌 **Nota Metodológica:** Los rangos siguen la clasificación de la variable `ing7c` del INEGI. 
El **Salario Mínimo de Referencia** detectado es de **${salario_minimo_base:,.2f}**. 
Observa cómo en el rango 4 (3-5 SM) la brecha de género comienza a ensancharse, culminando en la 
extrema volatilidad del rango 5.
""")

#-------------FORMALES VS INFORMALES---------------------
#----------------🛡️ ESTABILIDAD VS RIESGO: FORMALIDAD EN LA ÉLITE ----------------
#----------------🛡️ ESTABILIDAD VS RIESGO: FORMALIDAD EN LA ÉLITE ----------------
st.divider()
st.header("🛡️ Estabilidad vs. Riesgo: Formalidad en la cima")

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
st.header("🏆 La Élite Masculina: ¿A qué edad alcanzan la cima?")
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
fig_r = go.Figure(go.Bar(x=[1, ratio], y=['Hombres Estables', 'Mujeres Target'], orientation='h', marker_color=['#3498db', '#e74c3c']))
st.plotly_chart(fig_r, use_container_width=True)