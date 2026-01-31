import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import streamlit as st

# Configuración de página
st.set_page_config(page_title="Samuel Juárez Ortiz | Data Scientist", layout="wide")

# Inyectar el CSS que diseñamos
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_status=True)

# Crear el "Hero Section" con HTML dentro de Streamlit
st.markdown("""
    <section id="hero">
        <div class="hero-content">
            <h1>Samuel Juárez Ortiz</h1>
            <p class="subtitle">Científico de Datos | Especialista en Microdatos Económicos</p>
            <p class="description">
                Transformando datos complejos en decisiones estratégicas. Mi enfoque combina econometría y Machine Learning.
            </p>
        </div>
    </section>
""", unsafe_allow_status=True)

# Continúa con el resto de tu código de gráficas...

st.set_page_config(page_title="Análisis ENOE", layout="wide")

st.title("📊 Análisis de la Viabilidad Económica Por Género")
st.markdown("Estudio basado en microdatos de la ENOE (Q1-2024) SDEMT-1T-2024")

# Función para cargar datos (con caché para que sea rápido)
@st.cache_data
def load_data():
    # Asegúrate de que el CSV esté en la misma carpeta
    df = pd.read_csv('enoe_light.csv', encoding="latin1", low_memory=False)
    # Limpieza básica inmediata
    cols = ['eda', 'sex', 'clase2', 'fac_tri']
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

with st.spinner('Cargando base de datos pesada...'):
    df = load_data()

st.success("✅ Base de datos cargada correctamente")

# --- LÓGICA DE TU GRÁFICA ---
st.header("Dinámica Laboral por Edad y Género")
st.write("Sin importar la edad, las mujeres siempre se encuentran subcontratadas comparadas con los hombres.")

df_ocupados = df[(df['eda'] >= 15) & (df['eda'] < 99) & (df['clase2'] == 1)].copy()
resumen = df_ocupados.groupby(['eda', 'sex'])['fac_tri'].sum().unstack()

fig, ax = plt.subplots(figsize=(10, 5))
if 1 in resumen.columns:
    ax.plot(resumen.index, resumen[1], label='Hombres', color='#1f77b4', linewidth=2)
if 2 in resumen.columns:
    ax.plot(resumen.index, resumen[2], label='Mujeres', color='#d62728', linewidth=2)

ax.set_ylabel("Millones de personas")
ax.set_xlabel("Edad")
ax.legend()
# Formatear eje Y
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))

# EN STREAMLIT SE USA ESTO EN LUGAR DE PLT.SHOW()
st.pyplot(fig)


# --- MODELO PREDICTIVO ---------------------------------------------
st.divider()
st.header("🧠 Modelo Predictivo: ¿Quién participa en la economía?")
st.markdown("""
En esta sección utilizamos **Machine Learning** para entender qué factores determinan que una persona pertenezca a la 
Población Económicamente Activa (PEA).
""")

st.markdown("""
    ### 🤖 ¿Qué está haciendo la IA aquí?
    Imagina que este modelo es un **"Calculador de Probabilidades"**. En lugar de adivinar, toma 
    características como tu edad, educación y familia para calcular qué tan probable es (del 0% al 100%) 
    que una persona participe en el mercado laboral. 
    
    Es como una balanza: pone "pesos" a cada factor para ver hacia dónde se inclina la balanza del empleo.
    """)
    
st.info("💡 **Dato clave:** La Regresión Logística no solo dice 'Sí' o 'No', nos da la certeza detrás de esa respuesta.")

# Usamos un expander para no saturar la pantalla con procesos técnicos
with st.expander("Ver detalles del modelo (Regresión Logística)"):
    # 1. Preparación de datos (usando el DF cargado previamente)
    columnas_pnea = ['eda', 'sex', 'anios_esc', 'n_hij', 'e_con', 't_loc_tri', 'clase1']
    
    # Filtramos y limpiamos
    df_pnea = df[columnas_pnea].copy()
    for col in df_pnea.columns:
        df_pnea[col] = pd.to_numeric(df_pnea[col], errors='coerce')
    
    df_pnea = df_pnea[df_pnea['eda'] >= 15].dropna(subset=['clase1'])

    # 2. Ingeniería de variables
    df_pnea['target_participacion'] = (df_pnea['clase1'] == 1).astype(int)
    df_pnea['es_mujer'] = (df_pnea['sex'] == 2).astype(int)
    df_pnea['mujer_con_hijos'] = df_pnea['es_mujer'] * df_pnea['n_hij'].fillna(0)
    df_pnea['eda_2'] = df_pnea['eda'] ** 2
    df_pnea['es_urbano'] = df_pnea['t_loc_tri'].isin([1, 2]).astype(int)
    df_pnea['tiene_pareja'] = df_pnea['e_con'].isin([1, 5]).astype(int)
    df_pnea['anios_esc'] = df_pnea['anios_esc'].fillna(df_pnea['anios_esc'].median())

    # 3. Entrenamiento
    features = ['eda', 'eda_2', 'es_mujer', 'mujer_con_hijos', 'anios_esc', 'es_urbano', 'tiene_pareja']
    X = df_pnea[features]
    y = df_pnea['target_participacion']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo_pnea = LogisticRegression(max_iter=1000)
    modelo_pnea.fit(X_train, y_train)

    st.write("✅ Modelo entrenado con éxito.")

# --- VISUALIZACIÓN DE IMPACTO ---
st.subheader("Impacto de las variables en la participación laboral")

coef_pnea = pd.DataFrame({'Variable': features, 'Coeficiente': modelo_pnea.coef_[0]})
coef_pnea = coef_pnea.sort_values(by='Coeficiente')

# Graficamos los coeficientes
fig_ml, ax_ml = plt.subplots(figsize=(10, 6))
colors = ['#d62728' if x < 0 else '#2ca02c' for x in coef_pnea['Coeficiente']]
ax_ml.barh(coef_pnea['Variable'], coef_pnea['Coeficiente'], color=colors)
ax_ml.set_title("Coeficientes del Modelo (Impacto relativo)")
ax_ml.set_xlabel("Efecto Negativo <--- | ---> Efecto Positivo")

st.pyplot(fig_ml)

# Interpretación dinámica
col1, col2 = st.columns(2)

with col1:
    st.info("💡 **Aumentan la probabilidad:** Los años de escolaridad y vivir en zonas urbanas suelen ser los motores principales.")

with col2:
    st.warning("⚠️ **Disminuyen la probabilidad:** Aquí notarás que ser mujer y, sobre todo, 'mujer con hijos' tiene un impacto negativo severo en la probabilidad de trabajar.")

#-----------------PROBABILIDAD CON HIJOS-----------------
st.divider()
st.header("📉 Simulador: El 'Impuesto' a la Maternidad")
st.markdown("""
Esta herramienta utiliza los coeficientes de nuestra Regresión Logística para predecir la probabilidad 
de que una persona trabaje, comparando hombres vs. mujeres según su carga familiar.
""")

# --- CONTROLES INTERACTIVOS EN LA BARRA LATERAL O COLUMNAS ---
st.subheader("Configura el Perfil del Individuo")
col_sim1, col_sim2, col_sim3 = st.columns(3)

with col_sim1:
    edad_input = st.slider("Edad", 15, 80, 35)
with col_sim2:
    esc_input = st.slider("Años de Escolaridad", 0, 24, 12, help="12 años es Preparatoria")
with col_sim3:
    entorno_urbano = st.checkbox("¿Vive en zona Urbana?", value=True)

# --- LÓGICA DE CÁLCULO ---
# Extraemos coeficientes del modelo entrenado previamente
c = dict(zip(features, modelo_pnea.coef_[0]))
intercepto = modelo_pnea.intercept_[0]

def calcular_prob(es_mujer, n_hij):
    z = (intercepto + 
         c['eda'] * edad_input + 
         c['eda_2'] * (edad_input**2) + 
         c['es_mujer'] * es_mujer + 
         c['mujer_con_hijos'] * (es_mujer * n_hij) + 
         c['anios_esc'] * esc_input + 
         c['es_urbano'] * (1 if entorno_urbano else 0) + 
         c['tiene_pareja'] * 1) # Asumimos pareja fija para la comparación
    return 1 / (1 + np.exp(-z))

hijos_rango = np.arange(0, 6)
prob_hombres = [calcular_prob(0, h) for h in hijos_rango]
prob_mujeres = [calcular_prob(1, h) for h in hijos_rango]

# --- GRÁFICA ---
fig_prob, ax_prob = plt.subplots(figsize=(10, 5))
ax_prob.plot(hijos_rango, prob_hombres, marker='o', label='Hombres', linewidth=3, color='#1f77b4')
ax_prob.plot(hijos_rango, prob_mujeres, marker='o', label='Mujeres', linewidth=3, color='#d62728')

ax_prob.set_title(f'Probabilidad de trabajar (Perfil: {edad_input} años, {esc_input} años esc.)', fontsize=12)
ax_prob.set_xlabel('Número de Hijos')
ax_prob.set_ylabel('Probabilidad (0 a 1)')
ax_prob.set_ylim(0, 1.05)
ax_prob.grid(True, linestyle=':', alpha=0.7)
ax_prob.legend()

# Añadir anotación dinámica si hay brecha
brecha_0 = prob_hombres[0] - prob_mujeres[0]
ax_prob.annotate(f'Brecha inicial: {brecha_0:.1%}', 
                 xy=(0, prob_mujeres[0]), xytext=(0.5, prob_mujeres[0]-0.2),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

st.pyplot(fig_prob)

# --- MENSAJE FINAL DE IMPACTO ---
st.error(f"**Análisis:** Para una mujer con {edad_input} años, cada hijo adicional reduce su probabilidad de participar en la economía un **{abs(c['mujer_con_hijos']*100):.1f}%** (según el coeficiente del modelo).")


#----------EFECTOS MARGINALES PROMEDIO------------------
import statsmodels.api as sm

st.divider()
st.header("🎯 Efectos Marginales: El Impacto Real en %")
st.markdown("""
A diferencia de los coeficientes técnicos, los **Efectos Marginales Promedio (AME)** nos indican el cambio 
directo en la **probabilidad porcentual** de participar en el mercado laboral.
""")

with st.spinner("Calculando efectos marginales (esto puede tardar unos segundos con 1GB de datos)..."):
    # 1. Ajuste del modelo con statsmodels
    X_const = sm.add_constant(X)
    logit_model = sm.Logit(y, X_const).fit(disp=0) # disp=0 para que no ensucie la terminal

    # 2. Cálculo de Efectos Marginales
    ame = logit_model.get_margeff(at='overall', method='dydx')
    ame_df = ame.summary_frame() # Convertimos a DataFrame para graficar
    ame_df.index = features # Asignamos los nombres de las variables

# --- VISUALIZACIÓN PROFESIONAL ---
col_stats, col_info = st.columns([2, 1])

with col_stats:
    fig_ame, ax_ame = plt.subplots(figsize=(8, 6))
    
    y_pos = np.arange(len(ame_df.index))
    values = ame_df.iloc[:, 0]        # dy/dx
    conf_low = ame_df.iloc[:, 4]      # Conf_low
    conf_high = ame_df.iloc[:, 5]     # Conf_high
    
    # Dibujar las barras de error
    ax_ame.errorbar(values, y_pos, 
                    xerr=[values - conf_low, conf_high - values],
                    fmt='o', color='black', ecolor='#d62728', capsize=5, elinewidth=2)
    
    # --- AGREGAR LAS VIÑETAS DE VALOR EXACTO ---
    for i, v in enumerate(values):
        # Convertimos el valor a porcentaje (0.05 -> +5.0%)
        texto_porcentaje = f"{v*100:+.1f}%"
        
        # Colocamos el texto ligeramente a la derecha del límite superior del error
        ax_ame.text(conf_high[i] + 0.01, i, texto_porcentaje, 
                    va='center', fontsize=10, fontweight='bold',
                    color='#d62728' if v < 0 else '#2ca02c')

    # Estética final
    ax_ame.set_yticks(y_pos)
    ax_ame.set_yticklabels(ame_df.index)
    ax_ame.axvline(0, color='#1f77b4', linestyle='--', alpha=0.5)
    ax_ame.set_title("Efectos Marginales (Impacto Directo en Probabilidad)", fontsize=12, fontweight='bold')
    
    # Ajustamos el límite del eje X para que quepan las etiquetas
    ax_ame.set_xlim(min(conf_low) - 0.05, max(conf_high) + 0.15)
    
    st.pyplot(fig_ame)

with col_info:
    st.subheader("¿Cómo leer esto?")
    st.info("""
    Cada punto representa cuánto cambia la probabilidad (de 0 a 1). 
    
    **Ejemplo:** Si 'anios_esc' tiene **0.6%**, significa que cada año extra de estudio sube un **0.6%** la probabilidad de trabajar.
    """)
    
    # Resaltar el dato más importante para el video
    impacto_hijos = ame_df.loc['mujer_con_hijos', 'dy/dx']
    st.metric(label="Impacto por cada hijo (Mujeres)", 
              value=f"{impacto_hijos*100:.1f}%", 
              delta=f"{impacto_hijos*100:.1f}%", 
              delta_color="red")
    

    #------------MATRIZ DE CONFUSIÓN------------
    from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import seaborn as sns

st.divider()
st.header("🎯 Validación del Modelo (Accuracy)")
st.markdown("""
¿Qué tan bueno es nuestro modelo para predecir si una persona participa en la economía? 
Aquí validamos los resultados contra datos reales que el modelo nunca había visto.
""")

# --- ENTRENAMIENTO Y PREDICCIÓN ---
# Usamos el X y y que ya definimos en bloques anteriores
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

modelo_final = LogisticRegression(max_iter=1000)
modelo_final.fit(X_train, y_train)
y_pred = modelo_final.predict(X_test)

# --- MÉTRICAS EN TARJETAS ---
report = classification_report(y_test, y_pred, output_dict=True)
accuracy = report['accuracy']

col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Precisión General", f"{accuracy:.1%}")
col_m2.metric("Sensibilidad (Recall)", f"{report['1']['recall']:.1%}", help="Capacidad de detectar a quienes sí trabajan")
col_m3.metric("F1-Score", f"{report['1']['f1-score']:.2f}")

# --- MATRIZ DE CONFUSIÓN ESTILIZADA ---
st.subheader("Matriz de Confusión")
cm = confusion_matrix(y_test, y_pred)

fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=False,
            xticklabels=['No Participa', 'Participa'],
            yticklabels=['No Participa', 'Participa'])

ax_cm.set_xlabel('Predicción del Modelo', fontsize=12, fontweight='bold')
ax_cm.set_ylabel('Realidad (INEGI)', fontsize=12, fontweight='bold')
ax_cm.set_title('¿A cuántas personas clasificamos correctamente?', pad=20)

st.pyplot(fig_cm)

with st.expander("🤔 ¿Cómo leer esta matriz?"):
    st.write("""
    * **Diagonal Principal (Azul fuerte):** Son los éxitos. Personas que el modelo predijo correctamente.
    * **Esquinas opuestas:** Son los errores. 
    """)

    #------------FALSOS POSITIVOS--------------------
    st.divider()
st.header("🔍 Análisis de Potencial: Los Falsos Positivos")
st.markdown("""
¿Quiénes son las personas que el modelo predice que **deberían estar trabajando** pero el INEGI reporta que no lo están? 
Este grupo representa el **talento humano disponible** que la economía mexicana no está aprovechando.
""")

# --- CÁLCULO DE FALSOS POSITIVOS ---
# Unimos predicciones con datos originales
resultados = X_test.copy()
resultados['real'] = y_test
resultados['prediccion'] = y_pred

# Índices de Falsos Positivos (Predijo 1, Realidad 0)
indices_fp = resultados[(resultados['real'] == 0) & (resultados['prediccion'] == 1)].index
fp_completos = df_pnea.loc[indices_fp]

# --- VISUALIZACIÓN DE PERFIL ---
col_fp1, col_fp2 = st.columns([1, 2])

with col_fp1:
    st.subheader("📊 Estadísticas Clave")
    st.metric("Total de Personas (FP)", f"{len(fp_completos):,}")
    
    # Mostrar porcentaje de mujeres con un color destacado
    porc_mujeres = fp_completos['es_mujer'].mean() * 100
    st.write(f"**Porcentaje de Mujeres:**")
    st.progress(porc_mujeres / 100)
    st.write(f" {porc_mujeres:.2f}% de este grupo son mujeres.")

with col_fp2:
    st.subheader("📋 Perfil Promedio del Grupo")
    
    # Creamos un DataFrame bonito para mostrar el perfil
    datos_perfil = {
        "Atributo": ["Años de Escolaridad", "Edad Promedio", "Promedio de Hijos", "Tienen Pareja (%)"],
        "Valor": [
            f"{fp_completos['anios_esc'].mean():.1f} años",
            f"{fp_completos['eda'].mean():.1f} años",
            f"{fp_completos['n_hij'].mean():.2f}",
            f"{fp_completos['tiene_pareja'].mean()*100:.1f}%"
        ]
    }
    st.table(pd.DataFrame(datos_perfil))

# --- CONCLUSIÓN PARA EL VIDEO ---
st.info(f"""
👉 **Conclusión:** Este grupo tiene un promedio de **{fp_completos['anios_esc'].mean():.1f} años de estudio** (equivalente a { 'Preparatoria completa' if fp_completos['anios_esc'].mean() > 11 else 'Secundaria' }). 
Son personas calificadas que, por barreras sistémicas o falta de oportunidades, no están participando en la PEA.
""")
#--------------FALSOS POSITIVOS RESTANTES------------------
st.divider()
st.header("🕵️‍♂️ El Perfil del 'Potencial Masculino' Fuera del Mercado")
st.markdown("""
Analizamos específicamente a los **hombres** que el modelo clasifica como 'deberían estar trabajando'. 
Este grupo es pequeño en comparación con las mujeres, pero sus características son reveladoras.
""")

# --- CÁLCULO ESPECÍFICO ---
fp_hombres = fp_completos[fp_completos['es_mujer'] == 0]
total_fp_count = len(fp_completos)
n_hombres = len(fp_hombres)

# --- VISUALIZACIÓN ---
col_h1, col_h2 = st.columns([1, 1])

with col_h1:
    st.subheader("📊 Proporción Masculina")
    # Gráfico de dona sencillo
    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie([n_hombres, total_fp_count - n_hombres], 
               labels=['Hombres', 'Mujeres'], 
               autopct='%1.1f%%', 
               colors=['#1f77b4', '#d62728'],
               startangle=90, explode=(0.1, 0))
    st.pyplot(fig_pie)

with col_h2:
    st.subheader("📝 Ficha Técnica: El Hombre 'FP'")
    
    st.metric("Total identificados", f"{n_hombres:,}")
    
    # Datos en formato de lista de impacto
    st.write(f"🎓 **Escolaridad promedio:** {fp_hombres['anios_esc'].mean():.1f} años")
    st.write(f"🎂 **Edad promedio:** {fp_hombres['eda'].mean():.1f} años")
    st.write(f"👶 **Promedio de hijos:** {fp_hombres['n_hij'].mean():.2f}")
    st.write(f"💍 **Tienen pareja:** {fp_hombres['tiene_pareja'].mean()*100:.1f}%")

# --- CONCLUSIÓN DINÁMICA ---
st.warning(f"""
**Interpretación para el video:** Solo el **{(n_hombres/total_fp_count)*100:.1f}%** de las personas con potencial no aprovechado son hombres. 
Esto refuerza la tesis de que el sistema laboral absorbe a casi cualquier hombre calificado, 
mientras que a las mujeres calificadas las deja fuera con mucha más frecuencia.
""")

#----------------------DISTRIBUCION SALARIAL EN MEXICO----------------------
st.divider()
st.header("💰 La Realidad del 'Brokeman': Distribución Salarial Masculina")
st.markdown("""
¿Cuántos hombres en México realmente ganan lo suficiente para ser considerados 'proveedores' bajo los estándares actuales? 
Analizamos los ingresos reales mensuales reportados al INEGI.
""")

# --- PROCESAMIENTO ---
# Filtramos hombres ocupados con ingresos válidos
hombres_dist = df[(df['sex'] == 1) & (df['clase2'] == 1) & (df['ingocup'] < 999999)].copy()

# Agrupamos
dist_salarial = hombres_dist.groupby('ing7c', observed=False).agg({
    'fac_tri': 'sum',
    'ingocup': 'mean'
}).reset_index()

nombres_ingresos = {
    1: 'Hasta 1 SM', 2: '1 a 2 SM', 3: '2 a 3 SM', 
    4: '3 a 5 SM', 5: 'Más de 5 SM', 6: 'Sin ingresos', 7: 'No especificado'
}
dist_salarial['Rango'] = dist_salarial['ing7c'].map(nombres_ingresos)

# --- GRÁFICA ESTILIZADA ---
fig_sal, ax_sal = plt.subplots(figsize=(12, 7))

# Colores: Gris para la base, Azul para la "élite" (3 SM en adelante)
colores_dict = {
    'Hasta 1 SM': '#d3d3d3', '1 a 2 SM': '#d3d3d3', '2 a 3 SM': '#d3d3d3',
    '3 a 5 SM': '#1f77b4', 'Más de 5 SM': '#1a5a8a', 
    'Sin ingresos': '#f0f0f0', 'No especificado': '#f0f0f0'
}
colores = [colores_dict[r] for r in dist_salarial['Rango']]

sns.barplot(data=dist_salarial, x='Rango', y='fac_tri', palette=colores, ax=ax_sal)

ax_sal.set_title(
    'Distribución de Ingresos Mensuales (Hombres Ocupados)', 
    loc='left',       # <--- Esto lo mueve a la izquierda
    fontsize=16, 
    fontweight='bold', 
    pad=30,           # <--- Esto le da espacio hacia arriba para que no choque con las etiquetas
    color='#333333'
)

# Anotaciones duales (Personas + Salario $)
for i, p in enumerate(ax_sal.patches):
    millones = p.get_height() / 1_000_000
    salario_avg = dist_salarial.loc[i, 'ingocup']
    
    if millones > 0.05: # Solo anotar si hay suficiente gente
        texto = f'{millones:.1f}M\n${salario_avg:,.0f}'
        ax_sal.annotate(texto, 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='baseline', fontsize=9, fontweight='bold',
                        xytext=(0, 10), textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1f77b4", alpha=0.9))

ax_sal.set_ylabel("Millones de Hombres")
#ax_sal.set_title("Distribución de Ingresos Mensuales (Hombres Ocupados)", fontsize=14)
plt.xticks(rotation=15)
ax_sal.spines['top'].set_visible(False)
ax_sal.spines['right'].set_visible(False)

st.pyplot(fig_sal)

# --- MÉTRICA DE IMPACTO ---
elite_total = dist_salarial[dist_salarial['ing7c'].isin([4, 5])]['fac_tri'].sum()
porc_elite = (elite_total / dist_salarial['fac_tri'].sum()) * 100

col_inf1, col_inf2 = st.columns(2)
with col_inf1:
    st.info(f"**La Élite:** Solo el **{porc_elite:.1f}%** de los hombres ocupados ganan más de 3 Salarios Mínimos.")
with col_inf2:
    salario_top = dist_salarial[dist_salarial['ing7c'] == 5]['ingocup'].values[0]
    st.success(f"**El 1%:** El rango más alto promedia **${salario_top:,.2f} MXN** mensuales.")

    #-------------------DISTRIBUCION SALARIAL POR SEXO---------------------
    st.divider()
st.header("⚖️ La Brecha de Ingresos: Hombres vs. Mujeres")
st.markdown("""
Comparativa de la distribución salarial por género. Observa cómo la presencia femenina 
se diluye conforme aumentan los ingresos (el famoso 'techo de cristal').
""")

# --- PROCESAMIENTO ---
poblacion_ocupada = df[df['clase2'] == 1].copy()
dist_genero = poblacion_ocupada.groupby(['sex', 'ing7c'], observed=False)['fac_tri'].sum().reset_index()

nombres_ingresos = {
    1: 'Hasta 1 SM', 2: '1 a 2 SM', 3: '2 a 3 SM', 
    4: '3 a 5 SM', 5: 'Más de 5 SM', 6: 'Sin ingresos', 7: 'No especificado'
}
dist_genero['Rango'] = dist_genero['ing7c'].map(nombres_ingresos)
dist_genero['Sexo'] = dist_genero['sex'].map({1: 'Hombres', 2: 'Mujeres'})

# --- GRÁFICA COMPARATIVA ---
fig_gen, ax_gen = plt.subplots(figsize=(14, 8))

# Usamos colores contrastantes: Azul para hombres, Rojo/Coral para mujeres
sns.barplot(data=dist_genero, x='Rango', y='fac_tri', hue='Sexo', 
            palette=['#3498db', '#e74c3c'], ax=ax_gen)

# Título ajustado a la izquierda para evitar empalmes
ax_gen.set_title('Distribución Salarial en México por Género', 
                loc='left', fontsize=18, fontweight='bold', pad=40, color='#333333')

# Configuración de ejes
ax_gen.set_xlabel('Rango de Ingresos', fontsize=12)
ax_gen.set_ylabel('Millones de Personas', fontsize=12)
ax_gen.legend(title='Género', frameon=False)

# Etiquetas de millones sobre las barras
for p in ax_gen.patches:
    altura = p.get_height()
    if altura > 50000: # Solo anotar si hay datos significativos
        ax_gen.annotate(f'{altura/1_000_000:.1f}M', 
                        (p.get_x() + p.get_width() / 2., altura), 
                        ha='center', va='baseline', fontsize=9, fontweight='bold', 
                        xytext=(0, 10), textcoords='offset points')

# Estética final
ax_gen.spines['top'].set_visible(False)
ax_gen.spines['right'].set_visible(False)
ax_gen.yaxis.grid(True, linestyle='--', alpha=0.3)

st.pyplot(fig_gen)

# --- ANÁLISIS AUTOMÁTICO PARA EL VIDEO ---
# Calculamos la proporción de la élite por género
total_h = dist_genero[dist_genero['sex'] == 1]['fac_tri'].sum()
total_m = dist_genero[dist_genero['sex'] == 2]['fac_tri'].sum()
elite_h = dist_genero[(dist_genero['sex'] == 1) & (dist_genero['ing7c'].isin([4, 5]))]['fac_tri'].sum()
elite_m = dist_genero[(dist_genero['sex'] == 2) & (dist_genero['ing7c'].isin([4, 5]))]['fac_tri'].sum()

porc_h = (elite_h / total_h) * 100
porc_m = (elite_m / total_m) * 100

st.warning(f"""
💡 **Dato clave:** Mientras que el **{porc_h:.1f}%** de los hombres ocupados ganan más de 3 SM, 
solo el **{porc_m:.1f}%** de las mujeres alcanzan esos mismos rangos. 
¡La probabilidad de encontrar a un hombre **'ESTABLE'** es casi el doble que la de encontrar a una mujer!
""")

#--------------POBLACION DE MUJERES ENTRE 23-40 AÑOS--------------
st.divider()
st.header("🚺 El Universo Femenino (Target)")
st.markdown("""
Comenzamos delimitando nuestro grupo de estudio: mujeres en su etapa de mayor potencial profesional y personal (25 a 40 años).
""")

# --- CÁLCULO ---
# Usamos el DF original cargado al inicio
mujeres_target = df[(df['sex'] == 2) & (df['eda'] >= 25) & (df['eda'] <= 40)].copy()
total_mujeres_mexico = mujeres_target['fac_tri'].sum()

# --- VISUALIZACIÓN DE IMPACTO ---
col_t1, col_t2 = st.columns([1, 1])

with col_t1:
    st.metric(
        label="Mujeres en México (25-40 años)", 
        value=f"{total_mujeres_mexico/1_000_000:.2f} Millones",
        delta="Universo Inicial",
        delta_color="normal"
    )

with col_t2:
    st.info("""
    **Nota demográfica:** Este grupo representa el núcleo de la fuerza laboral femenina y el segmento con mayor impacto por las decisiones de maternidad y carrera.
    """)

# --- OPCIONAL: Gráfico de pastel para contexto ---
# Comparar este grupo contra el resto de las mujeres
total_mujeres_todas = df[df['sex'] == 2]['fac_tri'].sum()
resto_mujeres = total_mujeres_todas - total_mujeres_mexico

fig_uni, ax_uni = plt.subplots(figsize=(6, 4))
ax_uni.pie(
    [total_mujeres_mexico, resto_mujeres], 
    labels=['Target (25-40)', 'Otras edades'], 
    autopct='%1.1f%%', 
    colors=['#e74c3c', '#ffcccb'],
    startangle=140,
    explode=(0.1, 0)
)
ax_uni.set_title("Proporción del Segmento en la Población Femenina", fontsize=10)
st.pyplot(fig_uni)

#--------------------ELITE MASCULINA-----------------------
st.divider()
st.header("🏆 La Élite Masculina: ¿A qué edad alcanzan el éxito?")
st.markdown("""
Analizamos la distribución por edad de los hombres que ganan **más de 3 Salarios Mínimos**. 
Este gráfico muestra cuándo ocurre el pico de ingresos en la vida del hombre mexicano.
""")

# --- PROCESAMIENTO ---
# 1. Filtrar hombres de la élite (Usamos el DF cargado con caché)
hombres_3sm_mas_gen = df[
    (df['sex'] == 1) & 
    (df['ing7c'].isin([4, 5])) &
    (df['clase2'] == 1)
].copy()

# 2. Definir rangos de edad
bins = [20, 30, 40, 50, 100]
labels = ['20-29 años', '30-39 años', '40-49 años', '50+ años']

hombres_3sm_mas_gen['rango_edad'] = pd.cut(hombres_3sm_mas_gen['eda'], bins=bins, labels=labels, right=False)

# 3. Agrupar por población expandida
dist_edad_poblacional = hombres_3sm_mas_gen.groupby('rango_edad', observed=False)['fac_tri'].sum().reset_index()

# --- VISUALIZACIÓN ---
fig_age, ax_age = plt.subplots(figsize=(12, 7))
# Usamos 'magma' para un look más premium
colors = sns.color_palette('magma', len(labels))
sns.barplot(data=dist_edad_poblacional, x='rango_edad', y='fac_tri', palette=colors, ax=ax_age)

# Título y etiquetas con diseño limpio
ax_age.set_title('Distribución por Edad (Ingresos > 3 SM)', loc='left', fontsize=16, fontweight='bold', pad=30)
ax_age.set_ylabel("Estimación Nacional (Personas)", color='gray')
ax_age.set_xlabel("Rango de Edad", color='gray')

# Anotaciones sobre las barras
for p in ax_age.patches:
    ax_age.annotate(f'{int(p.get_height()):,}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='baseline', fontsize=11, fontweight='bold', 
                    xytext=(0, 10), textcoords='offset points',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

ax_age.spines['top'].set_visible(False)
ax_age.spines['right'].set_visible(False)

st.pyplot(fig_age)

# --- ANÁLISIS PARA EL VIDEO ---
max_rango = dist_edad_poblacional.loc[dist_edad_poblacional['fac_tri'].idxmax(), 'rango_edad']
total_elite_h = dist_edad_poblacional['fac_tri'].sum()

col_an1, col_an2 = st.columns(2)
with col_an1:
    st.info(f"📍 **El Pico del Éxito:** La mayor cantidad de hombres de alto ingreso se encuentra en el rango de **{max_rango}**.")
with col_an2:
    st.success(f"📈 **Volumen Total:** Hay **{total_elite_h:,.0f}** hombres en todo el país que cumplen con este perfil económico.")

#--------------FORMAL VS INFORMAL----------------------
st.divider()
st.header("🛡️ Estabilidad vs. Riesgo: Formalidad en la Élite")
st.markdown("""
¿Qué tan sólida es la posición de estos hombres? Analizamos quiénes cuentan con **Seguridad Social** (empleos formales) frente a quienes generan ingresos altos en la **Informalidad**.
""")

# --- PROCESAMIENTO ---
# 1. Filtrar hombres de la élite (Usamos el DF cargado con caché)
hombres_elite = df[
    (df['sex'] == 1) & 
    (df['ing7c'].isin([4, 5])) &
    (df['clase2'] == 1)
].copy()

# 2. Filtrar y agrupar por Seguridad Social
# 1: Tiene acceso, 2: No tiene acceso (según diseño ENOE)
resumen_social = hombres_elite[hombres_elite['seg_soc'].isin([1, 2])].copy()
dist_formalidad = resumen_social.groupby('seg_soc', observed=False)['fac_tri'].sum().reset_index()

# Mapeo
dist_formalidad['Estatus'] = dist_formalidad['seg_soc'].map({
    1: 'Formal (Con Seg. Social)', 
    2: 'Informal (Sin Seg. Social)'
})

# --- VISUALIZACIÓN DE MÉTRICAS ---
total_v = dist_formalidad['fac_tri'].sum()
formal_v = dist_formalidad[dist_formalidad['seg_soc'] == 1]['fac_tri'].sum()
informal_v = dist_formalidad[dist_formalidad['seg_soc'] == 2]['fac_tri'].sum()

col_f1, col_f2, col_f3 = st.columns(3)
col_f1.metric("Total Élite", f"{total_v/1e6:.1f}M")
col_f2.metric("Formales", f"{(formal_v/total_v)*100:.1f}%", delta="Estabilidad", delta_color="normal")
col_f3.metric("Informales", f"{(informal_v/total_v)*100:.1f}%", delta="Riesgo", delta_color="inverse")

# --- GRÁFICA ---
fig_form, ax_form = plt.subplots(figsize=(10, 6))
# Verde para estabilidad, Rojo para riesgo
sns.barplot(data=dist_formalidad, x='Estatus', y='fac_tri', palette=['#2ecc71', '#e74c3c'], ax=ax_form)

ax_form.set_title('Distribución de Seguridad Social (Ingresos > 3 SM)', loc='left', fontsize=14, fontweight='bold', pad=20)
ax_form.set_ylabel("Millones de Personas")

# Anotación de porcentaje sobre las barras
for p in ax_form.patches:
    porc = (p.get_height() / total_v) * 100
    ax_form.annotate(f'{porc:.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='baseline', fontsize=12, fontweight='bold', 
                    xytext=(0, 10), textcoords='offset points')

ax_form.spines['top'].set_visible(False)
ax_form.spines['right'].set_visible(False)

st.pyplot(fig_form)

# --- CONCLUSIÓN PARA EL GUIÓN ---
st.info(f"""
💡 **Dato:** Uno de cada **{int(1/(informal_v/total_v))}** hombres de la 'élite' 
realmente vive en la informalidad. Tienen el dinero, pero carecen de la red de protección del Estado. 
¿Es este un factor de riesgo para una familia? Definitivamente.
""")


#---------------- RATIO ----------------
# =========================================================
# CONCLUSIÓN FINAL: EL DATO DEMOLEDOR (FUERA DE TABS)
# =========================================================
st.divider()
st.header("🏁 Veredicto Final: El Ratio de Selección")

try:
    # 1. Recuperamos los totales usando el factor de expansión (fac_tri)
    # Total Mujeres Target (25-40 años)
    total_m_target = mujeres_target['fac_tri'].sum()
    
    # Total Hombres Élite (Hombres con > 3 SM)
    total_h_elite = hombres_3sm_mas_gen['fac_tri'].sum()
    
    if total_h_elite > 0:
        # 2. Calculamos el Ratio
        ratio_final = total_m_target / total_h_elite
        
        # 3. Diseño de Impacto con Métricas
        col_f1, col_f2 = st.columns([1, 1])
        
        with col_f1:
            st.metric(
                label="Ratio de Competencia Estadística", 
                value=f"{ratio_final:.1f} a 1",
                delta="Mujeres por cada Hombre de Élite",
                delta_color="inverse"
            )
            st.caption("Basado en proyecciones poblacionales de la ENOE 2024")

        with col_f2:
            with st.chat_message("assistant", avatar="🤖"):
                st.write(f"**Análisis finalizado:** En este segmento, existen aproximadamente **{ratio_final:.1f} mujeres** por cada **1 hombre** con ingresos superiores a 3 Salarios Mínimos.")
                st.write("Estadísticamente, esto representa un mercado altamente competitivo para este perfil socioeconómico.")

        # 4. Gráfica Visual de Proporción (Barra Horizontal)
        fig_crunch, ax_crunch = plt.subplots(figsize=(10, 2))
        
        # Dibujamos las barras
        ax_crunch.barh(["Hombres"], [1], color='#3498db', label="1 Hombre de Élite")
        ax_crunch.barh(["Mujeres"], [ratio_final], color='#e74c3c', label=f"{ratio_final:.1f} Mujeres Target")
        
        ax_crunch.set_title("Visualización de la Disponibilidad Relativa", loc='left', fontsize=12, fontweight='bold')
        ax_crunch.legend(loc='lower right', fontsize=9)
        
        # Limpieza estética de la gráfica
        ax_crunch.spines['top'].set_visible(False)
        ax_crunch.spines['right'].set_visible(False)
        ax_crunch.spines['left'].set_visible(False)
        ax_crunch.grid(axis='x', linestyle='--', alpha=0.3)
        
        st.pyplot(fig_crunch)

    else:
        st.warning("⚠️ No se detectaron suficientes datos para calcular el ratio.")

except Exception as e:
    st.error(f"Ocurrió un error al calcular el ratio final: {e}")
    st.info("Asegúrate de que las variables 'mujeres_target' y 'hombres_3sm_mas_gen' estén definidas en las pestañas anteriores.")


# --- BOTÓN DE FINALIZACIÓN OPTIMIZADO ---
st.divider()

# Usamos un contenedor para separar la lógica
if st.button("🚀 FINALIZAR ANÁLISIS"):
    # Ejecutamos solo las animaciones visuales
    st.balloons() 
    
    # Usamos un mensaje simple en lugar de volver a calcular todo
    st.success("✅ Reporte finalizado con éxito.")
    
    # Tip: No pongas cálculos pesados dentro del 'if st.button'
    # Solo pon las consecuencias visuales del botón.
