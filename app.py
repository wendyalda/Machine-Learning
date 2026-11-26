import streamlit as st
import pandas as pd
import os
from PIL import Image
from modulos.cnn_inference import CNNInferencia
from modulos.visualizaciones import graficar_probabilidades, graficar_curvas_entrenamiento

# Configuración de página
st.set_page_config(
    page_title="Sistema de Clasificación de Animales - UMSS",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar motor de inferencia
@st.cache_resource
def obtener_predictor():
    return CNNInferencia()

predictor = obtener_predictor()

# Barra lateral de navegación
st.sidebar.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=80)
st.sidebar.title("Navegación del Sistema")
st.sidebar.markdown("**Materia:** Aprendizaje Automático")
st.sidebar.markdown("**Docente:** Lic. Patricia Rodríguez Bilbao")
st.sidebar.markdown("**Área:** Clasificación y Optimización")
st.sidebar.markdown("---")

opcion_menu = st.sidebar.radio(
    "Seleccione un módulo:",
    ["🏠 Inicio y Descripción del Proyecto", 
     "🧠 Red Neuronal Convolucional (CNN)", 
     "📊 Métricas y Entrenamiento", 
     "🔮 Inferencia en Tiempo Real", 
     "🧩 Módulo No Supervisado (Próximamente)"]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip para la defensa:** Use la pestaña *Inferencia en Tiempo Real* para subir imágenes de prueba durante la revisión.")

# ------------------------------------------------------------------------------
# PESTAÑA 1: INICIO Y DESCRIPCIÓN
# ------------------------------------------------------------------------------
if opcion_menu == "🏠 Inicio y Descripción del Proyecto":
    st.title("🐾 Prototipo de Clasificación de Imágenes de Animales")
    st.markdown("### Universidad Mayor de San Simón - Facultad de Ciencias y Tecnología")
    
    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown("""
        Este sistema interactivo resuelve el problema de la **clasificación automática de imágenes de animales**
        mediante modelos de **Aprendizaje Profundo (Deep Learning)**.
        
        #### **Características Principales:**
        * **Dataset de origen:** Animals-10 (10 clases de animales).
        * **Arquitectura:** Red Neuronal Convolucional (CNN) secuencial de 4 bloques de extracción de características.
        * **Preprocesamiento:** Redimensionamiento a \\(128 \\times 128\\) px, normalización a \\([0.0, 1.0]\\) y Data Augmentation.
        * **Regularización:** Capas de Dropout (0.5), EarlyStopping y ReduceLROnPlateau.
        """)
    
    with col2:
        st.metric(label="Clases de Animales", value="10 Categorías")
        st.metric(label="Precisión en Prueba", value="67.07%")
        st.metric(label="Parámetros Entrenables", value="1.42 M")

# ------------------------------------------------------------------------------
# PESTAÑA 2: ARQUITECTURA DEL MODELO
# ------------------------------------------------------------------------------
elif opcion_menu == "🧠 Red Neuronal Convolucional (CNN)":
    st.title("🧠 Arquitectura del Modelo Convolucional")
    st.markdown("Estructura detallada de la red secuencial implementada en Keras y TensorFlow.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Esquema de Capas")
        df_capas = pd.DataFrame([
            {"Capa": "Conv2D_1 + ReLU", "Filtros / Neuronas": "32 filtros (3x3)", "Salida": "(126, 126, 32)", "Parámetros": 896},
            {"Capa": "MaxPooling2D_1", "Filtros / Neuronas": "Ventana 2x2", "Salida": "(63, 63, 32)", "Parámetros": 0},
            {"Capa": "Conv2D_2 + ReLU", "Filtros / Neuronas": "64 filtros (3x3)", "Salida": "(61, 61, 64)", "Parámetros": 18496},
            {"Capa": "MaxPooling2D_2", "Filtros / Neuronas": "Ventana 2x2", "Salida": "(30, 30, 64)", "Parámetros": 0},
            {"Capa": "Conv2D_3 + ReLU", "Filtros / Neuronas": "128 filtros (3x3)", "Salida": "(28, 28, 128)", "Parámetros": 73856},
            {"Capa": "MaxPooling2D_3", "Filtros / Neuronas": "Ventana 2x2", "Salida": "(14, 14, 128)", "Parámetros": 0},
            {"Capa": "Conv2D_4 + ReLU", "Filtros / Neuronas": "128 filtros (3x3)", "Salida": "(12, 12, 128)", "Parámetros": 147584},
            {"Capa": "MaxPooling2D_4", "Filtros / Neuronas": "Ventana 2x2", "Salida": "(6, 6, 128)", "Parámetros": 0},
            {"Capa": "Flatten", "Filtros / Neuronas": "Vectorización", "Salida": "(4608)", "Parámetros": 0},
            {"Capa": "Dense (Oculta)", "Filtros / Neuronas": "256 neuronas + ReLU", "Salida": "(256)", "Parámetros": 1179904},
            {"Capa": "Dropout", "Filtros / Neuronas": "Tasa 0.5", "Salida": "(256)", "Parámetros": 0},
            {"Capa": "Dense (Salida)", "Filtros / Neuronas": "10 neuronas + Softmax", "Salida": "(10)", "Parámetros": 2570},
        ])
        st.dataframe(df_capas, use_container_width=True)
        
    with col2:
        st.subheader("Hiperparámetros de Compilación")
        st.json({
            "Optimizador": "Adam (lr=0.001)",
            "Función de Pérdida": "Categorical Crossentropy",
            "Métrica Principal": "Accuracy",
            "Tamaño de Lote (Batch Size)": 32,
            "Épocas Ejecutadas": 14,
            "Callback Parada Temprana": "Paciencia = 5 épocas"
        })

# ------------------------------------------------------------------------------
# PESTAÑA 3: MÉTRICAS Y ENTRENAMIENTO
# ------------------------------------------------------------------------------
elif opcion_menu == "📊 Métricas y Entrenamiento":
    st.title("📊 Desempeño y Curvas de Entrenamiento")
    
    ruta_historial = "datos/historial_entrenamiento.csv"
    if os.path.exists(ruta_historial):
        df_hist = pd.read_csv(ruta_historial)
        st.subheader("Curvas de Aprendizaje (Accuracy & Loss)")
        fig_curvas = graficar_curvas_entrenamiento(df_hist)
        st.pyplot(fig_curvas)
    else:
        st.warning("⚠️ No se encontró el archivo `datos/historial_entrenamiento.csv`. Asegúrate de copiarlo a la carpeta `datos/`.")
        
    st.subheader("Métricas Formales en Conjunto de Prueba")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exactitud (Accuracy)", "67.07%")
    c2.metric("Precisión (Precision)", "67.80%")
    c3.metric("Sensibilidad (Recall)", "67.07%")
    c4.metric("F1-Score", "67.25%")

# ------------------------------------------------------------------------------
# PESTAÑA 4: INFERENCIA EN TIEMPO REAL
# ------------------------------------------------------------------------------
elif opcion_menu == "🔮 Inferencia en Tiempo Real":
    st.title("🔮 Clasificación de Imágenes en Tiempo Real")
    st.markdown("Suba una imagen de un animal desde su computadora para clasificarla con la red neuronal convolucional.")
    
    archivo_subido = st.file_uploader("Seleccione una imagen (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])
    
    if archivo_subido is not None:
        imagen_pil = Image.open(archivo_subido)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(imagen_pil, caption="Imagen Entrada", use_container_width=True)
            
        with col2:
            with st.spinner("Procesando imagen con la red convolucional..."):
                resultados = predictor.predecir(imagen_pil)
                
            if resultados:
                st.success(f"**Predicción Principal:** {resultados['clase']}")
                st.info(f"**Nivel de Confianza:** {resultados['confianza']:.2f}%")
                
                fig_probs = graficar_probabilidades(resultados["probabilidades"])
                st.pyplot(fig_probs)
            else:
                st.error("❌ No se pudo realizar la predicción. Verifique que el archivo `modelos/modelo_cnn_animales.keras` esté presente.")

# ------------------------------------------------------------------------------
# PESTAÑA 5: APRENDIZAJE NO SUPERVISADO (AVANCE 2)
# ------------------------------------------------------------------------------
elif opcion_menu == "🧩 Módulo No Supervisado (Próximamente)":
    st.title("🧩 Módulo de Aprendizaje No Supervisado")
    st.info("Este módulo integrará los algoritmos de Clustering (K-Means / DBSCAN) correspondientes al **Segundo Avance y Entrega Final**.")