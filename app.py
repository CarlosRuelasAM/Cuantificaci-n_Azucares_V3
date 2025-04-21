import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import io
from collections import defaultdict

# Función para suavizar imagen
def suavizar_imagen(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

# Función para normalizar color
def normalizar_color(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# Función para seleccionar área manualmente (en Streamlit, mediante sliders)
def seleccionar_area_con_sliders(img):
    h, w, _ = img.shape
    st.write("### Selecciona el área de análisis dentro de la imagen")
    x_start = st.slider("X inicio", 0, w-1, 0)
    x_end = st.slider("X fin", x_start+1, w, w)
    y_start = st.slider("Y inicio", 0, h-1, 0)
    y_end = st.slider("Y fin", y_start+1, h, h)
    
    # Dibuja un rectángulo en la imagen para mostrar el área seleccionada
    img_con_rectangulo = img.copy()
    cv2.rectangle(img_con_rectangulo, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)  # Rectángulo verde
    st.image(img_con_rectangulo, caption="Área de análisis seleccionada", use_column_width=True)
    
    # Calcular el área de fondo (5 píxeles alrededor de la región seleccionada)
    margen = 5
    fondo = img[max(y_start - margen, 0):min(y_end + margen, h), max(x_start - margen, 0):min(x_end + margen, w)]
    
    # Promediar el color del fondo
    color_fondo = np.mean(fondo, axis=(0, 1))
    
    # Extraer el área de análisis (sin el fondo)
    area_roi = img[y_start:y_end, x_start:x_end]
    
    return area_roi, color_fondo

# Función para extraer color promedio de una imagen
def extraer_color_promedio(imagen):
    promedio_bgr = np.mean(imagen, axis=(0, 1))
    return promedio_bgr[::-1]  # Convertir BGR a RGB

# Página principal
st.title("Cuantificación de azúcares con prueba de Fehling")

st.header("1. Subir imágenes estándar")
imagenes_estandar = st.file_uploader("Carga imágenes de estándares con concentraciones conocidas", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
concentraciones = []
colores_estandar = []

# Usar un diccionario para agrupar muestras del mismo estándar (concentración)
estandares_grupales = defaultdict(list)

if imagenes_estandar:
    for i, img_file in enumerate(imagenes_estandar):
        st.image(img_file, caption=f"Estándar {i+1}", width=300)
        conc = st.number_input(f"Concentración para Estándar {i+1} (mg/mL):", min_value=0.0, step=0.1, key=f"conc_{i}")
        imagen = Image.open(img_file)
        imagen_cv = cv2.cvtColor(np.array(imagen), cv2.COLOR_RGB2BGR)
        imagen_suave = suavizar_imagen(imagen_cv)
        imagen_norm = normalizar_color(imagen_suave)
        
        # Seleccionar área y obtener color promedio del fondo
        area_roi, color_fondo = seleccionar_area_con_sliders(imagen_norm)
        
        # Sustracción del fondo
        color_ajustado = extraer_color_promedio(area_roi) - color_fondo
        
        # Agrupar por concentración (promediar los valores de color para la misma concentración)
        estandares_grupales[conc].append(color_ajustado)

# Promediar las muestras de un mismo estándar
for conc in estandares_grupales:
    colores_estandar.append(np.mean(estandares_grupales[conc], axis=0))  # Promediar las características del color
    concentraciones.append(conc)

# Selección del modelo de regresión
modelo_seleccionado = st.selectbox("Selecciona el modelo de regresión", ["Regresión Lineal", "Regresión Polinómica", "Regresión SVR"])

# Crear modelo de calibración
modelo = None
r2_general = None
if concentraciones and colores_estandar:
    X = np.array(colores_estandar)
    y = np.array(concentraciones)

    # Modelo de regresión lineal
    if modelo_seleccionado == "Regresión Lineal":
        modelo = LinearRegression().fit(X, y)
        predicciones = modelo.predict(X)
        r2_general = r2_score(y, predicciones)
    
    # Modelo de regresión polinómica
    elif modelo_seleccionado == "Regresión Polinómica":
        poly = PolynomialFeatures(degree=2)  # Grado 2 como ejemplo
        X_poly = poly.fit_transform(X)
        modelo = LinearRegression().fit(X_poly, y)
        predicciones = modelo.predict(X_poly)
        r2_general = r2_score(y, predicciones)
    
    # Modelo de regresión de soporte vectorial
    elif modelo_seleccionado == "Regresión SVR":
        modelo = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
        modelo.fit(X, y)
        predicciones = modelo.predict(X)
        r2_general = r2_score(y, predicciones)

    # Mostrar el coeficiente de determinación (R²) general
    st.subheader(f"Coeficiente de determinación (R²) para la curva de calibración ({modelo_seleccionado}): {r2_general:.4f}")

    st.subheader("Curva de calibración")
    fig, ax = plt.subplots()
    ax.scatter(concentraciones, predicciones, label="Datos calibrados")
    ax.plot([min(concentraciones), max(concentraciones)], [min(concentraciones), max(concentraciones)], 'r--', label="Ideal")
    ax.set_xlabel("Concentración real (mg/mL)")
    ax.set_ylabel("Concentración estimada (mg/mL)")
    ax.legend()
    st.pyplot(fig)

st.header("2. Subir imágenes de muestra")
imagenes_muestra = st.file_uploader("Carga imágenes de muestras a analizar", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
resultados = []

if imagenes_muestra and modelo:
    for i, img_file in enumerate(imagenes_muestra):
        st.image(img_file, caption=f"Muestra {i+1}", width=300)
        imagen = Image.open(img_file)
        imagen_cv = cv2.cvtColor(np.array(imagen), cv2.COLOR_RGB2BGR)
        imagen_suave = suavizar_imagen(imagen_cv)
        imagen_norm = normalizar_color(imagen_suave)
        
        # Seleccionar área y obtener color promedio del fondo
        area_roi, color_fondo = seleccionar_area_con_sliders(imagen_norm)
        
        # Sustracción del fondo
        color_ajustado = extraer_color_promedio(area_roi) - color_fondo
        
        # Predicción de concentración
        if modelo_seleccionado == "Regresión Polinómica":
            X_poly = PolynomialFeatures(degree=2).fit_transform([color_ajustado])
            pred = modelo.predict(X_poly)[0]
        else:
            pred = modelo.predict([color_ajustado])[0]
        
        # Calcular R² para cada muestra (como un "coeficiente de estimación")
        r2_muestra = r2_score([concentraciones[0]], [pred])  # Usamos un valor conocido como base para R² de muestra
        
        resultados.append({"Muestra": f"Muestra {i+1}", "Concentración estimada (mg/mL)": round(pred, 2), "R² de estimación": round(r2_muestra, 4)})

    df_resultados = pd.DataFrame(resultados)
    st.subheader("Resultados de concentración estimada")
    st.dataframe(df_resultados)

    # Botón para descargar resultados
    csv = df_resultados.to_csv(index=False)
    st.download_button(label="Descargar resultados en CSV", data=csv, file_name='resultados_fehling.csv', mime='text/csv')
