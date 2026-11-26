import time
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Detección de objetos YOLOv8", page_icon="🔎", layout="wide")
st.title("🔎 Detección de objetos con YOLOv8")
st.write("Sube una imagen o toma una foto. El modelo YOLOv8 (entrenado con COCO, 80 clases) "
         "detecta personas, vehículos, animales, objetos de uso diario y más.")

@st.cache_resource
def cargar_modelo(nombre):
    return YOLO(nombre)

# --- Barra lateral ---
st.sidebar.header("⚙️ Parámetros")
nombre_modelo = st.sidebar.selectbox("Modelo", ["yolov8n.pt", "yolov8s.pt"],
                                     help="n = más rápido, s = más preciso")
conf = st.sidebar.slider("Confianza mínima", 0.05, 0.95, 0.25, 0.05)
iou = st.sidebar.slider("Umbral IoU (NMS)", 0.1, 0.9, 0.45, 0.05)
modelo = cargar_modelo(nombre_modelo)
clases = st.sidebar.multiselect("Filtrar clases (vacío = todas)", list(modelo.names.values()))

fuente = st.radio("Fuente", ["📁 Subir imagen", "📷 Cámara"], horizontal=True)
archivo = st.file_uploader("Imagen", type=["jpg", "jpeg", "png"]) if fuente == "📁 Subir imagen" \
          else st.camera_input("Toma una foto")

if archivo is not None:
    img = Image.open(archivo).convert("RGB")
    ids = [k for k, v in modelo.names.items() if v in clases] or None

    t0 = time.time()
    res = modelo.predict(np.array(img), conf=conf, iou=iou, classes=ids, verbose=False)[0]
    ms = (time.time() - t0) * 1000

    col1, col2 = st.columns(2)
    col1.subheader("Original")
    col1.image(img, use_container_width=True)
    col2.subheader("Detecciones")
    col2.image(res.plot()[:, :, ::-1], use_container_width=True)

    n = len(res.boxes)
    c1, c2 = st.columns(2)
    c1.metric("Objetos detectados", n)
    c2.metric("Tiempo de inferencia", f"{ms:.0f} ms")

    if n:
        df = pd.DataFrame({
            "Clase": [modelo.names[int(c)] for c in res.boxes.cls],
            "Confianza": res.boxes.conf.numpy().round(2),
            "Caja (x1, y1, x2, y2)": [tuple(map(int, b)) for b in res.boxes.xyxy.numpy()],
        })
        st.subheader("Detalle")
        st.dataframe(df, use_container_width=True)
        st.subheader("Conteo por clase")
        st.bar_chart(df["Clase"].value_counts())
    else:
        st.info("No se detectaron objetos. Prueba a bajar la confianza mínima.")
