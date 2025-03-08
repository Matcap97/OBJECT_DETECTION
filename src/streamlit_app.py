import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image
from ultralytics import YOLO 

if os.path.exists("./immagini_test/result.jpg"):
    os.remove("./immagini_test/result.jpg")

#caricamento del modello
model = YOLO("../models/yolo11n.pt")  # Using a pre-trained YOLOv8 model

st.title("Applicazione per il rilevamento delle targhe")
st.write("Carica un'immagine per rilevare la targa")

imm = st.file_uploader("Scegli immagine...", type=["jpg", "jpeg", "png"])

if imm:
    image = Image.open(imm)
    #effettuo la predizione e salvo la predizione
    model(image)[0].save(filename="./immagini_test/result.jpg")
    obj_image = Image.open("./immagini_test/result.jpg")
    st.image(obj_image, caption="Risultato", use_container_width=True)
