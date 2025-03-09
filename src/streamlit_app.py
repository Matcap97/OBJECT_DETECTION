import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image
from ultralytics import YOLO 

#seleziona modello
def model_selector(folder_path='./models'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox("Selezionare il modello per effettuare OBJECT DETECTION:",filenames,index=None,placeholder='Seleziona un modello...')
    return selected_filename

#rimuovo immagine predetta
if os.path.exists("./immagini_test/result.jpg"):
    os.remove("./immagini_test/result.jpg")

st.title("WebAPP OBJECT DETECTION")

model_selected = model_selector()

descrizione = {'yolo11n.pt': ['YOLO v11 NANO','rilevamento 80 classi del dataset COCO','le 80 classi in essa'],
               'yolo11s.pt': ['YOLO v11 SMALL','rilevamento 80 classi del dataset COCO','le 80 classi in essa']
               }

if os.path.exists(f"./models/{model_selected}") and model_selected!=None:
    st.write('Hai selezionato `%s`' % descrizione[model_selected][0])

    #caricamento del modello
    model = YOLO(f"./models/{model_selected}")  # Using a pre-trained YOLOv8 model

    st.title(f"Applicazione per {descrizione[model_selected][1]}")
    st.write(f"Carica un'immagine per rilevare {descrizione[model_selected][2]}")

    imm = st.file_uploader("Scegli immagine...", type=["jpg", "jpeg", "png"])

    if imm:
        image = Image.open(imm)
        #effettuo la predizione e salvo la predizione
        model(image)[0].save(filename="./immagini_test/result.jpg")
        obj_image = Image.open("./immagini_test/result.jpg")
        st.image(obj_image, caption="Risultato", use_container_width=True)
