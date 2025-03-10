# IMPORT PACCHETTI NECESSARI
from ultralytics import YOLO
import os
from PIL import Image

#parametri
epoche=100
lr=0.001
str_lr = str(lr).split('.')[1]
b_s=16
str_batch=str(b_s)
opt='Adam'
freezed=10
str_freeze=str(freezed)

# IMPORT MODELLO YOLOv11 NANO 
model_n = YOLO("./models/yolo11n.pt")

im1 = Image.open("./immagini_test/Bologna.jpg")
new_size = (im1.width // 3, im1.height // 3)
img_resized = im1.resize(new_size)

results = model_n.predict(source=im1, name='predict', save=True)
im1 = Image.open(f"{results[0].save_dir}\\Bologna.jpg")
new_size = (im1.width // 3, im1.height // 3)
img_resized = im1.resize(new_size)

# Durante il training viene applicata una pipeline per la Data augmentation.
# Nello specifico vengono applicate le seguenti trasformazioni:
# *   hue
# *   saturation
# *   brightness
# *   Translates the image
# *   Scales the image
# *   Flips the image  
# *   Mosaic (Combines four training images into one, simulating different scene compositions and object interactions. Highly effective for complex scene understanding.)
# *   Randomly erases a portion of the image during classification training, encouraging the model to focus on less obvious features for recognition
# *   Crops the classification image to a fraction of its size to emphasize central features and adapt to object scales, reducing background distractions
# 
# 

#modello di partenza
dataset_yaml=os.path.join(os.getcwd(),"license_plate\\data.yaml")
#training
results = model_n.train(data=dataset_yaml,
                        epochs=epoche,
                        imgsz=640,
                        exist_ok=True,
                        name=f'train_freezed_first_{str_freeze}_{opt}_LR{str_lr}_BATCH{str_batch}_cls01',
                        optimizer=opt,
                        seed=0,
                        freeze=freezed,
                        lr0=lr,
                        weight_decay=0.0005,
                        batch=b_s, 
                        cls=0.01, #Weight of the classification loss in the total loss function, affecting the importance of correct class prediction relative to other components.
                        patience=(epoche*10)//100  #circa il 10% delle epoche
                        )