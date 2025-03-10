# IMPORT PACCHETTI NECESSARI
from ultralytics import YOLO
import os
from PIL import Image

#parametri
best_run = "train_freezed_backbone_ADAM001_BATCH16"
b_s=16

# Load best model
best_model = YOLO(os.path.join(os.getcwd(),f"runs\\detect\\{best_run}\\weights\\best.pt"))
testset_yaml=os.path.join(os.getcwd(),"license_plate\\data_test.yaml")
# Customize validation settings
test_results = best_model.val(data=testset_yaml, imgsz=640, batch=b_s, device="0", name='test', exist_ok=True)

print("RISULTATI TEST SET")
print('mAP50-95:',test_results.box.map)
print('mAP50:',test_results.box.map50)  
print('mAP75:',test_results.box.map75)  
print("Recall:",test_results.box.r[0])
print("Precision:",test_results.box.p[0])