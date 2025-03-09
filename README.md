# Progetto per il riconoscimento delle targhe di autoveicoli
### OBBIETTIVO: Applicare tecnica del Transfer Learning in modo da poter riconoscere una classe non supportata

#### Analisi classi supportate
Il modello YOLO v11 di base è stato trainato sul dataset COCO contenente 80 classi. Tra le classi ci sono alcuni tipi di autoveicoli come auto e bus.
Sfrutto la presenza di queste classi e scelgo come nuova classe target, la targa di un autoveicolo qualsiasi. Questa scelta è dovuta al fatto che contenendo - il dataset COCO - già immagini di mezzi, il modello pretrainato avrà già la capacità di individuare determinate caratteristiche nelle immagini di autoveicoli. La capacità di individuare determinate caratteristiche nelle immagini mi permette inoltre di 'esaltare' la tecnica del Transfer Learning, infatti anche se congelerò tutto il backbone, il modello avrà già la capacità di riconoscere dettagli e caratteristiche di autoveicoli (e di conseguenza le loro targhe) nelle immagini. 
In questo modo, ho la possibilità di ridurre il costo computazionale riducendo il numero di layer da aggiornare.

### Dataset scelto:
Il datase scelto fa riferimento alla classe target delle targhe di autoveicoli.
Ho scaricato il seguente dataset da
[Roboflow](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e).

VERSIONE: v6 del 2024-10-23 6:33pm
| SET | NUMERO IMMAGINI | PERCENTUALE |
|:---:|:---:|:---:|
|TRAINING SET |7057 | 70% |
|VALIDATION SET |2048 | 20% |
|TEST SET |1020 | 10% |

### Hardware a disposizione:
GPU:  NVIDIA GeForce GTX 1650 Ti, 4096 Mb
RAM:  16 Gb

### DETTAGLI SUL MODELLO
Il modello pretrainato viene scaricato dalla libreria ufficiale di Ultralytics che verrà utilizzata per le operazioni di training, validazione, test ed inference. Dalla documentazione risultano diversi modelli di YOLOv11: nano, small, medium, large, extra. Per questioni di tempo, risorse ed in seguito ad alcune prove decido di utilizzare il modello nano congelando direttamente tutto il backbone del modello. (In questo modo addestro solamente l'head).
Per farlo ho bisogno di verificare la struttura del modello che trovo nello schema YAML del [modello](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/11/yolo11.yaml). Dallo schema apprendo che i layer da congelare sono i primi 10.

NOTA: la scelta delle parti da congelare si valuta in base a dataset(tipo e numero di dati), risorse e tempo a disposizione. Non c'è una regola specifica che consiglia di congelare sempre la backbone. Bisogna fare diverse prove per capire quale sia il compromesso migliore.


#### ADDESTRAMENTO
Il training viene fatto tramite notebook così da avere la possibilità di visualizzare i risultati e poter vedere anche la differenza del modello prima e dopo il trasfer learning.
##### CODICE ADDESTRAMENTO
    model_n = YOLO(model_path)
    results = model_n.train(data=data_folder),
                            epochs=epoche,
                            imgsz=img_sz,
                            exist_ok=True,
                            name=train_folder,
                            optimizer=opt,
                            seed=0,
                            freeze=freeze_layer,
                            lr0=lr,
                            weight_decay=0.0005,
                            batch=16, 
                            patience=(epoche*10)//100  #circa il 10% delle epoche
                            )

#### TUNING IPERPARAMETRI
Il tuning degli iperparametri (BATCH_SIZE, LEARNING_RATE, OPTIMIZER, EPOCHE) viene deciso effettuando diverse prove e valutazioni sul validation set. (per questioni di tempo non si usa k-fold)

##### IPERPARAMETRI STABILITI A PRIORI
PATIENCE (earlystopping): 10%N_EPOCHE, se il modello non migliora sulvalidation set dopo il 10% del numero di epoche totali, blocca il training

IMG_SIZE: 640,640 indica la dimensione che il modello andrà effettivamente ad elaborare, quindi se ci sono immagini con dimensioni diverse saranno resizate

SEED: 0 per la riproducibilità

WHEIGHT_DECAY: 0.0005 penalizza pesi con numeri elevati per prevenire l'overfitting

#### TEST

Raggiunti i risultati sperati su validation set si utilizza il modello sul test per poi trarne le conclusioni

### RISULTATI SUL TEST
| METRICA | SIGNIFICATO | RISULTATI | RIFERIMENTO ROBOFLOW |
|:---:|:---:|:---:|:---:|
| mAP50 | precisione media calcolata con una soglia di intersezione su unione (IoU) pari a 0,50 | | 97.6% |
| mAP50-95 |  la media della precisione media calcolata a diverse soglie IoU, che vanno da 0,50 a 0,95 | | ? |
| Precision | Numero di casi in cui il modello indovina effettivamente una classe rispetto al numero totale di casi predetti con quella classe | | 98.3% |
| Recall |  Numero di casi in cui il modello indovina effettivamente una classe rispetto al numero totale di casi predetti correttamente | | 95.1% |


### EXTRA: APPLICAZIONE WEB

Grazie alla libreria streamlit si crea una semplice app web per caricare e visualizzare in tempo reale i risultati delle predizioni 

    streamlit run .\src\streamlit_app.py 