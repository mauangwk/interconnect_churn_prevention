# params.py
# Librerias ----------------------------------------

import platform

# Sistema operativo ----------------------------------------

sistema_operativo = platform.system()

# Variables de control de ejecucion --------------------------------

# Si se requiere preprocesamiento
preprocess_required = False

# Si se requiere entrenamiento
training_required = True

# Sistema de archivos ----------------------------------------------
# El tipo de archivo parquet mantiene el tipo de dato al ser leido por pandas
FILE_TRAIN_PROCESSED_PATH = 'datasets/pre-processed/data_train_processed.parquet'
FILE_TEST_PROCESSED_PATH = 'datasets/pre-processed/data_test_processed.parquet'

MODEL_OUTPUT_PATH = 'files/outputs/models/interconnect_catboost_churn_model.pkl'
MODEL_METRICS_PATH = 'files/outputs/models/interconnect_catboost_churn_model_metrics.csv'

IMAGES_OUTPUT_PATH = 'files/outputs/images/'

# Segmentacion de datos en entrenamiento ---------------------------
validation_size = 0.25

# Etiquetas para estadisticas
accuracy_label = 'Accuracy Score'
recall_label = 'Recall Score'
f1_label = 'F1 Score'
aps_label = 'APS'
roc_auc_label = 'ROC AUC'