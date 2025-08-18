# params.py
# Librerias ----------------------------------------

import platform

# Systema operativo ----------------------------------------

sistema_operativo = platform.system()

# Variables de control de ejecucion --------------------------------
# Si requiere prueba de ejecucion de la estructura del proyecto
test_for_run_required = True

# Si se requiere preprocesamiento
preprocess_required = False

# Si se requiere entrenamiento
training_required = False

# Sistema de archivos ----------------------------------------------
# El tipo de archivo parquet mantiene el tipo de dato al ser leido por pandas
FILE_PREPROCESSED_PATH = 'datasets/pre-processed/interconnect_preprocessed.parquet'
MODEL_OUTPUT_PATH = 'files/outputs/models/interconnect_catboost_churn_model.pkl'

# Segmentacion de datos en entrenamiento ---------------------------
validation_size = 0.25
