# 02_create_model.py

import sys
import os
sys.path.append(os.getcwd())

# Librerias ----------------------------------------
from functions.general_functions import evaluate_model
import params as params
import pandas as pd
import joblib
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score



# Se recupera la informacion pre-procesada del checkpoint
data_train = pd.read_parquet(params.FILE_TRAIN_PROCESSED_PATH)
data_test = pd.read_parquet(params.FILE_TEST_PROCESSED_PATH)


# Separacion de features y target ------------------------------------------------------------

y_train = data_train['ceased_customer']
# customer_id puede ser utilizado para identificar al cliente posterior a las predicciones
x_train = data_train.drop(['ceased_customer', 'customer_id'], axis=1)

y_test = data_test['ceased_customer']
x_test = data_test.drop(['ceased_customer', 'customer_id'], axis=1)


print("\nData procesada cargada... \n")


model = CatBoostClassifier(
    loss_function='Logloss',
    eval_metric='AUC',
    verbose=False,
    random_seed=54321,
    iterations=200,
    learning_rate=0.05,
    depth=4,
    early_stopping_rounds=50
)


# Creamos un Pool de datos para la validación interna de CatBoost.
# Es una forma eficiente de pasar los datos al modelo.
train_pool = Pool(x_train, y_train)
test_pool = Pool(x_test, y_test)

# Entrenamos el modelo. 
# CatBoost automáticamente usa el conjunto de prueba para
# la validación y el early stopping si se lo proporcionas.
print("Entrenando modelo...")
model.fit(train_pool, eval_set=test_pool)

joblib.dump(model, params.MODEL_OUTPUT_PATH)
print(f"Modelo almacenado ({params.MODEL_OUTPUT_PATH})...")

# Obtenemos las probabilidades de la clase positiva (clase 1).
# CatBoost.predict_proba() devuelve las probabilidades para ambas clases.
y_pred_proba_train = model.predict_proba(x_train)[:, 1]
y_pred_proba_test = model.predict_proba(x_test)[:, 1]

predicted_test = model.predict(x_test)

evaluate_model(model, x_train, y_train, x_test, y_test, params.MODEL_METRICS_PATH)

results_df = pd.DataFrame({
    'customer_id': data_test['customer_id'],
    'cease_probability': y_pred_proba_test,
    'ceased': predicted_test
})

print(f"-------------------------------")
print(results_df.sample(10))
