# 02_create_model.py

import sys
import os
sys.path.append(os.getcwd())

# Librerias ----------------------------------------
import params as params
import pandas as pd
import joblib
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from functions.general_functions import printClassFrequency




# Se recupera la informacion desde el archivo donde fue almacenado un checkpoint
try:
    data = pd.read_parquet(params.FILE_PREPROCESSED_PATH)
except Exception as e:
    print(e)

# Separacion de features y target ------------------------------------------------------------

y = data['ceased_customer']
X = data.drop('ceased_customer', axis=1)

# Division de datos --------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params.validation_size, random_state=10)


printClassFrequency(y_train)

print("\n--------------------\n")

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
train_pool = Pool(X_train, y_train)
test_pool = Pool(X_test, y_test)

# Entrenamos el modelo. CatBoost automáticamente usa el conjunto de prueba para
# la validación y el early stopping si se lo proporcionas.
model.fit(train_pool, eval_set=test_pool)

# 4. Hacer predicciones y evaluar el modelo
# Obtenemos las probabilidades de la clase positiva (clase 1).
# CatBoost.predict_proba() devuelve las probabilidades para ambas clases.
y_pred_proba_train = model.predict_proba(X_train)[:, 1]
y_pred_proba_test = model.predict_proba(X_test)[:, 1]

predicted_test = model.predict(X_test)

print("Metricas resultantes:")
print("Precision score: {}".format(precision_score(y_test, predicted_test)))
print("Recall score:    {}".format(recall_score(y_test, predicted_test)))
print("F1 score:        {}".format(f1_score(y_test, predicted_test)))

print(f"ROC AUC Train score:   {roc_auc_score(y_train, y_pred_proba_train):.4f}")
print(f"ROC AUC Test score:   {roc_auc_score(y_test, y_pred_proba_test):.4f}")


results_df = pd.DataFrame({
    'cease_probability': y_pred_proba_test,
    'ceased': predicted_test
})

print(f"-------------------------------")
print(results_df.sample(10))


joblib.dump(model, params.MODEL_OUTPUT_PATH)
print(f"Modelo guardado exitosamente en: {params.MODEL_OUTPUT_PATH}")
