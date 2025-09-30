# 01_preprocess.py
import sys
import os
sys.path.append(os.getcwd())

# Librerias ----------------------------------------
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

import params as params
from functions.general_functions import printClassFrequency, displayClassFrequency

pd.set_option('future.no_silent_downcasting', True)

print("Pre-procesando datos...")

df_contract = pd.read_csv("datasets/raw/contract.csv")
df_personal = pd.read_csv("datasets/raw/personal.csv")
df_internet = pd.read_csv("datasets/raw/internet.csv")
df_phone = pd.read_csv("datasets/raw/phone.csv")
print("Datos cargados...")

# ### Tratamiento de dataset Contract
df_temp = df_contract.rename(columns={
    'customerID': 'customer_id',
    'BeginDate': 'begin_date',
    'EndDate': 'end_date',
    'Type': 'type',
    'PaperlessBilling': 'paperless_billing',
    'PaymentMethod': 'payment_method',
    'MonthlyCharges': 'monthly_charges',
    'TotalCharges': 'total_charges'
})

df_temp['paperless_billing'] = pd.to_numeric(
    df_temp['paperless_billing'].replace({'Yes': 1, 'No': 0}), downcast="unsigned")
df_temp["monthly_charges"] = pd.to_numeric(
    df_temp["monthly_charges"], downcast="float")
df_temp['total_charges'] = pd.to_numeric(
    df_temp['total_charges'], errors="coerce", downcast="float")
df_temp["total_charges"] = df_temp["total_charges"].fillna(0)
df_temp["ceased_customer"] = pd.to_numeric(
    (df_temp["end_date"] != 'No').astype(int), downcast="unsigned")
df_temp = df_temp.drop(["begin_date", "end_date"], axis=1)

# ### Tratamiento de dataset Personal

df_merged = pd.merge(df_temp, df_personal, left_on='customer_id',
                        right_on='customerID', how='left', copy='False')

df_temp = df_merged.rename(
    columns={
        'SeniorCitizen': 'senior_citizen',
        'Partner': 'partner',
        'Dependents': 'dependents'
    })

df_temp = df_temp.drop(["customerID"], axis=1)
df_temp['senior_citizen'] = pd.to_numeric(
    df_temp['senior_citizen'], downcast="unsigned")
df_temp['partner'] = pd.to_numeric(df_temp['partner'].replace(
    {'Yes': 1, 'No': 0}), downcast="unsigned")
df_temp['dependents'] = pd.to_numeric(
    df_temp['dependents'].replace({'Yes': 1, 'No': 0}), downcast="unsigned")
df_temp['gender'] = pd.to_numeric(df_temp['gender'].replace(
    {'Male': 1, 'Female': 0}), downcast="unsigned")

# ### Tratamiento de dataset Internet

df_merged = pd.merge(df_temp, df_internet, left_on='customer_id',
                        right_on='customerID', how='left', copy='False')

df_temp = df_merged.rename(
    columns={
        'InternetService': 'internet_service',
        'OnlineSecurity': 'online_security',
        'OnlineBackup': 'online_backup',
        'DeviceProtection': 'device_protection',
        'TechSupport': 'tech_support',
        'StreamingTV': 'streaming_tv',
        'StreamingMovies': 'streaming_movies'
    })

df_temp = df_temp.drop(["customerID"], axis=1)
df_temp["internet_service"] = df_temp["internet_service"].fillna("none")
df_temp["online_security"] = df_temp["online_security"].fillna(0)
df_temp["online_backup"] = df_temp["online_backup"].fillna(0)
df_temp["device_protection"] = df_temp["device_protection"].fillna(0)
df_temp["tech_support"] = df_temp["tech_support"].fillna(0)
df_temp["streaming_tv"] = df_temp["streaming_tv"].fillna(0)
df_temp["streaming_movies"] = df_temp["streaming_movies"].fillna(0)

df_temp['online_security'] = pd.to_numeric(
    df_temp['online_security'].replace({'Yes': 1, 'No': 0}), downcast="unsigned")
df_temp['online_backup'] = pd.to_numeric(
    df_temp['online_backup'].replace({'Yes': 1, 'No': 0}), downcast="unsigned")
df_temp['device_protection'] = pd.to_numeric(
    df_temp['device_protection'].replace({'Yes': 1, 'No': 0}), downcast="unsigned")
df_temp['tech_support'] = pd.to_numeric(
    df_temp['tech_support'].replace({'Yes': 1, 'No': 0}), downcast="unsigned")
df_temp['streaming_tv'] = pd.to_numeric(
    df_temp['streaming_tv'].replace({'Yes': 1, 'No': 0}), downcast="unsigned")
df_temp['streaming_movies'] = pd.to_numeric(
    df_temp['streaming_movies'].replace({'Yes': 1, 'No': 0}), downcast="unsigned")


# Tratamiento de dataset phone

df_merged = pd.merge(df_temp, df_phone, left_on='customer_id',
                     right_on='customerID', how='left', copy='False')

df_temp = df_merged.rename(
    columns={
        'customerID': 'phone_service',
        'MultipleLines': 'multiple_lines'
    }
)
# customerID -> Servirá para indicar si tiene servicio de telefonia o no
df_temp["phone_service"] = df_temp["phone_service"].fillna(0)
# Cualquier valor diferente de 0 (cualquier customerID) indica que si tiene servicio de telefonia
df_temp["phone_service"] = df_temp['phone_service'].mask(df_temp['phone_service'] != 0, 1)

df_temp["multiple_lines"] = df_temp["multiple_lines"].fillna(0)

df_temp['phone_service'] = pd.to_numeric(
    df_temp['phone_service'], downcast="unsigned")
df_temp['multiple_lines'] = pd.to_numeric(
    df_temp['multiple_lines'].replace({"Yes": 1, "No": 0}), downcast="unsigned")

# Seleccionando columnas finales (eliminando las que se consideran innecesarias)
df_temp = df_temp.drop(["gender"], axis=1)

# Division de datos --------------------------------------------------------------------------

data_train, data_test = train_test_split(df_temp, test_size=params.validation_size, random_state=10)

displayClassFrequency(data_train["ceased_customer"])

# Estandarizacion y normalizcion

cat_features = [
    'type',
    'payment_method',
    'internet_service'
]

num_features = [
    'monthly_charges',
    'total_charges'
]


preprocessor = ColumnTransformer(
    transformers=[
        # Aplicar One-Hot Encoding a las características categóricas
        ('onehot', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=2, sparse_output=False), 
         cat_features),
        # Aplicar StandardScaler a las características numéricas
        ('scaler', StandardScaler(), num_features)
    ],
    verbose_feature_names_out=False,
    remainder='passthrough'
)

# Ajuste y transformacion de los datos de entrenamiento
data_train_processed = preprocessor.fit_transform(data_train)

# Transformacion de los datos de prueba
data_test_processed = preprocessor.transform(data_test)

print("Datos procesados...")

# Almacenando el avance del trabajo de tratamiento de los datos en un archivo como checkpoint

data_train_processed = pd.DataFrame(data_train_processed, columns=preprocessor.get_feature_names_out())
data_test_processed = pd.DataFrame(data_test_processed, columns=preprocessor.get_feature_names_out())

data_train_processed.to_parquet(params.FILE_TRAIN_PROCESSED_PATH)
data_test_processed.to_parquet(params.FILE_TEST_PROCESSED_PATH)
print("Checkpoint de archivos guardados...")