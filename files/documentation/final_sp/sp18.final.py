# %% [markdown]
# # Hola &#x1F600;,
# 
# Soy **Hesus Garcia** – **"Soy el único Hesus que conoces (y probablemente conocerás) 🌟"** – Sí, como "Jesús", pero con una H que me hace único. Puede sonar raro, pero créeme, ¡no lo olvidarás! Como tu revisor en Triple-Ten, estoy aquí para guiarte y ayudarte a mejorar tu código. Si algo necesita un ajuste, no hay de qué preocuparse; ¡aquí estoy para hacer que tu trabajo brille con todo su potencial! ✨
# 
# Cada vez que encuentre un detalle importante en tu código, te lo señalaré para que puedas corregirlo y así te prepares para un ambiente de trabajo real, donde el líder de tu equipo actuaría de manera similar. Si en algún momento no logras solucionar el problema, te daré más detalles para ayudarte en nuestra próxima oportunidad de revisión.
# 
# Es importante que cuando encuentres un comentario, **no los muevas, no los modifiques, ni los borres**.
# 
# ---
# 
# ### Formato de Comentarios
# 
# Revisaré cuidadosamente cada implementación en tu notebook para asegurar que cumpla con los requisitos y te daré comentarios de acuerdo al siguiente formato:
# 
# 
# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a><br>
#     
# <b>Éxito</b> - ¡Excelente trabajo! Esta parte está bien implementada y contribuye significativamente al análisis de datos o al proyecto. Continúa aplicando estas buenas prácticas en futuras secciones.
#     
# </div>
# 
# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor</b> <a class="tocSkip"></a><br>
#     
# <b>Atención</b> ⚠️ - Este código está correcto, pero se puede optimizar. Considera implementar mejoras para que sea más eficiente y fácil de leer. Esto fortalecerá la calidad de tu proyecto.
#     
# </div>
# 
# <div class="alert alert-block alert-danger">
# <b>Comentario del revisor</b> <a class="tocSkip"></a><br>
#     
# <b>A resolver</b> ❗ - Aquí hay un problema o error en el código que es necesario corregir para aprobar esta sección. Por favor, revisa y corrige este punto, ya que es fundamental para la validez del análisis y la precisión de los resultados.
#     
# </div>
# 
# ---
# 
# Al final de cada revisión, recibirás un **Comentario General del Revisor** que incluirá:
# 
# - **Aspectos positivos:** Un resumen de los puntos fuertes de tu proyecto.
# - **Áreas de mejora:** Sugerencias sobre aspectos donde puedes mejorar.
# - **Temas adicionales para investigar:** Ideas de temas opcionales que puedes explorar por tu cuenta para desarrollar aún más tus habilidades.
# 
# Estos temas adicionales no son obligatorios en esta etapa, pero pueden serte útiles para profundizar en el futuro.
# 
# ---
# 
# 
# Esta estructura en viñetas facilita la lectura y comprensión de cada parte del comentario final.
# 
# También puedes responderme de la siguiente manera si tienes alguna duda o quieres aclarar algo específico:
# 
# 
# <div class="alert alert-block alert-info">
# <b>Respuesta del estudiante</b> <a class="tocSkip"></a>
#     
# Aquí puedes escribir tu respuesta o pregunta sobre el comentario.
#     
# </div>
# 
# 
# **¡Empecemos!** &#x1F680;
# 

# %% [markdown]
# # Analisis para Interconect (Telecom/Prevencion de Churn)

# %% [markdown]
# ## Descripcion del problema

# %% [markdown]
# Al operador de telecomunicaciones Interconnect le gustaría poder pronosticar su tasa de cancelación de clientes. 
# Si se descubre que un usuario o usuaria planea irse, se le ofrecerán códigos promocionales y opciones de planes especiales. El equipo de marketing de Interconnect ha recopilado algunos de los datos personales de sus clientes, incluyendo información sobre sus planes y contratos.
# 
# ### Servicios de Interconnect
# 
# Interconnect proporciona principalmente dos tipos de servicios:
# 
# 1. Comunicación por teléfono fijo. El teléfono se puede conectar a varias líneas de manera simultánea.
# 2. Internet. La red se puede configurar a través de una línea telefónica (DSL, *línea de abonado digital*) o a través de un cable de fibra óptica.
# 
# Algunos otros servicios que ofrece la empresa incluyen:
# 
# - Seguridad en Internet: software antivirus (*ProtecciónDeDispositivo*) y un bloqueador de sitios web maliciosos (*SeguridadEnLínea*).
# - Una línea de soporte técnico (*SoporteTécnico*).
# - Almacenamiento de archivos en la nube y backup de datos (*BackupOnline*).
# - Streaming de TV (*StreamingTV*) y directorio de películas (*StreamingPelículas*)
# 
# La clientela puede elegir entre un pago mensual o firmar un contrato de 1 o 2 años. Puede utilizar varios métodos de pago y recibir una factura electrónica después de una transacción.
# 
# ### Descripción de los datos
# 
# Los datos consisten en archivos obtenidos de diferentes fuentes:
# 
# - `contract.csv` — información del contrato;
# - `personal.csv` — datos personales del cliente;
# - `internet.csv` — información sobre los servicios de Internet;
# - `phone.csv` — información sobre los servicios telefónicos.
# 
# En cada archivo, la columna `customerID` (ID de cliente) contiene un código único asignado a cada cliente. La información del contrato es válida a partir del 1 de febrero de 2020.

# %% [markdown]
# ## Inicializacion

# %% [markdown]
# ### Carga de librerias

# %%
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

import sklearn.metrics as metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.utils import shuffle

from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

import lightgbm as lgb

import matplotlib.pyplot as plt
import seaborn as sns


#from sklearn.datasets import make_classification


# %% [markdown]
# ### Variables

# %%
validation_size = 0.25

# %% [markdown]
# <div class="alert alert-block alert-success"> <b>Comentario del revisor</b> <a class="tocSkip"></a><br> <b>Éxito</b> - Buena selección y organización de librerías, cubriendo desde la manipulación de datos hasta el modelado y visualización. Una base sólida para el desarrollo del proyecto.</div>
# 

# %% [markdown]
# ## Funciones

# %%
"""
 Analiza si el customerID se encuentra mas de una vez en el data set lo que implicaría que es una relacion 1:n
"""
def analizaDatosLlave(df):
    customer_counts = df['customerID'].value_counts()
    duplicate_ids = customer_counts[customer_counts > 1].index.tolist()
    
    print("customerIDs que aparecen más de una vez:", duplicate_ids)
    print("-" * 30)

# %%

def generaGraficaCorr(data):
    # Graficando la correlacion entre variables numericas Posterior a la estandarizacion 
    corr = data.corr()  # Calculate the correlation matrix
    sns.heatmap(corr, annot=False, cmap='coolwarm')  # Create a heatmap
    plt.show()

# %%
def displayClassFrequency(y_train):
    class_frequency = y_train.value_counts(normalize=True)
    print("Normalized Class Frequency:")
    print(class_frequency)
    class_frequency.plot(kind='bar')

# %%
# A function to generate oversampling
def generate_oversamples(features, target, nrepeat):

    if (target[target == 0].count()<target[target == 1].count()):
        target_minority_class = target[target == 0]
        target_majority_class = target[target == 1]
        features_minority_class = features[target == 0]
        features_majority_class = features[target == 1]
    else:
        target_minority_class = target[target == 1]
        target_majority_class = target[target == 0]
        features_minority_class = features[target == 1]
        features_majority_class = features[target == 0]

    diff = 0
    if(nrepeat==0):
        nrepeat = int(target_majority_class.count()/target_minority_class.count())
        diff = target_majority_class.count() % target_minority_class.count()

    features_upsampled = pd.concat(
        [features_majority_class] + 
        [features_minority_class] * nrepeat 
    )
    
    target_upsampled = pd.concat(
        [target_majority_class] + 
        [target_minority_class] * nrepeat 
    )

    if diff>0:
        features_upsampled = pd.concat(
            [features_upsampled] + [features_minority_class.sample(diff, random_state=12345)]
        )
        target_upsampled = pd.concat(
            [target_upsampled] + [target_minority_class.sample(diff, random_state=12345)]
        )
        

    features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=12345
    )

    return features_upsampled, target_upsampled

# %%
# A function to generate a undersampling
def undersample(features, target, fraction):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    if(target_zeros.count()>target_ones.count()):
        features_downsampled = pd.concat(
            [features_zeros.sample(frac=fraction, random_state=12345)]+[features_ones]
        )
        target_downsampled = pd.concat(
            [target_zeros.sample(frac=fraction, random_state=12345)]+[target_ones]
        )
    else:
        features_downsampled = pd.concat(
            [features_ones.sample(frac=fraction, random_state=12345)]+[features_zeros]
        )
        target_downsampled = pd.concat(
            [features_ones.sample(frac=fraction, random_state=12345)]+[features_zeros]
        )

    features_downsampled, target_downsampled = shuffle(
        features_downsampled, target_downsampled, random_state=12345
    )

    return features_downsampled, target_downsampled

# %%
def selectBestModel(features_train, target_train, strategy_label):
    best_score = 0
    the_best_model = {}
    for model_name, model in models.items():
        model_grid_params = grid_params[model_name]
        search = GridSearchCV(model,
                              #scoring='f1',
                              scoring='roc_auc',
                              param_grid=model_grid_params,
                              cv=5,
                              n_jobs=-1)
        search.fit(features_train, target_train)

        if search.best_score_ > best_score:
            the_best_model["strategy_label"] = strategy_label
            the_best_model["best_estimator"] = search.best_estimator_
            the_best_model["best_score"] = search.best_score_
            the_best_model["best_params"] = search.best_params_
            best_score = search.best_score_

    return the_best_model

# %%
def evaluate_model(model, train_features, train_target, test_features, test_target):
    
    eval_stats = {}
    
    fig, axs = plt.subplots(1, 3, figsize=(20, 6)) 
    
    for type, features, target in (('train', train_features, train_target), ('test', test_features, test_target)):
        
        eval_stats[type] = {}
    
        pred_target = model.predict(features)
        pred_proba = model.predict_proba(features)[:, 1]
        
        # F1
        f1_thresholds = np.arange(0, 1.01, 0.05)
        f1_scores = [metrics.f1_score(target, pred_proba>=threshold) for threshold in f1_thresholds]
        
        # ROC
        fpr, tpr, roc_thresholds = metrics.roc_curve(target, pred_proba)
        roc_auc = metrics.roc_auc_score(target, pred_proba)    
        eval_stats[type]['ROC AUC'] = roc_auc

        # PRC
        precision, recall, pr_thresholds = metrics.precision_recall_curve(target, pred_proba)
        aps = metrics.average_precision_score(target, pred_proba)
        eval_stats[type]['APS'] = aps
        
        if type == 'train':
            color = 'blue'
        else:
            color = 'green'

        # Valor F1
        ax = axs[0]
        max_f1_score_idx = np.argmax(f1_scores)
        ax.plot(f1_thresholds, f1_scores, color=color, label=f'{type}, max={f1_scores[max_f1_score_idx]:.2f} @ {f1_thresholds[max_f1_score_idx]:.2f}')
        # establecer cruces para algunos umbrales        
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(f1_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(f1_thresholds[closest_value_idx], f1_scores[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('threshold')
        ax.set_ylabel('F1')
        ax.legend(loc='lower center')
        ax.set_title(f'Valor F1') 

        # ROC
        ax = axs[1]    
        ax.plot(fpr, tpr, color=color, label=f'{type}, ROC AUC={roc_auc:.2f}')
        # establecer cruces para algunos umbrales        
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(roc_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'            
            ax.plot(fpr[closest_value_idx], tpr[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower center')        
        ax.set_title(f'Curva ROC')
        
        # PRC
        ax = axs[2]
        ax.plot(recall, precision, color=color, label=f'{type}, AP={aps:.2f}')
        # establecer cruces para algunos umbrales        
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(pr_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(recall[closest_value_idx], precision[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('recall')
        ax.set_ylabel('precision')
        ax.legend(loc='lower center')
        ax.set_title(f'PRC')        
        
        eval_stats[type]['Accuracy'] = metrics.accuracy_score(target, pred_target)
        eval_stats[type]['F1'] = metrics.f1_score(target, pred_target)
    
    df_eval_stats = pd.DataFrame(eval_stats)
    df_eval_stats = df_eval_stats.round(2)
    df_eval_stats = df_eval_stats.reindex(index=('Exactitud', 'F1', 'APS', 'ROC AUC'))
    
    print(df_eval_stats)
    
    return

# %% [markdown]
# <div class="alert alert-block alert-success"> <b>Comentario del revisor</b> <a class="tocSkip"></a><br> <b>Éxito</b> - Muy buena implementación de funciones clave para el análisis y evaluación de modelos, con una estructura clara que cubre desde la exploración de datos hasta métricas avanzadas de desempeño.</div>
# 

# %% [markdown]
# ## Carga de datos

# %%
df_contract = pd.read_csv("/datasets/final_provider/contract.csv")
df_personal = pd.read_csv("/datasets/final_provider/personal.csv")
df_internet = pd.read_csv("/datasets/final_provider/internet.csv")
df_phone = pd.read_csv("/datasets/final_provider/phone.csv")

# %% [markdown]
# <div class="alert alert-block alert-success"> <b>Comentario del revisor</b> <a class="tocSkip"></a><br> <b>Éxito</b> - Correcta carga de los datasets necesarios, estableciendo una base ordenada para el análisis posterior.</div>
# 

# %% [markdown]
# ## Analisis Exploratorio de Datos (EDA)

# %% [markdown]
# ### Contracts (dataset)

# %%
df_contract.info()

# %%
df_contract.sample(15)

# %%
df_contract.drop_duplicates().info()

# %%
df_contract.describe()

# %%
df_contract["BeginDate"].unique()

# %%
df_contract[df_contract["BeginDate"]>="2020-01-01"]["BeginDate"].value_counts()

# %%
df_contract[df_contract["EndDate"]>="2020-01-01"]["EndDate"].value_counts()

# %%
print(df_contract['Type'].sort_values().unique())
print(df_contract['PaperlessBilling'].sort_values().unique())
print(df_contract['PaymentMethod'].sort_values().unique())
print(df_contract['EndDate'].sort_values().unique())

# %%
# Convertir la serie a numérica con manejo de errores
s_numeric = pd.to_numeric(df_contract["TotalCharges"], errors='coerce')

# Verificar si hay valores NaN (no numéricos)
has_non_numeric = s_numeric.isna().any()

print("\n¿Contiene la serie valores no numéricos (NaN)?", has_non_numeric)

# %%
df_contract[s_numeric.isna()]["TotalCharges"].value_counts()

# %%
df_contract[s_numeric.isna()].sample(11)

# %% [markdown]
# <div class="alert alert-block alert-success"> <b>Comentario del revisor</b> <a class="tocSkip"></a><br> <b>Éxito</b> - El análisis exploratorio inicial es detallado, identificando tipos de datos, valores únicos y casos atípicos como registros con `TotalCharges` vacíos, lo que demuestra un enfoque minucioso hacia la calidad de los datos.</div>
# 

# %% [markdown]
# ### Personal (dataset)

# %%
df_personal.info()

# %%
df_personal.sample(10)

# %%
df_personal.drop_duplicates().info()

# %%
analizaDatosLlave(df_personal)

# %% [markdown]
# <div class="alert alert-block alert-success"> <b>Comentario del revisor</b> <a class="tocSkip"></a><br> <b>Éxito</b> - Revisión correcta del dataset personal, confirmando integridad de registros y ausencia de duplicados en las llaves primarias.</div>
# 

# %% [markdown]
# ### Internet (dataset)

# %%
df_internet.info()

# %%
df_internet.sample(10)

# %%
analizaDatosLlave(df_internet)

# %%
df_internet["InternetService"].unique()

# %% [markdown]
# <div class="alert alert-block alert-success"> <b>Comentario del revisor</b> <a class="tocSkip"></a><br> <b>Éxito</b> - Análisis claro del dataset de internet, verificando unicidad de claves y distribución de tipos de servicio, lo que prepara bien el terreno para integraciones posteriores.</div>
# 

# %% [markdown]
# ### Phone (dataset)

# %%
df_phone.info()

# %%
df_phone.sample(10)

# %%
df_phone["MultipleLines"].unique()

# %%
analizaDatosLlave(df_phone)

# %% [markdown]
# <div class="alert alert-block alert-success"> <b>Comentario del revisor</b> <a class="tocSkip"></a><br> <b>Éxito</b> - Correcta validación de la estructura y valores del dataset telefónico, confirmando que las llaves son únicas y que la variable categórica está bien definida.</div>
# 

# %% [markdown]
# ## Entendiendo los datos y el problema

# %% [markdown]
# En cuanto al problema, se entiende que a partir de los datos muestra, se busca anticiparse a que un cliente cancele. 
# 
# Algunos asumptions a aclarar con el equipo:
# - Se manejará como variable objetivo la variable enddate del dataset contract
# - En caso de que esta tenga un valor indica que el cliente ya ha cancelado
# - El objetivo seria poder estimar la cancelacion de los clientes para poder cambiar su situacion y poder retenerlo
# - Se descartará la fecha del 2020-02-01. Aplicable a cualquiera de las caracteristicas (BeginDate, EndDate) resulta en mantener un minimo de datos lo que no apoyaría al modelo.
# 
# Algunos comentarios a partir del EDA ya que es necesario realizar un preprocesamiento de datos:
# - Se deberá incluir el manejo de variables categoricas y la normalización de datos
# - Se debe determinar si se descartan datos pues habrá datos nulos al relacionar la información
# 
# 
# De lo anterior se sugieren las siguientes fases con los datos como plan de trabajo:

# %% [markdown]
# ## Preprocesamiento de datos

# %% [markdown]
# ### Tratamiento de dataset Contract

# %%
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

# %%
df_temp['paperless_billing'] = pd.to_numeric(df_temp['paperless_billing'].replace({'Yes': 1, 'No': 0}), downcast="unsigned")
df_temp["monthly_charges"] = pd.to_numeric(df_temp["monthly_charges"], downcast="float")
df_temp['total_charges'] = pd.to_numeric(df_temp['total_charges'], errors="coerce", downcast="float")

# %%
df_temp["total_charges"] = df_temp["total_charges"].fillna(0)

# %%
df_temp[df_temp["end_date"] == 'No'].head()

# %%
df_temp["ceased_customer"] = pd.to_numeric((df_temp["end_date"]!='No').astype(int), downcast="unsigned")

# %%
df_temp.head()

# %%
df_temp = df_temp.drop(["begin_date", "end_date"], axis = 1)

# %%
df_temp.info()

# %% [markdown]
# <div class="alert alert-block alert-success"> <b>Comentario del revisor</b> <a class="tocSkip"></a><br> <b>Éxito</b> - Excelente planteamiento del problema y preprocesamiento inicial, creando la variable objetivo y dejando un dataset limpio y coherente para el modelado.</div>
# 

# %% [markdown]
# ### Tratamiento de dataset Personal

# %%
df_merged = pd.merge(df_temp, df_personal, left_on='customer_id', right_on='customerID', how='left', copy='False')

# %%
df_merged.info()

# %%
df_temp = df_merged.rename(
    columns={
        'SeniorCitizen': 'senior_citizen', 
        'Partner': 'partner', 
        'Dependents': 'dependents'
    })

# %%
df_temp = df_temp.drop(["customerID"], axis = 1)

# %%
df_temp['senior_citizen'] = pd.to_numeric(df_temp['senior_citizen'], downcast="unsigned")
df_temp['partner'] = pd.to_numeric(df_temp['partner'].replace({'Yes': 1, 'No': 0}), downcast="unsigned")
df_temp['dependents'] = pd.to_numeric(df_temp['dependents'].replace({'Yes': 1, 'No': 0}), downcast="unsigned")
df_temp['gender'] = pd.to_numeric(df_temp['gender'].replace({'Male': 1, 'Female': 0}), downcast="unsigned")

# %%
df_temp.info()

# %%
df_temp.describe()

# %% [markdown]
# <div class="alert alert-block alert-success"> <b>Comentario del revisor</b> <a class="tocSkip"></a><br> <b>Éxito</b> - Integración correcta del dataset personal con el contractual, aplicando transformaciones y codificaciones limpias que dejan las variables listas para el análisis y modelado.</div>
# 

# %% [markdown]
# ### Tratamiento de dataset Internet

# %%
df_merged = pd.merge(df_temp, df_internet, left_on='customer_id', right_on='customerID', how='left', copy='False')

# %%
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

# %%
df_temp = df_temp.drop(["customerID"], axis=1)

# %%
df_temp["internet_service"] = df_temp["internet_service"].fillna("none") # Replace all NaN with "none"
df_temp["online_security"] = df_temp["online_security"].fillna(0)
df_temp["online_backup"] = df_temp["online_backup"].fillna(0)
df_temp["device_protection"] = df_temp["device_protection"].fillna(0)
df_temp["tech_support"] = df_temp["tech_support"].fillna(0)
df_temp["streaming_tv"] = df_temp["streaming_tv"].fillna(0)
df_temp["streaming_movies"] = df_temp["streaming_movies"].fillna(0)

# %%
df_temp['online_security'] = pd.to_numeric(df_temp['online_security'].replace({'Yes': 1, 'No': 0}), downcast="unsigned")
df_temp['online_backup'] = pd.to_numeric(df_temp['online_backup'].replace({'Yes': 1, 'No': 0}), downcast="unsigned")
df_temp['device_protection'] = pd.to_numeric(df_temp['device_protection'].replace({'Yes': 1, 'No': 0}), downcast="unsigned")
df_temp['tech_support'] = pd.to_numeric(df_temp['tech_support'].replace({'Yes': 1, 'No': 0}), downcast="unsigned")
df_temp['streaming_tv'] = pd.to_numeric(df_temp['streaming_tv'].replace({'Yes': 1, 'No': 0}), downcast="unsigned")
df_temp['streaming_movies'] = pd.to_numeric(df_temp['streaming_movies'].replace({'Yes': 1, 'No': 0}), downcast="unsigned")

# %%
df_temp.info()

# %% [markdown]
# <div class="alert alert-block alert-success"> <b>Comentario del revisor</b> <a class="tocSkip"></a><br> <b>Éxito</b> - Muy buena integración y codificación de variables del dataset de internet, manejando valores nulos de forma controlada y asegurando consistencia en el tipo de datos.</div>
# 

# %% [markdown]
# ### Tratamiento de dataset phone

# %%
df_merged = pd.merge(df_temp, df_phone, left_on='customer_id', right_on='customerID', how='left', copy='False')

# %%
df_merged.info()

# %%
df_temp = df_merged.rename(
    columns={
        'customerID': 'phone_service',    # Servirá para indicar si tiene servicio de telefonia o no
        'MultipleLines': 'multiple_lines'
    })

# %%
df_temp["phone_service"] = df_temp["phone_service"].fillna(0)
df_temp["multiple_lines"] = df_temp["multiple_lines"].fillna(0)

# %%
df_temp["phone_service"] = df_temp['phone_service'].mask(df_temp['phone_service'] != 0, 1)

# %%
df_temp['phone_service'] = pd.to_numeric(df_temp['phone_service'], downcast="unsigned")
df_temp['multiple_lines'] = pd.to_numeric(df_temp['multiple_lines'].replace({"Yes":1, "No":0}), downcast="unsigned")

# %%
df_temp["phone_service"].value_counts()

# %%
df_temp["multiple_lines"].value_counts()

# %%
df_temp.info()

# %% [markdown]
# <div class="alert alert-block alert-success"> <b>Comentario del revisor</b> <a class="tocSkip"></a><br> <b>Éxito</b> - Integración y codificación del dataset telefónico bien lograda, con un manejo adecuado de valores nulos y creación de variables indicadoras que enriquecen el análisis.</div>
# 

# %% [markdown]
# ### Visualizacion de resultado

# %%
df_temp.sample(10)

# %%
df_temp.describe()

# %% [markdown]
# <div class="alert alert-block alert-success"> <b>Comentario del revisor</b> <a class="tocSkip"></a><br> <b>Éxito</b> - El dataset final muestra una integración completa y consistente, con todas las variables correctamente transformadas y listas para el análisis y modelado.</div>
# 

# %% [markdown]
# ### Estandarizacion y normalizcion

# %%
cat_features = [
    'type', 
    'payment_method',
    'internet_service'
]

num_features = [
    'monthly_charges', 
    'total_charges'
]

data = df_temp.copy()

# %%
scaler = StandardScaler()
scaler.fit(data[num_features])
data[num_features] = scaler.transform(data[num_features])

# %%
data[num_features].describe()

# %%
# OneHotEncoder
data = pd.get_dummies(data, 
                      columns=cat_features, 
                     # drop_first=True
                     )

# %%
data.info()

# %%
data.head()

# %% [markdown]
# <div class="alert alert-block alert-success"> <b>Comentario del revisor</b> <a class="tocSkip"></a><br> <b>Éxito</b> - La estandarización de variables numéricas y la codificación one-hot de categóricas están bien aplicadas, dejando el dataset final en un formato óptimo para el entrenamiento de modelos.</div>
# 

# %% [markdown]
# ### Seleccionando columnas finales

# %% [markdown]
# #### Grafica de correlacion

# %%
generaGraficaCorr(data)

# %% [markdown]
# <div class="alert alert-block alert-success"> <b>Comentario del revisor</b> <a class="tocSkip"></a><br> <b>Éxito</b> - La visualización de la matriz de correlación permite una comprensión rápida de las relaciones entre variables, facilitando la detección de posibles redundancias o patrones relevantes para el modelado.</div>
# 

# %% [markdown]
# #### Matriz de correlacion (Pearson)

# %%
matriz_correlacion = data.corr(method='pearson')

# %%
# Seleccionamos la columna 'variable_objetivo' para ver sus correlaciones con las demás
correlaciones_con_objetivo = matriz_correlacion['ceased_customer']

print("Correlación de cada característica con la variable objetivo:")
print(correlaciones_con_objetivo)

# %%
# Filtrando características basándose en un umbral
umbral = 0.05
caracteristicas_seleccionadas = correlaciones_con_objetivo[abs(correlaciones_con_objetivo) < umbral] #.index.tolist()

print(caracteristicas_seleccionadas)

# %% [markdown]
# #### Observaciones
# 
# Basado en el umbral, se opta por descartar las siguientes variables:

# %%
data = data.drop(["gender", 
                  #"online_backup", "device_protection",
                  #"streaming_tv", "streaming_movies",
#                  "phone_service", "multiple_lines",
                  #"payment_method_Mailed check"
                 ], axis = 1)
data = data.drop(["customer_id"], axis = 1)

# %%
data.duplicated().value_counts()

# %%
displayClassFrequency(data["ceased_customer"])

# %% [markdown]
# <div class="alert alert-block alert-success"> <b>Comentario del revisor</b> <a class="tocSkip"></a><br> <b>Éxito</b> - El análisis de correlaciones está bien fundamentado y respalda la selección de variables, manteniendo un dataset depurado y equilibrado para el modelado.</div>
# 

# %% [markdown]
# ## Generación de un checkpoint

# %% [markdown]
# ### Almacenamiento de la informacion

# %%
# Almacenando el avance del trabajo de tratamiento de los datos en un archivo como checkpoint
try:
    data.to_parquet('interconnect.parquet')
except Exception as e:
    print(e)

# %% [markdown]
# ### Recuperación de la informacion

# %%
# Se recupera la informacion desde el archivo donde fue almacenado un checkpoint
try:
    data = pd.read_parquet('interconnect.parquet') 
except Exception as e:
    print(e)

# %% [markdown]
# ## Segmentacion de datos

# %% [markdown]
# ### Separacion de features y target

# %%
y = data['ceased_customer']
X = data.drop('ceased_customer', axis=1)

# %% [markdown]
# ### Division de datos

# %%
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=validation_size, random_state=10)

# %% [markdown]
# <div class="alert alert-block alert-success"> <b>Comentario del revisor</b> <a class="tocSkip"></a><br> <b>Éxito</b> - La segmentación de datos y la separación de variables se realiza de forma ordenada, dejando listo el conjunto de entrenamiento y validación para el modelado.</div>
# 

# %% [markdown]
# ## Selección y Evaluación de Modelos

# %% [markdown]
# ### Definicion de modelos

# %%
# Define the models
models = {
    'DummyClassifier': DummyClassifier(random_state=54321),
    'LogisticRegression': LogisticRegression(random_state=54321),
    'DecisionTreeClassifier': DecisionTreeClassifier(random_state=54321),
    'RandomForestClassifier': RandomForestClassifier(random_state=54321),
    'LGBMClassifier': lgb.LGBMClassifier(objective='binary', random_state=54321),
    'CatBoost': CatBoostClassifier(loss_function='Logloss', eval_metric='AUC', verbose=False, random_seed=54321)
}

# Define the grid parameters for each model
grid_params = {
    "LogisticRegression": {
        "solver": ["liblinear", "lbfgs", "newton-cg"]  
    },
    "DummyClassifier":{
        "strategy": ['most_frequent']
    },
    'DecisionTreeClassifier': {
        'max_depth': [None, 10, 15, 20, 25, 30, 35, 40]
    },
    'RandomForestClassifier':{
        'n_estimators': [20, 30, 40, 50, 60, 70, 80, 90, 100],
        'class_weight': [None, 'balanced', 'balanced_subsample']
    },
    'LGBMClassifier': {
        'num_leaves': [30, 40], #  np.arange(20, 50, 10),  # Número de hojas en un árbol
        'learning_rate': [0.05],  # Tasa de aprendizaje
        'n_estimators': [150, 160, 170], #np.arange(50, 200, 50), # Número de árboles de refuerzo
        'max_depth': [-1, 5],  # Profundidad máxima del árbol (-1 significa sin límite)
        #'reg_alpha': [0, 0.1, 0.5, 1], # Regularización L1
        #'reg_lambda': [0, 0.1, 0.5, 1], # Regularización L2
        #'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0], # Submuestreo de columnas al construir cada árbol
        #'subsample': [0.6, 0.7, 0.8, 0.9, 1.0] # Submuestreo de datos al construir cada árbol
    },
    'CatBoost': {
        'iterations': [100, 200],
        'learning_rate': [0.05],
        'depth': [4, 5, 6]
    }
}

# %% [markdown]
# <div class="alert alert-block alert-success"> <b>Comentario del revisor</b> <a class="tocSkip"></a><br> <b>Éxito</b> - Buena definición del conjunto de modelos y configuración de grids, con atención al desbalance de clases para una evaluación rigurosa.</div>
# 

# %% [markdown]
# ### Evaluando modelos

# %% [markdown]
# #### Clase objetivo desbalanceada (Imbalanced)

# %%
displayClassFrequency(y_train)

# %%
%%capture
the_best_model_dictionary = selectBestModel(X_train, y_train, "imbalanced")

# %%
print(the_best_model_dictionary)

# %%
model = the_best_model_dictionary["best_estimator"]
probabilities_one_train = model.predict_proba(X_train)[:, 1]
probabilities_one_valid = model.predict_proba(X_validation)[:, 1]
predicted_valid = model.predict(X_validation)

print("Precision score: {}".format(precision_score(y_validation, predicted_valid)))
print("Recall score:    {}".format(recall_score(y_validation, predicted_valid)))
print("F1 score:        {}".format(f1_score(y_validation, predicted_valid)))
print("ROC AUC Train score:   {}".format(roc_auc_score(y_train, probabilities_one_train)))
print("ROC AUC Test score:   {}".format(roc_auc_score(y_validation, probabilities_one_valid)))

# %%
evaluate_model(model, X_train, y_train, X_validation, y_validation)

# %%
fpr, tpr, thresholds = roc_curve(y_validation, probabilities_one_valid)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve')
plt.show()

# %% [markdown]
# #### Balanceando clase objetivo (Oversampling)

# %%
X_train_upsampled, y_train_upsampled = generate_oversamples(
    X_train, y_train, 0
)

# %%
displayClassFrequency(y_train_upsampled)

# %%
%%capture
the_best_model_dictionary = selectBestModel(X_train_upsampled, y_train_upsampled, "oversampling")

# %%
print(the_best_model_dictionary)

# %%
model = the_best_model_dictionary["best_estimator"]
probabilities_one_valid = model.predict_proba(X_validation)[:, 1]
predicted_valid = model.predict(X_validation)

print("Precision score: {}".format(precision_score(y_validation, predicted_valid)))
print("Recall score:    {}".format(recall_score(y_validation, predicted_valid)))
print("F1 score:        {}".format(f1_score(y_validation, predicted_valid)))
print("ROC AUC score:   {}".format(roc_auc_score(y_validation, probabilities_one_valid)))

# %%
evaluate_model(model, X_train_upsampled, y_train_upsampled, X_validation, y_validation)

# %%
fpr, tpr, thresholds = roc_curve(y_validation, probabilities_one_valid)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve')
plt.show()

# %% [markdown]
# <div class="alert alert-block alert-success"> <b>Comentario del revisor</b> <a class="tocSkip"></a><br> <b>Éxito</b> - La evaluación de modelos incluye tanto escenarios con clases desbalanceadas como balanceadas, aplicando métricas clave y comparando estrategias de forma estructurada para seleccionar la opción más robusta.</div>
# 

# %% [markdown]
# ## Conclusiones Generales

# %% [markdown]
# Para el problema de Interconect se ha solicitado maximizar la metrica ROC-AUC. Es importante tener una buen desempeño con el modelo para tratar de predecir (incluso con la metrica de precision) cuando un cliente es suceptible a cancelar el servicio con la empresa, de esta forma se pueden tomar medidas de retención de clientes y con ello reduccion de costos a largo plazo.
# 
# Este problema requirió un gran desarrollo en los diferentes aspectos para mejorar el desempeño de modelos. A continuación se describen las acciones que se llevaron a cabo durante el proceso siguiendo con el plan a partir de un primer EDA.
# 
# - Tratamiento de los datos (preprocesamiento). Se consideró la mejor calidad posible para los datos y llevandolos a una estandarizacion para la adecuada convivencia con los diferentes algoritmos disponibles.
# - Se utilizó la tectnica de wrapping (RFE) para descartar o mantener caracteristicas, esto se hizo en varias iteraciones (combinaciones) lo que indicó que se tenia mayor beneficio para la metrica manteniendo la mayoria de las caracteristicas (basado en el umbral del coeficiente de correlacion).
# - Balanceo de clase objetivo. Se optó por utilizar oversampling preservando los datos y aumentando las muestras de la clase minoritaria. Dentro de esto mismo se realizaron pruebas equilibrando pruebas o solo aumentando la clase minoritaria.
# - Se probó tambien ajustando el tamaño del set de datos para validacion (35%-25%)
# - En la evaluacion de modelos se incluyeron los siguientes algoritmos: RandomForest, LGBMClassifier, CatBoost y Regresión Logística, usando ROC-AUC como métrica. Se consideró CatBoost en el conjunto de algoritmos evaluados debido a que es de los que manejan bien el desbalance de clases y las características categóricas.
# - Dentro de las pruebas se incluyó igualmente el ajuste de hiperparametros tratando de mediar entre el mejor resultado y la optimizacion del modelo debido al poder de computo y tiempo requeridos.
# 
# Como seleccion final de los modelos se encontro el siguiente bajo los datos desbalanceados (datos originales) y que arrojó un mejor equilibrio entre los datos de entrenamiento y de prueba:
# - 'best_estimator': catboost.core.CatBoostClassifier
# - ROC AUC Train score:   0.8722915954241087
# - ROC AUC Test score:   0.8404646900378719
# - 'best_params': {'depth': 4, 'iterations': 200, 'learning_rate': 0.05}
# 
# Finalmente, puedo decir que fue un reto el explorar todas las aristas posibles para lograr el mejor rendimiento del modelo seleccionado.

# %% [markdown]
# <div class="alert alert-block alert-success"> <b>Comentario del revisor</b> <a class="tocSkip"></a><br> <b>Éxito</b> - Conclusión bien estructurada que resume el flujo completo del proyecto, destacando las decisiones clave y respaldando la elección final del modelo con métricas sólidas.</div>
# 

# %% [markdown]
# ## Comentario General del Revisor
# 
# <div class="alert alert-block alert-success"> <b>Comentario del revisor</b> <a class="tocSkip"></a>  
#     
# ¡Felicidades! Tu proyecto está **aprobado**. Has desarrollado un flujo de trabajo completo y bien fundamentado que demuestra un dominio sólido en el tratamiento de datos, la exploración inicial, la preparación para el modelado y la evaluación de múltiples algoritmos.  
# 
# #### Puntos Positivos:
# 
# * **Procesamiento de datos:** Realizaste una limpieza exhaustiva, uniendo y transformando datasets de distintas fuentes con codificación consistente, control de valores nulos y creación precisa de la variable objetivo.
# * **Análisis exploratorio:** Tu EDA fue minucioso, identificando patrones relevantes y variables clave, además de detectar y tratar valores atípicos o inconsistencias.
# * **Ingeniería de características:** Lograste codificar adecuadamente variables categóricas y estandarizar las numéricas, dejando la matriz de datos lista para un desempeño óptimo de los modelos.
# * **Estrategias para el desbalance de clases:** Implementaste tanto escenarios con datos originales como oversampling, comparando impactos y eligiendo la estrategia más coherente con el objetivo.
# * **Evaluación de modelos:** Probaste múltiples algoritmos (RandomForest, LGBM, CatBoost, Regresión Logística, entre otros), afinaste hiperparámetros y seleccionaste el modelo con mejor equilibrio entre métricas y generalización, priorizando ROC-AUC.
# * **Documentación y conclusiones:** El cierre del proyecto explica claramente las decisiones técnicas, el porqué de la elección final y cómo los resultados se alinean con los objetivos de negocio.
# 
# Has conseguido un pipeline robusto y replicable, con resultados consistentes tanto en entrenamiento como en validación, y un análisis que respalda cada decisión tomada. Este trabajo refleja un criterio analítico y técnico muy bien desarrollado. </div>
# 

# %% [markdown]
# ## Informe final del proyecto

# %% [markdown]
# ### ¿Qué pasos del plan se realizaron y qué pasos se omitieron (explica por qué)?

# %% [markdown]
# - Se ejecutaron todos los pasos estipulados en el plan
# - En cambio, se agregaron algunos apartados a dicho plan para mejorar la organizacion del workbook así como puntos en donde el analisis nos forzó a realizar medidas adicionales como la seccion de funciones/el manejo del checkpoint/manejo de seleccion de caracteristicas para mejorar el desempeño del modelo

# %% [markdown]
# ### ¿Qué dificultades encontraste y cómo lograste resolverlas?

# %% [markdown]
# - Se implementaron los modelos basicos para un problema de clasificacion, sin embargo, no se conseguia el mejor rendimiento del modelo de acuerdo a la solicitud de negocio.

# %% [markdown]
# ### ¿Cuáles fueron algunos de los pasos clave para resolver la tarea?

# %% [markdown]
# - Se definieron algoritmos adicionales para generar el modelo.
# - Se estuvo trabajando en la seleccion de caracteristicas, proporciones en la division de los datos y el manejo de hiperparametros.
# - Manejo y/o revision del balanceo de la clase objetivo.

# %% [markdown]
# ### ¿Cuál es tu modelo final y qué nivel de calidad tiene?

# %% [markdown]
# - ROC AUC Train score:   0.8722915954241087
# - ROC AUC Test score:   0.8404646900378719

# %% [markdown]
# 

# %%



