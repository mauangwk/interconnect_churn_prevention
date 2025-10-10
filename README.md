# Analisis para la telecom Interconect, Prevencion de Churn

## Objetivo

Tratar de prevenir el churn al identificar que un cliente tiene las caracteristicas para abandonar la compañia.
El equipo de marketing de Interconnect ha recopilado algunos de los datos personales de sus clientes, incluyendo información sobre sus planes y contratos.

## Estructura del Proyecto

El repositorio está organizado de la siguiente manera:

- **/datasets**: Contiene los datos brutos (`raw`) y los datos listos para ser procesados para la creacion del modelo(`pre-processed`).
- **/files/documentation**: Almacena la estructura/modelo de datos y notebooks de Jupyter para análisis exploratorio o referencias del proyecto en un bootcamp en tripleten.
- **/files/outputs**: Guarda los resultados del proyecto, como el modelo entrenado (`models`), imagenes (`images`)
- **/src**: Contiene el código fuente modularizado en scripts de Python.
- **`requirements.txt`**: Lista de dependencias del proyecto.
- En la raiz del proyecto se encuentran los archivos principales para la ejecución y prueba del mismo.

## Preparando el ambiente virtual
Por favor considerar el ejecutar las siguientes instrucciones para manejar el workspace dentro de un ambiente virtual:

```
mkdir venv
python3 -m venv venv
source venv/bin/activate
user$ pip install --upgrade pip
user$ python -m pip install -r requirements.txt
```

## Como ejecutar el proyecto

Se puede controlar las secciones del proceso a ejecutar por medio de variables dentro del modulo **params** en la raiz del proyecto segun se requiera. 

- preprocess_required = True/False
- training_required = True/False

Para la ejecucion del pipeline, es suficiente con la siguiente linea en raiz del proyecto:

```
python project_pipeline.py 

```
## Modelo Final y Resultados del entrenamiento

El modelo final es un `CatBoost` optimizado. Las siguientes fueron las metricas obtenidas:

<img width="2000" height="600" alt="model_evaluation" src="https://github.com/user-attachments/assets/b04abc7e-df73-4eb1-86f3-289b0153e37a" />

|        |train|test|
|--------------|-----|----|
|F1 Score      |0.63 |0.59|
|Accuracy Score|0.82 |0.81|
|Recall Score  |0.57 |0.54|
|APS           |0.72 |0.65|
|ROC AUC       |0.87 |0.84|
