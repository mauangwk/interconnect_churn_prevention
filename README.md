# Analisis para la telecom Interconect, Prevencion de Churn

## Objetivo

Tratar de prevenir el churn al identificar que un cliente tiene las caracteristicas para abandonar la compañia.
El equipo de marketing de Interconnect ha recopilado algunos de los datos personales de sus clientes, incluyendo información sobre sus planes y contratos.

### Descripción de los datos

Los datos consisten en archivos obtenidos de diferentes fuentes:

- `contract.csv` — información del contrato;
- `personal.csv` — datos personales del cliente;
- `internet.csv` — información sobre los servicios de Internet;
- `phone.csv` — información sobre los servicios telefónicos.

En cada archivo, la columna `customerID` (ID de cliente) contiene un código único asignado a cada cliente. 

<img width="542" height="534" alt="image" src="https://github.com/user-attachments/assets/79bf8d8c-a3e6-42e1-88f6-364e56fa1391" />

### Servicios de Interconnect que precisan los datos

Interconnect proporciona principalmente dos tipos de servicios:

1. Comunicación por teléfono fijo. El teléfono se puede conectar a varias líneas de manera simultánea.
2. Internet. La red se puede configurar a través de una línea telefónica (DSL, *línea de abonado digital*) o a través de un cable de fibra óptica.

Algunos otros servicios que ofrece la empresa incluyen:

- Seguridad en Internet: software antivirus (*ProtecciónDeDispositivo*) y un bloqueador de sitios web maliciosos (*SeguridadEnLínea*).
- Una línea de soporte técnico (*SoporteTécnico*).
- Almacenamiento de archivos en la nube y backup de datos (*BackupOnline*).
- Streaming de TV (*StreamingTV*) y directorio de películas (*StreamingPelículas*)

La clientela puede elegir entre un pago mensual o firmar un contrato de 1 o 2 años. Puede utilizar varios métodos de pago y recibir una factura electrónica después de una transacción.



## Preparando el ambiente virtual
Por favor considera ejecutar las siguientes instrucciones inicialmente para manejar el workspace dentro de un ambiente virtual:

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

## Estructura del Proyecto

El repositorio está organizado de la siguiente manera:

- **/datasets**: Contiene los datos brutos (`raw`) y los datos listos para ser procesados para la creacion del modelo(`pre-processed`).
- **/files/documentation**: Almacena la estructura/modelo de datos y notebooks de Jupyter para análisis exploratorio o referencias del proyecto en un bootcamp en tripleten.
- **/files/outputs**: Guarda los resultados del proyecto, como el modelo entrenado (`models`), imagenes (`images`)
- **/src**: Contiene el código fuente modularizado en scripts de Python.
- **`requirements.txt`**: Lista de dependencias del proyecto.



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



