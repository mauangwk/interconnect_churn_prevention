# Analisis para Interconect (una Telecom/Prevencion de Churn)

## Que ocurre aqui??

Al operador de telecomunicaciones Interconnect le gustaría poder pronosticar su tasa de cancelación de clientes. 
Si se descubre que un usuario o usuaria planea irse, se le ofrecerán códigos promocionales y opciones de planes especiales. Nuestro objetivo será tratar de prevenir el churn al identificar que un cliente tiene las caracteristicas para abandonar la compañia.

El equipo de marketing de Interconnect ha recopilado algunos de los datos personales de sus clientes, incluyendo información sobre sus planes y contratos.

### Servicios de Interconnect

Interconnect proporciona principalmente dos tipos de servicios:

1. Comunicación por teléfono fijo. El teléfono se puede conectar a varias líneas de manera simultánea.
2. Internet. La red se puede configurar a través de una línea telefónica (DSL, *línea de abonado digital*) o a través de un cable de fibra óptica.

Algunos otros servicios que ofrece la empresa incluyen:

- Seguridad en Internet: software antivirus (*ProtecciónDeDispositivo*) y un bloqueador de sitios web maliciosos (*SeguridadEnLínea*).
- Una línea de soporte técnico (*SoporteTécnico*).
- Almacenamiento de archivos en la nube y backup de datos (*BackupOnline*).
- Streaming de TV (*StreamingTV*) y directorio de películas (*StreamingPelículas*)

La clientela puede elegir entre un pago mensual o firmar un contrato de 1 o 2 años. Puede utilizar varios métodos de pago y recibir una factura electrónica después de una transacción.

### Descripción de los datos

Los datos consisten en archivos obtenidos de diferentes fuentes:

- `contract.csv` — información del contrato;
- `personal.csv` — datos personales del cliente;
- `internet.csv` — información sobre los servicios de Internet;
- `phone.csv` — información sobre los servicios telefónicos.

En cada archivo, la columna `customerID` (ID de cliente) contiene un código único asignado a cada cliente. 


## Preparando el ambiente virtual
Por favor considera ejecutar las siguientes instrucciones inicialmente para manejar el workspace dentro de un ambiente virtual:

```
user$ mkdir venv
user$ python3 -m venv venv
user$ source venv/bin/activate
(venv) user$ pip install --upgrade pip
(venv) user$ python -m pip install -r requirements.txt
```

## Como ejecutar el proyecto

Se puede controlar las secciones del proceso a ejecutar por medio de variables dentro del modulo **params** en la raiz del proyecto segun se requiera. 

- preprocess_required = True/False
- training_required = True/False

La variable **test_for_run_required** se sugiere ejecutar encendida la primera vez para comprobar la correcta ejecucion del proyecto desde la configuracion proporcionada por vscode. Una vez realizada la prueba se puede apagar y encender las variables del proyecto dependiendo la necesidad.

Para la ejecucion del pipeline, es suficiente con la siguiente linea en raiz del proyecto:

```
(venv) user$ python project_pipeline.py 
```

## Estructura del Proyecto

El repositorio está organizado de la siguiente manera:

- **/datasets**: Contiene los datos brutos (`raw`) y los datos listos para ser procesados para la creacion del modelo(`pre-processed`).
- **/notebooks/documentation**: Almacena notebooks de Jupyter para análisis exploratorio o referencias del proyecto, como la estructura del modelo de datos.
- **/files/outputs**: Guarda los resultados del proyecto, como el modelo entrenado (`models`).
- **/src**: Contiene el código fuente modularizado en scripts de Python.
- `requirements.txt`: Lista de dependencias del proyecto.



## Modelo Final y Resultados

El modelo final es un `CatBoost` optimizado. Las siguientes fueron las metricas obtenidas:

- Precision score: 0.6409574468085106
- Recall score:    0.5320088300220751
- F1 score:        0.5814234016887817
- **ROC AUC Train score:   0.8723**
- **ROC AUC Test score:   0.8405**