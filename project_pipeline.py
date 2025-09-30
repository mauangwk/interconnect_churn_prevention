# Librerias ----------------------------------------

import params as params
import os
import sys
import argparse

# Esto es para agregar al path la ruta de ejecución actual 
# para poder importar respecto a la ruta del proyecto, desde donde se debe ejecutar el código
sys.path.append(os.getcwd())

# Argumentos por linea de comandos ----------------------------------------

parser = argparse.ArgumentParser()
# parser.add_argument('--periodo', default=f'{params.periodo_YYYYMM}', help='periodo en formato YYYYMM')
args = parser.parse_args()

# Definir extension de ejecutables ----------------------------------------

if params.sistema_operativo == 'Windows':
    extension_binarios = ".exe"
else:
    extension_binarios = ""

print("\nproceso iniciado...")

# Preprocesamiento          ----------------------------------------
if params.preprocess_required:
    os.system(f"python{extension_binarios} src/01_preprocess.py")

# Creacion de Modelo        ----------------------------------------
if params.training_required:
    os.system(f"python{extension_binarios} src/02_create_model.py")

print("\nproceso terminado...")