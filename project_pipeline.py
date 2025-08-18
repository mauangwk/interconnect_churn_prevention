# Librerias ----------------------------------------

import params as params
import os
import sys
import argparse

# Esto es para agregar al path la ruta de ejecución actual y poder importar respecto a la ruta del proyecto, desde donde se debe ejecutar el código
sys.path.append(os.getcwd())

# Argumentos por linea de comandos ----------------------------------------

parser = argparse.ArgumentParser()
# parser.add_argument('--periodo', default=f'{params.periodo_YYYYMM}', help='periodo en formato YYYYMM')

try:
    args = parser.parse_args()
except argparse.ArgumentTypeError as e:
    print(f"Invalid argument: {e}")

# Definir extension de ejecutables ----------------------------------------

if params.sistema_operativo == 'Windows':
    extension_binarios = ".exe"
else:
    extension_binarios = ""

# Info ----------------------------------------

# print(f"---------------------- \nComenzando proceso para periodo: {args.periodo}\n-------------------------")
print(f"---------------------- \nprocess starts ... \n-------------------------")

# Prueba de ejecucion -------------------------------
if params.test_for_run_required:
    os.system(f"python{extension_binarios} src/00_test.py")
    print("project structure tested... ")

# Preproceso ----------------------------------------
try:
    if params.preprocess_required:
        os.system(f"python{extension_binarios} src/01_preprocess.py")
        print("data preprocesed... ")
except Exception as e:
    print(e)

# Creacion de Modelo ----------------------------------------
try:
    if params.training_required:
        os.system(f"python{extension_binarios} src/02_create_model.py")
        print("model created... ")
except Exception as e:
    print(e)

print("process done...")