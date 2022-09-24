from fastapi import FastAPI
from pydantic import BaseModel, conlist  # Para crear el modelo de datos
from typing import List
import pandas as pd  # pandas para lectura en formato dataframe
import json  # json para transformarlo en formato json
import csv  # csv para insertar los datos en él
import os  # para reconocer el entorno y usarlo como consola
import logging  # logging para capturar los errores en caso de que los hubiera
import pickle

# creamos la primera aplicación fastAPI llamada “app”:
app = FastAPI(title="Iris classifier API",
              description="API for Iris classification using ML",
              version="0.1.0")

# Configuramos el registro de la API para tener registro de los fallos
# en caso de no poder acceder a la consola.
my_logger = logging.getLogger()
my_logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG, filename='logs.log')

model = None


# Creamos el modelo de datos con pydantic
class Iris(BaseModel):
    """ 
        Clase del modelo de datos
    """
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    species: str

# Creamos el modelo para predecir  con pydantic


class Iris_predict(BaseModel):
    """ 
        Clase utilizada para la prediccion
    """
    data: List[conlist(float, min_items=4, max_items=4)]


MEDIA_ROOT = os.path.expanduser("~/atom/fastAPI_web/iris.csv")

model = None


@app.on_event("startup")
def load_model():
    """ 
        Carga del modelo para la predicción.
    """
    global model
    model = pickle.load(open("model.pkl", "rb"))


# Método GET a la url "/"
# llamaremos a nuestra aplicación (<app name> + <método permitido>)
@app.get("/", tags=["home"])
async def root():
    """ 
        Página de inicio de la aplicación
    """
    return "Welcome to FastAPI!!"  # Retornar el mensaje bienvenido a FastAPI


@app.get("/iris/",
         tags=["Dataset"])  # Metodo GET para visualización de los datos
async def home():
    """ 
        Método *GET* para visualización de los datos: 
            Parámetros
            ----------
            Parámetro 1 : tipo --> int -> sepal_length [cm]
                Longitud del sépalo en centímetros
            Parámetro 2 : tipo --> int -> sepal_width [cm]
                Ancho del sépalo en centímetros
            Parámetro 3 : tipo --> int -> petal_length [cm]
                Longitud del pétalo en centímetros
            Parámetro 4 : tipo --> int -> petal_width [cm]
                Ancho del pétalo en centímetros
            Parámetro 5 : tipo --> str -> species
                Variedad de Iris Setosa
    """
    df = pd.read_csv(MEDIA_ROOT)  # Cargamos el dataset con ayuda de pandas:
    data = df.to_json(orient="index")  # Lo transformamos a json:
    data = json.loads(
        data
    )  # Recargamos el dataset para mejora la visualización d elos datos
    return data  # Retornar el dataset


@app.post("/insertData/", tags=["Dataset"])
async def insertData(item: Iris):
    """ 
        Método *POST* para insertar datos en el dataset: 
            Parámetros
            ----------
            Parámetro 1 : tipo --> int -> sepal_length [cm]
                Longitud del sépalo en centímetros
            Parámetro 2 : tipo --> int -> sepal_width [cm]
                Ancho del sépalo en centímetros
            Parámetro 3 : tipo --> int -> petal_length [cm]
                Longitud del pétalo en centímetros
            Parámetro 4 : tipo --> int -> petal_width [cm]
                Ancho del pétalo en centímetros
            Parámetro 5 : tipo --> str -> species
                Variedad de Iris Setosa
    """
    with open(
            MEDIA_ROOT, 'a', newline=''
    ) as csvfile:  # Leemos el archivo iris.csv insertamos en la última línea los campos a escribir
        fieldnames = [
            'sepal_length', 'sepal_width', 'petal_length', 'petal_width',
            'species'
        ]  # Nombres de los campos
        writer = csv.DictWriter(csvfile,
                                fieldnames=fieldnames)  # Escritor para csv
        writer.writerow({
            'sepal_length': item.sepal_length,
            'sepal_width': item.sepal_width,
            'petal_length': item.petal_length,
            'petal_width': item.petal_width,
            'species': item.species
        })  # insertar en la última fila:
    return item


@app.put("/updateData/{item_id}", tags=["Dataset"])
async def updateData(item_id: int, item: Iris):
    """ 
        Método *PUT* para actualizar datos en el dataset según ID: 
            Parámetros
            ----------
            Parámetro 1 : tipo --> int -> sepal_length [cm]
                Longitud del sépalo en centímetros
            Parámetro 2 : tipo --> int -> sepal_width [cm]
                Ancho del sépalo en centímetros
            Parámetro 3 : tipo --> int -> petal_length [cm]
                Longitud del pétalo en centímetros
            Parámetro 4 : tipo --> int -> petal_width [cm]
                Ancho del pétalo en centímetros
            Parámetro 5 : tipo --> str -> species
                Variedad de Iris Setosa
            Parámetro 6 : tipo --> int -> item_ID
                ID requerido del dato que se quiere reemplazar.                  
        """
    df = pd.read_csv(MEDIA_ROOT)  # Leemos el csv con ayuda de pandas:
    df.loc[
        df.index[item_id],
        'sepal_length'] = item.sepal_length  # Modificamos el último dato con los valores que nos lleguen:
    df.loc[df.index[item_id], 'sepal_width'] = item.sepal_width
    df.loc[df.index[item_id], 'petal_length'] = item.petal_length
    df.loc[df.index[item_id], 'petal_width'] = item.petal_width
    df.loc[df.index[item_id], 'species'] = item.species
    df.to_csv(MEDIA_ROOT, index=False)  # convertir a csv
    return {"item_id": item_id,
            **item.dict()
            }  # Retornamos el id que hemos modificado y el dato en formato diccionario:


@app.delete("/deleteData/{item_id}", tags=["Dataset"]
            )  # Eliminar el dato con id seleccionado del dataframe
async def deleteData(item_id: int):
    """ 
        Método *DELETE* para eliminar 1 dato en el dataset según ID: 
            Parámetros
            ----------
            Parámetro 1 : tipo --> int -> item_ID
                ID requerido del dato que se quiere reemplazar.
    """
    df = pd.read_csv(MEDIA_ROOT)  # Leemos el csv con ayuda de pandas:
    # Eliminar el valor indicado en el id
    df.drop(df.index[item_id], inplace=True)
    df.to_csv(MEDIA_ROOT, index=False)  # convertir a csv
    return 'Eliminado'


@app.post("/API_Prediction/", tags=["Prediction"])
async def get_predictions(iris: Iris_predict):
    """ 
        Método para predecir el tipo de Iris que tenemos según los datos ingresados.
            Parámetros
            ----------
            Parámetro 1 : tipo --> int -> sepal_length [cm]
                Longitud del sépalo en centímetros
            Parámetro 2 : tipo --> int -> sepal_width [cm]
                Ancho del sépalo en centímetros
            Parámetro 3 : tipo --> int -> petal_length [cm]
                Longitud del pétalo en centímetros
            Parámetro 4 : tipo --> int -> petal_width [cm]
                Ancho del pétalo en centímetross
    """
    try:
        data = dict(iris)['data']
        #data_2 = data_1.to_json(orient="index")
        # print(data_1)
        iris_types = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        prediction = list(
            map(lambda x: iris_types[x],
                model.predict(data).tolist()))
        #log_proba = model.predict_log_proba(data).tolist()
        return {"prediction": prediction}  # ,"log_proba": log_proba
    except:
        my_logger.error("Something went wrong!")
        return {"prediction": "error"}
