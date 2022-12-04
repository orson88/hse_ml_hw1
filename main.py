import io
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
import csv
import codecs
from pydantic import BaseModel
from typing import List
from io import StringIO
import warnings
warnings.filterwarnings("ignore")


with open('inference_info.pickle', 'rb') as handle:
    REFERENCE_INFO = pickle.load(handle)



def get_predict(predict_line: list) -> list[float]:
    DATAFRAME = REFERENCE_INFO['shape_df']
    MODEL = REFERENCE_INFO['pipeline']
    DATAFRAME.loc[0] = predict_line
    return list(np.exp(MODEL.predict(DATAFRAME)))[0]


app = FastAPI()


class Item(BaseModel):
    name: str
    year: float
    km_driven: float
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float
    engine: float
    max_power: float
    seats: float
    nm: float
    rpm: float
    manufacturer: str

class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> list[float]:
    values = list(item.dict().values())
    return get_predict(values)

def to_df(file):
    data = file.file
    data = csv.reader(codecs.iterdecode(data,'utf-8'), delimiter=',')
    header = data.__next__()
    df = pd.DataFrame(data, columns=header)
    return df

@app.post("/upload_predict_items")
async def predict_items_csv(file: UploadFile):
    df = to_df(file).iloc[:5]
    df.drop(columns = [df.columns[0]], inplace = True )
    MODEL = REFERENCE_INFO['pipeline']
    df['preds'] = list(np.exp(MODEL.predict(df)))
    df.to_csv('export.csv')
    return FileResponse('export.csv')