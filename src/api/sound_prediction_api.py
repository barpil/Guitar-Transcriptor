from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

import config
from src.api.pipeline.model.prediction_model import PredictionModel

app = FastAPI()

model = None

sound_rec_pth = f"{config.MODEL_DIR_PATH}/sound_recognition_model.pth"
string_rec_pth = f"{config.MODEL_DIR_PATH}/string_recognition_model.pth"
sound_type_rec_pth = f"{config.MODEL_DIR_PATH}/sound_type_recognition_model.pth"
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("Starting sound prediction API")
    model = PredictionModel(sound_rec_pth, string_rec_pth, sound_type_rec_pth, 22050)
    yield
    print("Zamknięcie API — cleanup")

app = FastAPI(lifespan=lifespan)
@app.get("/")
async def test():
    return {"Hello":"World"}

@app.post("/predict/")
async def analyze_wav(file: UploadFile = File(...)):
    if not file.filename.endswith(".wav"):
        return JSONResponse(status_code=400, content={"Error": "Only .wav files are supported for prediction."})

    temp_file_path = f"{config.TMP_DIR_PATH}/{file.filename}"
    with open(temp_file_path, "wb") as f:
        f.write(await file.read())

    result = model.predict(file.file)
    return result

