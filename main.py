from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import logging

from schemas import ResponseBody, RequestBody

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

MODEL_PATH = "model.pkl"
try:
    model = joblib.load(MODEL_PATH)
    logger.info("Модель успешно загружена.")
except FileNotFoundError:
    raise RuntimeError(f"Файл модели не найден по пути {MODEL_PATH}")


class Features(BaseModel):
    feature_1: float = Field(..., ge=0, le=10, description="Признак 1, диапазон: 0-10")
    feature_2: float = Field(..., ge=0, le=10, description="Признак 2, диапазон: 0-10")
    feature_3: float = Field(..., ge=0, le=10, description="Признак 3, диапазон: 0-10")
    feature_4: float = Field(..., ge=0, le=10, description="Признак 4, диапазон: 0-10")


@app.post("/predict", response_model=ResponseBody)
async def predict(body: RequestBody):
    predictions = [sample.upper() for sample in body.samples]  # Пример обработки
    return ResponseBody(samples=body.samples, predictions=predictions)


@app.post("/upload_model")
async def upload_model(model_file: bytes):
    try:
        with open(MODEL_PATH, "wb") as f:
            f.write(model_file)
        global model
        model = joblib.load(MODEL_PATH)
        logger.info("Новая модель успешно загружена и обновлена.")
        return {"detail": "Модель успешно обновлена."}
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {str(e)}")
        raise HTTPException(status_code=500, detail="Ошибка загрузки модели.")
