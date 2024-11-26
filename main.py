from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


app = FastAPI()

model: MultinomialNB = None
vectorizer: CountVectorizer = None


class RequestBody(BaseModel):
    samples: list[str]


class ResponseBody(BaseModel):
    samples: list[str]
    predictions: list[str]


def load_model_and_vectorizer():
    global model, vectorizer
    try:
        with open("model.pkl", "rb") as model_file:
            model = pickle.load(model_file)
        with open("vector.pkl", "rb") as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        print("Модель и векторизатор успешно загружены.")
    except Exception as e:
        raise RuntimeError(f"Ошибка при загрузке модели или векторизатора: {e}")


@app.on_event("startup")
async def startup_event():
    load_model_and_vectorizer()


@app.post("/predict", response_model=ResponseBody)
async def predict(request: RequestBody):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")

    samples = request.samples
    features = vectorizer.transform(samples)
    predictions = model.predict(features)
    return ResponseBody(samples=samples, predictions=predictions.tolist())
