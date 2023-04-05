from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import torch
import json

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# translator_en_ru = pipeline("translation_en_to_ru", model="Helsinki-NLP/opus-mt-en-ru")
# translator_ru_en = pipeline("translation_ru_to_en", model="Helsinki-NLP/opus-mt-ru-en")
translator_en_ru = pipeline("translation_en_to_ru", model="facebook/wmt19-en-ru")
translator_ru_en = pipeline("translation_ru_to_en", model="facebook/wmt19-ru-en")
summarizer = pipeline("summarization", model="MurkatG/review-summarizer-en")
sentiment_analysis = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment")
sentiment_labels = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}
# summarizer = pipeline("summarization", model=r"C:\Users\gtim0\Desktop\models\BART")


@app.get('/')
def index():
    return {'message': 'This is the homepage of the API '}


@app.post('/ru')
async def ru(text: str):
    text = translator_ru_en(text)[0]["translation_text"]
    sentiment = sentiment_labels[sentiment_analysis(text)[0]["label"]]
    text = summarizer(text)[0]['summary_text']
    text = translator_en_ru(text)[0]['translation_text']
    response = {
        "summarization": text,
        "sentiment": sentiment
    }
    return response


@app.post('/en')
async def en(text: str):
    text = summarizer(text)[0]['summary_text']
    return text


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, port=8000)
