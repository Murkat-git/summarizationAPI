from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.on_event("startup")
def load_model():
    global translator_en_ru, translator_ru_en, summarizer
    translator_en_ru = pipeline("translation_en_to_ru", model="Helsinki-NLP/opus-mt-en-ru")
    translator_ru_en = pipeline("translation_ru_to_en", model="Helsinki-NLP/opus-mt-ru-en")
    summarizer = pipeline("summarization", model="MurkatG/review-summarizer-en")


@app.get('/')
def index():
    return {'message': 'This is the homepage of the API '}


@app.post('/ru')
async def ru(text: str):
    text = translator_ru_en(text)[0]["translation_text"]
    text = summarizer(text)[0]['summary_text']
    text = translator_en_ru(text)[0]['translation_text']
    return text


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, port=8000)
