from fastapi import FastAPI
from pydantic import BaseModel
from src.preprocessing import generate_diff
from api.inference import all_predictions
from pathlib import Path
from fastapi.staticfiles import StaticFiles

app = FastAPI()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIRECTORY = PROJECT_ROOT / 'frontend'

app.mount(
    '/frontend',
    StaticFiles(directory=FRONTEND_DIRECTORY),
    name='frontend'
)

class PredictionRequest(BaseModel):
    buggy_code: str
    fixed_code: str

@app.get('/')
async def root():
    return {'message': 'Hello world'}

@app.get('/health')
async def health_check():
    return {'status':'OK'}

@app.post('/predict')
async def predict(request: PredictionRequest):
    diff = generate_diff(request.buggy_code, request.fixed_code)
    predictions = all_predictions(diff)
    return {'diff': diff, 'all_predictions' : predictions}