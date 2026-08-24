from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import RedirectResponse
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
    return RedirectResponse('/frontend/index.html')

@app.get('/health')
async def health_check():
    return {'status':'OK'}

@app.post('/predict')
async def predict(request: PredictionRequest):
    diff = generate_diff(request.buggy_code, request.fixed_code)
    try:
        predictions = all_predictions(diff)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {'diff': diff, 'all_predictions' : predictions}