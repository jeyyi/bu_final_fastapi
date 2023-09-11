from fastapi import FastAPI,File, UploadFile, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import nltk
import app.router.nlp_router as nlp_router
import app.router.survey_router as survey_router
import app.router.csv_router as csv_router
import app.config.db as db, app.models.models as models  # Import database-related code and models

app = FastAPI()
nltk.download('stopwords')
# Configure CORS settings
origins = [
    "http://localhost:3000",  # Adjust this to the actual URL of your React app
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get('/')
async def root():
    return {"message": "hello world"}
#Router for database
app.include_router(survey_router.router_survey)
#router for csv file
app.include_router(csv_router.router_csv)
#router for nlp
app.include_router(nlp_router.router_nlp)