from fastapi import FastAPI,File, UploadFile, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from gensim import corpora
from pydantic import BaseModel
from typing import List
from gensim.models.ldamodel import LdaModel
from typing import Union
import pandas as pd
import numpy as np
from io import BytesIO
import re
import nltk
from collections import Counter
from sqlalchemy.orm import Session
import db, models  # Import database-related code and models
app = FastAPI()

# Dependency to get the database session
def get_db():
    db_session = db.SessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()

class TextItems(BaseModel):
    texts: List[str]
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
nltk.download('stopwords')
@app.get('/')
async def root():
    return {"message": "hello world"}

@app.post('/analyzecsv')
async def analyze_csv(file: UploadFile = File(...)):
    try:
        content = await file.read()
        data = BytesIO(content)
        if file.filename.endswith('.csv'):
            #read csv
            df = pd.read_csv(data)
        elif file.filename.endswith(('.xlsx','xls')):
            #read excel
            df = pd.read_excel(data)
        else:
            return JSONResponse(content={"error": "Invalid file format"}, status_code = 400)
        df = remove_null(df)
        return get_vals(df)
    except Exception as e:
        #If exception occurs while processing the file
        return JSONResponse(content = {"error": str(e)}, status_code=500)
    
def remove_null(df):
    null_percentage = df.isna().mean().round(4)*100
    #drop columns
    return df.drop(columns = null_percentage[null_percentage==100].index)

def classify_columns(df, cat_threshold=35):
    """
    Classify columns of a DataFrame as 'categorical', 'open_ended', or 'date'.
    
    Parameters:
    df (DataFrame): The DataFrame to classify
    cat_threshold (int): The number of unique values below which a column is considered 'categorical'.
                         Default is 35.
                         
    Returns:
    dict: A dictionary where keys are column names and values are column types
    """
    column_classification = {}
    
    if type(df)==pd.core.series.Series:
        if np.issubdtype(df.dtype,np.datetime64):
            column_classification[df.name]='date'
        elif df.nunique() <= cat_threshold:
            column_classification[df.name]='categorical'
        else:
            column_classification[df.name]='open_ended'
    elif type(df)==pd.core.frame.DataFrame:  
        for col in df.columns:
            # Check for date
            if np.issubdtype(df[col].dtype, np.datetime64):
                column_classification[col] = 'date'
            # Check for categorical
            elif df[col].nunique() <= cat_threshold:
                column_classification[col] = 'categorical'
            # Default to open_ended
            else:
                column_classification[col] = 'open_ended'
    else:
        None
    return column_classification

def get_vals(df):
    col_names = df.columns
    cats = classify_columns(df)
    dict = {}
    for col in col_names:
        if cats[col]=='categorical':
            dict[col]=df[col].value_counts().to_dict()
        elif cats[col] == 'open_ended':
            #temp_lstText =  get_text(df[col])
            #dict[col] = get_topics(temp_lstText)
            dict[col] = get_text(df[col])
    return dict

from nltk.corpus import stopwords
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove all non a-z0-9 characters using regex
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]
    
    # Join the cleaned words back into a string
    cleaned_text = ' '.join(cleaned_words)
    return cleaned_text

def get_text(series):
    series = series.dropna()
    series = series.apply(lambda x: clean_text(x))
    return series.tolist()

@app.post("/get_topics/")
def get_topics(texts, num_topics = 5):
    # Create a dictionary representation of the documents
    tokenized_texts = [text.split() for text in texts if text]
    dictionary = corpora.Dictionary(tokenized_texts)

    # Convert the list of texts to a list of vectors based on word frequency
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    topics = lda_model.print_topics(num_words=5)  # Adjust num_words to display more or fewer keywords per topic
    topics_as_word_lists = []
    num_words_per_topic =5
    for topic_id in range(lda_model.num_topics):
        topic_word_ids = [word_id for word_id, prob in lda_model.get_topic_terms(topic_id, num_words_per_topic)]
        topic_words = [dictionary[word_id] for word_id in topic_word_ids]
        topics_as_word_lists.append(topic_words)

    #return topics_as_word_lists
    return topics_as_word_lists

@app.post("/get_frequent/")
def get_frequent(texts, num_words=10):
    # Flatten the list of lists into a single list of words
    all_words = [word for text in texts for word in text.split()]

    # Use Counter to get word frequencies
    word_counts = Counter(all_words)

    # Return the top 'num_words' frequent words
    return dict(word_counts.most_common(num_words))

@app.get("/survey/")
def get_surveys(db: Session = Depends(get_db)):
    item = db.query(models.Survey).all()
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return item

@app.get("/questionnares/{survey_id}")
def get_questions(survey_id: int, db:Session = Depends(get_db)):
    try:
        items = db.query(models.Question).filter(models.Question.survey_id == survey_id).all()
        if items is None:
            raise HTTPException(status_code = 404, detail = "Item not found")
        return items
    except Exception as error:
         raise HTTPException(status_code=500, detail="Internal server error")