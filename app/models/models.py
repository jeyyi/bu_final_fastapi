from sqlalchemy import Column, Integer, String
from pydantic import BaseModel
from sqlalchemy.ext.declarative import declarative_base
from typing import List
# Create a SQLAlchemy Base class
Base = declarative_base()

# Define your database model
class Survey(Base):
    __tablename__ = "VisualizationApp_survey"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    description = Column(String)
    status = Column(Integer)
    description = Column(String)
    waiver = Column(String)
    start_date = Column(String)
    end_date = Column(String)
    added_at = Column(String)
    updated_at = Column(String)
    category_id = Column(Integer)

class Question(Base):
    __tablename__ = "VisualizationApp_surveyquestionnaire"
    id = Column(Integer, primary_key = True)
    type = Column(String)
    choices = Column(String)
    config = Column(String)
    question = Column(String)
    labels = Column(String)
    added_at = Column(String)
    updated_at = Column(String)
    survey_id = Column(Integer)

class Answer(Base):
    __tablename__ = "VisualizationApp_surveyquestionnaireresponse"
    id = Column(Integer, primary_key =True)
    answer = Column(String)
    type = Column(String)
    file = Column(String)
    added_at = Column(String)
    question_id = Column(Integer)
    survey_response_id = Column(Integer)

class TextItems(BaseModel):
    texts: List[str]
