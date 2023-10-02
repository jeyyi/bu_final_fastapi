from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import app.config.db as db, app.models.models as models


router_survey = APIRouter()



# Dependency to get the database session
def get_db():
    db_session = db.SessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()

@router_survey.get("/survey/")
def get_surveys(db: Session = Depends(get_db)):
    item = db.query(models.Survey).all()
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return item


@router_survey.get("/questionnaires/{survey_id}")
def get_questions(survey_id: int, db:Session = Depends(get_db)):
    try:
        items = db.query(models.Question).filter(models.Question.survey_id == survey_id).all()
        if items is None:
            raise HTTPException(status_code = 404, detail = "Item not found")
        return items
    except Exception as error:
         raise HTTPException(status_code=500, detail="Internal server error")

@router_survey.get("/get_question/{question_id}")
def get_single_question(question_id: int, db:Session = Depends(get_db)):
    try:
        item = db.query(models.Question).filter(models.Question.id==question_id).first()
        if item is None:
            raise HTTPException(status_code=404, detail = "Item not found")
        return item
    except Exception as error:
        raise HTTPException(status_code=500, detail = "Internal server error")

@router_survey.get("/responses/{question_id}")
def get_responses(question_id: int, db:Session = Depends(get_db)):
    try:
        items = db.query(models.Answer).filter(models.Answer.question_id == question_id).all()
        if items is None:
            raise HTTPException(status_code = 404, detail = "Item not found")     
        return [item.answer for item in items]
    except Exception as error:
        raise HTTPException(status_code = 500, detail = "Internal server error")