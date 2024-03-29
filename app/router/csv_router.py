from fastapi import APIRouter, UploadFile, File
import app.models.models as models
from io import BytesIO
import pandas as pd
import numpy as np
from fastapi.responses import JSONResponse

router_csv = APIRouter()


@router_csv.post('/analyzecsv')
async def analyze_csv(file: UploadFile = File(...)):
    try:
        content = await file.read()
        data = BytesIO(content)
        if file.filename.endswith('.csv'):
            # read csv
            df = pd.read_csv(data)
        elif file.filename.endswith(('.xlsx', 'xls')):
            # read excel
            df = pd.read_excel(data)
        else:
            return JSONResponse(content={"error": "Invalid file format"}, status_code=400)
        df = remove_null(df)
        return get_vals(df)
    except Exception as e:
        # If exception occurs while processing the file
        return JSONResponse(content={"error": str(e)}, status_code=500)


def remove_null(df):
    null_percentage = df.isna().mean().round(4)*100
    # drop columns
    return df.drop(columns=null_percentage[null_percentage == 100].index)


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

    if type(df) == pd.core.series.Series:
        if np.issubdtype(df.dtype, np.datetime64):
            column_classification[df.name] = 'date'
        elif df.nunique() <= cat_threshold:
            column_classification[df.name] = 'categorical'
        else:
            column_classification[df.name] = 'open_ended'
    elif type(df) == pd.core.frame.DataFrame:
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
    results_dict = {}
    for col in col_names:
        if cats[col] == 'categorical':
            results_dict[col] = df[col].value_counts().to_dict()
        elif cats[col] == 'open_ended':
            # temp_lstText =  get_text(df[col])
            # dict[col] = get_topics(temp_lstText)
            # dict[col] = get_text(df[col])
            results_dict[col] = df[col].apply(
                lambda x: None if pd.isna(x) else x).to_list()
    return results_dict
