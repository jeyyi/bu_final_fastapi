# bu_final_fastapi
This README provides instructions for setting up and running your FastAPI application. Please follow the steps below to get started.



## Features

* FastAPI for building APIs
* Uvicorn for ASGI server

## Requirements
Before running the FastAPI application, make sure you have the following prerequisites installed on your system:
- Python 3.7 or higher
- pip (Python package manager)

## Setup

Clone this repository to your local machine using Git:
```bash
git clone https://github.com/jeyyi/bu_final_fastapi.git
cd your-project-name
```
## Create a Virtual Environment (Optional but Recommended)
It's best practice to run Python applications within a virtual environment to isolate dependencies. You can create a virtual environment using the following commands:
```bash
# Create a virtual environment (replace 'venv' with your preferred name)
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS and Linux:
source venv/bin/activate
```
## Install Dependencies
```bash
pip install -r requirements.txt
```
## Running the FastAPI Application
Navigate to the app directory where the main.py file is located:
```bash
cd app
```
To run the FastAPI application, use the following command:
```bash
uvicorn main:app --reload
```
This command starts the development server and makes your FastAPI application accessible at http://localhost:8000.
## Access API Documentation
You can access the API documentation to try out the APIs by opening a web browser and navigating to:

http://localhost:8000/docs

The FastAPI interactive documentation (Swagger UI) allows you to explore, test, and interact with your APIs directly from your browser.
## Usage
You can now use the API endpoints as documented in the Swagger UI or by sending HTTP requests to the API using tools like curl or API testing tools like Postman.

## Additional Information
For more information on FastAPI and how to develop your application, refer to the [FastAPI documentation](https://fastapi.tiangolo.com/)
