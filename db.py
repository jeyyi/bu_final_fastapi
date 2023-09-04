from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Create a SQLAlchemy engine to connect to your MySQL database
DATABASE_URL = "mysql+mysqlconnector://root:root@localhost/testing_bu"
engine = create_engine(DATABASE_URL)

# Create a Session class for querying the database
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)