from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from models.models import Base
from constants import RECORD_MANAGER_DB_URL

def clear_sql_db():
    engine = create_engine(RECORD_MANAGER_DB_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    try:
        Base.metadata.drop_all(bind=engine)
        print("SQL Database cleared successfully.")
    except SQLAlchemyError as e:
        print(f"Error occurred while clearing SQL database: {e}")

if __name__ == "__main__":
    clear_sql_db()