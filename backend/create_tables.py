from backend.models.models import Base
from backend.constants import RECORD_MANAGER_DB_URL
from sqlalchemy import create_engine

# Create an engine
engine = create_engine(RECORD_MANAGER_DB_URL)

# Create the tables
Base.metadata.create_all(engine)

print("Tables created successfully!")