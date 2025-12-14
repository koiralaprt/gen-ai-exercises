1. Prerequisites

Docker & Docker Compose

Python 3.9+

AWS credentials 

Amazon Bedrock (Nova Lite)

2. Start PostgreSQL (Docker)
docker compose up -d


Verify database is running:

docker ps

3. Setup Python Environment
python -m venv venv
source venv/bin/activate


Install dependencies:

pip install langchain langchain-aws pydantic sqlalchemy psycopg2-binary

4. Configure Environment

Ensure .env contains:

POSTGRES_URI=postgresql+psycopg2://postgres:postgres@localhost:5432/company_db


AWS credentials can be configured through environment variables.

5. Run the Application
python -m app.main

6. Verify Data in PostgreSQL
psql -h localhost -U postgres company_db

SELECT * FROM companies;

