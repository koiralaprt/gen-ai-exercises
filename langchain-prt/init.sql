CREATE TABLE IF NOT EXISTS companies (
    id SERIAL PRIMARY KEY,
    company_name TEXT NOT NULL,
    founding_date DATE NOT NULL,
    founders TEXT[]
);