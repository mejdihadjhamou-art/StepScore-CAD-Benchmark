# CadEval MVP API Scaffold

This is an Express API that implements all endpoints defined in:
- `./docs/mvp/api_contract.yaml`

It uses PostgreSQL for persistence.

## Run
1. Install Node.js 18+.
2. Ensure Postgres is running and schema is applied from:
   - `./db/migrations/0001_init.up.sql`
3. Set database URL:
   - `export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/cadeval`
4. From `./services/api` run:
   - `npm install`
   - `npm run start`
5. Server starts on `http://localhost:8080`.

## Notes
- Health check: `GET /health`
- Browser UI: `GET /`
- API base: `/v1/*`
- Data is persisted in Postgres.
