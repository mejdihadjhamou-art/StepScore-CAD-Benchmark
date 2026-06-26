# CadEval MVP Worker

This worker polls queued runs and processes replicates into stored checks, metrics, and artifacts.

## Run
1. Install Node.js 18+.
2. Ensure Postgres is running with schema from:
   - `./db/migrations/0001_init.up.sql`
3. Set env:
   - `export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/cadeval`
   - `export WORKER_POLL_MS=1500`
4. From `./services/worker` run:
   - `npm install`
   - `npm run start`

## Behavior
- Claims one queued run at a time (`FOR UPDATE SKIP LOCKED`).
- Marks run and replicate lifecycle states.
- Writes metrics and pass/fail checks for MVP validation.
- Creates generated SCAD/STL artifact records linked to each replicate.
- If `OPENAI_API_KEY` is set, generated STL content is produced by ChatGPT and stored in `asset_contents`.
- If no key is set, fallback STL geometry is used.
- Runs `tools/advanced_geometry_metrics.py` per replicate and writes results into:
  - `replicate_metrics` keys prefixed with `adv_`
  - `replicate_checks` keys prefixed with `advanced:`
