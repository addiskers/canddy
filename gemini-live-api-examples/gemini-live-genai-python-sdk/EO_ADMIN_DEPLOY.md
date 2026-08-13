# Tring Tring AI — deploy notes

Deployment notes for the Tring Tring AI interview platform (FastAPI + Gemini Live + Plivo +
admin SPA). Internal identifiers keep their legacy `EO_*` / `eo` names on purpose — do not
rename env vars, the `/api/eo` API prefix, `eo.db`, or the `eo-data` volume.

## What runs where (all coexist)
| URL           | What it is                                          | Auth                             | Cost shown |
|---------------|-----------------------------------------------------|----------------------------------|------------|
| `/`           | **Internal test console** — gated by `EO_DEMO_ENABLED` (when off, `/` redirects to `/admin/`) | none (keep OFF in prod) | — |
| `/admin`      | Admin React SPA — campaigns, call logs, assessment review | per-user login             | **never**  |
| `/superadmin` | Legacy dashboard — transcripts + real costing       | shared secret `ANALYTICS_SECRET` | yes        |
| `/live`       | Live transcript viewer for phone calls              | none (transcript only)           | —          |
| `/plivo/*`    | Plivo answer/hangup webhooks + media WebSocket      | Plivo fetches via `PUBLIC_URL`   | —          |

## Python deps
```bash
pip install -r requirements.txt
```
(includes `openpyxl` for Excel import/template.)

## Build the SPA (Node 18+)
```bash
cd admin
npm install
npm run build      # emits admin/dist/, which FastAPI serves at /admin/*
```
FastAPI serves `admin/dist` automatically if present; if it's missing, `/admin` returns a
503 telling you to build. Re-run `npm run build` after any SPA change.

## Environment variables
Existing vars (Plivo, `PUBLIC_URL`, `GEMINI_API_KEY`, `ANALYTICS_SECRET`, `CALLBACK_*`,
`TT_LANGUAGE_CODE`, `TT_ASSESSMENT_*`) are documented in `.env.example`. Admin-platform vars:

| Var | Default | Purpose |
|-----|---------|---------|
| `EO_ADMIN_USER` | `eoadmin` | Username of the first admin, seeded **only if the users table is empty**. |
| `EO_ADMIN_PASS` | `eoadmin123` | Password for that seed user. **Change in prod**, then change again from the Profile page. |
| `EO_SESSION_SECRET` | falls back to `ANALYTICS_SECRET` | HMAC key for admin login tokens (14-day expiry). |
| `EO_DEMO_ENABLED` | `false` | Gates the `/` browser test console. Keep OFF in production. |
| `EO_CAMPAIGN_RUNNER_ENABLED` | `true` | Master switch for the campaign dialer loop. |
| `EO_CAMPAIGN_MAX_CONCURRENT` | `5` | Max simultaneous campaign calls in flight. |
| `EO_CAMPAIGN_MAX_PER_TICK` | `3` | New campaign dials started per 30s tick (pacing). |
| `EO_CAMPAIGN_POLL_INTERVAL` | `30` | Runner tick interval (seconds). |
| `EO_CAMPAIGN_NOANSWER_SECONDS` | `90` | Ring window before a dial with no call record counts as no-answer. |

The "Scheduler: ON/OFF" toggle (Dashboard/Scheduler pages) controls **both** the callback
scheduler and the campaign runner — OFF pauses all outbound dialing.

## Data
- SQLite DB `eo.db` (users, contacts, campaigns, campaign_contacts) is created next to the JSON
  call store under `DATA_DIR`, auto-migrated on startup. Back it up with the rest of `DATA_DIR`.
- Calls stay JSON files; each campaign dial's record carries a `campaign_id` for per-campaign logs.
  Assessments are stored on the call record and surfaced in the admin call drawer.
- The employee roster (`MEMBER_DIRECTORY_PATH`, default `data/employees.csv`) is PII — keep it
  out of version control.

## Run (single worker — required)
The in-process callback scheduler, campaign runner **and** assessment sweep assume ONE uvicorn
worker:
```bash
python main.py            # or: uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
```
On boot you should see: `Callback scheduler started` **and** `Campaign runner started`.

## First login
Go to `https://eo.globalvoxinc.com/admin`, sign in with `EO_ADMIN_USER` / `EO_ADMIN_PASS`,
then create the real users from the **Users** page and change the seed password from **Profile**.

---

## Docker deployment (recommended)
Files: `Dockerfile` (multi-stage: builds the SPA with Node, then runs uvicorn on Python),
`docker-compose.yml` (one service, `127.0.0.1:8000`, persistent `eo-data` volume for `eo.db` +
call logs), `.dockerignore`. The container runs **one** uvicorn worker.

> **Service rename note:** the compose service/container was renamed `eo-radha` → `tringtring`.
> On an existing host, bring it up with
> `docker compose up -d --build --remove-orphans`
> so the old `eo-radha` container is removed. The **`eo-data` volume name is intentionally
> unchanged** — all users/contacts/campaigns/call-logs carry over.

**On the server (in this app directory):**
```bash
cp .env.example .env          # then edit .env — see below
docker compose up -d --build  # build image + start
docker compose logs -f        # expect "Callback scheduler started" + "Campaign runner started"
```

**`.env` must set (real values):**
- `PUBLIC_URL=https://eo.globalvoxinc.com`  ← REQUIRED (Plivo fetches /plivo/answer here)
- `PLIVO_AUTH_ID` / `PLIVO_AUTH_TOKEN` / `PLIVO_FROM_NUMBER`, `GEMINI_API_KEY`, `MODEL`
- `EO_ADMIN_USER` / `EO_ADMIN_PASS`, `EO_SESSION_SECRET` (long random), `ANALYTICS_SECRET`
- `EO_DEMO_ENABLED=false` (production), `TT_ASSESSMENT_ENABLED` as decided per deployment
- leave `DATA_DIR` blank — compose sets it to `/var/eo-data` (the persistent volume)

**HTTPS via the host Caddy** (add to the Caddyfile, then `caddy reload`):
```
eo.globalvoxinc.com {
    reverse_proxy localhost:8000
}
```
Point the `eo.globalvoxinc.com` DNS A record at the server IP; Caddy auto-provisions the TLS cert.

> **Domain note:** if the platform moves to a Tring Tring domain, update **both** this Caddy
> block **and** `PUBLIC_URL` in `.env` — Plivo webhooks and the SPA are reached via `PUBLIC_URL`,
> so changing only one breaks phone calls.

**Update after code changes:**
```bash
git pull
docker compose up -d --build   # rebuilds SPA + app; the eo-data volume persists
```
The `eo-data` volume survives rebuilds, so users/contacts/campaigns/call-logs are never lost.
