# Tring Tring AI

**Tring Tring AI** is an AI phone interviewer built for **Canny Management Services** to screen its
**61 contract employees** about the **6 August 2026 collective work stoppage** at client **Baoxhin**
(the phone-storage policy protest). The agent conducts a **20-question structured interview** over a
real phone call — **Hindi-first**, mirroring the employee into **Gujarati or English** when they
switch — and records a per-question progress trail during the call.

After each completed phone interview, a **post-call scoring pipeline** produces a structured
assessment: a **0–100 score across 6 rubric categories**, a **red-flag level**
(None / Moderate / Critical), a **review status**, an involvement classification, and **verbatim
evidence quotes** from the transcript.

> **Every result is an input to human review. The AI never decides employment outcomes.**
> `human_review_required` is forced true in code on every assessment; scores, flags and statuses
> exist to prioritise and support reviewers at Canny and Baoxhin, not to replace them.

## Architecture

- **FastAPI** backend + **Gemini Live** realtime voice (via the [`google-genai`](https://github.com/googleapis/python-genai) Python SDK).
- **Plivo** bidirectional audio streaming for real phone calls (`/plivo/*` webhooks + media WebSocket).
- **Browser test console** at `/` (vanilla JS, WebSocket `/ws`) — internal testing only, gated by `EO_DEMO_ENABLED`.
- **Admin SPA** at `/admin` (React, built into `admin/dist`) — campaigns, call logs, call drawer with
  recording playback, assessment review panel.
- **Call store**: JSON call records under `DATA_DIR/calls/` + **SQLite `eo.db`** (users, contacts,
  campaigns) in the same `DATA_DIR`.
- **Campaign runner + callback scheduler**: outbound dial pacing, calling-hours window, and automatic
  re-dials — an **incomplete interview is re-dialed and resumes from question N** using the saved
  per-question progress of the original call.
- **Interview tools**: the model calls `record_interview` (final outcome) and a silent
  `mark_question` (per-question status: answered / partial / declined / dont_know / skipped, with a
  one-line gist) during the call.
- **Post-call scoring pipeline** in `assessment.py`: eligibility gating, scheduling with bounded
  concurrency, a Gemini text-model scoring pass, strict result validation, evidence-quote
  verification against the transcript, and retry/failure accounting.

## Quick start

```bash
# 1. Python deps
pip install -r requirements.txt

# 2. Build the admin SPA (Node 18+)
cd admin && npm install && npm run build && cd ..

# 3. Configure — copy the annotated example and fill in real values
cp .env.example .env

# 4. Run (single worker required — in-process schedulers)
uvicorn main:app --port 8000
```

At minimum set `GEMINI_API_KEY`; for phone calls set the `PLIVO_*` values and a public `PUBLIC_URL`
so Plivo can reach `/plivo/answer`.

## Interview flow

1. **Identity check** — the agent opens in polite Hindi, confirms it is speaking with the named
   employee (roster: `data/employees.csv` — phone, first name, employee ID), and states it is
   calling on behalf of Canny management.
2. **Consent + language** — the employee can refuse (recorded as `no`), ask for a callback, or
   continue in Hindi / Gujarati / English.
3. **20 structured questions** about the 6 Aug 2026 stoppage — each answer is silently marked via
   `mark_question`; interviews interrupted mid-way are called back and resume from question N.
4. **Outcome recorded** via `record_interview` — completed / refused / callback / voicemail /
   do-not-contact / wrong number.
5. **Post-call assessment** (phone interviews meeting the eligibility bar) is scored and queued for
   human review.

### Assessment rubric (100 points)

| Category | Points |
| --- | --- |
| Involvement in the stoppage | 30 |
| Conduct during the incident | 20 |
| Accountability / honesty | 20 |
| Future compliance | 15 |
| Communication | 10 |
| Overall suitability | 5 |

- **Red-flag levels**: `None`, `Moderate`, `Critical`.
- **Review statuses**: `Further consideration`, `Canny review`, `Baoxhin review`,
  `Critical human review`.
- Each assessment carries an involvement classification (passive → organiser/instigator bands),
  a summary, and **evidence quotes verified against the transcript**.

### Review workflow

- **Admin call drawer** (`/admin`) shows the assessment panel per call: score, category breakdown,
  red flag, review status, evidence — with reviewer overrides (`PATCH /api/eo/calls/{id}/review`)
  that keep the original values alongside the edit audit (`edited_by` / `edited_at`).
- **Campaign ranking**: `GET /api/eo/campaigns/{id}/ranking` sorts by concern or score
  (`?sort=concern|score_desc|score_asc`) and exports CSV (`?format=csv`). The call-log CSV export
  includes Score / Red flag / Review status / Involvement / Assessment columns.
- Manual (re)scoring: `POST /api/eo/calls/{id}/assess` and `POST /api/eo/campaigns/{id}/assess`.

## Endpoints

| Route | Purpose |
| --- | --- |
| `GET /` | Internal browser **test console** — only when `EO_DEMO_ENABLED=true`, else redirects to `/admin/` |
| `WS /ws` | Browser test call (proxied to Gemini Live) |
| `GET /admin` | Admin SPA (per-user login) — campaigns, contacts, call logs, assessment review |
| `GET /superadmin` | Legacy cost/transcript dashboard (shared key `ANALYTICS_SECRET`) |
| `GET /live`, `WS /live/ws` | Live transcript viewer for phone calls |
| `POST /call-me` | Place an outbound phone call |
| `GET/POST /plivo/*` | Plivo answer/hangup webhooks + media stream WebSocket |
| `GET/POST /api/eo/*` | Admin API — highlights: `POST /api/eo/calls/{id}/assess`, `POST /api/eo/campaigns/{id}/assess`, `GET /api/eo/campaigns/{id}/ranking`, `PATCH /api/eo/calls/{id}/review`, CSV exports |
| `GET /api/admin/*` | Legacy superadmin API (`X-Admin-Key`) |

## Configuration

All settings live in **`.env.example`** (annotated) — copy it to `.env` and fill in real values.
Notable settings:

- `TT_LANGUAGE_CODE` (default `hi-IN`) — TTS pronunciation bias for the Hindi-first interview.
- `TT_ASSESSMENT_*` — the scoring pipeline: `TT_ASSESSMENT_ENABLED` (explicit opt-in),
  `TT_ASSESSMENT_MODEL`, `TT_ASSESSMENT_MIN_SECONDS`, `TT_ASSESSMENT_MAX_CONCURRENT`,
  `TT_ASSESSMENT_MAX_ATTEMPTS`, `TT_ASSESSMENT_TIMEOUT_S`, `TT_ASSESS_BROWSER`.
- `CALL_MAX_SECONDS=1200` — a full 20-question Hindi interview runs ~12–15 minutes; keep
  `EO_RECORD_MAX_SECONDS` equal to it or recordings get truncated.
- `EO_DEMO_ENABLED` — gates the `/` test console (keep `false` in production).

## Deploy

See **[EO_ADMIN_DEPLOY.md](EO_ADMIN_DEPLOY.md)** — Docker Compose service, Caddy TLS, env checklist,
and the single-worker requirement.

## Internal naming

Legacy `eo_` / `rsvp_` identifiers — Python module names, the `/api/eo` API prefix, `eo.db`, the
`eo-data` volume, `EO_*` env var names, and stored JSON fields like `rsvp_outcome_status` /
`booking_created` — are **intentional**: renaming them would break stored data, API contracts and
deployments. The rebrand to Tring Tring AI is applied at the display layer only (UI labels, docs,
prompts), which is fully rebranded.
