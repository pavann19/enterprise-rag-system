# Enterprise RAG — Web Frontend

A Next.js (App Router) + Tailwind frontend for the FastAPI backend in
[`../service/api.py`](../service/api.py). See the main
[README's Web Frontend section](../README.md#-web-frontend-nextjs) for why
this exists alongside `streamlit_app.py`, and for deployment instructions.

## Run it

```bash
# from the repo root, in a separate terminal:
CORS_ALLOWED_ORIGINS=http://localhost:3000 uvicorn service.api:app --reload

cp .env.local.example .env.local
npm install
npm run dev
```

## Layout

- `src/app/page.tsx` — the entire UI: query input, streamed answer, sources
- `src/lib/api.ts` — the fetch/SSE client for `service/api.py`
- `src/components/HealthBadge.tsx` — corpus-loaded status indicator
- `src/app/globals.css` — design tokens (colors, one accent, light/dark)

No state management library, no component library, no CSS-in-JS — the
surface area here is small enough that plain React state and Tailwind
utility classes are the right amount of tooling.

## Testing

Vitest + React Testing Library, 30 tests, 100% line/function coverage:

```bash
npm test              # run once
npm run test:watch    # watch mode
npm run test:coverage # with a coverage report
```

- `src/lib/api.test.ts` — the SSE parser (`streamAnswer`): the `[SOURCES]`
  preamble, tokens split across chunk boundaries, `[ERROR]`/`[DONE]`
  sentinels, auth headers, non-JSON error bodies
- `src/components/HealthBadge.test.tsx` — checking/online/offline states,
  singular vs. plural document count, no state update after unmount
- `src/app/page.test.tsx` — the query flow: streaming, errors, the
  empty-response warning, Clear resetting state
- `src/app/layout.test.tsx` — renders children, exports metadata

Runs in CI (`.github/workflows/ci.yml`'s `web` job) alongside lint and a
production build — a broken frontend fails the same way a broken backend
does, not silently.
