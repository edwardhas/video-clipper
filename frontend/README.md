# Local Clipper — web UI

Vue 3 + TypeScript + Vite, plain CSS (tokens in `src/style.css`), vue-router (history mode).

- `npm run dev` — Vite dev server; `/api` and `/media` are proxied to the backend on `127.0.0.1:8765`.
- `npm run build` — type-checks (`vue-tsc`) and writes the production bundle to `dist/`, which the
  backend serves. The bundle only uses relative URLs; no host is baked in.
- `npm test` — unit tests for the pure helpers and the API client (Node's built-in test runner).

Layout: `src/api` (types mirroring `backend/clipper/api/schemas.py`, fetch/XHR client),
`src/lib` (pure helpers), `src/stores` (tiny shared state), `src/components`, `src/views`.
