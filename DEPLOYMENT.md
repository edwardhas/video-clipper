# Deploying Local Clipper on a server

Local Clipper is built to run on the owner's computer (`scripts/start.sh`). This guide covers the
optional **hosted mode**: the same app on a persistent Linux server behind HTTPS, with a
mandatory owner sign-in.

> **Status:** the deployment package (Dockerfile, compose file, Caddyfile, systemd examples) has
> been written and its configuration validated locally (`docker compose config`, `caddy adapt` of
> the Caddyfile, `nginx -t` of the nginx example on nginx 1.22/1.24/1.26, unit tests of the hosted
> settings; see `backend/tests/test_deploy.py`). **No live domain, DNS record or HTTPS
> certificate has been verified yet**, and the container image has not been built on a server.
> `clipper.example.com` below is a placeholder hostname: replace it with your own.

## 1. What runs where

```
browser ──HTTPS──> Caddy (ports 80/443) ──HTTP──> api (uvicorn, CLIPPER_PORT 8765, not published)
                                                   │ SQLite + media on the /data volume
                                       worker ─────┘ FFmpeg, yt-dlp, OpenAI (outbound HTTPS)
```

* **api** serves the web interface, the JSON API and media files. It needs no internet access.
* **worker** processes one job at a time (download, transcription, selection, rendering). It
  needs outbound HTTPS to `api.openai.com` and YouTube.
* Both share one **SQLite database and media folder on a local disk**. Run exactly **one api and
  one worker**; no replicas, no load balancing across hosts.

Not supported:

* **Serverless or static hosting** (Vercel/Netlify functions, Cloud Run scale-to-zero, Lambda,
  static site hosts). Jobs run for many minutes, need FFmpeg and a persistent disk.
* **Hosting only the frontend while it calls the API on your laptop.** That is a different
  architecture (cross-origin API, tunnels, CORS, cookies across sites) and is not supported. The
  frontend is always served by the same server as the API.
* Ordinary shared website hosting is **not assumed** to meet these requirements; check it
  against section 2 before using it.

## 2. Host requirements

| | Minimum / recommendation |
|---|---|
| Server | Persistent Linux VM or dedicated server (x86-64 or arm64), always on |
| CPU / RAM | ≥ 2 vCPU and 4 GB RAM recommended (rendering uses CPU libx264; no hardware encoder on Linux) |
| Disk | ≥ 50 GB **persistent local** disk (SSD) for `/data`; sources up to 2 h plus renders and caches |
| Software | Docker Engine with the Compose plugin (v2.24+), **or** native Python 3.12, FFmpeg, deno ≥ 2.3, Node 22 (build only) |
| Network | Inbound TCP 80 + 443 (and UDP 443 for HTTP/3) to the proxy; outbound HTTPS to `api.openai.com`, `*.youtube.com`/`*.googlevideo.com`, and the ACME CA (Let's Encrypt/ZeroSSL) |
| DNS | A (and AAAA if the server has IPv6) record for the chosen hostname |

SQLite must live on a local filesystem: **not** NFS/SMB/CIFS network shares, not object storage
mounts (s3fs etc.), and not ephemeral container storage. Use the named volume (or a bind mount of
a local disk path).

## 3. Hostname, DNS and HTTPS

1. Choose the hostname (example: `clipper.example.com`).
2. Create DNS records pointing that name at the server: `A <server IPv4>` and, if available,
   `AAAA <server IPv6>`. Wait until `dig +short clipper.example.com` returns the server address.
3. Open ports 80 and 443 in the server firewall / cloud security group. Caddy obtains and renews
   the certificate automatically (HTTP-01/TLS-ALPN-01 challenges need both ports).
4. Set `CLIPPER_PUBLIC_ORIGIN=https://<hostname>` and `CLIPPER_HOSTNAME=<hostname>`.

Cookie and HSTS notes:

* The session cookie is **host-only** (no `Domain` attribute), `HttpOnly`, `SameSite=Lax` and
  `Secure`. It is never shared with the parent domain or other subdomains. Do not add a cookie domain.
* Avoid hosting untrusted content on sibling subdomains of the same parent domain.
* The Caddyfile sends `Strict-Transport-Security: max-age=31536000` **without**
  `includeSubDomains`/`preload`, so other hosts under the parent domain are unaffected. Once browsers have
  seen it they insist on HTTPS for this hostname for a year; if you are unsure about keeping
  HTTPS on this name, start with a small `max-age` (e.g. 300) and raise it later.

## 4. Configuration

Copy the example and fill it in (the file is git-ignored; keep it private):

```bash
cp deploy/env.hosted.example deploy/.env.hosted
chmod 600 deploy/.env.hosted
```

| Variable | Example | Notes |
|---|---|---|
| `CLIPPER_MODE` | `hosted` | Sign-in becomes mandatory; startup fails without secrets. |
| `CLIPPER_HOST` | `0.0.0.0` | Container interface only; the port is never published. Natively behind nginx on the same host: `127.0.0.1`. |
| `CLIPPER_PORT` | `8765` | Internal port the api listens on and the proxy talks to. Compose passes the same value to both containers (Caddy dials `api:{$CLIPPER_PORT}`); natively, use it in the proxy address. Never published. |
| `CLIPPER_PUBLIC_ORIGIN` | `https://clipper.example.com` | Exact origin users open; must be `https://`. Used for Host/Origin checks. |
| `CLIPPER_HOSTNAME` | `clipper.example.com` | Caddy only: the site name to get a certificate for. |
| `CLIPPER_DATA_DIR` | `/data` | Database, media, logs. Persistent local disk. |
| `CLIPPER_TRUSTED_PROXIES` | `172.30.57.10` | IPs/CIDRs allowed to set `X-Forwarded-*` (Caddy's address in `compose.yaml`, i.e. `CLIPPER_PROXY_IP`; `127.0.0.1` for nginx on the same host). |
| `CLIPPER_PROXY_SUBNET` | `172.30.57.0/24` | Compose only, optional: subnet of the `internal` network. Change it only if it collides with another network on the host. |
| `CLIPPER_PROXY_IP` | `172.30.57.10` | Compose only, optional: Caddy's fixed address inside `CLIPPER_PROXY_SUBNET`. Keep it equal to `CLIPPER_TRUSTED_PROXIES`. |
| `CLIPPER_OWNER_USERNAME` | `owner` | Optional. |
| `CLIPPER_OWNER_PASSWORD_HASH` | `'$argon2id$v=19$…'` | From `scripts/hash-password.sh`. **Keep the single quotes**: Docker Compose otherwise expands the `$` signs and corrupts the hash. |
| `CLIPPER_SESSION_SECRET` | 64 random characters | `python3 -c "import secrets; print(secrets.token_urlsafe(48))"`. Changing it signs everyone out. |
| `OPENAI_API_KEY` | `sk-…` | Server-side only; never sent to browsers or logged. |
| `CLIPPER_MAX_UPLOAD_MB` | `4096` | App limit; Caddy's body limit uses the same value. |
| `CLIPPER_MAX_SOURCE_MINUTES` | `120` | Optional. |
| `CLIPPER_MIN_FREE_DISK_MB` | `1024` | Optional disk reserve. |
| `CLIPPER_IMAGE` | `local-clipper:3f2a9c1` | Optional compose image tag (see upgrades/rollback). |

Every other `CLIPPER_*` option is documented in `.env.example`. Generate the password hash on
any machine with the repository set up:

```bash
scripts/hash-password.sh            # asks twice, prints only the hash
# or, on the server once the image is built:
docker compose --env-file deploy/.env.hosted -f deploy/compose.yaml run --rm --no-deps api \
  python -m clipper.cli hash-password
```

## 5. Deploy with Docker Compose (recommended)

From the repository root on the server:

```bash
git rev-parse --short HEAD                    # note the version
export CLIPPER_IMAGE=local-clipper:$(git rev-parse --short HEAD)   # or set it in deploy/.env.hosted
docker compose --env-file deploy/.env.hosted -f deploy/compose.yaml build
docker compose --env-file deploy/.env.hosted -f deploy/compose.yaml up -d
docker compose --env-file deploy/.env.hosted -f deploy/compose.yaml ps        # api/worker "healthy"
docker compose --env-file deploy/.env.hosted -f deploy/compose.yaml run --rm --no-deps api \
  python -m clipper.cli doctor
```

`--env-file` is needed on every compose command (it supplies `CLIPPER_HOSTNAME` to Caddy); to
avoid typing it, `export COMPOSE_ENV_FILES=deploy/.env.hosted COMPOSE_FILE=deploy/compose.yaml`.

What the compose file sets up:

* `api`: image default command (`python -m clipper.cli serve-api` → uvicorn with
  `--proxy-headers --forwarded-allow-ips $CLIPPER_TRUSTED_PROXIES`), **no published ports**,
  only on the `internal` network (`internal: true`, no outbound internet).
* `worker`: same image, `python -m clipper.worker`, on the `egress` network (outbound HTTPS for
  OpenAI/YouTube, nothing published), `stop_grace_period: 30s` so it can stop FFmpeg and mark the
  current job *interrupted* (resumable; see "Interrupted jobs" in section 10).
* `proxy`: `caddy:2`, the only service publishing ports (80, 443, 443/udp), on `internal` (fixed
  address `172.30.57.10`) and `edge`. It proxies to `api:$CLIPPER_PORT` (default 8765); compose
  hands the same `CLIPPER_PORT` to the api, so set it only in `deploy/.env.hosted`. If
  `172.30.57.0/24` collides with another network on the host (`docker network inspect` shows
  "Pool overlaps"), set `CLIPPER_PROXY_SUBNET` and
  `CLIPPER_PROXY_IP` in `deploy/.env.hosted` (compose reads them via `--env-file`) and set
  `CLIPPER_TRUSTED_PROXIES` to the same IP.
* Volumes: `clipper_data` (`/data` for api and worker), `caddy_data` (certificates; keep it),
  `caddy_config`. Containers run as UID 10001, with tini as PID 1 and `restart: unless-stopped`.

Logs: `docker compose … logs -f api worker proxy`. Caddy's access log omits query strings,
cookies and authorization headers; the app never logs secrets.

## 6. Native install (alternative, systemd)

On Debian/Ubuntu with Python 3.12 and its venv module (Ubuntu 24.04: `apt install python3.12 python3.12-venv`),
FFmpeg (`apt install ffmpeg`), deno ≥ 2.3 and Node 22:

```bash
sudo useradd --system --home /opt/local-clipper --shell /usr/sbin/nologin clipper
sudo git clone <repository> /opt/local-clipper && sudo chown -R clipper: /opt/local-clipper
sudo install -d -o clipper -g clipper -m 700 /var/lib/local-clipper
cd /opt/local-clipper && sudo -u clipper scripts/setup.sh --prod
```

Put the hosted settings (section 4, with `CLIPPER_HOST=127.0.0.1`,
`CLIPPER_DATA_DIR=/var/lib/local-clipper`, `CLIPPER_TRUSTED_PROXIES=127.0.0.1`) in
`/opt/local-clipper/.env` (owned by `clipper`, mode 600); the app reads it directly.

`/etc/systemd/system/local-clipper-api.service`:

```ini
[Unit]
Description=Local Clipper API
After=network-online.target
Wants=network-online.target

[Service]
User=clipper
Group=clipper
WorkingDirectory=/opt/local-clipper
Environment=PYTHONPATH=/opt/local-clipper/backend
ExecStart=/opt/local-clipper/backend/.venv/bin/python -P -m clipper.cli serve-api
Restart=on-failure
RestartSec=5
TimeoutStopSec=20
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/local-clipper

[Install]
WantedBy=multi-user.target
```

`/etc/systemd/system/local-clipper-worker.service`: the same, except

```ini
Description=Local Clipper worker
ExecStart=/opt/local-clipper/backend/.venv/bin/python -P -m clipper.worker
# SIGTERM to the worker only; it stops its FFmpeg process group and marks the job interrupted.
KillMode=mixed
TimeoutStopSec=30
```

Then `sudo systemctl daemon-reload && sudo systemctl enable --now local-clipper-api local-clipper-worker`.
Use Caddy (`apt install caddy`, the same Caddyfile with `reverse_proxy 127.0.0.1:8765` (your
`CLIPPER_PORT`) and the other `{$…}` placeholders replaced) or nginx:

```nginx
log_format clipper '$remote_addr [$time_local] "$request_method $uri $server_protocol" '
                   '$status $body_bytes_sent $request_time';   # $uri: no query string

server {
    listen 80;
    listen [::]:80;
    server_name clipper.example.com;
    return 301 https://$host$request_uri;
}

server {
    # Works with the nginx of Ubuntu 24.04 (1.24) and Debian 12 (1.22). On nginx >= 1.25.1 this
    # form only prints a deprecation warning; there you may write `listen 443 ssl;` + `http2 on;`.
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name clipper.example.com;
    ssl_certificate     /etc/letsencrypt/live/clipper.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/clipper.example.com/privkey.pem;
    access_log /var/log/nginx/clipper.access.log clipper;

    client_max_body_size 4096m;          # = CLIPPER_MAX_UPLOAD_MB
    add_header Strict-Transport-Security "max-age=31536000" always;

    location / {
        proxy_pass http://127.0.0.1:8765;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $remote_addr;   # overwrite, never append client values
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-Host $host;
        proxy_request_buffering off;     # stream uploads to the app
        proxy_buffering off;             # stream video/ZIP responses
        proxy_read_timeout 1800s;
        proxy_send_timeout 1800s;
        client_body_timeout 300s;
        send_timeout 1800s;
    }
}
```

### Local mode behind a proxy or tunnel

If you keep `CLIPPER_MODE=local` and `CLIPPER_HOST=127.0.0.1` but let a reverse proxy or tunnel on
the same computer forward a public hostname to the app (that hostname in `CLIPPER_PUBLIC_ORIGIN` or
`CLIPPER_ALLOWED_HOSTS`), sign-in turns on automatically and needs the password hash and session
secret. Also set `CLIPPER_TRUSTED_PROXIES=127.0.0.1` in `.env`: the proxy connects from 127.0.0.1
for every visitor, so without it all visitors share one sign-in throttle and
`CLIPPER_LOGIN_MAX_FAILURES` wrong passwords from anyone lock the owner out as well. With it, the
app reads the visitor's address from the last `X-Forwarded-For` entry that is not a trusted proxy,
so the proxy must set or append the visitor's address there (as the nginx example above does).
Local mode still starts uvicorn with `--no-proxy-headers`; only the sign-in throttle uses the
header. `scripts/doctor.sh` warns when this setup has `CLIPPER_TRUSTED_PROXIES` empty.

## 7. Persistence and backups

* All state is in `/data` (`CLIPPER_DATA_DIR`): `clipper.db` (SQLite, WAL), `projects/`
  (sources, transcripts, renders), `uploads/`, `cache/` (paid AI results), `logs/`.
* Back up **while no job is processing or rendering** (the command refuses when a job is running
  unless `--force`). The database copy uses SQLite's online backup API, so it is consistent even
  while the app is up. Each backup has a `manifest.json` with the database SHA-256 and file sizes.

```bash
# Local
scripts/backup.sh --out /Volumes/Backup/clipper --archive [--include-uploads] [--include-cache]

# Docker: the backup folder must be writable by UID 10001
sudo install -d -o 10001 -g 10001 /srv/clipper-backups
docker compose --env-file deploy/.env.hosted -f deploy/compose.yaml run --rm --no-deps \
  -v /srv/clipper-backups:/backups api python -m clipper.cli backup --out /backups --archive
```

* Copy backups **off the host** (another server, object storage, an external disk) and keep
  several generations. `cache/` is optional (it avoids paying again for transcription).

Restore (and rehearse it once — a restore drill — before relying on backups):

```bash
# Local: refuses while the app runs (pid file, port, or a live worker heartbeat)
scripts/stop.sh
scripts/restore.sh --from /Volumes/Backup/clipper/clipper-backup-20261006-120000.tar.gz

# Docker
docker compose --env-file deploy/.env.hosted -f deploy/compose.yaml stop api worker
docker compose --env-file deploy/.env.hosted -f deploy/compose.yaml run --rm --no-deps \
  -v /srv/clipper-backups:/backups api python -m clipper.cli restore \
  --from /backups/clipper-backup-20261006-120000.tar.gz --yes
docker compose --env-file deploy/.env.hosted -f deploy/compose.yaml up -d
```

Restore verifies the manifest, database checksum, `integrity_check`, schema version and file
sizes **before** touching anything, then moves the current data aside (never deletes it): to
`data.bak-<timestamp>` next to the data folder, or inside `/data` when the data folder is a mount
point (Docker volume). Delete the old copy once the restored app looks right.

## 8. Upgrades and rollback

1. Back up (section 7).
2. `git pull`, then build a **new image tag**:
   `export CLIPPER_IMAGE=local-clipper:$(git rev-parse --short HEAD)` (or update `deploy/.env.hosted`),
   `docker compose … build && docker compose … up -d`. Keep the previous tag (`docker image ls local-clipper`).
3. Check `docker compose … ps`, the smoke test (section 9) and `clipper.cli doctor`.

`up -d` recreates the worker: a job that is processing at that moment is stopped and shows as
*interrupted*; press **Resume** once the new version runs (see "Interrupted jobs" in section 10).
To avoid this, upgrade while no project is processing or rendering.

Rollback: set `CLIPPER_IMAGE` back to the previous tag and `docker compose … up -d`. The database
stores its schema version; an older app **refuses to start** on a database migrated to a newer
schema ("Database schema version N is newer than this application"). In that case restore the
backup taken before the upgrade. Native installs: `git checkout <previous tag>`,
`scripts/setup.sh --prod`, restart both services.

## 9. Smoke test checklist (after every deploy)

Replace the hostname with yours. Unauthenticated checks with curl:

```bash
H=https://clipper.example.com
curl -sS "$H/api/health"                                         # {"status":"ok"}
curl -sSI "http://clipper.example.com/" | head -3                # redirect to https
curl -sSI "$H/" | grep -iE 'strict-transport|content-security'   # both headers present
for p in /api/projects /api/system /api/settings /media/projects/p_000000000000/thumb.jpg; do
  printf '%s %s\n' "$(curl -s -o /dev/null -w '%{http_code}' "$H$p")" "$p"   # expect 401, never 200
done
curl -s -o /dev/null -w '%{http_code}\n' -X POST -H 'Origin: https://evil.example' \
  -H 'Content-Type: application/json' -d '{}' "$H/api/auth/login"           # expect 403
```

In a browser (private window):

- [ ] The sign-in page appears; a wrong password is rejected; repeated failures are throttled.
- [ ] Sign in; Settings shows the OpenAI key as configured (never its value) and the expected models.
- [ ] Upload a small speech video (a few minutes), start analysis, watch real progress.
- [ ] Clips appear; play one, download it and its captions (`.srt`), export a ZIP.
- [ ] Sign out; the back button and a reload of a project URL lead to the sign-in page.
- [ ] `docker compose … logs api worker | grep -c 'sk-'` prints `0`.
- [ ] `docker compose … port api 8765` reports no published port, and `curl http://<server-ip>:8765/` from
      another machine fails to connect (only the proxy is reachable).

## 10. Troubleshooting

* **App refuses to start**: `docker compose … logs api` shows the configuration problems
  (missing hash/secret, non-https origin). Fix `deploy/.env.hosted`, then `up -d`.
* **Every login looks like it comes from one IP / throttling hits everyone**:
  `CLIPPER_TRUSTED_PROXIES` does not match the proxy's address. This applies to local mode
  behind a proxy or tunnel too (see the end of section 6); `scripts/doctor.sh` and
  the `check-config` command warn when it is empty in either setup.
* **"This address is not allowed for this app" (400)**: the browser's hostname differs from
  `CLIPPER_PUBLIC_ORIGIN`.
* **Certificate errors**: DNS not pointing at the server yet, or ports 80/443 blocked; see
  `docker compose … logs proxy`.
* **YouTube imports fail**: the worker needs outbound access; the YouTube error text in the UI
  explains the cause (bot check, private, age-restricted…). Uploading the file always works.
* **502 Bad Gateway from Caddy**: the api is not answering on `CLIPPER_PORT`; check
  `docker compose … ps` and `logs api`. After changing `CLIPPER_PORT`, run `docker compose … up -d`
  (not `restart`) so the api and the proxy are both recreated with the new value.
* **Interrupted jobs (after a restart, upgrade, crash or out-of-memory kill)**: jobs are never
  restarted automatically. On SIGTERM the worker stops FFmpeg and marks its current job
  *interrupted*; the project then says "Processing was interrupted (the app or worker stopped).
  Resume to continue where it left off." Press **Resume**: finished stages are skipped and paid
  OpenAI results already received are reused from the data folder's `cache/`, so only the unfinished work runs
  again. An interrupted re-selection keeps the previous clips; an interrupted re-render of one
  clip is simply started again from that clip.
  * If the worker could not do that itself (killed after its grace period, crashed, out of
    memory), the job keeps showing as processing for a while. The worker looks for jobs whose
    worker is gone at startup and periodically while it runs, and marks them *interrupted* within
    about a minute of the job's last heartbeat; no second restart is needed. Until then Resume,
    Re-select and Delete answer "This project is still processing", and a Cancel is remembered
    and applied when the job is recovered.
  * A SIGTERM that arrives during a long OpenAI request (transcription or clip selection can take
    minutes, up to `CLIPPER_OPENAI_TIMEOUT_SECONDS`) can outlast the worker's grace period
    (`stop_grace_period: 30s` in compose, `TimeoutStopSec=30` in systemd). The worker is then
    killed, the job is recovered as *interrupted* as above, and that one request is sent again on
    Resume. To let such a request finish during planned restarts, raise the grace period (for
    example to `330s`), or restart while nothing is processing.
