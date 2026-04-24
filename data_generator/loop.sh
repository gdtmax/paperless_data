#!/bin/sh
# Continuous generator loop.
#
# The generator.py CLI exits after --duration seconds. For unattended
# operation we wrap it in an infinite restart loop with a small sleep
# between cycles.
#
# Required env: PAPERLESS_URL, PAPERLESS_TOKEN
# Optional env: RATE (default 0.5), CYCLE_DURATION (default 600),
#               UPLOAD_ONLY (default true), SLEEP_BETWEEN (default 5)
#
# Waits for Paperless to be reachable before starting — on a fresh bring-up
# the webserver container needs ~30s to finish migrations, and hammering it
# during that window just produces connection errors.

set -e

: "${PAPERLESS_URL:?PAPERLESS_URL not set}"
: "${PAPERLESS_TOKEN:?PAPERLESS_TOKEN not set}"
: "${RATE:=0.5}"
: "${CYCLE_DURATION:=600}"
: "${UPLOAD_ONLY:=true}"
: "${SLEEP_BETWEEN:=5}"

echo "[loop] waiting for Paperless at ${PAPERLESS_URL} ..."
# Probe a real authenticated endpoint with a known 200-or-401 contract.
# /api/ returns 302 → /api/schema/view/, which urllib auto-follows and
# can fail confusingly; /api/documents/ returns 200 with auth or 401
# without, but both prove the server is UP. We accept any HTTP status
# code (200, 302, 401, 403) and only treat network-level errors
# (ConnectionRefused, DNS failure, timeout) as "still booting".
i=0
while [ $i -lt 60 ]; do
    CODE=$(python3 -c "
import sys, urllib.request, urllib.error
req = urllib.request.Request(
    '${PAPERLESS_URL}/api/documents/?page_size=1',
    headers={'Authorization': 'Token ${PAPERLESS_TOKEN}'},
)
try:
    r = urllib.request.urlopen(req, timeout=3)
    print(r.status); sys.exit(0)
except urllib.error.HTTPError as e:
    # Any HTTP response = server is up
    print(e.code); sys.exit(0)
except Exception as e:
    # Network-level failure = still booting
    print('NET_ERR:%s' % type(e).__name__); sys.exit(1)
" 2>/dev/null) && {
        echo "[loop] Paperless reachable (HTTP ${CODE})."
        break
    }
    echo "[loop] not ready yet: ${CODE}"
    i=$((i + 1))
    sleep 5
done
if [ $i -ge 60 ]; then
    echo "[loop] Paperless never became reachable — aborting to prevent restart storm"
    exit 1
fi

CYCLE=0
while true; do
    CYCLE=$((CYCLE + 1))
    echo ""
    echo "[loop] ================================================================"
    echo "[loop] cycle ${CYCLE} starting"
    echo "[loop] rate=${RATE}/s duration=${CYCLE_DURATION}s upload_only=${UPLOAD_ONLY}"
    echo "[loop] ================================================================"

    ARGS="--rate ${RATE} --duration ${CYCLE_DURATION}"
    case "${UPLOAD_ONLY}" in
        true|True|1|yes) ARGS="${ARGS} --upload-only" ;;
    esac

    # shellcheck disable=SC2086
    python3 generator.py ${ARGS} || echo "[loop] cycle ${CYCLE} exited non-zero; continuing"

    echo "[loop] sleeping ${SLEEP_BETWEEN}s before next cycle..."
    sleep "${SLEEP_BETWEEN}"
done
