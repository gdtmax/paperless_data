#!/bin/sh
# Continuous generator loop.
#
# The generator.py CLI exits after --duration seconds. For unattended
# operation we wrap it in an infinite restart loop with a small sleep
# between cycles. Between cycles the bot is idle — that tiny gap is
# harmless.
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
i=0
while [ $i -lt 60 ]; do
    # Reachable = any HTTP response (including 4xx auth errors). Only
    # network errors (connection refused, DNS failure, timeout) count as
    # not-ready. Using urllib's HTTPError to distinguish the two.
    if python3 -c "
import sys, urllib.request, urllib.error
req = urllib.request.Request('${PAPERLESS_URL}/api/')
try:
    urllib.request.urlopen(req, timeout=3)
    sys.exit(0)
except urllib.error.HTTPError:
    # Server responded with a 4xx/5xx — that still means it's up
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
        echo "[loop] Paperless reachable."
        break
    fi
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
