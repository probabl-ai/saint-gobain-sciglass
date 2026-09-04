#!/usr/bin/env bash
# Mint / reuse a Hub key via ``skore agent --harness copilot``, then start
# interactive Copilot CLI or print GitHub Copilot desktop setup.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_ID="skore-agent"
FORCE=0
MODE=""

usage() {
  cat <<'EOF'
Usage: scripts/skore-copilot.sh {cli|desktop} [--force]

  cli       Starts Copilot CLI session with skore-agent.
  desktop   Retrieves and copy the Hub API key to the clipboard and print GitHub Copilot
            app (desktop) install steps.
  --force   Run ``skore agent --harness copilot`` even if ``.skore`` already exists.

``skore agent`` runs only when ``.skore`` is missing, or with ``--force``.
The Hub key is stored in gitignored ``.skore``. This script does not read ``.env``.
EOF
}

for arg in "$@"; do
  case "$arg" in
    -h | --help)
      usage
      exit 0
      ;;
    --force)
      FORCE=1
      ;;
    cli | desktop)
      if [[ -n "$MODE" ]]; then
        usage >&2
        exit 1
      fi
      MODE="$arg"
      ;;
    *)
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$MODE" ]]; then
  usage >&2
  exit 1
fi

SKORE_FILE="$ROOT/.skore"
command -v python3 >/dev/null || {
  echo "python3 is required to read .skore." >&2
  exit 1
}

run_skore_agent() {
  command -v skore >/dev/null || {
    echo "skore is not on PATH (need skore-cli with --harness copilot)." >&2
    exit 1
  }

  local shadow
  shadow="$(mktemp -d "${TMPDIR:-/tmp}/skore-copilot.XXXXXX")"
  cleanup_shadow() { rm -rf "$shadow"; }
  trap cleanup_shadow EXIT
  cat >"$shadow/code" <<'EOF'
#!/bin/sh
exit 0
EOF
  chmod +x "$shadow/code"
  cp "$shadow/code" "$shadow/code-insiders"

  echo "Running skore agent --harness copilot (VS Code window suppressed; may still write user chatLanguageModels.json)..."
  PATH="$shadow:$PATH" skore agent --harness copilot --workspace "$ROOT" --model-id "$MODEL_ID"
  cleanup_shadow
  trap - EXIT
}

if [[ ! -f "$SKORE_FILE" || "$FORCE" -eq 1 ]]; then
  run_skore_agent
  [[ -f "$SKORE_FILE" ]] || {
    echo "skore agent did not write $SKORE_FILE" >&2
    exit 1
  }
fi

[[ -f "$SKORE_FILE" ]] || {
  echo "missing $SKORE_FILE; run without an existing key or pass --force." >&2
  exit 1
}

eval "$(
  python3 - "$SKORE_FILE" <<'PY'
import json, shlex, sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
key = data.get("api_key") or ""
hub = (data.get("hub_url") or "").rstrip("/")
if not key or not hub:
    sys.stderr.write("invalid .skore: missing hub_url or api_key\n")
    sys.exit(1)
print("HUB_URL=" + shlex.quote(hub))
print("HUB_KEY=" + shlex.quote(key))
PY
)"
BASE_URL="${HUB_URL}/v1"

if [[ "$MODE" == "desktop" ]]; then
  if command -v pbcopy >/dev/null && printf '%s' "$HUB_KEY" | pbcopy; then
    echo "Hub API key copied to the clipboard."
  elif command -v wl-copy >/dev/null && printf '%s' "$HUB_KEY" | wl-copy; then
    echo "Hub API key copied to the clipboard."
  elif command -v xclip >/dev/null && printf '%s' "$HUB_KEY" | xclip -selection clipboard; then
    echo "Hub API key copied to the clipboard."
  else
    echo "Could not copy to a clipboard tool. Paste the api_key field from .skore (gitignored)."
  fi
  cat <<EOF

GitHub Copilot desktop app (not VS Code, not Copilot CLI)
---------------------------------------------------------
1. Open the GitHub Copilot app.
2. Settings → Model providers → Add provider.
3. Choose OpenAI-compatible.
4. Base URL: ${BASE_URL}
5. API format / wire API: Chat completions.
6. API key: paste from the clipboard (header X-API-Key).
   If the form has Custom headers, also add X-API-Key with the same value.
7. Add a model:
   - Display name: Skore Agent
   - Model identifier: ${MODEL_ID}
8. Pick that model in the session picker.

There is no supported file/API to write desktop providers from a script;
credentials live in the OS keychain. Re-run this command if the key is missing.

EOF
  exit 0
fi

command -v copilot >/dev/null || {
  echo "copilot CLI is not on PATH." >&2
  exit 1
}

export COPILOT_PROVIDER_TYPE=openai
export COPILOT_PROVIDER_BASE_URL="$BASE_URL"
export COPILOT_PROVIDER_WIRE_API=completions
export COPILOT_MODEL="$MODEL_ID"
export COPILOT_PROVIDER_HEADERS="X-API-Key: ${HUB_KEY}"

cd "$ROOT"
exec copilot
