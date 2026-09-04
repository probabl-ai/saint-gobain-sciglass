# Mint / reuse a Hub key via ``skore agent --harness copilot``, then start
# interactive Copilot CLI or print GitHub Copilot desktop setup.
# Windows counterpart of skore-copilot.sh (that script is macOS / Linux).
param(
    [Parameter(Position = 0)]
    [string]$Mode
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Show-Usage {
    @"
Usage: scripts/skore-copilot.ps1 {cli|desktop}

  cli       Run ``skore agent --harness copilot``, then start an interactive
            Copilot CLI session against Skore Hub (no extra copilot flags).
  desktop   Same key retrieval, copy the Hub API key to the clipboard when
            possible, and print GitHub Copilot app (desktop) install steps.
            Does not print the key.

The Hub key is stored in gitignored ``.skore``. This script does not read ``.env``.
"@
}

if ($Mode -in @("-h", "--help", "-?", "/?")) {
    Show-Usage
    exit 0
}
if ($Mode -notin @("cli", "desktop")) {
    Show-Usage
    exit 1
}

if (-not (Get-Command skore -ErrorAction SilentlyContinue)) {
    Write-Error "skore is not on PATH (need skore-cli with --harness copilot)."
}

$shadow = Join-Path ([System.IO.Path]::GetTempPath()) ("skore-copilot-" + [guid]::NewGuid().ToString("n"))
New-Item -ItemType Directory -Path $shadow | Out-Null
$stub = "@echo off`r`nexit /b 0`r`n"
Set-Content -Path (Join-Path $shadow "code.cmd") -Value $stub -NoNewline
Set-Content -Path (Join-Path $shadow "code-insiders.cmd") -Value $stub -NoNewline
# shutil.which("code") also matches code.exe on Windows.
Set-Content -Path (Join-Path $shadow "code.bat") -Value $stub -NoNewline
Set-Content -Path (Join-Path $shadow "code-insiders.bat") -Value $stub -NoNewline

$env:PATH = "$shadow;$env:PATH"
try {
    Write-Host "Running skore agent --harness copilot (VS Code window suppressed; may still write user chatLanguageModels.json)..."
    & skore agent --harness copilot --workspace $Root
    if ($LASTEXITCODE -ne 0) {
        throw "skore agent failed with exit code $LASTEXITCODE"
    }
}
finally {
    Remove-Item -Recurse -Force $shadow -ErrorAction SilentlyContinue
}

$skoreFile = Join-Path $Root ".skore"
if (-not (Test-Path $skoreFile)) {
    Write-Error "skore agent did not write $skoreFile"
}

$cfg = Get-Content -Raw -Path $skoreFile | ConvertFrom-Json
$hubKey = [string]$cfg.api_key
$hubUrl = ([string]$cfg.hub_url).TrimEnd("/")
if (-not $hubKey -or -not $hubUrl) {
    Write-Error "invalid .skore: missing hub_url or api_key"
}
$baseUrl = "$hubUrl/v1"

if ($Mode -eq "desktop") {
    try {
        Set-Clipboard -Value $hubKey
        Write-Host "Hub API key copied to the clipboard."
    }
    catch {
        Write-Host "Could not copy to the clipboard. Paste the api_key field from .skore (gitignored)."
    }
    @"

GitHub Copilot desktop app (not VS Code, not Copilot CLI)
---------------------------------------------------------
1. Open the GitHub Copilot app.
2. Settings → Model providers → Add provider.
3. Choose OpenAI-compatible.
4. Base URL: $baseUrl
   (stop at /v1 — do not append /chat/completions)
5. API format / wire API: Chat completions.
6. API key: paste from the clipboard (Hub key; header Hub expects is X-API-Key).
   If the form has Custom headers, also add X-API-Key with the same value.
7. Add a model:
   - Display name: Skore Agent
   - Model identifier: skore-agent
8. Pick that model in the session picker.

There is no supported file/API to write desktop providers from a script;
credentials live in the OS credential store. Re-run this command if the key is missing.

Docs: https://docs.github.com/en/copilot/how-tos/github-copilot-app/use-byok-models
"@
    exit 0
}

if (-not (Get-Command copilot -ErrorAction SilentlyContinue)) {
    Write-Error "copilot CLI is not on PATH."
}

$env:COPILOT_PROVIDER_TYPE = "openai"
$env:COPILOT_PROVIDER_BASE_URL = $baseUrl
$env:COPILOT_PROVIDER_WIRE_API = "completions"
$env:COPILOT_MODEL = "skore-agent"
$env:COPILOT_PROVIDER_HEADERS = "X-API-Key: $hubKey"

Set-Location $Root
& copilot
exit $LASTEXITCODE
