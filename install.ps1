# shellm installer (Windows). makes a private python environment in
# %USERPROFILE%\.shellm and a `shellm` command. run it again any time to update.
#
#   from a checkout:   powershell -ExecutionPolicy Bypass -File install.ps1
#
# knobs: $env:SHELLM_HOME, $env:SHELLM_BRANCH (default main), $env:SHELLM_REPO
$ErrorActionPreference = "Stop"

$Repo    = if ($env:SHELLM_REPO)   { $env:SHELLM_REPO }   else { "https://github.com/ArthurJClack/clack" }
$Branch  = if ($env:SHELLM_BRANCH) { $env:SHELLM_BRANCH } else { "main" }
$HomeDir = if ($env:SHELLM_HOME)   { $env:SHELLM_HOME }   else { Join-Path $HOME ".shellm" }
$BinDir  = Join-Path $HomeDir "bin"

function Say($msg) { Write-Host "> $msg" -ForegroundColor Cyan }
function Die($msg) { Write-Host "x $msg" -ForegroundColor Red; exit 1 }

# 1. python 3.10+
$Py = $null
foreach ($cand in @(@("py", "-3"), @("python"), @("python3"))) {
    $exe = $cand[0]
    $pre = @()
    if ($cand.Count -gt 1) { $pre = $cand[1..($cand.Count - 1)] }
    if (Get-Command $exe -ErrorAction SilentlyContinue) {
        $ok = $null
        try { $ok = & $exe @pre -c "import sys; print(int(sys.version_info >= (3, 10)))" 2>$null } catch { }
        if ("$ok".Trim() -eq "1") { $Py = @{ Exe = $exe; Pre = $pre }; break }
    }
}
if (-not $Py) { Die "python 3.10 or newer is needed: https://www.python.org/downloads/ (tick 'Add python.exe to PATH')" }
$PyExe = $Py.Exe
$PyPre = $Py.Pre
Say ("using " + (& $PyExe @PyPre --version))

# 2. the code: this folder when the script sits in a checkout, otherwise a clone
$Src = $null
if ($PSScriptRoot -and (Test-Path (Join-Path $PSScriptRoot "shellm\tui.py"))) {
    $Src = $PSScriptRoot
} else {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Die "git is needed to download shellm: https://git-scm.com/download/win" }
    $Src = Join-Path $HomeDir "src"
    if (Test-Path (Join-Path $Src ".git")) {
        Say "updating $Src"
        git -C $Src pull -q --ff-only
    } else {
        Say "downloading shellm ($Branch)"
        New-Item -ItemType Directory -Force -Path $HomeDir | Out-Null
        git clone -q --depth 1 -b $Branch $Repo $Src
    }
    if ($LASTEXITCODE -ne 0) { Die "download failed" }
}

# 3. a private environment, so nothing else on the machine is touched
Say "setting up $HomeDir\venv"
& $PyExe @PyPre -m venv (Join-Path $HomeDir "venv")
if ($LASTEXITCODE -ne 0) { Die "could not create a python environment" }
$VPy = Join-Path $HomeDir "venv\Scripts\python.exe"
& $VPy -m pip install -q --upgrade pip
Say "installing shellm (the first time, this downloads torch and can take a few minutes)"
& $VPy -m pip install -q -e $Src
if ($LASTEXITCODE -ne 0) { Die "install failed" }

# 4. the command
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
$Exe = Join-Path $HomeDir "venv\Scripts\shellm.exe"
Set-Content -Path (Join-Path $BinDir "shellm.cmd") -Value "@echo off`r`n`"$Exe`" %*" -Encoding ASCII
& $Exe info | Out-Null
if ($LASTEXITCODE -ne 0) { Die "installed, but the bundled model did not load" }

$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (-not $UserPath) { $UserPath = "" }
if (($UserPath -split ";") -notcontains $BinDir) {
    [Environment]::SetEnvironmentVariable("Path", ($UserPath.TrimEnd(";") + ";" + $BinDir).TrimStart(";"), "User")
    Say "added $BinDir to your PATH (open a new terminal for it to take effect)"
}
Say "done. open a new terminal and start it with:  shellm"
