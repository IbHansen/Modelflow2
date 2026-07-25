@echo off
REM Rebuild the mf-workbench JupyterLab extension.
REM Assumes the book314 conda env is already activated (needs node + jupyterlab).
REM After it finishes: hard-refresh the JupyterLab tab (Ctrl+Shift+R) for TS changes,
REM and restart the notebook kernel for kernel.py changes.

REM Run from this script's own folder regardless of the caller's cwd.
cd /d "%~dp0"

echo === Rebuilding mf-workbench (jlpm run build) ===
call jlpm run build
if errorlevel 1 (
    echo.
    echo *** Build FAILED ***
    exit /b 1
)

echo.
echo === Build OK ===
echo Hard-refresh JupyterLab (Ctrl+Shift+R) for TS changes;
echo restart the notebook kernel for kernel.py changes.
