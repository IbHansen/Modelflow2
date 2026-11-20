@echo off
rem ==========================================================
rem  Build a fresh ModelFlow dev environment (default name book314b)
rem
rem  Usage (plain cmd or Miniforge Prompt):
rem     make_book_env.bat                       creates book314b
rem     make_book_env.bat book315               creates another name
rem     make_book_env.bat book314b updatebase   updates Miniforge base first
rem
rem  Limits and why:
rem     pandas <3.1      gamspy needs it
rem     jupyterlab <4.6  jupyterlab-myst 2.7.0 does not work on 4.6
rem     py2eviews comes from the eviews channel
rem  If anaconda.org gives HTTP 503 (ibh / eviews channels), run again later.
rem ==========================================================
setlocal

set "ENVNAME=%~1"
if "%ENVNAME%"=="" set "ENVNAME=book314b"
set "MINIFORGE=C:\Users\ibhan\miniforge3"
set "MODELFLOW=C:\modelflow2\modelflow"

call "%MINIFORGE%\Scripts\activate.bat" base
if errorlevel 1 goto :fail

rem --- refuse to overwrite an existing environment ---
if exist "%MINIFORGE%\envs\%ENVNAME%\conda-meta" (
    echo *** Environment %ENVNAME% already exists - remove it first or use another name:
    echo     conda env remove -y -n %ENVNAME%
    exit /b 1
)

rem === 1. Optional: update Miniforge base ===
if /i "%~2"=="updatebase" (
    echo === Updating Miniforge base ===
    call conda update -y -n base -c conda-forge conda mamba
    if errorlevel 1 goto :fail
)

rem === 2. Create the environment ===
echo === Creating %ENVNAME% ===
call conda create -y -n %ENVNAME% -c ibh -c conda-forge python=3.14 ibh::modelflow "pandas>=3.0,<3.1" "jupyterlab>=4.5,<4.6" papermill jupyterlab_rise spyder-kernels=3.1 eviews::py2eviews jupyterlab-myst anywidget ffmpeg nodejs python-mumps pymetis
if errorlevel 1 goto :fail
call conda activate %ENVNAME%
if errorlevel 1 goto :fail

rem === 3. Pins, so later conda updates keep the limits ===
echo pandas ^<3.1>> "%CONDA_PREFIX%\conda-meta\pinned"
echo jupyterlab ^<4.6>> "%CONDA_PREFIX%\conda-meta\pinned"

rem === 4. Pip extras ===
echo === Pip extras ===
python -m pip install jupyter-book pypardiso dash-interactive-graphviz gamspy
if errorlevel 1 goto :fail

rem === 5. mf-workbench JupyterLab extension (editable) ===
echo === mf-workbench ===
python -m pip install -e "%MODELFLOW%\mf-workbench"
if errorlevel 1 goto :fail
call "%MODELFLOW%\mf-workbench\rebuild.bat"
if errorlevel 1 goto :fail
cd /d C:\modelflow2

rem === 6. Build the Dash JupyterLab extension (a few minutes; failure is only a warning) ===
echo === jupyter lab build ===
call jupyter lab build
if errorlevel 1 echo *** jupyter lab build failed - only needed for @plotly/dash-jupyterlab

rem === 7. Check ===
echo.
echo === Check ===
python -c "import pypardiso, gamspy, mumps, pandas, modelclass; print('imports ok, pandas', pandas.__version__)"
call conda list pandas | findstr /b "pandas"
call jupyter labextension list
echo --- pinned:
type "%CONDA_PREFIX%\conda-meta\pinned"

echo.
echo === Done: %ENVNAME% ===
echo pandas should be 3.0.x from conda-forge (not pypi);
echo jupyterlab-myst and mf-workbench should say "enabled ok" (mathjax3-extension X is harmless).
echo To use it for Claude, set line 2 of C:\modelflow2\launch_claude.bat to:
echo     call %MINIFORGE%\Scripts\activate.bat %ENVNAME%
exit /b 0

:fail
echo.
echo *** Failed - see the output above.
exit /b 1
