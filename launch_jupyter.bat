@echo off
call "%USERPROFILE%\miniforge3\condabin\conda.bat" activate book314
cd /d "%~dp1"
jupyter notebook "%~nx1"

rem ------------------------------------------------------------
rem How to use this from Windows Explorer:
rem
rem 1. Save this file, for example as:
rem       <FOLDER>\launch_jupyter.bat
rem
rem 2. In Explorer, right-click an .ipynb file.
rem
rem 3. Choose:
rem       Open with -> Choose another app
rem
rem 4. Scroll down and choose:
rem       Choose an app on your PC
rem
rem 5. Browse to:
rem       <FOLDER>\launch_jupyter.bat
rem
rem 6. Tick:
rem       Always use this app to open .ipynb files
rem
rem 7. Click:
rem       Open
rem
rem After this, double-clicking an .ipynb file should start
rem Jupyter Notebook in the book314 conda environment.
rem ------------------------------------------------------------