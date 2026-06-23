@echo off
call "%USERPROFILE%\miniforge3\condabin\conda.bat" activate book314
cd /d "%~dp1"
jupyter notebook "%~nx1"

rem to use with double clock_: 
rem assoc .ipynb=Jupyter.Notebook
rem ftype Jupyter.Notebook="C:\modelflow2\launch_jupyter.bat" "%1"