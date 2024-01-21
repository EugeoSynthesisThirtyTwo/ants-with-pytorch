@echo off

set "VENV_PATH=%cd%\venv"
set "CONDA_PATH=C:\Users\%USERNAME%\miniconda3\Scripts"

if not exist "%CONDA_PATH%" (
    set "CONDA_PATH=C:\ProgramData\miniconda3\Scripts"
)

if exist "%CONDA_PATH%" (
    "%VENV_PATH%\python" main.py
) else (
    "%VENV_PATH%\Scripts\python" main.py
)

pause > nul