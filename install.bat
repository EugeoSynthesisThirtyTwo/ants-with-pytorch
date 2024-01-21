@echo off
setlocal enabledelayedexpansion

set VENV_PATH=venv
set "CONDA_PATH=C:\Users\%USERNAME%\miniconda3\Scripts"

if not exist "%CONDA_PATH%" (
    set "CONDA_PATH=C:\ProgramData\miniconda3\Scripts"
)

if exist "%CONDA_PATH%" (
    echo Installation avec Conda
    %CONDA_PATH%\conda create -p %VENV_PATH% python=3.10.13 -y
    %VENV_PATH%\python -m pip install --upgrade pip
    %VENV_PATH%\Scripts\pip install -r requirements.txt
) else (
    echo Installation avec Python
    python -m venv %VENV_PATH%
    %VENV_PATH%\Scripts\python -m pip install --upgrade pip
    %VENV_PATH%\Scripts\python -m pip install -r requirements.txt
)

pause