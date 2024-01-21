@echo off

set VENV_PATH=venv
set "CONDA_PATH=C:\Users\%USERNAME%\miniconda3\Scripts"

if not exist "%CONDA_PATH%" (
    set "CONDA_PATH=C:\ProgramData\miniconda3\Scripts"
)

if exist "%CONDA_PATH%" (
    cmd /k "%CONDA_PATH%\activate "%cd%\%VENV_PATH%""
) else (
    cmd /k "%VENV_PATH%\Scripts\activate"
)
