@echo off
setlocal
pushd "%~dp0"

set "PYTHON_BIN="
if defined NEUROCOREG_PYTHON (
    if exist "%NEUROCOREG_PYTHON%" set "PYTHON_BIN=%NEUROCOREG_PYTHON%"
)
if not defined PYTHON_BIN (
    if exist ".venv\Scripts\python.exe" set "PYTHON_BIN=.venv\Scripts\python.exe"
)
if not defined PYTHON_BIN (
    if exist "%LOCALAPPDATA%\neurocoreg-venv\Scripts\python.exe" set "PYTHON_BIN=%LOCALAPPDATA%\neurocoreg-venv\Scripts\python.exe"
)
if not defined PYTHON_BIN (
    where python >nul 2>nul && set "PYTHON_BIN=python"
)
if not defined PYTHON_BIN (
    where py >nul 2>nul && set "PYTHON_BIN=py -3"
)

%PYTHON_BIN% -c "import numpy, scipy, nibabel" >nul 2>nul
if errorlevel 1 (
    echo Setting up user-space Python environment in "%LOCALAPPDATA%\neurocoreg-venv" ^(no Administrator privileges required^)...
    %PYTHON_BIN% -m venv "%LOCALAPPDATA%\neurocoreg-venv"
    "%LOCALAPPDATA%\neurocoreg-venv\Scripts\python.exe" -m pip install --upgrade pip >nul 2>nul
    "%LOCALAPPDATA%\neurocoreg-venv\Scripts\python.exe" -m pip install -r requirements.txt
    set "PYTHON_BIN=%LOCALAPPDATA%\neurocoreg-venv\Scripts\python.exe"
)

if not exist "sample_data\nifti\sub-01_T1w.nii.gz" (
    echo Generating sample single-subject multimodal NIfTI ^& DICOM datasets...
    %PYTHON_BIN% generate_sample_data.py
)

if not defined PORT set "PORT=8765"
echo Starting NeuroCoreg Studio on http://127.0.0.1:%PORT% ...
%PYTHON_BIN% server.py
popd
