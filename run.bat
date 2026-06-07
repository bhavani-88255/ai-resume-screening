@echo off
echo ========================================
echo    Resume Screening System
echo ========================================
echo.

:: Change to the folder where this bat file is located
cd /d "%~dp0"

:: Find Python
for /f "delims=" %%i in ('where python 2^>nul') do set PYTHON=%%i
if "%PYTHON%"=="" (
    for /f "delims=" %%i in ('where py 2^>nul') do set PYTHON=py
)
if "%PYTHON%"=="" (
    echo Python not found! Please install Python from python.org
    pause
    exit
)

echo Python found: %PYTHON%
echo.

echo Installing required libraries...
%PYTHON% -m pip install streamlit pandas numpy scikit-learn nltk pdfplumber --quiet
echo.
echo Starting app... Browser will open automatically!
echo.
%PYTHON% -m streamlit run app.py
pause
