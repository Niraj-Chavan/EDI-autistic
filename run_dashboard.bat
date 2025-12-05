@echo off
echo Starting Autism Detection Dashboard...
echo.

REM Activate virtual environment
call venv310\Scripts\activate.bat

REM Update app.py to use original model
echo Updating app to use original model...

REM Run streamlit
streamlit run app.py

pause
