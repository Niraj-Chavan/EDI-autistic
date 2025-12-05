@echo off
echo Creating Python 3.10 virtual environment...

REM Create virtual environment with Python 3.10
py -3.10 -m venv venv310

echo.
echo Activating virtual environment...
call venv310\Scripts\activate.bat

echo.
echo Installing dependencies...
python -m pip install --upgrade pip
pip install tensorflow==2.15.0
pip install streamlit
pip install opencv-python
pip install pillow

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo To use this environment:
echo 1. Run: venv310\Scripts\activate.bat
echo 2. Run: streamlit run app.py
echo.
pause
