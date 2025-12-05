@echo off
echo Setting up Kaggle API...

REM Create .kaggle directory
if not exist "%USERPROFILE%\.kaggle" mkdir "%USERPROFILE%\.kaggle"

REM Create kaggle.json file
echo {"username":"your_username","key":"KGAT_22c3a416170d4af5b92ca5148857d8f3"} > "%USERPROFILE%\.kaggle\kaggle.json"

echo.
echo Kaggle API configured!
echo.
echo Now downloading the model...
call venv310\Scripts\activate.bat
kaggle kernels output ibrahimnibrahim/autism-spectrum-test-accuracy-96 -p .

echo.
echo Done! Check for downloaded files.
pause
