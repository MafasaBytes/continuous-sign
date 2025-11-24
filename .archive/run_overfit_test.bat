@echo off
REM Run baseline MobileNetV3 overfit test

echo ==========================================
echo Running MobileNetV3 Baseline Overfit Test
echo ==========================================

python overfit_test.py

echo.
echo Test complete! Check the following files:
echo   - overfit_test_results.png
echo   - overfit_test_report.txt

pause

