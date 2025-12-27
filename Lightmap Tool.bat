@echo off
REM Launcher for Lightmap Tool (Windows batch)
REM Place this file in the same folder as "Lightmap Tool.py" and double-click to run.

chcp 65001 >nul
pushd "%~dp0"

REM Prefer the Windows Python launcher if available
where py >nul 2>nul
if %ERRORLEVEL%==0 (
    py -3 "Lightmap Tool.py" %*
) else (
    where python >nul 2>nul
    if %ERRORLEVEL%==0 (
        python "Lightmap Tool.py" %*
    ) else (
        echo Python 3 not found in PATH.
        echo Install Python 3 and ensure "python" or the "py" launcher is available.
        pause
    )
)

popd
echo.
echo Press any key to exit...
pause >nul
