@echo off
REM Simple launcher for "Lightmap Tool.py"
REM Behavior: prefer venv's pythonw (no console), fall back to python; start the GUI in a separate process and exit this script.

SETLOCAL
SET "SCRIPT_DIR=%~dp0"

REM Prefer pythonw (no console) in virtualenv
IF EXIST "%SCRIPT_DIR%\.venv\Scripts\pythonw.exe" (
    START "" "%SCRIPT_DIR%\.venv\Scripts\pythonw.exe" "%SCRIPT_DIR%Lightmap Tool.py"
    GOTO :EOF
)

REM Fallback to python in virtualenv
IF EXIST "%SCRIPT_DIR%\.venv\Scripts\python.exe" (
    START "" "%SCRIPT_DIR%\.venv\Scripts\python.exe" "%SCRIPT_DIR%Lightmap Tool.py"
    GOTO :EOF
)

REM Try system pythonw (no console) if available
where pythonw >nul 2>&1
IF %ERRORLEVEL%==0 (
    START "" pythonw "%SCRIPT_DIR%Lightmap Tool.py"
    GOTO :EOF
)

REM Final fallback to system python
START "" python "%SCRIPT_DIR%Lightmap Tool.py"

:EOF
ENDLOCAL
EXIT /B 0
