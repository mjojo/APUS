@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

:: Always operate relative to the repository root where start_apus.bat is located
cd /d "%~dp0"
title APUS // Asynchronous Paged Ultra-Streaming MoE Engine [by mjojo]

set "APUS_BIN=target\release\apus.exe"
set "MODE=%~1"

echo ===============================================================================
echo    APUS // ASYNC PAGED ULTRA-STREAMING MoE ENGINE v0.1.0 (by mjojo)
echo ===============================================================================

:: 1. Ensure release binary is built
if not exist "%APUS_BIN%" (
    echo [APUS Launcher] Release binary not found. Building optimized binary...
    cargo build --release --bin apus
    if errorlevel 1 (
        echo [APUS Launcher] ERROR: Cargo build failed.
        exit /b 1
    )
)

:: 2. Ensure OOBE Hardware Wizard has initialized apus.toml
if not exist "apus.toml" (
    echo [APUS Launcher] First launch detected. Running OOBE Hardware Wizard...
    "%APUS_BIN%" wizard apus.toml
)

:: 3. Dispatch non-interactive CLI modes if passed as arguments
if /I "%MODE%"=="--test" goto :RUN_FULL_TEST
if /I "%MODE%"=="test" goto :RUN_FULL_TEST
if /I "%MODE%"=="--headless" goto :RUN_HEADLESS
if /I "%MODE%"=="tui" goto :RUN_TUI
if /I "%MODE%"=="serve" goto :RUN_SERVE
if /I "%MODE%"=="wizard" goto :RUN_WIZARD
if /I "%MODE%"=="profiles" goto :RUN_PROFILES
if /I "%MODE%"=="pull" goto :RUN_PULL
if /I "%MODE%"=="bench" goto :RUN_BENCH

:MENU
echo.
echo  Select launch mode:
echo    [1] Launch 60 FPS Aerospace Telemetry TUI (apus run)
echo    [2] Start OpenAI / Anthropic HTTP+SSE Server on port 8080 (apus serve)
echo    [3] Run Hardware ^& Storage Auto-Tuner Wizard (apus wizard)
echo    [4] List Supported Frontier MoE Profiles ^& Disk Readiness (apus profiles)
echo    [5] Open Interactive Model Hub / Importer (apus pull)
echo    [6] Run Full End-to-End Diagnostic ^& Benchmark Suite (apus --test)
echo    [0] Exit
echo.
set /p "CHOICE=Enter selection [0-6]: "

if "%CHOICE%"=="1" goto :RUN_TUI
if "%CHOICE%"=="2" goto :RUN_SERVE
if "%CHOICE%"=="3" goto :RUN_WIZARD
if "%CHOICE%"=="4" goto :RUN_PROFILES
if "%CHOICE%"=="5" goto :RUN_PULL
if "%CHOICE%"=="6" goto :RUN_FULL_TEST
if "%CHOICE%"=="0" exit /b 0
goto :MENU

:RUN_TUI
"%APUS_BIN%" run %2 %3
exit /b %ERRORLEVEL%

:RUN_HEADLESS
"%APUS_BIN%" run --headless
exit /b %ERRORLEVEL%

:RUN_SERVE
set "PORT=%~2"
if "%PORT%"=="" set "PORT=8080"
"%APUS_BIN%" serve %PORT% %3
exit /b %ERRORLEVEL%

:RUN_WIZARD
"%APUS_BIN%" wizard apus.toml %2
exit /b %ERRORLEVEL%

:RUN_PROFILES
"%APUS_BIN%" profiles
exit /b %ERRORLEVEL%

:RUN_PULL
"%APUS_BIN%" pull %2
exit /b %ERRORLEVEL%

:RUN_BENCH
"%APUS_BIN%" bench %2
exit /b %ERRORLEVEL%

:RUN_FULL_TEST
echo.
echo [Step 1/7] Running OOBE Hardware ^& Disk Volume Audit...
"%APUS_BIN%" wizard apus.toml
if errorlevel 1 exit /b 1

echo.
echo [Step 2/7] Checking Frontier MoE Profiles Catalog...
"%APUS_BIN%" profiles
if errorlevel 1 exit /b 1

echo.
echo [Step 3/7] Testing GGUF v3 -^> 4KB Page-Aligned .apus Importer...
set "TMP_APUS=%TEMP%\apus_batch_test.apus"
"%APUS_BIN%" import sample_moe.gguf "%TMP_APUS%" 2 4
if errorlevel 1 exit /b 1

echo.
echo [Step 4/7] Inspecting 512B Header, LBA Offset Table ^& BLAKE3 Root...
"%APUS_BIN%" inspect "%TMP_APUS%"
if errorlevel 1 exit /b 1

echo.
echo [Step 5/7] Running End-to-End SSD Streaming, AVX2 FMA3 ^& .roost Benchmark...
"%APUS_BIN%" bench "%TMP_APUS%"
if errorlevel 1 exit /b 1

echo.
echo [Step 6/7] Running Mathematical SQNR/Perplexity Audit ^& 8-Thread Stress Test...
"%APUS_BIN%" audit
if errorlevel 1 exit /b 1
"%APUS_BIN%" stress
if errorlevel 1 exit /b 1

echo.
echo [Step 7/7] Verifying 60 FPS Boot Splash ^& Telemetry HUD Rendering...
"%APUS_BIN%" run --headless
if errorlevel 1 exit /b 1

if exist "%TMP_APUS%" del /f /q "%TMP_APUS%" >nul 2>&1
if exist "coding_rust.roost" del /f /q "coding_rust.roost" >nul 2>&1

echo.
echo ===============================================================================
echo  [APUS Launcher] ALL 7 END-TO-END DIAGNOSTIC STAGES PASSED SUCCESSFULLY!
echo ===============================================================================
exit /b 0
