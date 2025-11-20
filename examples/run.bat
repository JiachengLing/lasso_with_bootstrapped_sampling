@echo off
setlocal enabledelayedexpansion

pushd "%~dp0.."

set "DATADIR=examples\data"
set "OUTROOT=examples\outputs"

if not exist "%OUTROOT%" mkdir "%OUTROOT%"

set "CMD=python -m lasso_with_bss.cli"

echo Running double-mode Lasso for all datasets...
echo.

for %%F in ("%DATADIR%\*.csv") do (
    set "FILE=%%~nxF"
    set "BASENAME=%%~nF"

    REM ==========================
    REM ====== Mode E (drop) =====
    REM ==========================
    set "OUTDIR=%OUTROOT%\!BASENAME!_E"
    if not exist "!OUTDIR!" mkdir "!OUTDIR!"

    %CMD% ^
        --data "%DATADIR%" ^
        --pattern "!FILE!" ^
        --target y ^
        --explicit-drop ^
        --n-boot 1000 ^
        --n-folds 10 ^
        --n-alphas 50 ^
        --plot-path "!OUTDIR!\alpha_mae.png" ^
        --coef-matrix-csv "!OUTDIR!\coef_matrix.csv" ^
        --coef-summary-csv "!OUTDIR!\coef_summary.csv" ^
        --report-json "!OUTDIR!\report.json"

    REM ==========================
    REM ====== Mode G (keep) =====
    REM ==========================
    set "OUTDIR=%OUTROOT%\!BASENAME!_G"
    if not exist "!OUTDIR!" mkdir "!OUTDIR!"

    %CMD% ^
        --data "%DATADIR%" ^
        --pattern "!FILE!" ^
        --target y ^
        --n-boot 1000 ^
        --n-folds 10 ^
        --n-alphas 50 ^
        --plot-path "!OUTDIR!\alpha_mae.png" ^
        --coef-matrix-csv "!OUTDIR!\coef_matrix.csv" ^
        --coef-summary-csv "!OUTDIR!\coef_summary.csv" ^
        --report-json "!OUTDIR!\report.json"
)

echo.
echo All done!
popd

