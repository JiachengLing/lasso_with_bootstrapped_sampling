@echo on
setlocal enabledelayedexpansion

rem ===== go to repo root =====
pushd %~dp0..

set OUTDIR=examples\outputs
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

set TARGET=y

rem ===== prefer console entry, fallback to python -m =====
where lasso-bss >nul 2>&1
if %errorlevel%==0 (
  set CMD=lasso-bss
) else (
  echo [Info] 'lasso-bss' not found, fallback to 'python -m lasso_with_bss.cli'
  set CMD=python -m lasso_with_bss.cli
)

rem ===== run: stdout -> file, stderr -> console (so tqdm shows) =====
%CMD% ^
  --data examples\data ^
  --target %TARGET% ^
  --n-boot 50 ^
  --n-folds 5 ^
  --n-alphas 50 ^
  --seed 42 ^
  --plot-path "%OUTDIR%\alpha_mae_curves.png" ^
  --coef-matrix-csv "%OUTDIR%\coef_matrix.csv" ^
  --coef-summary-csv "%OUTDIR%\coef_summary.csv" ^
  --report-json "%OUTDIR%\report.json" ^
  1> "%OUTDIR%\_last_stdout.txt"

if errorlevel 1 (
  echo.
  echo [ERROR] Failed. See stdout log at %OUTDIR%\_last_stdout.txt
  popd
  pause
  exit /b 1
)

echo.
echo Done. See %OUTDIR%
type "%OUTDIR%\_last_stdout.txt"

popd
pause
