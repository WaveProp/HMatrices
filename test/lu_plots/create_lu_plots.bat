@echo off

:: First argument: path to HMatrices project
:: Second argument: path where to store plots 

if "%1"=="" (
  set JULIA_PROJECT_PATH=%~dp0\..\..
) else (
  set JULIA_PROJECT_PATH=%1
)

if "%2"=="" (
  set PLOTS_PATH=%~dp0
) else (
  set PLOTS_PATH=%2
)

:: Path to the Julia project environment
set JULIA_ENVIROMENT_PATH=%JULIA_PROJECT_PATH%\test

:: Paths to the scripts
set RUN_LU_PATH=%JULIA_PROJECT_PATH%\test\lu_plots\run_lu.jl
set CREATE_PLOTS_PATH=%JULIA_PROJECT_PATH%\test\lu_plots\create_plots.jl

:: Space-separated list of thread counts to use
set THREAD_COUNTS=1 2 4 6 8

:: Loop over each thread count
for %%t in (%THREAD_COUNTS%) do (
  echo Running LU with %%t threads...
  set JULIA_NUM_THREADS=%%t
  julia --project=%JULIA_ENVIROMENT_PATH% %RUN_LU_PATH% %PLOTS_PATH%
)

:: Create the plots
echo Creating plots...
julia --project=%JULIA_ENVIROMENT_PATH% %CREATE_PLOTS_PATH% %PLOTS_PATH%
