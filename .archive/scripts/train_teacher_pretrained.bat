@echo off
REM ################################################################################
REM Train Teacher Model with Pre-trained Weights (Windows)
REM ################################################################################

setlocal enabledelayedexpansion

set DATA_DIR=data/teacher_features/mediapipe_full
set OUTPUT_DIR=checkpoints/teacher
set BATCH_SIZE=16
set NUM_WORKERS=0
set SEED=42

echo =====================================
echo Teacher Model Training Script
echo With Pre-trained Weights Support
echo =====================================
echo.

REM Check if strategy is provided
if "%~1"=="" (
    echo Usage: %0 [STRATEGY] [OPTIONS]
    echo.
    echo Strategies:
    echo   baseline              - Train from scratch
    echo   feature-extraction    - Freeze backbone, train classifier only
    echo   fine-tune             - Fine-tune top layers
    echo   progressive           - Progressive unfreezing (recommended)
    echo   full-finetune         - Fine-tune all layers
    echo.
    echo Examples:
    echo   %0 progressive
    echo   %0 fine-tune
    echo.
    exit /b 1
)

set STRATEGY=%1
shift

REM Strategy-specific configurations
if "%STRATEGY%"=="baseline" (
    echo Strategy: Baseline (No pre-trained weights^)
    set PRETRAINED=
    set EPOCHS=50
    set LR=5e-4
    set EXTRA_ARGS=
) else if "%STRATEGY%"=="feature-extraction" (
    echo Strategy: Feature Extraction
    set PRETRAINED=checkpoints/teacher/best_i3d.pth
    set EPOCHS=20
    set LR=1e-3
    set EXTRA_ARGS=--freeze_backbone --freeze_until_layer mixed_5c
) else if "%STRATEGY%"=="fine-tune" (
    echo Strategy: Fine-tuning Top Layers
    set PRETRAINED=checkpoints/teacher/best_i3d.pth
    set EPOCHS=30
    set LR=5e-4
    set EXTRA_ARGS=--freeze_backbone --freeze_until_layer mixed_4f
) else if "%STRATEGY%"=="progressive" (
    echo Strategy: Progressive Unfreezing (Recommended^)
    set PRETRAINED=checkpoints/teacher/best_i3d.pth
    set EPOCHS=50
    set LR=1e-4
    set EXTRA_ARGS=--freeze_backbone --freeze_until_layer mixed_4f --progressive_unfreeze --unfreeze_stages 4
) else if "%STRATEGY%"=="full-finetune" (
    echo Strategy: Full Fine-tuning
    set PRETRAINED=checkpoints/teacher/best_i3d.pth
    set EPOCHS=50
    set LR=1e-5
    set EXTRA_ARGS=
) else (
    echo Unknown strategy: %STRATEGY%
    exit /b 1
)

REM Check if pre-trained weights exist
if not "%PRETRAINED%"=="" (
    if not "%PRETRAINED%"=="i3d_kinetics400" (
        if not exist "%PRETRAINED%" (
            echo Error: Pre-trained weights not found at: %PRETRAINED%
            echo.
            echo Options:
            echo   1. Train baseline model first: %0 baseline
            echo   2. Use different weights path
            exit /b 1
        )
    )
)

echo.
echo Configuration:
echo   Data directory: %DATA_DIR%
echo   Output directory: %OUTPUT_DIR%
echo   Pre-trained weights: %PRETRAINED%
echo   Epochs: %EPOCHS%
echo   Learning rate: %LR%
echo   Batch size: %BATCH_SIZE%
echo.

echo Starting training...
echo.

REM Build and run the command
set CMD=python src/training/train_teacher.py --data_dir %DATA_DIR% --output_dir %OUTPUT_DIR% --batch_size %BATCH_SIZE% --epochs %EPOCHS% --learning_rate %LR% --num_workers %NUM_WORKERS% --seed %SEED%

if not "%PRETRAINED%"=="" (
    set CMD=!CMD! --pretrained_weights %PRETRAINED%
)

if not "%EXTRA_ARGS%"=="" (
    set CMD=!CMD! %EXTRA_ARGS%
)

echo Executing: !CMD!
echo.

!CMD!

if %ERRORLEVEL% EQU 0 (
    echo.
    echo =====================================
    echo Training completed successfully!
    echo =====================================
    echo.
    echo Results saved to: %OUTPUT_DIR%
    echo Check training curves: figures/teacher/training_curves.png
) else (
    echo.
    echo =====================================
    echo Training failed!
    echo =====================================
    exit /b 1
)

endlocal

