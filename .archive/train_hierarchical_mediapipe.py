"""
Thin entrypoint that delegates to the modular teacher pipeline.
"""

from teacher.train_mediapipe import main as teacher_main

if __name__ == "__main__":
    teacher_main()


