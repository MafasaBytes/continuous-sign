#!/bin/bash
# Run I3D Teacher overfit test

echo "=========================================="
echo "Running I3D Teacher Overfit Test"
echo "=========================================="

python overfit_test_teacher.py

echo ""
echo "Test complete! Check the following files:"
echo "  - overfit_test_teacher_results.png"
echo "  - overfit_test_teacher_report.txt"

