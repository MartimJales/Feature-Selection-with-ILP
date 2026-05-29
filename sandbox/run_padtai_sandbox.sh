#!/bin/bash
# Run PADTAI sandbox runner for clusters 0 and 1 with 10min timeout

# LT-only grounded operator
GROUNDED=${GROUNDED:-"lt:LTOperator"}

python3 sandbox/run_padtai_sandbox.py --clusters 0 1 --timeout 600 --intcols auto --grounded "$GROUNDED"
