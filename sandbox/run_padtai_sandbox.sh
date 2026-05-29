#!/bin/bash
#!/bin/bash
# Run PADTAI sandbox runner for clusters 0 and 1 with 10min timeout

# Example grounded operator: sum:SumOperator or lt:LtOperator
GROUNDED1=${GROUNDED1:-"sum:SumOperator"}
GROUNDED2=${GROUNDED2:-"lt:LtOperator"}

python3 sandbox/run_padtai_sandbox.py --clusters 0 1 --timeout 600 --intcols auto --grounded "$GROUNDED1" "$GROUNDED2"
