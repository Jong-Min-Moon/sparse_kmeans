#!/bin/bash

# Configuration
P_VALUES=(5000 6000 7000 8000 9000)
FDR_LEVEL=0.4

for P in "${P_VALUES[@]}"
do
    echo "Submitting simulation array for p = $P with FDR = $FDR_LEVEL"
    sbatch --export=P=$P,FDR=$FDR_LEVEL submit_greedy.sh
done
