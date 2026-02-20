#!/bin/bash

# Configuration
# p values: 3000, 3500, 4000, 4500, 5000
P_VALUES=(4000  5000)

for p in "${P_VALUES[@]}"
do
    echo "Submitting job for p=$p..."
    
    # Create a temporary job script for this P
    JOB_SCRIPT="submit_p${p}.sh"
    
    # Calculate n_iter: 200 + (p - 3000) / 2
    n_iter=300
    
    # Replace placeholders using sed
    sed "s/%P%/$p/g; s/%NITER%/$n_iter/g" submit_template.sh > "$JOB_SCRIPT"
    
    # Submit
    sbatch "$JOB_SCRIPT"
    
    # Optional: Delete script after submission?
    # rm "$JOB_SCRIPT"
done

echo "All jobs submitted."
