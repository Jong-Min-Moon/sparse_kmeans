#!/bin/bash

# Default values
USERNAME="jongminm"
HOSTNAME="discovery.usc.edu"
REMOTE_BASE="~/sparse_kmeans_project"

# Help message
usage() {
    echo "Usage: $0 [-u username] [-h hostname] [-r remote_base]"
    exit 1
}

# Parse arguments
while getopts "u:h:r:" opt; do
    case "$opt" in
        u) USERNAME=$OPTARG ;;
        h) HOSTNAME=$OPTARG ;;
        r) REMOTE_BASE=$OPTARG ;;
        *) usage ;;
    esac
done

SIM_NAME="thompson_unknowncov_naive"
# Use script directory as local base (relative path logic)
LOCAL_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REMOTE_DIR="${REMOTE_BASE}/simulations/production/${SIM_NAME}"

# SSH options to "remember" the password during the script execution
# ControlMaster=auto + ControlPersist ensures we only authenticate once for multiple scp calls.
SSH_OPTS="-o ControlMaster=auto -o ControlPath=/tmp/ssh_mux_%h_%p_%r -o ControlPersist=60s"

echo -e "\033[0;36mRetrieving simulation results and logs from HPC for $SIM_NAME...\033[0m"

# Prepare local repository struct internally to absorb outputs natively
mkdir -p "${LOCAL_DIR}/results_raw"
mkdir -p "${LOCAL_DIR}/logs"

# Sync results_raw
echo "Syncing results_raw..."
scp $SSH_OPTS -r "${USERNAME}@${HOSTNAME}:${REMOTE_DIR}/results_raw/*" "${LOCAL_DIR}/results_raw/"

# Sync logs
echo "Syncing logs..."
scp $SSH_OPTS -r "${USERNAME}@${HOSTNAME}:${REMOTE_DIR}/logs/*" "${LOCAL_DIR}/logs/"

echo -e "\033[0;32mSync process complete. You can now execute aggregate_results.R locally.\033[0m"

# Tip for persistent "remembering" via SSH keys (the standard Mac/Unix way)
echo ""
echo -e "\033[0;33mTo remember your password permanently on this Mac:\033[0m"
echo "1. Run: ssh-copy-id ${USERNAME}@${HOSTNAME}"
echo "2. Your Mac will then use your SSH key automatically for all future connections."
