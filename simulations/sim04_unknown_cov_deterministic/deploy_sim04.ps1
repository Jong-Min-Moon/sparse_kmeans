# Deploy Simulation 04 to HPC

$Server = "147.46.20.103"
$User = "jongmin"
$RemotePath = "/home/jongmin/sparse_kmeans/simulations/sim04_unknown_cov_deterministic"

cat "Deploying Simulation 04 to $Server..."

# 1. Create remote directory
ssh "$User@$Server" "mkdir -p $RemotePath"

# 2. Upload files
scp driver.R "$User@$Server:$RemotePath/"
scp submit.sh "$User@$Server:$RemotePath/"

cat "Deployment complete."
cat "To run: ssh $User@$Server 'cd $RemotePath && sbatch submit.sh'"
