#!/bin/bash

sep=2
n=200
p=1000
n_draw=10000
baseline=0

s_list=(6 32 210)
cluster_1_ratio_list=(0.1 0.3 0.5)


for s in "${s_list[@]}"; do
  for ratio in "${cluster_1_ratio_list[@]}"; do
    for id in {1..6}; do
      beta_seed=$((RANDOM % 1000000))
      script_name="/home1/jongminm/sparse_kmeans/experiment/26_04_24_2025/objective_plot_baseline_0_pen/job_s${s}_r${ratio}_v${id}.slurm"

cat <<EOF > "$script_name"
#!/bin/bash
#SBATCH --job-name=s${s}_r${ratio}_v${id}
#SBATCH --output=/home1/jongminm/sparse_kmeans/experiment/26_04_24_2025/objective_plot_baseline_0_pen/s${s}_r${ratio}_v${id}.out
#SBATCH --error=/home1/jongminm/sparse_kmeans/experiment/26_04_24_2025/objective_plot_baseline_0_pen/s${s}_r${ratio}_v${id}.err
#SBATCH --time=23:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4

module purge
module load legacy/CentOS7
module load matlab/2022a

matlab -nodisplay -r "addpath(genpath('/home1/jongminm/sparse_kmeans')); try, n=${n}; p=${p}; s=${s}; sep=${sep}; cluster_1_ratio=${ratio}; n_draw=${n_draw}; baseline=${baseline}; beta_seed=${beta_seed}; \
if ${id}==1; compare_cluster_support_distributions_pen(n, p, s, sep, baseline, cluster_1_ratio, 1:s, ((ceil(s/2)+1):(s+ceil(s/2))), n_draw, beta_seed); \
elseif ${id}==2; compare_cluster_support_distributions_pen(n, p, s, sep, baseline, cluster_1_ratio, 1:s, ((ceil(s/2)+1):(s+ceil(s/2)+ceil(s/2))), n_draw, beta_seed); \
elseif ${id}==3; compare_cluster_support_distributions_pen(n, p, s, sep, baseline, cluster_1_ratio, 1:s, ((ceil(s/2)+1):(s+ceil(s/2)-floor(3*s/4))), n_draw, beta_seed); \
elseif ${id}==4; compare_cluster_support_distributions_pen(n, p, s, sep, baseline, cluster_1_ratio, 1:s, ((s+1):(2*s)), n_draw, beta_seed); \
elseif ${id}==5; compare_cluster_support_distributions_pen(n, p, s, sep, baseline, cluster_1_ratio, 1:s, ((s+1):(2*s+ceil(s/2))), n_draw, beta_seed); \
elseif ${id}==6; compare_cluster_support_distributions_pen(n, p, s, sep, baseline, cluster_1_ratio, 1:s, ((s+1):(2*s-floor(3*s/4))), n_draw, beta_seed); \
end; catch e; disp(getReport(e)); end; exit;"
EOF

      # Submit the job
      sbatch "$script_name"
      if [ $? -eq 0 ]; then
        echo "Submitted: $script_name"
      else
        echo "Failed to submit: $script_name"
      fi
    done
  done
done
