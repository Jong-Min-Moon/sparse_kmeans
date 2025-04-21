#!/bin/bash
#SBATCH --job-name=s32_r0.3_v6
#SBATCH --output=/home1/jongminm/sparse_kmeans/experiment/26_04_24_2025/objective_plot_baseline_10/s32_r0.3_v6.out
#SBATCH --error=/home1/jongminm/sparse_kmeans/experiment/26_04_24_2025/objective_plot_baseline_10/s32_r0.3_v6.err
#SBATCH --time=23:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4

module purge
module load legacy/CentOS7
module load matlab/2022a

matlab -nodisplay -r "addpath(genpath('/home1/jongminm/sparse_kmeans')); try, n=200; p=1000; s=32; sep=2; cluster_1_ratio=0.3; n_draw=10000; baseline=10; beta_seed=19164; if 6==1; compare_cluster_support_distributions(n, p, s, sep, baseline, cluster_1_ratio, 1:s, ((ceil(s/2)+1):(s+ceil(s/2))), n_draw, beta_seed); elseif 6==2; compare_cluster_support_distributions(n, p, s, sep, baseline, cluster_1_ratio, 1:s, ((ceil(s/2)+1):(s+ceil(s/2)+ceil(s/2))), n_draw, beta_seed); elseif 6==3; compare_cluster_support_distributions(n, p, s, sep, baseline, cluster_1_ratio, 1:s, ((ceil(s/2)+1):(s+ceil(s/2)-floor(3*s/4))), n_draw, beta_seed); elseif 6==4; compare_cluster_support_distributions(n, p, s, sep, baseline, cluster_1_ratio, 1:s, ((s+1):(2*s)), n_draw, beta_seed); elseif 6==5; compare_cluster_support_distributions(n, p, s, sep, baseline, cluster_1_ratio, 1:s, ((s+1):(2*s+ceil(s/2))), n_draw, beta_seed); elseif 6==6; compare_cluster_support_distributions(n, p, s, sep, baseline, cluster_1_ratio, 1:s, ((s+1):(2*s-floor(3*s/4))), n_draw, beta_seed); end; catch e; disp(getReport(e)); end; exit;"
