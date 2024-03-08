#!/bin/bash
#
#
cluster_home="/home1/jongminm"
project_name="sparse_kmeans"
meeting_date="17_03_08_2024"
experiment_name="ISEE_noisy"
extension_code="m"
extension_result="csv"

project_dir="${cluster_home}/${project_name}"
code_dir="${project_dir}/experiment/${meeting_date}/${experiment_name}"


#화면에 텍스트 표시:
echo "code_dir = ${code_dir}"



rho=20
Delta=4
p=100

s=10
n=500

for rep in 1 2 3 4 5 .. 100
do
    #filename of code
    filename_code="rho${rho}_Delta${Delta}_p${p}_rep_${rep}"
    echo "file_name = ${filename}"

    #filename of result
    path_result="${code_dir}/result/${filename_code}.${extension_result}"

    #python file
    touch ${code_dir}/temp_py

    echo "data = np.load('${code_dir}/${filename_code}.pkl', allow_pickle=True)" >> ${code_dir}/temp_py
    echo "model.fit(data)"                                                       >> ${code_dir}/temp_py
    echo "savedict = {'Omega_est_now' : model.precision_}"                       >> ${code_dir}/temp_py
    echo "sio.savemat('${code_dir}/${filename_code}.mat', savedict)"             >> ${code_dir}/temp_py
    mv ${code_dir}/temp_py ${code_dir}/${filename_code}.py





    # code
    touch ${code_dir}/temp_code
    cat ${code_dir}/matlab_parallel_usc >> ${code_dir}/temp_code
    cat ${code_dir}/sql_matlab_connec   >> ${code_dir}/temp_code
    echo "addpath(genpath(${project_dir}));" >> ${code_dir}/temp_code
    echo "rho = ${rho};" >> ${code_dir}/temp_code
    echo "p = ${p}" >> ${code_dir}/temp_code
    echo "Delta = ${Delta}" >> ${code_dir}/temp_code
    echo "s = ${s}" >> ${code_dir}/temp_code
    echo "n = ${n}" >> ${code_dir}/temp_code
    echo "rep = ${rep}" >> ${code_dir}/temp_code
    cat ${code_dir}/skeleton_code.${extension_code} >> ${code_dir}/temp_code
    mv ${code_dir}/temp_code ${code_dir}/${filename_code}.${extension_code}

    


    # job
    touch ${code_dir}/temp_job
    cat ${code_dir}/skeleton_job.job >> ${code_dir}/temp_job
    echo "#SBATCH --job-name=${filename_code}" >> ${code_dir}/temp_job
    echo "#SBATCH --output=${code_dir}/result/${filename_code}.out" >> ${code_dir}/temp_job
    echo "cd ${code_dir}" >> ${code_dir}/temp_job
    echo "${tool} -batch \"${filename_code}\"" >> ${code_dir}/temp_job

    mv ${code_dir}/temp_job ${code_dir}/${filename_code}.job

    sbatch ${code_dir}/${filename_code}.job
done




