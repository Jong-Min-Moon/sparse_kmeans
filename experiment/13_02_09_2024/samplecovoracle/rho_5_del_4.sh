#!/bin/bash
#
#

project_name="sparse_kmeans"
meeting_date="13_02_09_2024"
experiment_name="samplecovoracle"
extension_code="m"
extension_result="csv"
tool="/usr/local/MATLAB/R2023b/bin/matlab"

code_dir="/mnt/nas/users/user213/${project_name}/experiment/${meeting_date}/${experiment_name}"


#화면에 텍스트 표시:
echo "code_dir = ${code_dir}"


rho=5
Delta=4
for p in {200..2000..200}
do
    #filename of code
    filename_code="rho${rho}_Delta${Delta}_p${p}"
    echo "file_name = ${filename}"

    #filename of result
    path_result="${code_dir}/result/${filename_code}.${extension_result}"



    # code
    touch ${code_dir}/temp_code
    echo "rho = ${rho};" >> ${code_dir}/temp_code
    echo "p = ${p}" >> ${code_dir}/temp_code
    echo "Delta = ${Delta}" >> ${code_dir}/temp_code
    echo "path_result = '${path_result}'" >> ${code_dir}/temp_code

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




