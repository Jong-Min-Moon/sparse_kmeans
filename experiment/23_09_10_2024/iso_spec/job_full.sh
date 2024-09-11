#!/bin/bash
#
#
n_iter_max=100
cluster_home="/home1/jongminm"
project_name="sparse_kmeans"
meeting_date="23_09_10_2024"
experiment_name="iso_spec"
extension_code="m"
extension_result="csv"
table_name="test"
project_dir="${cluster_home}/${project_name}"
code_dir="${project_dir}/experiment/${meeting_date}/${experiment_name}"
data_setting_dir="${project_dir}/code_data_setting"
db_dir="/home1/jongminm/sparse_kmeans/sparse_kmeans.db"

s=10
sample_size=200

for rho in 0
    do
    for separation in  5
        do
        for dimension in  50
            do
            for ii in {1..50}
                do
                    #filename of code
                    filename_code="rho${rho}_Delta${separation}_p${dimension}_rep_${ii}"
                    echo "file_name = ${filename_code}"

                    #filename of result
                    path_result="${code_dir}/result/${filename_code}.${extension_result}"

                
                    # code
                        touch ${code_dir}/temp_code
                        echo "addpath(genpath('${project_dir}'));" >> ${code_dir}/temp_code

                    ## parallel computing toolbox
                        cat ${project_dir}/code_basic/matlab_parallel_usc >> ${code_dir}/temp_code
                    ## parameters
                        echo "n_iter_max = ${n_iter_max};" >> ${code_dir}/temp_code
                        echo "ii = ${ii};" >> ${code_dir}/temp_code #delete this line for non-iterative algorithm
                        echo "rho = ${rho};" >> ${code_dir}/temp_code
                        echo "dimension = ${dimension}" >> ${code_dir}/temp_code
                        echo "separation = ${separation}" >> ${code_dir}/temp_code
                        echo "sample_size = ${sample_size}" >> ${code_dir}/temp_code
                        echo "table_name = '${table_name}'" >> ${code_dir}/temp_code
                        echo "db_dir = '${db_dir}'" >> ${code_dir}/temp_code
                    ## method-specific skeleton code
                        cat ${code_dir}/skeleton_code.${extension_code} >> ${code_dir}/temp_code
                
                    mv ${code_dir}/temp_code ${code_dir}/${filename_code}.${extension_code}
                    sleep 2
                    
                    # job
                    touch ${code_dir}/temp_job
                    echo "#!/bin/bash" >> ${code_dir}/temp_job
                    echo "#SBATCH --output=${code_dir}/${filename_code}.out" >> ${code_dir}/temp_job
                    cat ${code_dir}/skeleton_job.job >> ${code_dir}/temp_job
                    
                    echo "cd ${code_dir}" >> ${code_dir}/temp_job
                    sleep 4
                    echo "matlab -batch ${filename_code}" >> ${code_dir}/temp_job


                    mv ${code_dir}/temp_job ${code_dir}/${filename_code}.job

                    sbatch ${code_dir}/${filename_code}.job
                    sleep 3
                    rm ${code_dir}/${filename_code}.job
                done
            done
        done
    done

