#!/bin/bash
#
#
cluster_home="/home1/jongminm"
project_name="sparse_kmeans"
meeting_date="19_04_01_2024"
experiment_name="ISEE_dirty"
extension_code="m"
extension_result="csv"
table_name="sparse_kmeans_isee_dirty"
project_dir="${cluster_home}/${project_name}"
code_dir="${project_dir}/experiment/${meeting_date}/${experiment_name}"
data_setting_dir="${project_dir}/code_data_setting"


s=10
n=500
n_iter_max=100

for rho in 20 35 45
    do
    for Delta in 4 3
        do
        for p in  100 200 300 400
            do
            for ii in {1..10}
                do
                    #filename of code
                    filename_code="rho${rho}_Delta${Delta}_p${p}_rep_${ii}"
                    echo "file_name = ${filename_code}"

                    #filename of result
                    path_result="${code_dir}/result/${filename_code}.${extension_result}"

                
                    # code
                        touch ${code_dir}/temp_code
                        echo "addpath(genpath('${project_dir}'));" >> ${code_dir}/temp_code

                    ## parallel computing toolbox
                        cat ${project_dir}/code_basic/matlab_parallel_usc >> ${code_dir}/temp_code
                    ## parameters
                        echo "ii = ${ii};" >> ${code_dir}/temp_code #delete this line for non-iterative algorithm
                        echo "rho = ${rho};" >> ${code_dir}/temp_code
                        echo "n_iter_max = ${n_iter_max};" >> ${code_dir}/temp_code
                        echo "p = ${p}" >> ${code_dir}/temp_code
                        echo "Delta = ${Delta}" >> ${code_dir}/temp_code
                        echo "s = ${s}" >> ${code_dir}/temp_code
                        echo "n = ${n}" >> ${code_dir}/temp_code
                        echo "table_name = '${table_name}'" >> ${code_dir}/temp_code
                    ## data setting
                        cat ${data_setting_dir}/sparse_innovation.${extension_code} >> ${code_dir}/temp_code
                    ## method-specific skeleton code
                        cat ${code_dir}/skeleton_code.${extension_code} >> ${code_dir}/temp_code
                
                    mv ${code_dir}/temp_code ${code_dir}/${filename_code}.${extension_code}
                    sleep 4
                    
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
                    sleep 5
                    rm ${code_dir}/${filename_code}.job
                done
            done
        done
    done

