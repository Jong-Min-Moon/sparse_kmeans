#!/bin/bash
#
#
n_iter_max=100
cluster_home="/home1/jongminm"
project_name="sparse_kmeans"
meeting_date="25_04_03_2025"
experiment_name="witten"
extension_code="R"
table_name="iso_witten"
project_dir="${cluster_home}/${project_name}"
code_dir="${project_dir}/experiment/${meeting_date}/${experiment_name}"
db_dir="/home1/jongminm/sparse_kmeans/sparse_kmeans.db"

s=10
sample_size=200

    for separation in  4 5 
        do
        for dimension in  50 500 1000 1500 2000 2500 3000 3500 4000 4500
            do
            for ii in {1..10}
                do
                    #filename of code
                    filename_code="witten_Delta${separation}_p${dimension}_rep_${ii}"
                    echo "file_name = ${filename_code}"


                    rep_start=$(( (ii - 1) * 200 + 1 ))
                    rep_end=$(( ii * 200 ))
                    # code
                        touch ${code_dir}/temp_code


                        echo "rep_start = ${rep_start}" >> ${code_dir}/temp_code 
                        echo "rep_end = ${rep_end}" >> ${code_dir}/temp_code 
                        echo "dimension  = ${dimension}"  >> ${code_dir}/temp_code
                        echo "separation = ${separation}" >> ${code_dir}/temp_code
 
                    ## method-specific skeleton code
                        cat ${code_dir}/witten_code.${extension_code} >> ${code_dir}/temp_code
                
                    mv ${code_dir}/temp_code ${code_dir}/${filename_code}.${extension_code}
                    sleep 2
                    
                    # job
                    touch ${code_dir}/temp_job
                    echo "#!/bin/bash" >> ${code_dir}/temp_job
                    echo "#SBATCH --output=${code_dir}/${filename_code}.out" >> ${code_dir}/temp_job
                    cat ${code_dir}/witten.job >> ${code_dir}/temp_job
                    
                    echo "Rscript  ${code_dir}/${filename_code}.R" >> ${code_dir}/temp_job
                    sleep 4
      


                    mv ${code_dir}/temp_job ${code_dir}/${filename_code}.job

                    sbatch ${code_dir}/${filename_code}.job
                    sleep 3
                    rm ${code_dir}/${filename_code}.job
                done
            done
        done

