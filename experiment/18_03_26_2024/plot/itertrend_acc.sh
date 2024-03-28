#!/bin/bash
#
#
cluster_home="/home1/jongminm"
project_name="sparse_kmeans"
table_name_big="unknowncov_lowdim"
table_name_small="itertrend_acc"
table_name="${table_name_big}_${table_name_small}"

meeting_date="18_03_26_2024"
experiment_name="plot"
extension_code="tex"
project_dir="${cluster_home}/${project_name}"
code_dir="${project_dir}/experiment/${meeting_date}/${experiment_name}"


for del in 3 4 5
    do
    for rho in 05 2 45
        do
            #filename of code
        filename_table="table_${table_name}_del${del}rho${rho}"
        echo "file_name = ${filename_table}"
        touch ${code_dir}/temp_code
        echo "\input{table/table_${table_name}_axis}" >> ${code_dir}/temp_code
        #
        echo "\addplot[red, solid, mark options=solid]" >> ${code_dir}/temp_code
        echo "table [x=iter, y= ${del}0.${rho}, col sep = comma ]" >> ${code_dir}/temp_code
        echo "{data_for_figure/sparse_kmeans_isee_denoise_${table_name}.csv};" >> ${code_dir}/temp_code
        echo "\addlegendentry{ iter\_ISEE\_rate }" >> ${code_dir}/temp_code
        #
        echo "\addplot[blue, solid, mark options=solid]" >> ${code_dir}/temp_code
        echo "table [x=iter, y= ${del}0.${rho}, col sep = comma ]" >> ${code_dir}/temp_code
        echo "{data_for_figure/sparse_kmeans_isee_${table_name}.csv};" >> ${code_dir}/temp_code
        echo "\addlegendentry{ iter\_ISEE\_gaussmax }" >> ${code_dir}/temp_code
        #
        echo "\addplot[orange, solid, mark options=solid]" >> ${code_dir}/temp_code
        echo "table [x=iter, y= ${del}0.${rho}, col sep = comma ]" >> ${code_dir}/temp_code
        echo "{data_for_figure/sparse_kmeans_glasso_${table_name}.csv};" >> ${code_dir}/temp_code
        echo "\addlegendentry{ iter\_glasso }" >> ${code_dir}/temp_code        
        #
        echo "\input{table/exis_end_common}" >> ${code_dir}/temp_code  
        mv ${code_dir}/temp_code ${code_dir}/${filename_table}.${extension_code}
        done
    done

