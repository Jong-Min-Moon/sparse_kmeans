#!/bin/bash
#
# Author : Rocky Documentation Team
# Date: March 2022
# Version 1.0.0: "Hello world!"
#

project_name="sparse_kmeans"
meeting_date="13_02_09_2024"
experiment_name="AR1"

code_dir="/mnt/nas/users/user213/${project_name}/experiment/${meeting_date}/${experiment_name}"


#화면에 텍스트 표시:
echo "code_dir = ${code_dir}"

#filename
rho=9
p=1600
Delta=4
filename="rho${rho}_Delta${Delta}_p${p}"
echo "file_name = ${filename}"

# write matlab code
touch ${code_dir}/${filename}.m



