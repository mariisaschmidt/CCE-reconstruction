#!/usr/bin/env bash
file_name=${1}

local_file_name=$(echo "${file_name}" | cut -f8 -d/)
bare_local_file_name=$(echo "${local_file_name}" | cut -f1 -d.)

wget ${file_name}

./write_to_filtered_file.py ${local_file_name}

# gzip -c "${bare_local_file_name}.txt" > "${bare_local_file_name}.txt.gz"

# rm "${bare_local_file_name}.txt"
rm ${local_file_name}