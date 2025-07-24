#!/bin/bash


echo "Starting composition calculation"

python3 renyi_per_instance_sum_compo.py "./res_rm100/res_concat_rm.csv"

echo "Composition calculation completed."


# chmod +x run_compo.sh       --> to make executable
# ./run_compo.sh              --> to run
# nohup ./run_compo.sh > output_compo.log 2>&1 &   --> to run in background
# tail -f output_compo.log    --> to watch logs
