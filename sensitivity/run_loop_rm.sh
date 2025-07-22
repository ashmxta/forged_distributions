#!/usr/bin/env bash
#
# run_loop_rm.sh — iterate runs & stages, dropping the top-100 removed points each time

# absolute path to your removed-points file
REMOVED_FILE="/h/321/ashmita/forged_distributions/sensitivity/compo_res/removed_points.txt"

if [[ ! -f "$REMOVED_FILE" ]]; then
  echo "ERROR: removed-points file not found at $REMOVED_FILE" >&2
fi

# read into a bash array
read -r -a removed <<< "$(< "$REMOVED_FILE")"
echo "Loaded ${#removed[@]} points to remove."

for run in {1..1}; do
  for i in {0..40}; do
    # fractional stage 0.000 … 1.000
    stage=$(echo "scale=3; $i / 40" | bc)

    echo "Starting run ${run}, stage ${stage} (dropping ${#removed[@]} points)"
    python3 compute_sensitivity_rm_list.py \
      --stage "$stage" \
      --save-name "ckpt${run}" \
      --res-name "res${run}" \
      --remove-points "${removed[@]}"
  done
done

# make exec: chmod +x run_loop_rm.sh
# run: ./run_loop_rm.sh
# bg: nohup ./run_loop_rm.sh > output.log 2>&1 &
