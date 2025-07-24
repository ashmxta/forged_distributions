#!/usr/bin/env bash
#
# run_loop_rm.sh — iterate runs & stages, dropping hardcoded points

# Hardcoded points to remove
removed=(293 818 522 152 678 206 816 115 707 192 310 787 40 288 435 676 90 269 217 880 604 359 39 326 355 972 524 920 994 290 397 904 77 573 773 639 99 214 831 123 50 245 640 562 309 445 122 738 408 230 270 710 208 711 473 553 959 401 698 819 838 378 885 642 102 969 231 766 484 454 67 860 505 820 238 458 422 17 925 798 387 382 421 567 685 184 460 14 52 320 299 877 349 171 809 630 174 429 72 303)

echo "Loaded ${#removed[@]} points to remove."

for run in {1..10}; do
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
