# Forged training distributions in DP-SGD
The scripts in this repo contains scripts to compute per-instance per-step privacy costs - model training and grad norm computation scripts are in the 'sensitivity' folder, the set-up is adapted from [Gradients Look Alike: Sensitivity is Often Overestimated in DP-SGD](https://arxiv.org/abs/2307.00310).

## MNIST - Obtain per-point per step privacy guarantees:
- We need per-step per-point gradient norms to compute privacy costs
- Gradient norms show the model's sensitivity to a datapoint at a given step
    - Use run_loop.sh for grad norms (1K points, 40 epochs)
        - make the script executable: chmod +x run.sh
        - run in background: nohup ./run.sh > output.log 2>&1 &
        - checkpoints saved to models/ckptX (X = run)
        - results saved to sensitivity/res/resX.csv; X = run, 10 runs for this experiment
        - res_per_point contains CSV files per point per run 
        - naming convention: res_runX_pointX / res_runA_pointX (runA = average across runs)
- To obtain the privacy loss, per data point, per step:
    - Run renyi_per_instance_sum_compo.py, use bash script run_compo.sh if preferred
        - use sensitivity/res/res_concat.py to combine data over multiple runs
        - chmod +x run_compo.sh
        - nohup ./run_compo.sh
        - results saved to sensitivity/compo_res

## MNIST - Non-recursive removal from 1K sample:
- Obtain indicies of points of lowest impact:
    - compo_res.CSV files contain privacy costs per step, to obtain the total privacy cost they must first be summed, and then they can be ranked using compo_res/rank.py.
    - exp1: remove 100 pts of lowest impact + compare per point costs after re-training
        - does the ranking of the remaining 900 remain mostly consistent?
        - do privacy costs themselves change much (magnitude)?
        - metrics to compare similarity of the models themselves?
            - grad norms (training trajectory)?
            - prediction behaviour?

 ## Plotting privacy cost curves (approximation techniques)
- Summer 2024 [to be added to current pipeline]
    - offset interpolation
    - neural network approximation
    - linear interpolation
