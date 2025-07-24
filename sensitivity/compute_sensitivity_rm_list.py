#!/usr/bin/env python3
import os
import argparse
import torch
import pandas as pd
import numpy as np
import train_rm_list as train   # renamed module
import utils  # renamed module

"""
notes: 
- running this script creates a res folder in project root directory (same level as sensitivity folder)
- each file in res corresponds to a single run - res${run}.csv
- data is saved for each point at multiple stages in training 

modifications: 
- changed np.random to rng for reproducibility
- default random seed = 24
- added --remove-points to drop arbitrary list of data indices before training
- switched import from train.py to train_rm_list.py
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--points',
        nargs="+",
        type=int,
        default=list(range(1000)),
        help='indices of data points to compute sensitivity'
    )
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-iters', type=int, default=20, help='only useful for renyi')
    parser.add_argument('--alpha', type=int, default=8, help='only useful for renyi')
    parser.add_argument('--num-batches', type=int, default=100, help='only useful for renyi')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--cn', type=float, default=1, help='clipping norm')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--dp', type=int, default=1)
    parser.add_argument('--eps', type=float, default=10)
    parser.add_argument('--optimizer', type=str, default="sgd")
    parser.add_argument('--dataset', type=str, default="MNIST")
    parser.add_argument('--model', type=str, default="lenet")
    parser.add_argument(
        '--norm-type',
        type=str,
        default="gn",
        help="Note that batch norm is not compatible with DPSGD"
    )
    parser.add_argument('--save-freq', type=int, default=100, help='frequency of saving checkpoints')
    parser.add_argument('--save-name', type=str, default='ckpt', help='checkpoints will be saved under models/[save-name]')
    parser.add_argument('--res-name', type=str, default='res', help='sensitivity will be saved in res/[res-name].csv')
    parser.add_argument('--gamma', type=float, default=None, help='for learning rate schedule')
    parser.add_argument('--dec-lr', nargs="+", type=int, default=None, help='for learning rate schedule')
    parser.add_argument('--id', type=str, default='', help="experiment id")
    parser.add_argument('--seed', type=int, default=24)
    parser.add_argument('--overwrite', type=int, default=0, help="whether overwrite existing result files")
    parser.add_argument('--poisson-train', type=int, default=1, help="should always be 1 for correct DPSGD")
    parser.add_argument(
        '--stage',
        type=str,
        default='initial',
        help='initial, middle, final, or 0 to 1 where 0 means not training done and 1 means training finished'
    )
    parser.add_argument(
        '--reduction',
        type=str,
        default='sum',
        help="update rule, mean or sum"
    )
    parser.add_argument(
        '--exp',
        type=str,
        default='eps_delta',
        help='experiment type: eps_delta, or renyi'
    )
    parser.add_argument(
        '--remove-points',
        nargs="+",
        type=int,
        default=[],
        help="List of dataset indices to remove before training"
    )

    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    remove_points = set(args.remove_points)  # use set for faster lookup
    point_to_do = np.array(
        [p for p in args.points if p not in remove_points],
        dtype=int
    )

    print(f"Filtered {len(args.points) - len(point_to_do)} points (from {len(args.points)} total) due to --remove-points.")

    # instantiate training object
    train_fn = train.train_fn(
        args.lr,
        args.batch_size,
        args.dataset,
        args.model,
        exp_id=args.id,
        save_freq=args.save_freq,
        optimizer=args.optimizer,
        epochs=args.epochs,
        dp=args.dp,
        cn=args.cn,
        eps=args.eps,
        dec_lr=args.dec_lr,
        gamma=args.gamma,
        seed=args.seed,
        norm_type=args.norm_type,
        poisson=args.poisson_train,
        save_name=args.save_name,
        remove_points=remove_points,    # now accepts a list
        reduction=args.reduction
    )

    trainset_size = len(train_fn.trainset)
    p = args.batch_size / trainset_size
    all_indices = np.arange(trainset_size)

    # prepare results directory
    if not os.path.exists("res_rm100"):
        os.mkdir("res_rm100")
    out_dir = "/h/321/ashmita/forged_distributions/sensitivity/res_rm100"
    os.makedirs(out_dir, exist_ok=True)

    res_dir      = os.path.join(out_dir, f"{args.res_name}.csv")
    temp_res_dir = os.path.join(out_dir, f"temp_{args.res_name}.csv")
    print(f"Writing results to {res_dir}")
    if os.path.exists(temp_res_dir):
        os.remove(temp_res_dir)
    print(f"path to result file: {res_dir}")

    # checkpoint logic
    step = utils.find_ckpt(args.stage, trainset_size, args.batch_size, args.save_freq, args.epochs)
    cur_path = f"{train_fn.save_dir}/model_step_{step}"
    if not os.path.exists(cur_path):
        print("checkpoints not found, starting training")
        train_fn.save(-1)
        for s in range(train_fn.sequence.shape[0]):
            train_fn.train(s)
        train_fn.validate()
        step = utils.find_ckpt(args.stage, trainset_size, args.batch_size, args.save_freq, args.epochs)
        cur_path = f"{train_fn.save_dir}/model_step_{step}"
        # re-init to incorporate any dynamic state
        train_fn = train.train_fn(
            args.lr, args.batch_size, args.dataset, args.model,
            exp_id=args.id, save_freq=args.save_freq, optimizer=args.optimizer,
            epochs=args.epochs, dp=args.dp, cn=args.cn, eps=args.eps,
            dec_lr=args.dec_lr, gamma=args.gamma, seed=args.seed,
            norm_type=args.norm_type, poisson=args.poisson_train,
            remove_points=remove_points,
            reduction=args.reduction
        )

    train_fn.load(cur_path)
    accuracy = train_fn.validate()

    # existing result check
    if os.path.exists(res_dir) and not args.overwrite:
        temp_df = pd.read_csv(res_dir)
        if "renyi" in args.exp and args.reduction == "mean":
            temp_df = temp_df[(temp_df['type'] == args.stage) & (temp_df['alpha'] == args.alpha)]
        else:
            temp_df = temp_df[temp_df['type'] == args.stage]
        if temp_df.shape[0] != 0:
            done = temp_df["point"].unique()
            # keep everything except done, as an ndarray
            point_to_do = np.setdiff1d(point_to_do, done)
            print(f"found {done.shape[0]} done, {point_to_do.shape[0]} to analyze")
            

        else:
            print(f"{len(point_to_do)} points to analyze")
    else:
        print(f"{len(point_to_do)} points to analyze")
        if args.overwrite and os.path.exists(res_dir):
            os.remove(res_dir)

    if point_to_do.size == 0:  # or use len(point_to_do) == 0
        print("no points left, use --overwrite 1 to re-run")
        return

    # renyi branch
    if "renyi" in args.exp and args.reduction == "mean":
        for idx in point_to_do:
            remove1 = np.delete(all_indices, idx)
            size1, size2, inds1, inds2 = (
                (trainset_size, trainset_size-1, all_indices, remove1)
                if "reverse" in args.exp else
                (trainset_size-1, trainset_size, remove1, all_indices)
            )
            for b in range(args.num_iters):
                rng.shuffle(inds1); rng.shuffle(inds2)
                tgt = inds1[rng.binomial(1, p, size1).astype(bool)]
                batches = [
                    np.concatenate([
                        inds2[rng.binomial(1, p, size2).astype(bool)]
                        for _ in range(args.alpha)
                    ])
                    for _ in range(args.num_batches)
                ]
                for batch in batches:
                    dist = train_fn.sensitivity_renyi(tgt, batch, args.alpha, cn=args.cn)[0]
                    df = (
                        pd.read_csv(temp_res_dir) if os.path.exists(temp_res_dir)
                        else pd.read_csv(res_dir) if os.path.exists(res_dir)
                        else pd.DataFrame()
                    )
                    # build a single‐row record, only with scalars or same‐length arrays

                    # build a single‐row record matching the no‐remove schema + extra
                    record = {
                        f"distance ({args.reduction})": dist,
                        "step":                step,
                        "p":                   p,
                        "batch":               b,
                        "point":               idx,
                        "sigma":               train_fn.sigma,
                        "accuracy":            accuracy,
                        "type":                args.stage,

                        # — all original parameters —
                        "points":              args.points,
                        "batch_size":          args.batch_size,
                        "num_iters":           args.num_iters,
                        "alpha":               args.alpha,
                        "num_batches":         args.num_batches,
                        "lr":                  args.lr,
                        "cn":                  args.cn,
                        "dataset":             args.dataset,
                        "dec_lr":              args.dec_lr,
                        "dp":                  args.dp,
                        "eps":                 args.eps,
                        "optimizer":           args.optimizer,
                        "model":               args.model,
                        "norm_type":           args.norm_type,
                        "save_freq":           args.save_freq,
                        "save_name":           args.save_name,
                        "res_name":            args.res_name,
                        "gamma":               args.gamma,
                        "id":                  args.id,
                        "seed":                args.seed,
                        "overwrite":           args.overwrite,
                        "poisson_train":       args.poisson_train,
                        "stage":               args.stage,
                        "reduction":           args.reduction,
                        "exp":                 args.exp,
                        "less_point":          0,

                        # — our new field —
                        "remove_points_count": len(remove_points),
                    }
                    new_df = pd.DataFrame([record])
                    df     = pd.concat([df, new_df], ignore_index=True)
                    df.to_csv(temp_res_dir, index=False)

                    torch.cuda.empty_cache()
        os.replace(temp_res_dir, res_dir)

    # eps-delta branch
    else:
        dist, corr = train_fn.sensitivity(
            indices=point_to_do,
            cn=args.cn,
            expected_batch_size=args.batch_size
        )
        df = (
            pd.read_csv(temp_res_dir) if os.path.exists(temp_res_dir)
            else pd.read_csv(res_dir) if os.path.exists(res_dir)
            else pd.DataFrame()
        )
        # build a full DataFrame row matching the no‐remove schema + our extra field
        new_df = pd.DataFrame([
            {
                f"distance ({args.reduction})": d,
                "step": step,
                "real batch size": len(point_to_do),
                "p": p,
                "point": int(idx),
                "sigma": train_fn.sigma,
                "correct": int(c),
                "accuracy": accuracy,
                "type": args.stage,

                # --- all original parameters ---
                "batch_size": args.batch_size,
                "num_iters": args.num_iters,
                "alpha": args.alpha,
                "num_batches": args.num_batches,
                "lr": args.lr,
                "cn": args.cn,
                "dataset": args.dataset,
                "dec_lr": args.dec_lr,
                "dp": args.dp,
                "eps": args.eps,
                "optimizer": args.optimizer,
                "model": args.model,
                "norm_type": args.norm_type,
                "save_freq": args.save_freq,
                "save_name": args.save_name,
                "res_name": args.res_name,
                "gamma": args.gamma,
                "id": args.id,
                "seed": args.seed,
                "overwrite": args.overwrite,
                "poisson_train": args.poisson_train,
                "stage": args.stage,
                "reduction": args.reduction,
                "exp": args.exp,
                "less_point": 0,
                "remove_points_count": len(remove_points),
            }
            for idx, d, c in zip(point_to_do, dist, corr)
        ])

        # append & write
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(temp_res_dir, index=False)
        os.replace(temp_res_dir, res_dir)

if __name__ == "__main__":
    main()
