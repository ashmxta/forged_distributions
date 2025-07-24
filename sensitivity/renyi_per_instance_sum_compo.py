import pandas as pd
import numpy as np
import scipy
import math
import os
from scipy.special import comb

'''
- Computes the per-step contributions to the over per-instance composition guarantee (applied to the sum update rule).
- Saves a new CSV with per-step privacy costs to `compo_res/compo_resX.csv`
'''

def binom(n, k):
    return math.factorial(n) / math.factorial(k) / math.factorial(n - k)

def gpi_alpha(order, alpha, i):
    res = alpha
    for j in range(i):
        res = res * order / (order - 1) - 1 / order
    return res

def stepi_compo(alpha, sigma, q, cn, i, order):
    alpha = int(np.ceil(gpi_alpha(order, alpha, int(i))))
    res = []
    for k in range(alpha + 1):
        coeff = np.log(binom(alpha, k) * math.pow(1 - q, alpha - k) * math.pow(q, k))
        expect = math.pow(cn, 2) * k * (k - 1) / (2 * math.pow(sigma, 2))
        res.append(coeff + expect)
    divergence = scipy.special.logsumexp(res) / (alpha - 1)
    return divergence * order * (alpha - 1)

def scale(x, i, order, alpha):
    temp = i * np.log(order - 1) - (i + 1) * np.log(order) + np.log(x) - np.log(alpha - 1)
    return np.exp(temp)

import sys

if len(sys.argv) < 2:
    print("Usage: python renyi_per_instance_sum_compo.py res/resX.csv")
    sys.exit(1)

res_file = sys.argv[1]
feature = 'distance (sum)'
dataset = "MNIST"
model = "lenet"
cn = 1
eps = 10
alpha = 8

df = pd.read_csv(res_file)
# a = []
# import copy
# for t in range(10):
#     temp = copy.deepcopy(df)
#     temp['step'] = t
#     a.append(temp)
# df = pd.concat(a)
print(f"Using {len(df['step'].unique())} checkpoints")
assert len(df['sigma'].unique()) == 1
assert len(df['batch_size'].unique()) == 1
sigma = df['sigma'].unique()[0]
bs = df['batch_size'].unique()[0]
order = df['step'].max() * 3

if dataset == "MNIST":
    p = bs / 60000
elif dataset == "CIFAR10":
    p = bs / 50000
else:
    raise NotImplementedError

#First compute the scaled per-instance guarantee for composition for each step (before expectation)
df[feature] = df.apply(lambda x: stepi_compo(alpha, sigma, p, x[feature], x['step'], order), axis=1)
#Now compute the expectation over the trials to obtain the per-step contribution in composition
df = df.groupby(["point", "step"], as_index=False)[feature].apply(lambda grp: scipy.special.logsumexp(grp) - np.log(grp.count()))
df = df.rename(columns={feature: "Privacy cost"})
# print(df)

#File saving
os.makedirs("compo_res_rm100", exist_ok=True)
input_filename = os.path.basename(res_file)
output_filename = input_filename.replace("res_rm100", "compo_res_rm100")
output_path = os.path.join("compo_res_rm100", output_filename)
df.to_csv(output_path, index=False)
print(f"Saved per-step privacy costs to: {output_path}")
