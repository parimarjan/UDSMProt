import pickle
import pdb
import os
import pandas as pd
import numpy as np

def save_object(file_name, data):
    with open(file_name, "wb") as f:
        res = f.write(pickle.dumps(data))

def load_object(file_name):
    res = None
    print(os.path.exists(file_name))
    if os.path.exists(file_name):
        with open(file_name, "rb") as f:
            res = pickle.loads(f.read())

    return res

def summarize(FN):
    results = load_object(FN)
    print(len(results))
    auc = [r[1][-3] for r in results]
    acc = [r[1][-1] for r in results]
    # print(sum(acc) / len(acc))
    auc = auc[0:10]
    acc = acc[0:10]

    print("accuracy: ")
    print(np.round(acc, 2))
    print("auc: ")
    print(np.round(auc, 2))
    print("auc: ", sum(auc) / len(auc))
    print(sum(acc) / len(acc))
    pdb.set_trace()

FN = "combined_results_orig/lm_h128/results.pkl"
# FN = "combined_results/lm_h512/results.pkl"
# FN = "combined_results/from_scratch-nh128/results.pkl"
FN3 = "./combined_results_orig/lm_h128_shuffled/results.pkl"
# FN = "./combined_results_orig/lm_h512/results.pkl"
# FN2 = "./combined_results_orig/lm_h128/results.pkl"
# FN4 = "./combined_results/lm_h256/results.pkl"
# FNS = [FN, FN2, FN4]
FNS = ["./combined_results/lm_sprot_uniref_fwd/results.pkl", FN, FN3]
for fn in FNS:
    print(fn)
    summarize(fn)
# summarize(FN)
# summarize(FN2)
# summarize(FN3)
