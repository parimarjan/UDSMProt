import glob
import argparse
import pdb
import os
import subprocess as sp
import numpy as np
import pickle
import time

CLAS_TMP = '''python modelv1.py classification --from_scratch={FROM_SCRATCH} \
--pretrained_folder={PRETRAINED_FOLDER} --epochs=10 --bs=128 \
--metrics=["binary_auc","binary_auc50","accuracy"] --early_stopping=binary_auc \
--bs=64 --lr=0.05 --fit_one_cycle=False \
--working_folder={WORKING_FOLDER} \
--export_preds=True --eval_on_val_test=True --pretrained_model_filename {PRETRAINED_NAME} \
--epochs 10 --train=True --gradual_unfreezing={UNFREEZE} \
--nh {NH} --nl {NL}'''

SCOP_DIR = '/data/pari/UDSMProt/datasets/clas_scop/'

def save_object(file_name, data):
    with open(file_name, "wb") as f:
        res = f.write(pickle.dumps(data))

def load_object(file_name):
    res = None
    if os.path.exists(file_name):
        with open(file_name, "rb") as f:
            res = pickle.loads(f.read())
    return res

def save_combined_results(results, output_dir):
    print(output_dir)
    fn = output_dir + "/results.pkl"
    old = load_object(fn)
    if old is None:
        old = []
    old.append(results)
    save_object(fn, old)

def load_results(scop_dir):
    res_file = scop_dir + "/result.npy"
    if os.path.exists(res_file):
        return np.load(res_file)
    else:
        return None

def clean_scop(scop_dir):
    '''
    deletes results files / model files from all the SCOP dirs
    '''
    model_dir = scop_dir + "/models"
    model_files = list(glob.glob(model_dir+"/*.pth"))
    for m in model_files:
        os.remove(m)

    res_file = scop_dir + "/result.npy"
    try:
        os.remove(res_file)
    except:
        pass

# train model on all scop directories and evaluate

# collect all results

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_scratch", type=str, required=False,
            default="False")
    parser.add_argument("--pretrained_folder", type=str, required=False,
            default=None)
    parser.add_argument("--pretrained_name", type=str, required=False,
            default="model_3")
    parser.add_argument("--output_dir", type=str, required=False,
            default="./combined_results/")
    parser.add_argument("--gradual_unfreeze", type=str, required=False,
            default="True")
    parser.add_argument("--nh", type=int, required=False,
            default=1150)
    parser.add_argument("--nl", type=int, required=False,
            default=3)

    return parser.parse_args()

def main():
    scop_dirs = list(glob.glob(SCOP_DIR+"/*"))
    output_dir = args.output_dir + "/" + os.path.basename(args.pretrained_folder)
    print(output_dir)
    p = sp.Popen("mkdir -p {}".format(output_dir), shell=True)
    p.wait()
    print("going to save results at: ", output_dir)
    for sci, scop_dir in enumerate(scop_dirs):
        start=time.time()
        basename = os.path.basename(scop_dir)
        if "scop" not in basename:
            continue
        testfile = output_dir + "/" + basename + ".log"
        if os.path.exists(testfile):
            print("skipping: ", basename)
            continue

        logfile = output_dir + "/" + basename + ".log2"
        lf = open(logfile, "w+")
        clean_scop(scop_dir)

        exec_cmd = CLAS_TMP.format(FROM_SCRATCH = args.from_scratch,
                        PRETRAINED_FOLDER = args.pretrained_folder,
                        WORKING_FOLDER = scop_dir,
                        UNFREEZE = args.gradual_unfreeze,
                        PRETRAINED_NAME = args.pretrained_name,
                        NL = args.nl,
                        NH = args.nh)
        p = sp.Popen(exec_cmd, shell=True, stdout=lf)
        # p = sp.Popen(exec_cmd, shell=True)
        p.wait()
        lf.close()

        # load results, and clean result files
        results = load_results(scop_dir)
        if results is None:
            continue
        else:
            res_file = output_dir + "/" + basename + ".res"
            lf = open(res_file, "w+")
            lf.write("1")
            lf.close()

        print("{}, results: {}".format(basename, results))
        save_combined_results(results, output_dir)
        clean_scop(scop_dir)

        print("took: ", time.time()-start)

if __name__ == "__main__":

    print(SCOP_DIR)
    args = read_flags()
    main()
