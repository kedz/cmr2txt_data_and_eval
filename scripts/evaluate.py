import argparse
import pathlib
from tempfile import NamedTemporaryFile as tf
import json
from subprocess import check_output, DEVNULL
import numpy as np
import pandas as pd
from collections import Counter


def run_e2echal_eval_script(script_path, ref_path, pred_path):
    out = check_output([str(script_path), ref_path, pred_path],
                       stderr=DEVNULL)
    lines = out.decode('utf8').strip().split("\n")[-5:]

    results = {}
    for line in lines:
        m, v = line.split(": ")
        results[m] = float(v)

    return results


def lmr_error_counts(true_lmr, pred_lmr):
    true_counts = Counter(true_lmr)
    pred_counts = Counter(pred_lmr)

    for slot_filler in pred_counts.keys():

        for _ in range(pred_counts[slot_filler]):
            if true_counts[slot_filler] > 0:
                true_counts[slot_filler] -= 1
                pred_counts[slot_filler] -= 1
    true_slot_counts = Counter(
        [sf.split("=")[0] for sf, c in true_counts.items()
         if c > 0])

    incorrect = Counter()
    for sf in pred_counts.keys():
        c = pred_counts[sf]
        if c == 0: continue

        slot = sf.split("=")[0]
        if true_slot_counts[slot] > 0:
            incorrect[slot] += 1
            true_slot_counts[slot] -= 1
            for key in true_counts.keys():
                if key.startswith(slot) and true_counts[key] > 0:
                    true_counts[key] -= 1
                    break
            pred_counts[sf] -= 1
    missing = {k: v for k, v in true_counts.items() if v > 0}
    added = {k: v for k, v in pred_counts.items() if v > 0}
    tot_incorrect = sum(incorrect.values())
    tot_missing = sum(missing.values())
    tot_added = sum(added.values())
    tot_all = tot_incorrect + tot_missing + tot_added

    return {
        "incorrect": tot_incorrect,
        "missing": tot_missing,
        "added": tot_added,
        "total": tot_all,
    }

def evaluate_path(path, eval_script, corrections):
    errors = []
    total_slots = 0
    order_correct = []
    with open(path, 'r') as fp, tf("w") as ref_fp, tf("w") as mod_fp:
        for line in fp:
            # Load the next example from the json line.
            ex = json.loads(line)

            # Get the slot/values (i.e. meaning representation)
            # that were fed into the nlg model. 
            true_slot_fillers = ex['input_slot_fillers']
            total_slots += len(true_slot_fillers)
 
            # Get the best beam output from the nlg model.            
            output = ex['outputs'][ex['reranked_beam_output_index']]

            # Print the reference strings to file 
            print(ex['references'], end="\n\n", file=ref_fp)
            # Print the nlg output string to file.
            print(output['pretty'], file=mod_fp)
           
            KEY = " ".join(output['tokens'])
            if KEY in corrections:
                _, pred_slot_fillers = corrections[KEY]

            else:
                pred_slot_fillers = output['pred_lmr']

            err = lmr_error_counts(true_slot_fillers, pred_slot_fillers)
            errors.append(
                [err[x] for x in ['missing', 'incorrect', 'added', 'total']]
            )

            order_correct.append(pred_slot_fillers == true_slot_fillers) 

        mod_fp.flush()
        ref_fp.flush()
        autometrics = run_e2echal_eval_script(eval_script, ref_fp.name, mod_fp.name)
    results = {m: autometrics[m] * 100 for m in ['BLEU', 'ROUGE_L', "METEOR"]}
    results["CIDEr"] = autometrics["CIDEr"]
    results["NIST"] = autometrics["NIST"]
    results["path"] = path.name
    errors = np.array(errors)
    all_correct = (errors[:,-1] == 0).sum()
    results["Perf (%)"] = 100.0 * all_correct / len(order_correct)
    for v, n in zip(errors.sum(axis=0).tolist(), ['missing', 'incorrect', 'added', 'all']):
        results[n] = v
        if n == 'all':
            results['SER (%)'] = v / total_slots * 100
    results['Order (Acc. %)'] = 100.0 * np.sum(order_correct) / len(order_correct)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--corrections", type=pathlib.Path, default=None)
    parser.add_argument("eval_script", type=pathlib.Path)
    parser.add_argument("paths", nargs="+", type=pathlib.Path)

    args = parser.parse_args()

    corrections = {}
   
    if args.corrections:
        with args.corrections.open('r') as fp:
            for line in fp:
                k, v = json.loads(line)
                corrections[k] = v
        print(f"Read {len(corrections)} corrections ...")

    results = []
    for path in args.paths:
        print(path)
        results.append(evaluate_path(path, args.eval_script, corrections))

    cols = [
        "path", 'BLEU', 'NIST', 'METEOR', 'ROUGE_L', 'CIDEr',
        'missing', 'incorrect', 'added', 'all', "SER (%)", "Order (Acc. %)", "Perf (%)", 
    ]
    df = pd.DataFrame(results, columns=cols)
    mean_df = df[cols[1:]].mean().to_frame().T
    mean_df["path"] = ["mean"]
    df = pd.concat([df, mean_df])
    print(df)

if __name__ == "__main__":
    main() 
