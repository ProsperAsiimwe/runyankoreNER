
'''
python3 check_performance_files.py \
  --files ct_f1_mbert.csv ct_f1_xlmr.csv zs_f1_mbert.csv zs_f1_xlmr.csv
'''

import argparse, os, re
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    args = ap.parse_args()

    pat = re.compile(r'(?P<setup>ct|zs).*?(?P<model>mbert|xlmr)', re.IGNORECASE)
    all_rows = []
    for f in args.files:
        fn = os.path.basename(f)
        m = pat.search(fn)
        if not m:
            print(f"[skip] {fn} (cannot infer setup/model)")
            continue
        setup = "co-train" if m.group("setup").lower()=="ct" else "zero-shot"
        model = m.group("model").lower()
        df = pd.read_csv(f)
        print(f"[read] {fn}  -> setup={setup}, model={model}, rows={len(df)}")
        if not {"language","f1"}.issubset(df.columns):
            print("       !! missing columns language,f1")
        else:
            bad = df["f1"].isna().sum()
            print(f"       f1 NaNs: {bad}")

if __name__ == "__main__":
    main()
