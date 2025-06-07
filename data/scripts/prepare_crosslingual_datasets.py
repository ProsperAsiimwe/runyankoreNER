import os
import shutil

# === CONFIGURATION === #
base_dir = os.path.join(os.path.dirname(__file__), "../")
runyankore_sources = ["SALT", "MPTC", "COMBINED"]
masakha_dir = os.path.join(base_dir, "MasakhaNER2.0")
output_dir = os.path.join(base_dir, "CrossLingualCombined")

# MasakhaNER 2.0 languages (folder names)
masakha_languages = [
    "bam", "bbj", "ewe", "fon", "hau", "ibo", "kin", "lug", "luo",
    "mos", "nya", "pcm", "sna", "swa", "tsn", "twi", "wol", "xho", "yor", "zul"
]

# === SCRIPT EXECUTION === #
os.makedirs(output_dir, exist_ok=True)

for lang in masakha_languages:
    for runyankore in runyankore_sources:
        # Define input paths
        lang_train_path = os.path.join(masakha_dir, lang, "train.txt")
        run_train_path = os.path.join(base_dir, runyankore, "train.txt")
        run_dev_path = os.path.join(base_dir, runyankore, "dev.txt")
        run_test_path = os.path.join(base_dir, runyankore, "test.txt")

        # Define output directory
        out_folder = os.path.join(output_dir, f"{lang}_{runyankore}")
        os.makedirs(out_folder, exist_ok=True)

        # === Combine train.txt ===
        out_train_file = os.path.join(out_folder, "train.txt")
        with open(out_train_file, "w", encoding="utf-8") as outfile:
            for src in [lang_train_path, run_train_path]:
                with open(src, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read().strip() + "\n\n")
        print(f"[✓] Combined train.txt for {lang}_{runyankore}")

        # === Copy dev.txt and test.txt ===
        shutil.copy(run_dev_path, os.path.join(out_folder, "dev.txt"))
        shutil.copy(run_test_path, os.path.join(out_folder, "test.txt"))
        print(f"[✓] Copied dev.txt and test.txt for {lang}_{runyankore}")
