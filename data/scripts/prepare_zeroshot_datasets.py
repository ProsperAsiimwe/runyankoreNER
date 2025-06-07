import os
import shutil

# === CONFIGURATION === #
base_dir = os.path.join(os.path.dirname(__file__), "../")
runyankore_sources = ["SALT", "MPTC", "COMBINED"]
masakha_dir = os.path.join(base_dir, "MasakhaNER2.0")
output_dir = os.path.join(base_dir, "ZeroShotTransfer")

# MasakhaNER 2.0 languages (folder names)
masakha_languages = [
    "bam", "bbj", "ewe", "fon", "hau", "ibo", "kin", "lug", "luo",
    "mos", "nya", "pcm", "sna", "swa", "tsn", "twi", "wol", "xho", "yor", "zul"
]

# === SCRIPT EXECUTION === #
os.makedirs(output_dir, exist_ok=True)

for lang in masakha_languages:
    lang_folder = os.path.join(masakha_dir, lang)
    lang_train = os.path.join(lang_folder, "train.txt")
    lang_dev = os.path.join(lang_folder, "dev.txt")

    for runyankore in runyankore_sources:
        run_test_path = os.path.join(base_dir, runyankore, "test.txt")

        # Output directory
        out_folder = os.path.join(output_dir, f"{lang}_{runyankore}")
        os.makedirs(out_folder, exist_ok=True)

        # Copy train.txt and dev.txt from MasakhaNER2.0 language
        shutil.copy(lang_train, os.path.join(out_folder, "train.txt"))
        shutil.copy(lang_dev, os.path.join(out_folder, "dev.txt"))

        # Copy test.txt from Runyankore dataset
        shutil.copy(run_test_path, os.path.join(out_folder, "test.txt"))

        print(f"[✓] Created Zero-Shot setup: {lang}_{runyankore}")
