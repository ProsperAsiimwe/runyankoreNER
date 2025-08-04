import os
import shutil

# === CONFIGURATION === #
base_dir = os.path.join(os.path.dirname(__file__), "../")  # Adjust path as needed
runyankore_sources = ["COMBINED"]
typological_aux_langs = ["lug", "kin", "nya"]
masakha_dir = os.path.join(base_dir, "MasakhaNER2.0")
output_dir = os.path.join(base_dir, "EmbeddingGroups/COMBINED")

# === SCRIPT EXECUTION === #
os.makedirs(output_dir, exist_ok=True)

for runyankore in runyankore_sources:
    # Input: Runyankore files
    run_train_path = os.path.join(base_dir, runyankore, "train.txt")
    run_dev_path = os.path.join(base_dir, runyankore, "dev.txt")
    run_test_path = os.path.join(base_dir, runyankore, "test.txt")

    # Output folder (combined with typological group)
    out_folder = os.path.join(output_dir, f"mbert_{runyankore}")
    os.makedirs(out_folder, exist_ok=True)

    # === Combine train.txt ===
    out_train_file = os.path.join(out_folder, "train.txt")
    with open(out_train_file, "w", encoding="utf-8") as outfile:
        # Add Runyankore train
        with open(run_train_path, "r", encoding="utf-8") as infile:
            outfile.write(infile.read().strip() + "\n\n")
        
        # Add each auxiliary language train
        for lang in typological_aux_langs:
            aux_train_path = os.path.join(masakha_dir, lang, "train.txt")
            with open(aux_train_path, "r", encoding="utf-8") as infile:
                outfile.write(infile.read().strip() + "\n\n")

    print(f"[✓] Created combined train.txt for mbert_{runyankore}")

    # === Copy dev and test from Runyankore ===
    shutil.copy(run_dev_path, os.path.join(out_folder, "dev.txt"))
    shutil.copy(run_test_path, os.path.join(out_folder, "test.txt"))
    print(f"[✓] Copied dev.txt and test.txt for mbert_{runyankore}")
