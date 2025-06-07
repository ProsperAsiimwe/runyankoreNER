import os

# Define file paths
base_dir = os.path.join(os.path.dirname(__file__), "../")
datasets = ["SALT", "MPTC"]
output_dir = os.path.join(base_dir, "COMBINED")
splits = ["train.txt", "dev.txt", "test.txt"]

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def combine_files(split):
    combined_path = os.path.join(output_dir, split)
    with open(combined_path, "w", encoding="utf-8") as combined_file:
        for dataset in datasets:
            input_path = os.path.join(base_dir, dataset, split)
            with open(input_path, "r", encoding="utf-8") as infile:
                combined_file.write(infile.read().strip() + "\n\n")  # Ensure sentence separation
    print(f"Combined {split} written to {combined_path}")

# Combine all splits
for split in splits:
    combine_files(split)
