import os

# Set input file path
DATA_DIR = os.path.join(os.path.dirname(__file__), "../runyankore/multilingual_harvard_dataset/")
input_file = os.path.join(DATA_DIR, "test.txt")  # Change this to dev.txt or test.txt if needed

# Set output file path
base_name = os.path.basename(input_file)
output_file = os.path.join(DATA_DIR, f"cleaned_{base_name}")

def clean_blank_lines(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cleaned_lines = []
    blank_count = 0
    removed_excess_lines = 0

    for line in lines:
        if line.strip() == "":
            blank_count += 1
        else:
            if blank_count > 1:
                # We only keep 1 blank line, so the rest are excess
                removed_excess_lines += (blank_count - 1)
                cleaned_lines.append("\n")
            elif blank_count == 1:
                cleaned_lines.append("\n")
            cleaned_lines.append(line)
            blank_count = 0

    # Add a newline if the last line isn't already one
    if cleaned_lines and cleaned_lines[-1].strip() != "":
        cleaned_lines.append("\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(cleaned_lines)

    print(f"✅ Cleaned version saved to: {os.path.basename(output_path)}")
    print(f"🧹 Removed {removed_excess_lines} excess blank line(s)")

# Run the function
clean_blank_lines(input_file, output_file)
