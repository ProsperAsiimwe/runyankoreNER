import json
import os
import re

data_base_path = './DATA/'

input_file = os.path.join(data_base_path, 'SALT_DATA_SPACE_TOKENIZED_DOCANNO/test_salt_tokenized-eng-nyn-doccano.jsonl')  # Your original dataset file
output_file = os.path.join(data_base_path, 'CLEAN_DATA_SALT_SPACE_TOKENIZED_DOCCANO/test_salt_tokenized-eng-nyn-doccano.jsonl')

# Regex pattern to remove slashes attached to words
slash_pattern = re.compile(r"/(\w+)|/ ")

# Regex pattern to fix spacing around apostrophes
apostrophe_pattern = re.compile(r"(\w)' (\w)")

# Regex pattern to replace multiple spaces with a single space
space_pattern = re.compile(r"\s+")

print("🚀 Starting data cleaning...")

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        data = json.loads(line)
        original_text = data["text"]

        # Step 1: Remove slashes from words
        cleaned_text = slash_pattern.sub(r"\1", original_text)
        
        print(f"✅ Removed slashes:\n🔹 Before: {original_text}\n🔹 After:  {cleaned_text}\n")

        # Step 2: Fix incorrect white spacing around apostrophes
        cleaned_text = apostrophe_pattern.sub(r"\1'\2", cleaned_text)

        print(f"✅ Fixed apostrophes:\n🔹 {cleaned_text}\n")

        # Step 3: Ensure proper spacing
        cleaned_text = space_pattern.sub(" ", cleaned_text).strip()

        print(f"✅ Fixed spacing:\n🔹 {cleaned_text}\n")

        # Save cleaned text
        data["text"] = cleaned_text
        json.dump(data, outfile, ensure_ascii=False)
        outfile.write("\n")

print("🎉 Cleaning complete! Processed dataset saved to:", output_file)