import json
import os

data_base_path = './DATA/'

input_file = os.path.join(data_base_path, 'CLEAN_DATA_SALT_SPACE_TOKENIZED_DOCCANO/test_salt_tokenized-eng-nyn-doccano.jsonl')  # The cleaned JSONL file
output_file = os.path.join(data_base_path, 'CLEAN_DATA_SALT_SPACE_TOKENIZED_DOCCANO/test_salt_english_sentences.txt')  # File to store extracted translations

print("🚀 Extracting English translations...")

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        data = json.loads(line)
        translation = data.get("metadata", {}).get("translation")  # Safely access translation
        if translation:  # Ensure translation exists
            outfile.write(translation + "\n")

print(f"✅ Extraction complete! Saved to: {output_file}")
