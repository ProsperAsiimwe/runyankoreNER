import json
import os

data_base_path = './DATA/'

input_file = os.path.join(data_base_path, 'CLEAN_DATA_SALT_SPACE_TOKENIZED_DOCCANO/train_salt_tokenized-eng-nyn-doccano.jsonl')  # The cleaned JSONL file
output_file = os.path.join(data_base_path, 'CLEAN_DATA_SALT_SPACE_TOKENIZED_DOCCANO/train_salt_runyankore_sentences.txt')  # File to store extracted sentences

print("🚀 Extracting Runyankore sentences...")

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        data = json.loads(line)
        sentence = data["text"]
        outfile.write(sentence + "\n")

print(f"✅ Extraction complete! Saved to: {output_file}")
