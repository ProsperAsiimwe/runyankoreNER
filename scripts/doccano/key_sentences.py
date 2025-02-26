import json
import os
from transformers import pipeline

data_base_path = './DATA/'

# Load an English NER model (choose a good one)
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03", tokenizer="dbmdz/bert-large-cased-finetuned-conll03")

# Input dataset file
jsonl_file = os.path.join(data_base_path, 'CLEAN_DATA_SALT_SPACE_TOKENIZED_DOCCANO/test_salt_tokenized-eng-nyn-doccano.jsonl')  # Update with actual file path
output_file = os.path.join(data_base_path, 'RELEVANT_SENTENCES/filtered_runyankore_sentences.txt')

# Entity categories we are interested in
target_entities = {"PER", "LOC", "ORG", "DATE"}

print("🚀 Processing English translations for NER...")

# Read JSONL file and extract relevant sentences
with open(jsonl_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    total_sentences = 0
    selected_sentences = 0

    for line in infile:
        total_sentences += 1
        data = json.loads(line)

        runyankore_text = data["text"]
        english_translation = data["metadata"]["translation"]

        # Run NER on the English translation
        ner_results = ner_pipeline(english_translation)

        # Check if at least one target entity is found
        found_entities = {entity["entity"] for entity in ner_results if entity["entity"] in target_entities}

        if found_entities:
            selected_sentences += 1
            outfile.write(f"{runyankore_text}\n")

    print(f"✅ Processed {total_sentences} sentences.")
    print(f"🔍 Identified {selected_sentences} sentences with named entities.")
    print(f"📂 Saved to {output_file}")
