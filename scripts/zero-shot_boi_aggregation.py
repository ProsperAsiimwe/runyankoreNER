from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models/")
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/luganda/")

# Specify the path to your fine-tuned Luganda NER model
trained_model_path = os.path.join(MODEL_DIR, "luganda_xlmr")  # Update this path

print(f"🚀 Loading your fine-tuned Luganda NER model from: {trained_model_path}")

# Load the fine-tuned Luganda NER model
tokenizer = AutoTokenizer.from_pretrained(trained_model_path)
model = AutoModelForTokenClassification.from_pretrained(trained_model_path)

# Load NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy=None)

# Input and output files
input_file = os.path.join(DATA_DIR, "combined_runyankore_sentences.txt") 
output_file = os.path.join(DATA_DIR, "runyankore_ner_predictions.conll")

print("🚀 Performing zero-shot NER with BIO format on Runyankore sentences...")

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        sentence = line.strip()
        if not sentence:
            continue  # Skip empty lines

        # Run the model for NER prediction
        ner_results = ner_pipeline(sentence)

        # Process raw entity predictions
        entity_map = {}
        for entity in ner_results:
            word = entity["word"].replace("##", "")  # Fix subword tokenization
            label = entity["entity"]

            # Store entity positions
            if word in entity_map:
                entity_map[word].append(label)
            else:
                entity_map[word] = [label]

        # Reconstruct full sentence with BIO format
        prev_label = "O"
        for word in sentence.split():
            if word in entity_map:
                raw_label = entity_map[word][0]  # Get the first entity label

                # Ensure correct BIO format
                if "-" in raw_label:
                    prefix, label = raw_label.split("-", 1)
                    if prefix not in ["B", "I"]:
                        ner_label = f"B-{label}"
                    else:
                        ner_label = raw_label
                else:
                    ner_label = f"B-{raw_label}"

                prev_label = ner_label  # Track previous label
            else:
                ner_label = "O"
                prev_label = "O"

            outfile.write(f"{word} {ner_label}\n")

        outfile.write("\n")  # Separate sentences by a newline

print(f"✅ Zero-shot NER predictions saved in BIO format to: {output_file}")
