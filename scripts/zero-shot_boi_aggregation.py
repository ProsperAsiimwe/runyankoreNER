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
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

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

        # Track assigned entities
        entity_map = {}
        for entity in ner_results:
            word = entity["word"]
            label = entity["entity"]  # ✅ Use "entity" instead of "entity_group"

            # Fix subword tokenization issues
            if word.startswith("##"):  # Handle BERT-like subwords
                word = word.replace("##", "")

            if word in entity_map:
                entity_map[word].append(label)
            else:
                entity_map[word] = [label]

        # Write in BIO format
        prev_label = "O"
        for word in sentence.split():
            if word in entity_map:
                raw_label = entity_map[word][0]  # Get the predicted label
                if "-" in raw_label:
                    prefix, label = raw_label.split("-", 1)  # Extract the prefix
                    if prefix in ["B", "I"]:
                        ner_label = raw_label  # Already in correct BIO format
                    else:
                        ner_label = f"B-{label}"  # Force BIO compliance if incorrect format detected
                else:
                    ner_label = f"B-{raw_label}"  # Assign "B-" for new entities
                
                prev_label = raw_label  # Track previous label
            else:
                ner_label = "O"
                prev_label = "O"
            
            outfile.write(f"{word} {ner_label}\n")

        outfile.write("\n")  # Separate sentences by a newline

print(f"✅ Zero-shot NER predictions saved in BIO format to: {output_file}")
