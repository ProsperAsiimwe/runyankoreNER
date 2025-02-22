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
            label = entity["entity_group"]
            
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
                labels = entity_map[word]
                
                # Determine BIO tag
                if prev_label == "O" or labels[0] != prev_label:  
                    ner_label = f"B-{labels[0]}"  # Beginning of entity
                else:
                    ner_label = f"I-{labels[0]}"  # Inside entity
                
                prev_label = labels[0]  # Track previous label
            else:
                ner_label = "O"
                prev_label = "O"
            
            outfile.write(f"{word} {ner_label}\n")

        outfile.write("\n")  # Separate sentences by a newline

print(f"✅ Zero-shot NER predictions saved in BIO format to: {output_file}")