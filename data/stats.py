import os
from collections import Counter, defaultdict
from typing import List, Tuple, Dict


def read_conll_sentences(path: str) -> List[List[Tuple[str, str]]]:
    """
    Read a CoNLL-style NER file.
    Returns a list of sentences, where each sentence is a list of (token, label) pairs.
    Assumes the label is in the last column of each non-empty line.
    """
    sentences = []
    current = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue

            parts = line.split()
            token = parts[0]
            label = parts[-1]
            current.append((token, label))

    if current:
        sentences.append(current)

    return sentences


def count_entity_spans(labels: List[str]) -> int:
    """
    Count entity spans in a sequence of BIO labels.
    E.g. B-PER I-PER O B-LOC -> 2 spans.
    """
    spans = 0
    prev = "O"

    for lab in labels:
        if lab.startswith("B-"):
            spans += 1
        elif lab.startswith("I-"):
            # If we see I-XXX after O or a different type, treat as a new span (robustness)
            if prev == "O" or (prev[2:] != lab[2:] and not prev.startswith("B-") and not prev.startswith("I-")):
                spans += 1
        prev = lab

    return spans


def compute_stats_for_file(path: str) -> Dict[str, float]:
    sentences = read_conll_sentences(path)

    num_sentences = len(sentences)
    num_tokens = 0
    num_entity_tokens = 0
    label_counts = Counter()
    total_spans = 0

    for sent in sentences:
        tokens, labels = zip(*sent)
        num_tokens += len(tokens)

        # entity tokens = anything that's not O
        entity_labels = [lab for lab in labels if lab != "O"]
        num_entity_tokens += len(entity_labels)

        # count entity spans for this sentence
        total_spans += count_entity_spans(list(labels))

        # accumulate label counts
        label_counts.update(labels)

    avg_tokens_per_sentence = num_tokens / num_sentences if num_sentences > 0 else 0.0
    avg_entities_per_sentence = total_spans / num_sentences if num_sentences > 0 else 0.0
    entity_token_ratio = num_entity_tokens / num_tokens if num_tokens > 0 else 0.0

    return {
        "num_sentences": num_sentences,
        "num_tokens": num_tokens,
        "num_entity_tokens": num_entity_tokens,
        "num_entity_spans": total_spans,
        "avg_tokens_per_sentence": avg_tokens_per_sentence,
        "avg_entities_per_sentence": avg_entities_per_sentence,
        "entity_token_ratio": entity_token_ratio,
        "label_counts": label_counts,
    }


def print_stats(corpus_name: str, split_name: str, stats: Dict[str, float]):
    print(f"\n=== {corpus_name} :: {split_name} ===")
    print(f"Sentences           : {stats['num_sentences']}")
    print(f"Tokens              : {stats['num_tokens']}")
    print(f"Entity tokens       : {stats['num_entity_tokens']}")
    print(f"Entity spans        : {stats['num_entity_spans']}")
    print(f"Avg tokens/sentence : {stats['avg_tokens_per_sentence']:.2f}")
    print(f"Avg entities/sentence: {stats['avg_entities_per_sentence']:.2f}")
    print(f"Entity token ratio  : {stats['entity_token_ratio']:.3f} (entity tokens / all tokens)")
    print("Label distribution  :")
    for label, count in stats["label_counts"].most_common():
        print(f"  {label:7s} -> {count}")


def main():
    # Adjust these paths to your actual folder layout
    base_dir = "./"
    corpora = {
        "RunyaNER-SALT": os.path.join(base_dir, "SALT"),
        "RunyaNER-MPTC": os.path.join(base_dir, "MPTC"),
    }

    splits = ["train.txt", "dev.txt", "test.txt"]

    for corpus_name, corpus_dir in corpora.items():
        for split in splits:
            path = os.path.join(corpus_dir, split)
            if not os.path.exists(path):
                print(f"[WARN] File not found, skipping: {path}")
                continue

            stats = compute_stats_for_file(path)
            split_name = split.replace(".txt", "")
            print_stats(corpus_name, split_name, stats)


if __name__ == "__main__":
    main()
