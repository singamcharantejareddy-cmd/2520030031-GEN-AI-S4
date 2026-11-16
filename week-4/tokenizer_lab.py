
from transformers import AutoTokenizer
import csv

# STEP 1: Input sentences in three languages

sentences = {
    "English": "Artificial Intelligence is transforming education and healthcare.",
    "Telugu": "కృత్రిమ మేధస్సు విద్య మరియు వైద్య రంగాలను మారుస్తోంది.",
    "Hindi": "कृ त्रिम बुद्धिमत्ता शिक्षा और स्वास्थ्य सेवा को बदल रही है।"
}

# STEP 2: Select three tokenizer models

tokenizer_models = {
    "BPE (GPT-2)": {
        "model": "openai-community/gpt2",
        "use_fast": True
    },

    "WordPiece (Multilingual BERT)": {
        "model": "google-bert/bert-base-multilingual-cased",
        "use_fast": True
    },

    "SentencePiece (mT5 Unigram)": {
        "model": "google/mt5-small",
        "use_fast": False
    }
}

# STEP 3: Load tokenizers

loaded_tokenizers = {}

print("\nLoading Tokenizers...")

for name, config in tokenizer_models.items():

    print("\nLoading:", name)

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"],
        use_fast=config["use_fast"]
    )

    loaded_tokenizers[name] = tokenizer

    print("Loaded successfully!")

# STEP 4: Tokenize sentences

results = []

print("\n========== TOKENIZATION RESULTS ==========")

for language, sentence in sentences.items():

    print("\n--------------------------------------")
    print("Language:", language)
    print("Original Sentence:", sentence)

    word_count = len(sentence.split())

    for tokenizer_name, tokenizer in loaded_tokenizers.items():

        token_ids = tokenizer.encode(
            sentence,
            add_special_tokens=False
        )

        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        token_count = len(token_ids)

        token_ratio = (
            token_count / word_count if word_count else 0
        )

        print("\nTokenizer:", tokenizer_name)
        print("Tokens:", tokens)
        print("Token Count:", token_count)
        print("Word Count:", word_count)
        print("Token Ratio:", round(token_ratio, 3))

        results.append({
            "Language": language,
            "Tokenizer": tokenizer_name,
            "Word_Count": word_count,
            "Token_Count": token_count,
            "Token_Ratio": round(token_ratio, 3)
        })

# STEP 5: Display comparison table

print("\n\n========== TOKEN RATIO COMPARISON ==========")

print(
    f"{'Language':<12}"
    f"{'Tokenizer':<35}"
    f"{'Words':<10}"
    f"{'Tokens':<10}"
    f"{'Ratio':<10}"
)

print("-" * 77)

for row in results:

    print(
        f"{row['Language']:<12}"
        f"{row['Tokenizer']:<35}"
        f"{row['Word_Count']:<10}"
        f"{row['Token_Count']:<10}"
        f"{row['Token_Ratio']:<10}"
    )

# STEP 6: Export results to CSV

with open(
    "tokenizer_results.csv",
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    fieldnames = [
        "Language",
        "Tokenizer",
        "Word_Count",
        "Token_Count",
        "Token_Ratio"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)

print("\nResults saved to tokenizer_results.csv")

print("Week 4 experiment completed successfully!")
