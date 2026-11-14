
import csv
try:
    from transformers import AutoTokenizer
except ImportError:
    print("\nERROR: The 'transformers' library is not installed.")
    print("Install it using: pip install transformers")
    exit(1)


# STEP 1: Input sentences in three languages
# These sentences follow the Week-4 practical document.
sentences = {
    "English": "Artificial Intelligence is transforming education and healthcare.",
    "Telugu": "కృత్రిమ మేధస్సు విద్య మరియు వైద్య రంగాలను మారుస్తోంది.",
    "Hindi": "कृ त्रिम बुद्धिमत्ता शिक्षा और स्वास्थ्य सेवा को बदल रही है।",
}

# STEP 2: Select three tokenizer models
tokenizer_models = {
    "BPE (GPT-2)": {
        "model": "openai-community/gpt2",
        "use_fast": True,
    },
    "WordPiece (Multilingual BERT)": {
        "model": "google-bert/bert-base-multilingual-cased",
        "use_fast": True,
    },
    "SentencePiece (mT5 Unigram)": {
        "model": "google/mt5-small",
        "use_fast": False,
    },
}


def main():
    # STEP 3: Load the tokenizers
    loaded_tokenizers = {}

    print("\nLoading Tokenizers...")
    for name, config in tokenizer_models.items():
        print(f"\nLoading: {name}")
        tokenizer = AutoTokenizer.from_pretrained(
            config["model"],
            use_fast=config["use_fast"],
        )
        loaded_tokenizers[name] = tokenizer
        print("Loaded successfully!")

    # STEP 4: Tokenize the sentences
    results = []

    print("\n========== TOKENIZATION RESULTS ==========")
    for language, sentence in sentences.items():
        print("\n--------------------------------------")
        print("Language:", language)
        print("Original Sentence:", sentence)

        # Count whitespace-separated words
        word_count = len(sentence.split())

        for tokenizer_name, tokenizer in loaded_tokenizers.items():
            # Convert sentence into token IDs
            token_ids = tokenizer.encode(
                sentence,
                add_special_tokens=False,
            )

            # Convert IDs into readable token pieces
            tokens = tokenizer.convert_ids_to_tokens(token_ids)

            # Count tokens and calculate token ratio
            token_count = len(token_ids)
            token_ratio = token_count / word_count if word_count else 0

            print("\nTokenizer:", tokenizer_name)
            print("Tokens:", tokens)
            print("Token Count:", token_count)
            print("Word Count:", word_count)
            print("Token Ratio:", round(token_ratio, 3))

            # Save results in a list
            results.append({
                "Language": language,
                "Tokenizer": tokenizer_name,
                "Word_Count": word_count,
                "Token_Count": token_count,
                "Token_Ratio": round(token_ratio, 3),
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
    output_file = "tokenizer_results.csv"
    with open(output_file, "w", newline="", encoding="utf-8-sig") as file:
        fieldnames = [
            "Language",
            "Tokenizer",
            "Word_Count",
            "Token_Count",
            "Token_Ratio",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_file}")
    print("Week 4 experiment completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except OSError as error:
        print("\nA network or model-loading error occurred.")
        print("Check your internet connection and try running the program again.")
        print("Details:", error)
    except Exception as error:
        print("\nThe program could not complete.")
        print("Details:", error)
