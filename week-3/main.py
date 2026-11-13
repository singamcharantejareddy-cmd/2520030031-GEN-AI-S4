from dotenv import load_dotenv

from llm_client import (
    LLMClient,
    GeminiProvider,
    OllamaProvider
)

from mongo_logger import MongoLogger


def main():

    load_dotenv()

    logger = MongoLogger()

    try:

        print("\n===== LLM MONITORING APPLICATION =====")

        print("1. Gemini")
        print("2. Ollama")

        choice = input("Select provider: ").strip()

        if choice == "1":
            provider = GeminiProvider()

        elif choice == "2":
            provider = OllamaProvider()

        else:
            print("Invalid selection")
            return

        prompt = input("\nEnter your question: ").strip()

        if not prompt:
            print("Please enter a question")
            return

        client = LLMClient(
            provider,
            logger
        )

        result = client.generate(prompt)

        print("\n===== MODEL RESPONSE =====")
        print(result["text"])

        print("\n===== USAGE DETAILS =====")

        for key, value in result["metrics"].items():
            print(key, ":", value)

    except Exception as error:

        print("Error:", error)

    finally:
        logger.close()


if __name__ == "__main__":
    main()