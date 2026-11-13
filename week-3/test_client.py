import unittest

from llm_client import LLMProvider, LLMClient
from cost_calculator import calculate_cost


# Fake provider for testing
class FakeProvider(LLMProvider):

    model = "fake-test-model"

    def generate(self, prompt):

        return {
            "text": "Fake answer: " + prompt,
            "input_tokens": 5,
            "output_tokens": 7
        }


class FakeLogger:

    def __init__(self):
        self.records = []

    def log_call(self, record):
        self.records.append(record)
        return True


class TestLLMClient(unittest.TestCase):

    def test_response(self):

        logger = FakeLogger()

        client = LLMClient(
            FakeProvider(),
            logger,
            max_retries=1
        )

        result = client.generate("Hello")

        self.assertEqual(
            result["text"],
            "Fake answer: Hello"
        )

        self.assertEqual(
            result["metrics"]["total_tokens"],
            12
        )

        self.assertEqual(len(logger.records), 1)

    def test_ollama_cost(self):

        self.assertEqual(
            calculate_cost("llama3.2", 100, 100),
            0.0
        )


if __name__ == "__main__":
    unittest.main()