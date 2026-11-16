import unittest
from project_demo.data import prompt


class PromptParityTests(unittest.TestCase):
    def test_v07_column_mapped_plain_prompt(self):
        # Official v0.7 ColumnMapped dataset when use_hf_chat_template=False.
        row = {"context": "source facts", "question": "What is the result?"}
        self.assertEqual(prompt(row), "source facts What is the result? ")


class ChatFormattingTests(unittest.TestCase):
    def test_checkpoint_chat_does_not_duplicate_bos(self):
        from project_demo.inference import encode_prompt

        class Tokenizer:
            chat_template = "shipped-template"

            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                self.kwargs = kwargs
                return "<bos>formatted<assistant>"

            def __call__(self, text, **kwargs):
                return {"text": text, **kwargs}

        tokenizer = Tokenizer()
        actual = encode_prompt(
            tokenizer, {"context": "facts", "question": "question"}, "checkpoint-chat"
        )
        self.assertFalse(actual["add_special_tokens"])
        self.assertEqual(
            tokenizer.messages, [{"role": "user", "content": "facts question "}]
        )
        self.assertTrue(tokenizer.kwargs["add_generation_prompt"])
