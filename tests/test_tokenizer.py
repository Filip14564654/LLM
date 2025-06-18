import unittest
from tokenizer.bpe_tokenizer import BPETokenizer

class TokenizerTest(unittest.TestCase):
    def test_basic_training_and_tokenize(self):
        tok = BPETokenizer(vocab_size=10)
        corpus = ["hello world", "hello there"]
        tok.train(corpus)
        encoded = tok.tokenize("hello")
        self.assertTrue(len(encoded) > 0)

if __name__ == '__main__':
    unittest.main()
