import unittest
import importlib.util

torch_spec = importlib.util.find_spec("torch")
torch_available = torch_spec is not None
if torch_available:
    import torch
    from model.transformer import TransformerModel

@unittest.skipUnless(torch_available, "requires PyTorch")
class ModelTest(unittest.TestCase):
    def test_forward_pass(self):
        model = TransformerModel(vocab_size=20, embed_dim=16, num_heads=2, ff_dim=32, num_layers=1, max_len=8)
        inp = torch.randint(0, 20, (2, 5))
        out = model(inp)
        self.assertEqual(out.shape, (2, 5, 20))

if __name__ == '__main__':
    unittest.main()
