# tests/test_models.py

import unittest
import torch
from app.models.base_model import BaseModel
from app.models.fastgrok_model import FastGrokModel
from app.models.noise_model import NoiseModel

# test_models.py

class TestModels(unittest.TestCase):
    def setUp(self):
        self.input_size = 128
        self.hidden_size = 256
        self.output_size = 2
        self.batch_size = 64
        self.device = 'cpu'
        self.sample_input = torch.randn(self.batch_size, self.input_size)
        self.sample_target = torch.randint(0, self.output_size, (self.batch_size,))
    
    def test_base_model(self):
        model = BaseModel(self.input_size, self.hidden_size, self.output_size, use_gat=False).to(self.device)
        logits = model(self.sample_input)
        self.assertEqual(logits.shape, (self.batch_size, self.output_size))
    
    def test_fastgrok_model(self):
        model = FastGrokModel(self.input_size, self.hidden_size, self.output_size, use_gat=False).to(self.device)
        logits = model(self.sample_input)
        self.assertEqual(logits.shape, (self.batch_size, self.output_size))
    
    def test_noise_model(self):
        model = NoiseModel(self.input_size, self.hidden_size, self.output_size, use_gat=False, noise_level=0.1).to(self.device)
        model.train()
        logits = model(self.sample_input)
        self.assertEqual(logits.shape, (self.batch_size, self.output_size))
        model.eval()
        logits = model(self.sample_input)
        self.assertEqual(logits.shape, (self.batch_size, self.output_size))

if __name__ == '__main__':
    unittest.main()
