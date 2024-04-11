import unittest
import torch
import torch.nn as nn
from your_model_file import DiscreteTimeNN


class TestDiscreteTimeNN(unittest.TestCase):
    def setUp(self):
        # Setup code for the tests; instantiate the model here if needed
        self.hidden_layer_sizes = [10, 20, 30]  # Example sizes
        self.num_bins = 5
        self.model = DiscreteTimeNN(
            hidden_layer_sizes=self.hidden_layer_sizes, num_bins=self.num_bins
        )

    def test_initialization(self):
        # Test whether the model initializes correctly
        self.assertIsInstance(self.model, DiscreteTimeNN)
        self.assertEqual(len(self.model.layers), len(self.hidden_layer_sizes))
        for layer, expected_size in zip(
            self.model.layers, self.hidden_layer_sizes
        ):
            self.assertTrue(
                isinstance(layer, nn.LazyLinear)
            )  # Assuming LazyLinear was meant to be added to layers
        self.assertIsInstance(self.model.prediction_head, nn.LazyLinear)
        self.assertEqual(
            self.model.prediction_head.out_features, self.num_bins + 1
        )

    def test_forward_pass(self):
        # Test the forward pass with a sample input
        sample_input = torch.randn(
            1, self.hidden_layer_sizes[0]
        )  # Adjust the input shape as necessary
        output = self.model(sample_input)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (1, self.num_bins + 1))

    def test_activation_softmax(self):
        # Test if activation and softmax are correctly applied
        self.assertIsInstance(
            self.model.activation, nn.ReLU
        )  # Or check for the specified activation function
        # Checking softmax is a bit trickier directly; usually, we ensure the output sums to 1 for a probability distribution
