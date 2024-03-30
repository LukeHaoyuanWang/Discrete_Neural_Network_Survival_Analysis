"""A library that facilitate Discrete Neural Network Survival Analaysis.

The functions
"""

from typing import List, Type

import torch
import torch.nn as nn


class DiscreteTimeNN(nn.Module):
    """A discrete time neural network model implementing encoder layers.

    The encoder layer has optional batch normalization and an optional
    activation function.

    Parameters:

        hidden_layer_sizes(list):
            indcates number of linear layers and neuronsin the model.

        num_bins(int):
            indicates number of bins that time is partitioned into.

        batch_norm(boolean):
            indicates whether batch normalization is performed.The default
            value is False, which means the batch normalizationis not
            perfromed.

        activation(nn.Module):
            Activation function. Without specification, the
            model will use ReLu.

    Attribute:

        layers (nn.ModuleList):
            A list of layers comprising the encoder layers
            with batch normalization and activation, followed by the prediction
            head.

        prediction_head (nn.LazyLinear):
            A linear layer with lazy initialization that serves as the
            prediction head of the model.
    """

    def __init__(
        self,
        hidden_layer_sizes: List[int],
        num_bins: int,
        batch_norm: bool = False,
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        """Initializing the parameters for the model."""
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList()
        for size in hidden_layer_sizes:
            self.layers.append(nn.LazyLinear(size))
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(size))
            self.layers.append(activation())

        self.prediction_head: nn.LazyLinear = nn.LazyLinear(num_bins + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass.

        A forward pass with customized hidden layer size and activation
        function.
        """
        for layer in self.layers:
            x = layer(x)

        x = self.prediction_head(x)
        return x


class DiscreteFailureTimeNLL(torch.nn.Module):
    """A PyTorch module for calculating the Negative Log-Likelihood Loss.

    Attributes:
        bin_starts (torch.Tensor): The start points of each bin.
        bin_ends (torch.Tensor): The end points of each bin.
        bin_lengths (torch.Tensor): The lengths of the bins.
        tolerance (float): A small value added for numerical stability
    """

    def __init__(self, bin_boundaries: torch.Tensor, tolerance: float = 1e-8):
        """Initializes parameters for DiscreteFailureTimeNLL."""
        super().__init__()
        if not isinstance(bin_boundaries, torch.Tensor):
            bin_boundaries = torch.tensor(bin_boundaries, dtype=torch.float32)

        self.bin_starts = bin_boundaries[:-1]
        self.bin_ends = bin_boundaries[1:]
        self.bin_lengths = self.bin_ends - self.bin_starts
        self.tolerance = tolerance

    def _discretize_times(self, times: torch.Tensor) -> torch.Tensor:
        """Discretizes the given event times based on the bin boundaries.

        Parameters:
            times (torch.Tensor): The event times to discretize.

        Returns:
            torch.Tensor: A binary tensor indicating whether each time falls
            into each bin.
        """
        return (times[:, None] > self.bin_starts) & (
            times[:, None] <= self.bin_ends
        )

    def _get_proportion_of_bins_completed(
        self, times: torch.Tensor
    ) -> torch.Tensor:
        """Calculates the proportion of each bin that is completed.

        Parameters:
            times (torch.Tensor): The event times to evaluate.

        Returns:
            torch.Tensor: The proportion of each bin that is completed by each
            time.
        """
        proportions = (times[:, None] - self.bin_starts) / self.bin_lengths
        return torch.clamp(proportions, min=0, max=1)

    def forward(
        self,
        predictions: torch.Tensor,
        event_indicators: torch.Tensor,
        event_times: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the negative log-likelihood loss.

        Parameters:
            predictions (torch.Tensor):
                The predicted probabilities for each bin.

            event_indicators (torch.Tensor):
                Binary indicators of whether an event occurred or was
                censored.(1 stands for occurrance and 0 for censored)

            event_times (torch.Tensor):
                The times at which events occurred or were censored.

        Returns:
            torch.Tensor: The mean negative log-likelihood of the given data.
        """
        event_likelihood = (
            torch.sum(
                self._discretize_times(event_times) * predictions[:, :-1],
                dim=1,
            )
            + self.tolerance
        )
        nonevent_likelihood = (
            1
            - torch.sum(
                self._get_proportion_of_bins_completed(event_times)
                * predictions[:, :-1],
                dim=1,
            )
            + self.tolerance
        )

        log_likelihood = event_indicators * torch.log(event_likelihood) + (
            1 - event_indicators
        ) * torch.log(nonevent_likelihood)
        return -torch.mean(log_likelihood)
