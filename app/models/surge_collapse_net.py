# app/models/surge_collapse_net.py

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from collections import deque

class GATFeatureExtractor(nn.Module):
    """
    GAT-based feature extractor for input features.
    """
    def __init__(self, input_dim=128, gat_hidden_dim=64, gat_out_dim=64, heads=4):
        super(GATFeatureExtractor, self).__init__()
        self.gat_conv = GATConv(input_dim, gat_hidden_dim, heads=heads, concat=True, dropout=0.6)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(gat_hidden_dim * heads, gat_out_dim)
        self.batch_norm = nn.BatchNorm1d(gat_out_dim)
    
    def forward(self, x):
        """
        x: [batch_size, input_dim]
        """
        batch_size, input_dim = x.size()
    
        # Create a fully connected edge_index
        edge_index = self.create_fully_connected_edges(input_dim).to(x.device)
    
        # Transpose x to [input_dim, batch_size]
        x = x.transpose(0, 1)  # [input_dim, batch_size]
    
        # Apply GATConv
        x = self.gat_conv(x, edge_index)  # [input_dim, gat_hidden_dim * heads]
        x = self.relu(x)
    
        # Apply fully connected layer to reduce dimensionality
        x = self.fc(x)  # [input_dim, gat_out_dim]
        x = self.batch_norm(x)
    
        # Aggregate node features (e.g., mean pooling)
        x = x.mean(dim=0)  # [batch_size, gat_out_dim]
    
        return x
    
    def create_fully_connected_edges(self, num_nodes):
        """
        Create a fully connected edge_index tensor for a graph.
        """
        row, col = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes), indexing="ij")
        edge_index = torch.stack([row.flatten(), col.flatten()], dim=0)
        return edge_index

class StableMaxCrossEntropy(nn.Module):
    """
    Cross-Entropy loss with a stable max-based function to prevent numerical issues.
    """
    def __init__(self):
        super(StableMaxCrossEntropy, self).__init__()
    
    def forward(self, logits, targets):
        """
        logits: [batch_size, n_classes]
        targets: [batch_size] with class indices
        """
        # Define a "StableMax" function:
        #   s(x) = x + 1 for x >= 0
        #        = 1 / (1 - x) for x < 0
        stable_logits = torch.where(
            logits >= 0,
            logits + 1,
            1.0 / (1.0 - logits)
        )
        # Sum across classes to get denominator
        sum_logits = torch.sum(stable_logits, dim=1, keepdim=True) + 1e-8
        probs = stable_logits / sum_logits
    
        # Negative log likelihood
        batch_indices = torch.arange(logits.shape[0], device=logits.device)
        correct_probs = probs[batch_indices, targets]
        loss = -torch.log(correct_probs + 1e-8).mean()
    
        return loss

class SurgeCollapseNet(nn.Module):
    """
    SurgeCollapseNet architecture with standard linear layers.
    """
    def __init__(
        self,
        input_size=128,
        hidden_size=256,
        output_size=2,
        use_batch_norm=True,
        use_dropout=False,
        dropout_rate=0.5,
        activation_func='relu',
        debug=False
    ):
        super(SurgeCollapseNet, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.debug = debug

        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = self._get_activation(activation_func)
        
        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_size)
        
        if self.use_dropout:
            self.dropout = nn.Dropout(self.dropout_rate)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        if self.use_batch_norm:
            self.bn2 = nn.BatchNorm1d(hidden_size)
        
        if self.use_dropout:
            self.dropout2 = nn.Dropout(self.dropout_rate)
        
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        # Hooks for activations and gradients
        self.activations = {}
        self.gradients = {}
        self._register_hooks()

    def _get_activation(self, activation_func):
        """
        Return the activation function based on the name.
        """
        if activation_func == 'relu':
            return nn.ReLU()
        elif activation_func == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif activation_func == 'selu':
            return nn.SELU()
        elif activation_func == 'swish':
            return nn.SiLU()  # Swish is known as SiLU in PyTorch
        else:
            raise ValueError(f"Invalid activation function: {activation_func}")

    def _register_hooks(self):
        # Hooks to capture activations and gradients
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook

        def get_gradient(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook

        # Register hooks for activations
        self.fc1.register_forward_hook(get_activation('fc1'))
        self.activation.register_forward_hook(get_activation('activation'))
        self.fc2.register_forward_hook(get_activation('fc2'))
        self.activation.register_forward_hook(get_activation('activation2'))
        self.fc3.register_forward_hook(get_activation('fc3'))

        # Register hooks for gradients
        self.fc1.register_full_backward_hook(get_gradient('fc1'))
        self.fc2.register_full_backward_hook(get_gradient('fc2'))
        self.fc3.register_full_backward_hook(get_gradient('fc3'))

    def forward(self, x):
        x = self.fc1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.activation(x)
        if self.use_dropout:
            x = self.dropout(x)

        x = self.fc2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = self.activation(x)
        if self.use_dropout:
            x = self.dropout2(x)

        logits = self.fc3(x)
        return logits, self.activations
