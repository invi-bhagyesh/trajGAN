import torch
import torch.nn as nn

class TrajectoryNet(nn.Module):
    def __init__(self, latent_dim, path_length, hidden_dim, num_layers):
        super(TrajectoryNet, self).__init__()
        self.latent_dim = latent_dim
        self.path_length = path_length
        self.hidden_dim = hidden_dim
        
        # Initial projection of input latent
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        
        # LSTM for generating sequence
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output projection to latent space
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, z0):
        batch_size = z0.size(0)
        
        # Project input latent to hidden dimension
        x = self.input_proj(z0)
        x = x.unsqueeze(1).repeat(1, self.path_length, 1)
        
        # Generate sequence through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Project back to latent space
        trajectory = self.output_proj(lstm_out)
        
        # Concatenate input latent with generated sequence
        full_trajectory = torch.cat([z0.unsqueeze(1), trajectory], dim=1)
        
        return full_trajectory  # Shape: [batch_size, path_length + 1, latent_dim] 