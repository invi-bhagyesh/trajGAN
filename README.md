# Latent Trajectory Optimization for Semantic Image Editing

This project implements a GAN-based approach for generating smooth and semantically meaningful trajectories in latent space for image editing. The implementation consists of two main components:

1. A GAN trained from scratch to map latent vectors to images
2. A Trajectory Network that generates smooth sequences of latent vectors

## Project Structure

```
latent_trajectory_edit/
├── train_gan.py                 # GAN training script
├── train_trajectory.py          # Latent path generator training script
├── models/
│   ├── gan.py                   # Generator & Discriminator
│   ├── trajectory_net.py        # Model that learns latent sequences
├── losses.py                    # Adversarial + smoothness losses
├── dataset.py                   # Image dataset loader
├── utils/
│   ├── visualize.py             # Show latent trajectories as image sequences
│   ├── latent_utils.py          # Latent sampling, interpolation utils
└── config.yaml                  # Configuration file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/latent_trajectory_edit.git
cd latent_trajectory_edit
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Weights & Biases:
```bash
wandb login
```

## Usage

### 1. Training the GAN

First, train the GAN from scratch:

```bash
python train_gan.py
```

This will:
- Train the Generator and Discriminator
- Save model checkpoints
- Log training progress to Weights & Biases
- Generate sample images

### 2. Training the Trajectory Network

After the GAN is trained, train the trajectory network:

```bash
python train_trajectory.py
```

This will:
- Load the pretrained GAN
- Train the trajectory network
- Generate and visualize latent trajectories
- Log training progress to Weights & Biases

## Configuration

Edit `config.yaml` to modify:
- Model architectures and hyperparameters
- Training parameters
- Dataset settings
- Logging configuration

## Key Features

- **GAN Training**:
  - Generator maps latent vectors to images
  - Discriminator classifies real vs. generated images
  - Standard adversarial training

- **Trajectory Network**:
  - Generates smooth sequences of latent vectors
  - Uses LSTM for sequence generation
  - Optimized with smoothness and adversarial losses

- **Visualization**:
  - Real-time training progress
  - Generated image sequences
  - Latent space trajectories

## Results

The trained models can generate:
- High-quality images from random latent vectors
- Smooth transitions between different semantic attributes
- Consistent and meaningful image edits

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{latent_trajectory_edit,
  author = {Your Name},
  title = {Latent Trajectory Optimization for Semantic Image Editing},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/latent_trajectory_edit}}
}
``` 