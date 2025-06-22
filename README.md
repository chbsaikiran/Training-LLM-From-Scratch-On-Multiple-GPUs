# Multi-GPU GPT Training Implementation

This repository contains an implementation of a GPT-style language model with distributed training capabilities across multiple GPUs. The implementation is based on Sebastian Raschka's work under Apache License 2.0.

## Features

- Multi-GPU training using PyTorch's Distributed Data Parallel (DDP)
- Support for both Windows (gloo) and Linux (nccl) environments
- GPT architecture with transformer blocks and multi-head attention
- Mixed precision training with bfloat16
- Performance monitoring and visualization
- Text generation capabilities
- ~1.42 billion parameters

## Model Architecture

- 1.42B parameter configuration
- 24 transformer layers
- 16 attention heads
- 2048 embedding dimension
- 1024 token context length
- Sliding window approach for text processing

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU(s)
- See `requirements.txt` for complete dependencies

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Code

### Prerequisites
- Ensure you have CUDA-capable GPUs available
- Install all dependencies from requirements.txt
- Make sure you have enough GPU memory (recommended: at least 24GB per GPU)

### Training Options

1. **Single GPU Training**:
```bash
python 02_opt_multi_gpu_ddp.py
```

2. **Multi-GPU Training on Local Machine**:
```bash
torchrun --nproc_per_node=NUM_GPUS 02_opt_multi_gpu_ddp.py
```

3. **Multi-GPU Training on AWS**:
```bash
# First, activate your virtual environment if using one
source venv/bin/activate

# Then run the training
torchrun --nproc_per_node=4 02_opt_multi_gpu_ddp.py
```

Replace `NUM_GPUS` with the number of GPUs you want to use (e.g., 2, 4, 8, etc.).

### Important Notes for Training

1. **Memory Management**:
   - Default batch size is 8 per GPU
   - Uses gradient accumulation (4 steps)
   - Effective batch size = 8 * 4 = 32 per GPU
   - If you run into OOM errors, reduce batch size or increase gradient accumulation steps

2. **Training Duration**:
   - Runs for 50 epochs by default
   - No early stopping implemented
   - Training progress is logged in real-time

3. **Monitoring**:
   - Training and validation losses are printed every few steps
   - GPU memory usage is displayed
   - Sample text generations are shown periodically

## Training Results and Visualization

### Training Logs
The complete training logs for 50 epochs on AWS can be found here:
[AWS Training Logs](AWS_Training_Logs.log) - Contains detailed training progress, including loss values, tokens per second, and GPU utilization metrics for the entire training run.

### Multi-GPU Usage
Visual confirmation of multiple GPU utilization on AWS:
[GPU Utilization Screenshot](Screenshot_Of_AWS_Using_Multiple_GPUs.png) - Shows all 4 GPUs being utilized effectively during training, with memory usage and processing load distribution.

### Loss Visualization
Training and validation loss curves:
[Loss Plot](loss.pdf) - Visualizes the training and validation loss progression over 50 epochs, showing model convergence and learning patterns.

## Model Configuration

The default configuration (1.42B parameters) can be modified in the script:

```python
GPT_CONFIG_124M = {
    "vocab_size": 50304,
    "context_length": 1024,
    "emb_dim": 2048,
    "n_heads": 16,
    "n_layers": 24,
    "drop_rate": 0.1,
    "qkv_bias": False
}
```

## Training Settings

Default training settings:

```python
OTHER_SETTINGS = {
    "learning_rate": 5e-4 * world_size,
    "num_epochs": 50,
    "batch_size": 8,
    "weight_decay": 0.1,
    "gradient_accumulation_steps": 4
}
```

## Output

- Training and validation loss plots are saved as `loss.pdf`
- Training progress and metrics are displayed in real-time
- Memory usage statistics for CUDA devices
- Sample text generations during training

## License

This project is licensed under the Apache License 2.0 - see the LICENSE.txt file for details.

## Acknowledgments

Based on the work by Sebastian Raschka from "Build a Large Language Model From Scratch". 