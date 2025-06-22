# Multi-GPU GPT Training Implementation

This repository contains an implementation of a GPT-style language model with distributed training capabilities across multiple GPUs. The implementation is based on Sebastian Raschka's work under Apache License 2.0.

## Features

- Multi-GPU training using PyTorch's Distributed Data Parallel (DDP)
- Support for both Windows (gloo) and Linux (nccl) environments
- GPT architecture with transformer blocks and multi-head attention
- Mixed precision training with bfloat16
- Performance monitoring and visualization
- Text generation capabilities

## Model Architecture

- 124M parameter configuration
- 12 transformer layers
- 12 attention heads
- 768 embedding dimension
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

## Usage

### Single GPU Training

```bash
python 02_opt_multi_gpu_ddp.py
```

### Multi-GPU Training

For Linux:
```bash
torchrun --nproc_per_node=NUM_GPUS 02_opt_multi_gpu_ddp.py
```

For Windows:
```bash
torchrun --nproc_per_node=NUM_GPUS 02_opt_multi_gpu_ddp.py
```

Replace `NUM_GPUS` with the number of GPUs you want to use.

## Training Data

The model trains on the "Middlemarch" novel text, which will be automatically downloaded during the first run.

## Model Configuration

The default configuration (124M parameters) can be modified in the script:

```python
GPT_CONFIG_124M = {
    "vocab_size": 50304,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
```

## Training Settings

Default training settings:

```python
OTHER_SETTINGS = {
    "learning_rate": 5e-4,
    "num_epochs": 50,
    "batch_size": 32,
    "weight_decay": 0.1
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