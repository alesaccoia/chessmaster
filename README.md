A PyTorch implementation of deep reinforcement learning for chess, inspired by AlphaZero's self-play approach. The agent learns chess entirely through playing against itself and improves through experience.

## Key Features
- Deep neural network combining policy and value networks
- Self-play reinforcement learning
- Automatic game recording with video generation
- CUDA support for GPU acceleration

## Requirements
```bash
pip install torch python-chess cairosvg numpy
apt-get install ffmpeg  # for video generation
```

## Usage
```bash
python chess_rl.py
```

The agent will start training through self-play, saving periodic videos of its games to track progress.