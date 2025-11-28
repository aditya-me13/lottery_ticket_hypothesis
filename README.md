# Lottery Ticket Hypothesis (ES667)

Implementation of "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks"

Simple, reproducible implementation of iterative pruning to find "winning tickets" — sparse subnetworks that train nearly as well as the original dense network.

## Team Members:
- Aditya Mehta (22110017)
- Nikhil Goyal (23110218)
- Shardul Junagade (23110297)

## Setup
- Requirements are listed in `requirements.txt`.
- Python 3.10+ recommended; CUDA is optional but speeds up training.

Install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## How It Works
- Unified runner in [experiments/experiment_runner.py](experiments/experiment_runner.py) orchestrates training, pruning, and reinitialization.
- Masks are enforced during training (see `utils/trainer.py`) and pruning logic lives in `utils/pruning.py`.
- After each round, results are saved as CSV with columns: round, remaining, ES, val, test.

## Quick Start (Script)
Run a magnitude-pruning experiment on MNIST with LeNet-5 (Conv):

```python
from experiments.experiment_runner import ExperimentRunner
import torch

runner = ExperimentRunner(
	dataset="mnist",           # mnist | fashion | cifar10
	model_name="lenet_conv",   # lenet_fc | lenet_conv | conv6
	pruning_type="magnitude",  # magnitude | random
	pruning_scope="layerwise", # layerwise | global
	reinit_method="rewind",    # rewind | random | none
	pruning_rate=0.3,
	num_rounds=16,
	iterations=40000,
	learning_rate=0.0012,
	batch_size=60,
	device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
	results_dir="results/fashion_conv"  # choose your folder
)

results = runner.run()
runner.save_results("fashion_conv_magnitude_rewind.csv")
```


## Datasets, Models, Outputs
- Datasets: MNIST, Fashion-MNIST, CIFAR-10.
- Models: `LeNet-300-100`, `LeNet-5 (Conv)`, `Conv-6`.
- Typical config: 30% pruning per round, 16 rounds, 40k iterations, Adam optimizer, batch size 60.
- CSV per run: columns `round, remaining, ES, val, test`.
- Example: [results/fashion_conv/fashion_conv_magnitude_rewind.csv](results/fashion_conv/fashion_conv_magnitude_rewind.csv).



## Project Structure
- Core runner: [experiments/experiment_runner.py](experiments/experiment_runner.py)
- Utilities: [utils/data_loader.py](utils/data_loader.py), [utils/pruning.py](utils/pruning.py), [utils/trainer.py](utils/trainer.py)
- Models: [models/lenet_fc.py](models/lenet_fc.py), [models/lenet_conv.py](models/lenet_conv.py), [models/conv6.py](models/conv6.py)

