# Split Learning

A distributed privacy-preserving machine learning framework implemented with PyTorch and MPI.

## Overview

Split learning partitions a neural network across multiple machines: each client node holds the initial layers and processes its own raw data locally, sending only intermediate activations (the "split layer" output) to the server. The server holds the deeper layers, computes the loss, and sends gradients back. Raw data never leaves the client.

This repo implements split learning on FashionMNIST and compares it against standard centralized training.

## Architecture

```
Client 1 ──┐
Client 2 ──┼──→ [split layer tensors] ──→ Server (deeper layers + loss)
Client N ──┘         ←── [gradients] ──────────────────────────────────
```

**Model split (LeNet-style CNN):**
- **Client:** Conv2d(1→6) → ReLU → MaxPool2d
- **Server:** Conv2d(6→16) → ReLU → MaxPool2d → FC(256→120) → FC(120→84) → FC(84→10)

## Communication Protocol (MPI)

Nodes communicate via tagged MPI messages:

| Tag | Direction | Purpose |
|-----|-----------|---------|
| `tensor_and_labels` | Client → Server | Split layer activations + labels |
| `gradients` | Server → Client | Backpropagated gradients |
| `worker_done` / `epoch_done` | Client ↔ Server | Epoch synchronization |
| `Comienzo` | Client → Client | Barrier sync (ring coordination) |
| `training_complete` | Server → Client | Signal end of training |

Clients form a coordination ring — each waits for a signal from its predecessor before starting the next iteration, ensuring all nodes progress through epochs in lockstep.

## Setup

**Requirements:** Python 3, PyTorch, torchvision, MPI (`mpirun`)

```bash
pip install -r requirements.txt
```

Configure your nodes in `hostfile.txt` (one IP per line):
```
192.168.1.1
192.168.1.2
192.168.1.3
```

## Running

**Split learning (distributed):**
```bash
mpirun -np 3 -hostfile ~/Split_Learning/hostfile.txt python ~/Split_Learning/split_learning.py
```

**Regular (centralized) baseline:**
```bash
python regular_learning.py --epochs 10 --batch_size 64 --learning_rate 0.01
```

## Arguments (`regular_learning.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--batch_size` | 64 | Training batch size |
| `--test_batch_size` | 1000 | Test batch size |
| `--epochs` | 10 | Number of epochs |
| `--learning_rate` | 0.01 | SGD learning rate |
| `--log_steps` | 50 | Logging frequency (batches) |

## Output

Plots saved to `~/Split_Learning/plots/`:
- `test_loss_reg.pdf` — test loss curve
- `test_acc_reg.pdf` — test accuracy curve

## References

- [Split Learning for Health](https://arxiv.org/abs/1812.00564) — Vepakomma et al., MIT
- [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist)
