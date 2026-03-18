# 🚀 End-to-End Guide: Training Transformer on Google Colab (T4 GPU)

This guide walks you through running your English→Hindi Transformer on Colab with CUDA acceleration, saving weights at each epoch.

---

## Step 1: Set Up T4 GPU Runtime

1. Open [Google Colab](https://colab.research.google.com/)
2. Go to **Runtime → Change runtime type**
3. Set **Hardware accelerator** to **T4 GPU**
4. Click **Save**

> [!IMPORTANT]
> Without this step, your code will fall back to CPU and be extremely slow.

---

## Step 2: Create a New Notebook & Upload Your Files

Create a new notebook, then upload all four files using a code cell:

```python
# Cell 1: Upload your transformer files
from google.colab import files

uploaded = files.upload()  # Upload: config.py, dataset.py, model.py, train.py
```

**Select all 4 files** when prompted:
- `config.py`
- `dataset.py`
- `model.py`
- `train.py`

---

## Step 3: Install Dependencies

```python
# Cell 2: Install required packages
!pip install tiktoken datasets torch tensorboard tqdm
```

All these come pre-installed on Colab except `tiktoken` and `datasets`, but running this ensures the correct versions.

---

## Step 4: Verify CUDA / T4 GPU is Available

```python
# Cell 3: Verify GPU availability
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU device: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

**Expected output:**
```
PyTorch version: 2.x.x+cu121
CUDA available: True
CUDA version: 12.1
GPU device: Tesla T4
GPU memory: 15.8 GB
```

---

## Step 5: Fix the Deprecation Warning in `model.py`

Your `model.py` uses `nn.init.xavier_uniform` (deprecated). Fix it in a cell:

```python
# Cell 4: Patch model.py to fix deprecation warning
import fileinput, sys

with open('model.py', 'r') as f:
    content = f.read()

content = content.replace('nn.init.xavier_uniform(p)', 'nn.init.xavier_uniform_(p)')

with open('model.py', 'w') as f:
    f.write(content)

print("✅ Fixed xavier_uniform deprecation in model.py")
```

---

## Step 6: Run Training with Epoch-Level Logging

```python
# Cell 5: Train the model
import warnings
warnings.filterwarnings('ignore')

from config import get_config
from train import build_training

config = get_config()

# Print training configuration
print("=" * 60)
print("📋 TRAINING CONFIGURATION")
print("=" * 60)
print(f"  Dataset:        {config['datasource']}")
print(f"  Batch size:     {config['batch_size']}")
print(f"  Learning rate:  {config['learning_rate']}")
print(f"  Context length: {config['context_length']}")
print(f"  Feature dims:   {config['feature_dims']}")
print(f"  Epochs:         {config['num_epochs']}")
print(f"  Source lang:    {config['src_language']}")
print(f"  Target lang:    {config['tgt_language']}")
print(f"  Weights folder: {config['model_folder']}")
print("=" * 60)

# Start training
build_training(config)
```

This will:
- Auto-detect CUDA (your `train.py` already handles this)
- Download the English-Hindi dataset
- Filter sentences by context length
- Train for **20 epochs**, showing progress bars with live loss
- **Save weights after every epoch** to `weights/tmodel_XX.pt`

---

## Step 7: Monitor Training with TensorBoard

```python
# Cell 6: Launch TensorBoard (run in parallel with training or after)
%load_ext tensorboard
%tensorboard --logdir runs/tmodel
```

This shows interactive loss curves directly inside Colab.

---

## Step 8: Check Saved Weights

```python
# Cell 7: List all saved weight files
import os

weights_dir = 'weights'
if os.path.exists(weights_dir):
    files = sorted(os.listdir(weights_dir))
    print(f"\n📁 Saved weights ({len(files)} files):")
    for f in files:
        size_mb = os.path.getsize(os.path.join(weights_dir, f)) / (1024 * 1024)
        print(f"  ✅ {f}  ({size_mb:.1f} MB)")
else:
    print("❌ No weights directory found yet.")
```

---

## Step 9: Inspect a Specific Epoch's Weights

```python
# Cell 8: Load and inspect a specific epoch checkpoint
import torch

epoch_to_inspect = 5  # Change this to any epoch number (0-19)
checkpoint = torch.load(f'weights/tmodel_{epoch_to_inspect:02d}.pt')

print(f"📦 Checkpoint for Epoch {epoch_to_inspect}:")
print(f"  Epoch:       {checkpoint['epoch']}")
print(f"  Global Step: {checkpoint['global_step']}")
print(f"  Model keys:  {len(checkpoint['model_state_dict'])} layers")

# Print layer names and shapes
print(f"\n🧠 Model Layers:")
for name, param in checkpoint['model_state_dict'].items():
    print(f"  {name}: {list(param.shape)}")
```

---

## Step 10: Resume Training from a Checkpoint

If Colab disconnects or you want to continue training:

```python
# Cell 9: Resume training from a specific epoch
from config import get_config
from train import build_training

config = get_config()
config['preload'] = '05'       # Resume from epoch 5
config['num_epochs'] = 30      # Train up to epoch 30 now

build_training(config)
```

---

## Step 11: Download Weights to Local Machine

```python
# Cell 10: Download a specific weight file
from google.colab import files

# Download the final epoch weights
files.download('weights/tmodel_19.pt')

# Or download all weights as a zip
!zip -r weights.zip weights/
files.download('weights.zip')
```

---

## ⚡ Quick Reference — All Cells in Order

| Cell | Purpose | Time Estimate |
|------|---------|---------------|
| 1 | Upload files | 30 seconds |
| 2 | Install dependencies | 1-2 minutes |
| 3 | Verify GPU | Instant |
| 4 | Fix deprecation warning | Instant |
| 5 | **Train model** (20 epochs) | **30-90 min** |
| 6 | TensorBoard visualization | Instant |
| 7 | Check saved weights | Instant |
| 8 | Inspect checkpoint | Instant |
| 9 | Resume training (optional) | Varies |
| 10 | Download weights | Varies |

---

## 🛠 Troubleshooting

| Issue | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce `batch_size` to `4` or `2` in `config.py` |
| `ModuleNotFoundError` | Re-run `!pip install tiktoken datasets` |
| Colab disconnects | Use **Step 10** to resume from last saved epoch |
| Training is slow | Verify GPU is enabled (Step 4); reduce `context_length` to `128` |
| `RuntimeError` | Ensure all tensors are `.to(device)` |
