# Installation Guide

This guide walks you through installing the Hyper-Connections implementation using `uv`, a fast Python package installer.

## Prerequisites

- Python 3.9 or higher
- Git (optional, for cloning)

## Installation Steps

### Step 1: Install uv

`uv` is a fast Python package installer and resolver. Install it using one of these methods:

#### On macOS and Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### On Windows:
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Alternative (using pip):
```bash
pip install uv
```

### Step 2: Navigate to the Project Directory

```bash
cd cursor_manus_hc
```

### Step 3: Create a Virtual Environment

```bash
uv venv
```

This creates a `.venv` directory in your project.

### Step 4: Activate the Virtual Environment

#### On macOS/Linux:
```bash
source .venv/bin/activate
```

#### On Windows (Command Prompt):
```cmd
.venv\Scripts\activate.bat
```

#### On Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```

### Step 5: Install PyTorch

Install PyTorch based on your system:

#### CPU-only (faster install):
```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

#### With CUDA 11.8 (for NVIDIA GPUs):
```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu118
```

#### With CUDA 12.1 (for newer NVIDIA GPUs):
```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Step 6: Install the Project

```bash
uv pip install -e .
```

### Step 7: (Optional) Install Development Dependencies

```bash
uv pip install -e ".[dev]"
```

## Verify Installation

Run the test suite to verify everything is working:

```bash
python test_implementation.py
```

You should see output like:
```
======================================================================
Running Hyper-Connections Implementation Tests
======================================================================
Testing HyperConnection block...
✓ HyperConnection block tests passed!
...
🎉 All tests passed successfully!
```

Run the examples:

```bash
python example_usage.py
```

## Quick Start (One-liner)

If you already have `uv` installed:

```bash
# CPU version
uv venv && source .venv/bin/activate && uv pip install torch --index-url https://download.pytorch.org/whl/cpu && uv pip install -e .

# CUDA 11.8 version
uv venv && source .venv/bin/activate && uv pip install torch --index-url https://download.pytorch.org/whl/cu118 && uv pip install -e .
```

On Windows (PowerShell), replace `source .venv/bin/activate` with `.venv\Scripts\Activate.ps1`

## Troubleshooting

### "uv: command not found"

Make sure `uv` is in your PATH. After installing, you may need to restart your terminal or run:

```bash
source ~/.bashrc  # or ~/.zshrc on macOS
```

### "No module named 'torch'"

Make sure you've:
1. Activated the virtual environment
2. Installed PyTorch using the commands in Step 5

### CUDA out of memory errors

If you encounter CUDA memory errors during testing or training:
- Reduce batch size
- Reduce sequence length
- Use gradient checkpointing (not included in basic implementation)
- Use CPU version for testing

### Import errors

Make sure you're in the project directory and have activated the virtual environment:

```bash
cd cursor_manus_hc
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

## Alternative: Using Standard pip

If you prefer using standard pip:

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # .venv\Scripts\activate on Windows

# Install PyTorch
pip install torch

# Install project
pip install -e .
```

## Next Steps

Once installed:

1. **Run tests**: `python test_implementation.py`
2. **Try examples**: `python example_usage.py`
3. **Read the README**: See `README.md` for usage details
4. **Check the paper**: See `HYPER-CONNECTIONS.pdf` for theory

## System Requirements

### Minimum:
- CPU: Any modern CPU
- RAM: 4GB
- Disk: 500MB

### Recommended for Training:
- CPU: Multi-core processor
- RAM: 16GB+
- GPU: NVIDIA GPU with 8GB+ VRAM
- Disk: 5GB+ for datasets

## Getting Help

If you encounter issues:

1. Check this installation guide
2. Review the README.md
3. Check PyTorch installation: `python -c "import torch; print(torch.__version__)"`
4. Verify environment: `which python` (should point to `.venv`)
