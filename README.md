# Saint-Gobain SciGlass

A workspace for working with GlassPy, a Python library for glass science and engineering.

## Setup

### Virtual Environment

The project uses a Python virtual environment (`.venv`). To activate it:

```bash
source .venv/bin/activate
```

### Dependencies

All dependencies are installed in the virtual environment. To reinstall or update:

```bash
pip install -r requirements.txt
```

### GlassPy

GlassPy is installed and ready to use. See the [GlassPy documentation](https://github.com/drcassar/glasspy) for usage examples.

## PyTorch

PyTorch is installed for CPU. If you need GPU support, reinstall PyTorch with:

```bash
# For CUDA (GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Replace `cu118` with your CUDA version (e.g., `cu121` for CUDA 12.1).
