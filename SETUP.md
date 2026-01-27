# Isaac-GR00T on RTX 5090 (SM 12.0) — Minimal, Correct venv Setup

This guide is designed to avoid the exact failure modes you hit:
- **Torch / torchvision / torchaudio mismatch** (e.g., stable torch with nightly torchvision)
- **flash-attn ABI mismatch** after changing torch
- **nvcc too old** for `compute_120`
- **wrong arch format** (`120` vs `12.0`)

It assumes:
- You have a recent NVIDIA driver (your `nvidia-smi` showed driver CUDA 12.9 support).
- You have the **CUDA 12.9 toolkit installed under `/usr/local/cuda-12.9`**.

---

## 0) One-time system prerequisites

Install build essentials:

```bash
sudo apt-get update
sudo apt-get install -y python3.10-dev build-essential ninja-build git
```

**If you previously installed the Ubuntu meta package:**

```bash
sudo apt-get purge -y nvidia-cuda-toolkit
sudo apt-get autoremove -y
```

Confirm CUDA 12.9 toolkit exists:

```bash
ls -d /usr/local/cuda-12.9
```

---

## 1) Create a fresh venv

```bash
cd ~/heizmany/Isaac-GR00T

rm -rf .venv_gr00t
python3.10 -m venv .venv_gr00t
source .venv_gr00t/bin/activate
```

Upgrade packaging tools:

```bash
pip install -U pip setuptools wheel
```

---

## 2) Install Isaac-GR00T base deps

```bash
pip install -e .[base]
pip install python-dotenv jupyter
```

---

## 3) Install a matching PyTorch nightly set (Blackwell-safe)

Install **torch + torchvision + torchaudio together** from the same nightly channel:

```bash
pip uninstall -y torch torchvision torchaudio
pip install --no-cache-dir --pre torch torchvision torchaudio   --index-url https://download.pytorch.org/whl/nightly/cu128
```

Sanity check (versions should share the same nightly date):

```bash
python -c "import torch, torchvision, torchaudio; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('torchaudio', torchaudio.__version__); print('runtime', torch.version.cuda); print('gpu', torch.cuda.get_device_name(0)); print('cap', torch.cuda.get_device_capability(0))"
```

You want:
- All three versions with the **same nightly date** and `+cu128`
- `cap (12, 0)` on the RTX 5090

---

## 4) Build flash-attn 2.8.2 correctly (SM 12.0)

**Critical:** Use **CUDA 12.9 nvcc** for `compute_120`.

```bash
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

which nvcc
nvcc --version
```

Clean any old flash-attn artifacts:

```bash
pip uninstall -y flash-attn
pip cache purge
```

Build with correct arch flags:

```bash
export PIP_NO_BUILD_ISOLATION=1
export TORCH_CUDA_ARCH_LIST="12.0"
export FLASH_ATTN_CUDA_ARCHS="120"
export MAX_JOBS=2

pip install --no-cache-dir --no-build-isolation --no-binary=flash-attn   flash-attn==2.8.2
```

---

## 5) Validate

Torchvision NMS:

```bash
python - <<'PY'
import torch
from torchvision.ops import nms
boxes = torch.tensor([[0,0,10,10],[0,0,9,9]], device="cuda", dtype=torch.float32)
scores = torch.tensor([0.9, 0.8], device="cuda")
print("nms ok", nms(boxes, scores, 0.5))
PY
```

Flash-Attn:

```bash
python - <<'PY'
import torch
from flash_attn import flash_attn_func
q = torch.randn(1,128,8,64, device="cuda", dtype=torch.float16)
k = torch.randn(1,128,8,64, device="cuda", dtype=torch.float16)
v = torch.randn(1,128,8,64, device="cuda", dtype=torch.float16)
out = flash_attn_func(q,k,v, dropout_p=0.0, causal=False)
print("flash-attn ok", out.shape)
PY
```

---

## 6) Important rule (prevents the undefined-symbol loop)

If you **ever reinstall or change**:
- `torch`
- `torchvision`
- `torchaudio`

then you **must rebuild flash-attn** afterward.

Rebuild command:

```bash
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export PIP_NO_BUILD_ISOLATION=1
export TORCH_CUDA_ARCH_LIST="12.0"
export FLASH_ATTN_CUDA_ARCHS="120"
export MAX_JOBS=2

pip install --no-cache-dir --no-build-isolation --no-binary=flash-attn   --force-reinstall flash-attn==2.8.2
```




lastly do this to not enforce the cuda version needed to install flash attn on runtime

unset CUDA_HOME
unset LD_LIBRARY_PATH
hash -r

---

## 7) Run finetune

```bash
python scripts/gr00t_finetune.py 2>&1 | tee finetuning.log
```

---

## Summary of the “must be correct” points

- **Nightly cu128** is needed for RTX 5090 support.
- **Torch / torchvision / torchaudio must be installed as a matching set** from the same index/channel.
- **`TORCH_CUDA_ARCH_LIST="12.0"`** (correct format).
- **`FLASH_ATTN_CUDA_ARCHS="120"`** (FlashAttention’s flag for sm_120).
- **Build flash-attn with CUDA 12.9 nvcc**, not 11.x/12.4.




rm -rf ~/snap/code/215/.local/share/Trash/files


rsync -av --ignore-existing --progress ./Isaac-GR00T/models/ /media/innovation-hacking/flatboi-archive/heizmany/ur5_chess/models/

additionally

pip install python-dotenv
