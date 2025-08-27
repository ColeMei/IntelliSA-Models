#!/usr/bin/env python3
import os, sys, platform, importlib, subprocess

def try_import(name):
    try:
        m = importlib.import_module(name)
        return "OK", getattr(m, "__version__", "unknown")
    except Exception as e:
        return "MISSING", str(e)

def run(cmd):
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
        return True, out.strip()
    except subprocess.CalledProcessError as e:
        return False, e.output.strip()

print("=== Python & System ===")
print(f"Python: {sys.version.split()[0]} ({sys.executable})")
print(f"Hostname: {platform.node()}")
print()

print("=== Env Vars ===")
for k in ["PROJECT_ID","PROJECT_ROOT","TEMP_ROOT","HF_HOME","HF_DATASETS_CACHE","TRANSFORMERS_CACHE","CUDA_VISIBLE_DEVICES","SLURM_JOB_ID","SLURM_NODELIST","SLURM_STEP_GPUS"]:
    print(f"{k}={os.environ.get(k,'')}")
print()

print("=== Packages ===")
for name in ["torch","torchvision","torchaudio","transformers","datasets","accelerate","peft","pandas","numpy","scikit_learn","jsonlines"]:
    status, info = try_import(name)
    print(f"{name:15s}: {status} ({info})")
print()

print("=== GPU ===")
ok, out = run("nvidia-smi")
print(out if ok else "nvidia-smi not available")
try:
    import torch
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"device_count: {torch.cuda.device_count()}")
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        print(f"GPU0: {torch.cuda.get_device_name(0)}  CC={torch.cuda.get_device_capability(0)}")
except Exception as e:
    print(f"Torch CUDA check error: {e}")
print()

print("=== Disk ===")
print(run("df -h $PROJECT_ROOT || true")[1])
print(run("du -sh $PROJECT_ROOT || true")[1])
print()
print("If all OK and GPU visible, you’re ready to train.")