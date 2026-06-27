import torch

print(f"PyTorch versione: {torch.__version__}")
print(f"CUDA disponibile: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU rilevata: {torch.cuda.get_device_name(0)}")
    print(f"Memoria GPU totale: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("ATTENZIONE: CUDA non disponibile, il training userebbe la CPU (molto lento)")