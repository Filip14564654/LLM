import torch

print("CUDA verze:", torch.version.cuda)
print("CUDA dostupná:", torch.cuda.is_available())
print("Zařízení:", torch.cuda.get_device_name(0))