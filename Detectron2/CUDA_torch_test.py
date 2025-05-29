import torch
print(torch.cuda.is_available())  # True ならGPUが使える
print(torch.cuda.get_device_name(0))  # GPU名が出る
