import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

import dgl
print("DGL version:", dgl.__version__)

from D4CMPP import train
print("D4CMPP imported OK")