import io, requests
import torch
import torch.nn as nn

from src.losses.dall_e.encoder import Encoder
from src.losses.dall_e.decoder import Decoder
from src.losses.dall_e.dall_e_utils   import map_pixels, unmap_pixels

def load_model(path: str) -> nn.Module:
    if path.startswith('http://') or path.startswith('https://'):
        resp = requests.get(path)
        resp.raise_for_status()
            
        with io.BytesIO(resp.content) as buf:
            return torch.load(buf)
    else:
        with open(path, 'rb') as f:
            return torch.load(f)