import GPUtil
import torch

def get_available_device():
    allowed_gpus = GPUtil.getAvailable()
    device = torch.device(f"cuda:{allowed_gpus[0]}"
                          if (torch.cuda.is_available() and len(allowed_gpus) > 0)
                          else "cpu")

    return device
