import os
import random
import numpy as np
import torch

def seed_everything(seed: int = 42) -> None:
    """
    Fixe toutes les graines aléatoires pour garantir la reproductibilité absolue des expériences.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 Graine aléatoire fixée : SEED = {seed}")

if __name__ == "__main__":
    seed_everything(42)
