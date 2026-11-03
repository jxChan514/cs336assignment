"""Atomic local checkpoints with model, optimizer, counters and random states."""
from pathlib import Path
import random
import numpy as np
import torch


def save_checkpoint(path, model, optimizer, data_rng, progress, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
             "optimizer": optimizer.state_dict(), "progress": progress, "config": config,
             "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                     "torch": torch.get_rng_state(), "data": data_rng.getstate(),
                     "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def restore_checkpoint(path, model, optimizer, data_rng):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"])
    data_rng.setstate(state["rng"]["data"])
    if state["rng"]["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    return state
