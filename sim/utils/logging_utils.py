import os
import json
import torch
import torch.nn as nn
import time

def calculate_bitsize(model: nn.Module):
    """Calculates number of parameters and estimated size in bytes."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Assuming float32 -> 4 bytes per param
    size_bytes = total_params * 4
    return total_params, size_bytes

def save_metrics(run_name, metrics, directory="data"):
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, f"{run_name}.json")
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
        
class Logger:
    def __init__(self, use_wandb=False, project="tensor-rl-thesis", run_name="test"):
        self.use_wandb = use_wandb
        self.metrics = {"episodes": [], "rewards": [], "lengths": [], "time": []}
        self.start_time = time.time()
        self.run_name = run_name

        if self.use_wandb:
            import wandb
            wandb.init(project=project, name=run_name)
            
    def log(self, episode, reward, length, bitsize=None):
        elapsed = time.time() - self.start_time
        
        self.metrics["episodes"].append(episode)
        self.metrics["rewards"].append(reward)
        self.metrics["lengths"].append(length)
        self.metrics["time"].append(elapsed)
        
        stat = {"episode": episode, "reward": reward, "length": length, "time": elapsed}
        if bitsize:
            stat["params"] = bitsize[0]
            stat["bytes"] = bitsize[1]
            
        if self.use_wandb:
            import wandb
            wandb.log(stat)
            
    def finish(self):
        save_metrics(self.run_name, self.metrics, directory="sim/data")
        if self.use_wandb:
            import wandb
            wandb.finish()
