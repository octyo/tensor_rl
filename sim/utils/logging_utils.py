import os
import json
import torch
import torch.nn as nn
import time


def calculate_bitsize(model: nn.Module):
    """Calculates number of parameters and estimated size in bytes."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_bytes = total_params * 4  # float32
    return total_params, size_bytes


def save_metrics(run_name, metrics, directory="data"):
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, f"{run_name}.json")
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)


class Logger:
    def __init__(self, use_wandb=False, project="tensor-rl-thesis", run_name="test"):
        self.use_wandb = use_wandb
        self.metrics = {
            "episodes": [], "rewards": [], "lengths": [], "time": [],
            "loss": [], "mean_q": [], "grad_norm": [], "eval_reward": [],
        }
        self.start_time = time.time()
        self.run_name = run_name

        if self.use_wandb:
            import wandb
            wandb.init(project=project, name=run_name)

    def log(self, episode, reward, length, bitsize=None,
            loss=None, mean_q=None, grad_norm=None, eval_reward=None):
        elapsed = time.time() - self.start_time

        self.metrics["episodes"].append(episode)
        self.metrics["rewards"].append(reward)
        self.metrics["lengths"].append(length)
        self.metrics["time"].append(elapsed)
        self.metrics["loss"].append(loss if loss is not None else 0.0)
        self.metrics["mean_q"].append(mean_q if mean_q is not None else 0.0)
        self.metrics["grad_norm"].append(grad_norm if grad_norm is not None else 0.0)
        self.metrics["eval_reward"].append(eval_reward if eval_reward is not None else 0.0)

        stat = {"episode": episode, "reward": reward, "length": length, "time": elapsed,
                "loss": loss, "mean_q": mean_q, "grad_norm": grad_norm,
                "eval_reward": eval_reward}
        if bitsize:
            stat["params"] = bitsize[0]
            stat["bytes"] = bitsize[1]
            # store once in metrics too
            if "params" not in self.metrics:
                self.metrics["params"] = bitsize[0]
                self.metrics["bytes"] = bitsize[1]

        if self.use_wandb:
            import wandb
            wandb.log({k: v for k, v in stat.items() if v is not None})

    def finish(self):
        save_metrics(self.run_name, self.metrics, directory="sim/data")
        if self.use_wandb:
            import wandb
            wandb.finish()
