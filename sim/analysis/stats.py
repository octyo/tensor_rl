import os
import json
import matplotlib.pyplot as plt
import numpy as np

def load_data(directory="sim/data"):
    # Load all matching json files
    data = []
    for file in os.listdir(directory):
        if file.endswith('.json'):
            path = os.path.join(directory, file)
            with open(path, 'r') as f:
                content = json.load(f)
                data.append({"name": file.replace(".json", ""), "metrics": content})
    return data

def quick_plot(data):
    """Generates basic comparison graphs of steady-state rewards and training speeds."""
    plt.figure()
    for d in data:
        metrics = d['metrics']
        plt.plot(metrics['episodes'], metrics['rewards'], label=d['name'], alpha=0.8)
    
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('RL Training Progress Comparison')
    plt.legend()
    plt.grid()
    plt.show() # Can also be saved to file using plt.savefig()

if __name__ == "__main__":
    # Placeholder execution
    data = load_data()
    # quick_plot(data)
