import os
import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_event_file(run_dir):
    matches = glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True)
    if matches:
        return max(matches, key=os.path.getsize)
    return None

def extract_scalar(event_file, tag):
    ea = EventAccumulator(event_file)
    ea.Reload()
    if tag in ea.Tags().get('scalars', []):
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        return steps, values
    return None, None

runs = {
    '1M Baseline': 'runs/dqn_pong_1M_baseline/',
    '1M Low LR': 'runs/dqn_pong_1M_low_lr/',
    '1M High LR': 'runs/dqn_pong_1M_high_lr/',
    '1M Large Batch': 'runs/dqn_pong_1M_large_batch/'
}

metrics = {
    'Reward': ('rollout/ep_rew_mean', 'Episode Reward Mean'),
    'Episode Length': ('rollout/ep_len_mean', 'Sequence Length Mean'),
    'Loss': ('train/loss', 'Q-Network Loss'),
    'Exploration Rate': ('rollout/exploration_rate', 'Epsilon')
}

os.makedirs('plots_1M', exist_ok=True)

for metric_name, (tag, ylabel) in metrics.items():
    plt.figure(figsize=(10, 6))
    
    plotted = False
    for run_name, run_dir in runs.items():
        event_file = get_event_file(run_dir)
        if event_file:
            steps, values = extract_scalar(event_file, tag)
            if steps and values:
                plt.plot(steps, values, label=run_name, alpha=0.8)
                plotted = True
                
    if plotted:
        plt.title(f'{metric_name} Comparison (1M Steps)')
        plt.xlabel('Timesteps')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        
        filename = metric_name.lower().replace(" ", "_") + '.png'
        plt.savefig(os.path.join('plots_1M', filename), dpi=300)
        print(f"Saved plots_1M/{filename}")
    else:
        print(f"No data found for {metric_name}. (Runs might not be finished yet)")

