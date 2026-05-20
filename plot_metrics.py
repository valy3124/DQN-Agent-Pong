import os
import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_event_file(run_dir):
    matches = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))
    if not matches:
        # Search recursively
        matches = glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True)
    
    # Return the one with the highest size/latest if multiple
    if matches:
        return max(matches, key=os.path.getsize)
    return None

def extract_scalar(event_file, tag):
    ea = EventAccumulator(event_file)
    ea.Reload()
    if tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        return steps, values
    return None, None

runs = {
    'Baseline': 'runs/dqn_pong_run_1_baseline/',
    'Low LR': 'runs/dqn_pong_run_2_low_lr/',
    '10M Steps': 'runs/dqn_pong_run_3_10M/'
}

metrics = {
    'Reward': ('rollout/ep_rew_mean', 'Episode Reward Mean'),
    'Episode Length': ('rollout/ep_len_mean', 'Sequence Length Mean'),
    'Loss': ('train/loss', 'Q-Network Loss')
}

os.makedirs('plots', exist_ok=True)

for metric_name, (tag, ylabel) in metrics.items():
    plt.figure(figsize=(10, 6))
    
    for run_name, run_dir in runs.items():
        event_file = get_event_file(run_dir)
        if event_file:
            steps, values = extract_scalar(event_file, tag)
            if steps and values:
                plt.plot(steps, values, label=run_name, alpha=0.8)
                
    plt.title(f'{metric_name} Comparison')
    plt.xlabel('Timesteps')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    
    filename = metric_name.lower().replace(" ", "_") + '.png'
    plt.savefig(os.path.join('plots', filename), dpi=300)
    print(f"Saved {filename}")

