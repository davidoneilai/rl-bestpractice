from tensorboard.backend.event_processing import event_accumulator
import csv
import sys
import os

logdir = sys.argv[1]   # ex: results/logs/DQN_lr_0.01
tag = sys.argv[2]      # ex: 'Average Reward (100 episodes)'
out = sys.argv[3]      # ex: out.csv

ea = event_accumulator.EventAccumulator(logdir)
ea.Reload()
scalars = ea.Scalars(tag)
with open(out, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['wall_time','step','value'])
    for s in scalars:
        writer.writerow([s.wall_time, s.step, s.value])

print(f'Exported {len(scalars)} rows to {out}')