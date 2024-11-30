import os

# Define the commands
commands = [
    "python data_virtual_timeline.py --w 2",
    "python train.py --v biLSTM --t virtual",
    "python data_virtual_timeline.py --w 3",
    "python train.py --v biLSTM --t virtual",
    "python data_virtual_timeline.py --w 5",
    "python train.py --v biLSTM --t virtual",
    "python data_virtual_timeline.py --w 10",
    "python train.py --v biLSTM --t virtual",
    "python data_virtual_timeline.py --w 20",
    "python train.py --v biLSTM --t virtual"
]

# Execute the commands in sequence
for command in commands:
    print(f"Executing: {command}")
    os.system(command)
