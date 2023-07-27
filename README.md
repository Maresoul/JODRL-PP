## Project Name

Repository for 'Privacy-Preserving Offloading Scheme in Multi-Access Edge Computing Based on MADRL'

## Experiment Environment

- Python Version: 3.8.5
- PyTorch Version: 1.8.1
- GPU Enabled: Yes

## Introduction

This algorithm is a code implementation of the model and algorithm proposed in the article "Privacy-Preserving Offloading Scheme in Multi-Access Edge Computing Based on MADRL." In the experiment, the intelligent agent interacts with the environment, receives rewards, and gradually learns better offloading strategies to optimize the performance and privacy of the model.


## Running Instructions

1. Install Dependencies:
Make sure you have Python 3.8.5 and PyTorch 1.8.1 installed and have a GPU-enabled environment ready.

2. Clone the Project: Use the following command to clone the project to your local machine:
````
git clone https://github.com/Maresoul/JODRL-PP.git
````

3. Change Directory: Enter the project folder:
````
cd pytorch-jodrl_pp
````

4. Run the Program: Execute the following command to run the main program:
````
python main.py
````

5. Compare Algorithms:
Navigate to the compare_algorithm folder and execute the corresponding algorithm.

## File Structure

- `main.py`: Main program entry point, runs the experimental design.
- `compare_algorithm/`: Contains program files for comparative experiments.
- `discen.py`: Baseline algorithm 1, decentralized learning policy network.
- `local.py`: Baseline algorithm 2, all tasks executed locally.
- `near.py`: Baseline algorithm 3, tasks offloaded to the nearest edge node.
- `qmix/`: QMIX algorithm program folder.
- `mappo/`: MAPPO algorithm program folder.
- `MEC_env.py`: MEC environment settings for interaction with the intelligent agent.

## Usage Notes

This program has undergone multiple modifications and corrections to ensure proper functioning on this machine. If you encounter any errors during execution, it is likely that GPU parameters are misconfigured. Please double-check the settings carefully.







