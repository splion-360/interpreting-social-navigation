# Social Navigation using Modified Social Attention

This code here is the extension of the work carried out in the [Social Attention](https://arxiv.org/abs/1710.04689) to fit the [MaBe](https://www.aicrowd.com/challenges/multi-agent-behavior-challenge-2022/problems/mabe-2022-mouse-triplets#dataset) dataset for future trajectory prediction and behaviour analysis. <br>

## Getting Started

- Clone the repository - `git clone https://github.com/BRAINML-GT/social_navigate`
- Run the `setup.sh` bash file to setup the directories for logging.   
- Setup the conda environment using the `env.yml` configuration file. 
- Activate the environment using `conda activate snav`
- Download the dataset (`user_train.npy`) from the website and place it inside the `data/MaBe/`. You may want to change the filename depending on the name in the `train.py`
- Execute `python train.py` to train the model. 

Feel free to explore with the hyperparameters to accomodate your device requirements. This code was tested on Ubuntu 20.04 LTS containing NVIDIA GeForce RTX 3050 GPU with CUDA 12.0. 



