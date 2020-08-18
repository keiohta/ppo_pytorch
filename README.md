# ppo_pytorch
PPO implementation using pytorch

## Prepare virtual environment using Anaconda
```bash
$ conda create -n pytorch python=3.8 anaconda
$ conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
$ pip install gym
```

## Train Pendulum-v0
```bash
$ python train_ppo.py
```
