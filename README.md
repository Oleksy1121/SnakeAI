# SnakeAI 🐍🤖

An implementation of the classic **Snake** game and an **AI agent** trained to play it using **Reinforcement Learning (RL)** with **Stable-Baselines3**.  
The project includes a custom **Gymnasium environment** with both **image-based** and **feature vector** observations.

## 📌 Project Overview
- Built from scratch in **Python (3.12, Conda environment)** with **PyGame** for rendering.
- Custom environment `SnakeEnv` compatible with **Gymnasium API**.
- Two RL algorithms were tested: **DQN** and **PPO**.
- **PPO with CNN + feature vector input** achieved the best results.

## 🎮 Environment (`snake_env.py`)
The environment extends `gym.Env` and provides:
- **Action space:** `Discrete(3)` → `straight`, `right`, `left`
- **Observation space:**
    - **Image**: grid representation of the board (`head`, `body`, `apples`)
    - **Vector**: 20 numerical features, including distance to apple, direction encoding, nearby walls/tail, etc.

### Reward shaping

- +1 for eating an apple
- Small positive reward for survival
- Distance-based shaping rewards (+ closer to apple, - farther)
- Wall proximity penalties
- Body proximity penalties
- -1 on game over

This design helps stabilize learning and encourages the snake to actively seek apples.

## 🤖 Training
Two agents were trained:
- **DQN** (`train-dqn.py`)
- **PPO** (`train-ppo.py`)

The PPO agent performed **significantly better** in this setup.

### PPO training details
- **Policy:** `MultiInputPolicy` (CNN for grid + MLP for feature vector)
- **Total steps:** 50M (best model reached ~40M)
- **Environments:** 16 parallel `SubprocVecEnv`
- **Dynamic apple count**: gradually reduced from 40 → 1 during training

### Results
- **Eval mean episode length:** ~3000
- **Eval mean reward:** ~278
- Depending on randomness, the snake is capable of eating **40–120 apples** in one run.
- Training was stopped at **50M steps**, with the **best checkpoint around 40M**.

📈 Training evaluation curves (TensorBoard):  
![](attachments/Pasted%20image%2020250902210444.png)

🎥 Example gameplay:  
![](attachments/snake%20demo.gif)

The snake successfully collects apples, but sometimes **traps itself** by building walls around its own body. Future work will focus on solving this problem.

## 🛠 Installation
Clone the repository and install dependencies from `requirements.txt`:
`conda create -n snakeai python=3.12 conda activate snakeai pip install -r requirements.txt`

## 🚀 Usage
### 1. Check the environment
`python env-check.py`

### 2. Train a model
- PPO:
    `python train-ppo.py`
- DQN:
    `python train-dqn.py`
    
Models and logs will be stored in the `models/` and `logs/` directories.

### 3. Run a trained model
`python model-check.py`

## 📂 Repository structure

```
├── env/  
│   └── snake_env.py       # Custom Gymnasium environment
├── train-ppo.py           # PPO training script
├── train-dqn.py           # DQN training script
├── model-check.py         # Test a trained PPO model
├── env-check.py           # Debug/visualize environment
├── game.py                # Classic Snake implementation (PyGame)
├── requirements.txt       # Dependencies
└── README.md              # Project description
```

## 🔮 Future Work
- Preventing the snake from **self-trapping**.
- Experimenting with **curriculum learning** for more robust policies.
- Comparing additional algorithms (e.g., A2C, SAC).
- Optimizing CNN architecture for faster training.