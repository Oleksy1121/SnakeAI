import os
import shutil
import glob
import time
import importlib
import logging
from typing import Any, Dict, Callable

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback

# =======================
# Logging configuration
# =======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# =======================
# Global configuration
# =======================
ENV_MODULE_NAME = "env.snake_env"
NUM_ENVS = 16
CONFIG: Dict[str, Any] = {
    "learning_rate": 1e-4,
    "n_steps": 2048,
    "batch_size": 512,
    "n_epochs": 10,
    "total_timesteps": 200_000_000,
    "start_apples": 40,
    "end_apples": 1,
}


# =======================
# Callbacks
# =======================
class ApplesControlCallback(BaseCallback):
    """Callback that dynamically decreases the number of apples in the environment."""

    def __init__(self, total_timesteps: int, num_envs: int, start_apples: int, end_apples: int, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.num_envs = num_envs
        self.start_apples = start_apples
        self.end_apples = end_apples

    def _on_step(self) -> bool:
        # Linear interpolation of apple count based on training progress
        progress = min(self.num_timesteps / (self.total_timesteps * 0.5), 1.0)
        current_apples = int(self.start_apples - progress * (self.start_apples - self.end_apples))
        current_apples = max(self.end_apples, current_apples)

        # Apply the change across all environments
        self.training_env.env_method("set_max_apples", current_apples)

        if self.verbose and self.num_timesteps % 10_000 == 0:
            logger.info(f"Current number of apples: {current_apples}")
        return True


class AdditionalMetricsCallback(BaseCallback):
    """Callback that logs additional metrics such as average score and snake length."""

    def _on_step(self) -> None:
        # Required abstract method, does nothing in this callback
        return True

    def _on_rollout_end(self) -> None:
        # Collect metrics from info dict
        infos = self.locals.get("infos", [])
        scores = [info["score"] for info in infos if "score" in info and "terminal_observation" in info]
        snake_lengths = [info["snake_length"] for info in infos if "snake_length" in info and "terminal_observation" in info]

        if scores:
            mean_score = sum(scores) / len(scores)
            self.logger.record("rollout/ep_score_mean", mean_score)

        if snake_lengths:
            mean_length = sum(snake_lengths) / len(snake_lengths)
            self.logger.record("rollout/ep_snake_length_mean", mean_length)


# =======================
# Main training function
# =======================
def main() -> None:
    # Prepare directories for saving models and logs
    base_dir = "models"
    run_number = len(glob.glob(os.path.join(base_dir, "PPO-*"))) + 1
    model_name = f"PPO-{run_number}"
    models_dir = os.path.join(base_dir, model_name)
    logdir = os.path.join("logs", model_name)

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    # Copy training and environment scripts for reproducibility
    current_script = os.path.abspath(__file__)
    env_script = os.path.join(os.path.dirname(current_script), ENV_MODULE_NAME.replace(".", os.sep) + ".py")
    scripts_dir = os.path.join(models_dir, "scripts")
    os.makedirs(scripts_dir, exist_ok=True)

    for src in [current_script, env_script]:
        try:
            shutil.copy(src, scripts_dir)
            logger.info(f"Copied {os.path.basename(src)} to {scripts_dir}")
        except FileNotFoundError:
            logger.warning(f"File not found: {src}")

    # Import the environment class dynamically
    try:
        env_module = importlib.import_module(ENV_MODULE_NAME)
        EnvClass: Callable = getattr(env_module, "SnakeEnv")
    except (ImportError, AttributeError) as e:
        logger.exception(f"Environment import failed: {e}")
        return

    # Factory function for creating monitored environments
    def make_env() -> Callable:
        return lambda: Monitor(EnvClass(max_apples=5, initial_length=8), info_keywords=("score", "snake_length"))

    # Vectorized training and evaluation environments
    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])
    eval_env = Monitor(EnvClass())

    # Callbacks
    apples_callback = ApplesControlCallback(
        total_timesteps=CONFIG["total_timesteps"],
        num_envs=NUM_ENVS,
        start_apples=CONFIG["start_apples"],
        end_apples=CONFIG["end_apples"],
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(save_freq=CONFIG["total_timesteps"], save_path=models_dir)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(models_dir, "best_model"),
        log_path=logdir,
        eval_freq=50_000,
        deterministic=True,
        render=False,
        verbose=1,
    )
    metrics_callback = AdditionalMetricsCallback()

    # Define the PPO model
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        learning_rate=CONFIG["learning_rate"],
        n_steps=CONFIG["n_steps"],
        batch_size=CONFIG["batch_size"],
        n_epochs=CONFIG["n_epochs"],
        tensorboard_log=logdir,
    )

    # Start training
    model.learn(
        total_timesteps=CONFIG["total_timesteps"],
        tb_log_name="log",
        callback=[checkpoint_callback, eval_callback, metrics_callback, apples_callback],
    )


if __name__ == "__main__":
    main()
