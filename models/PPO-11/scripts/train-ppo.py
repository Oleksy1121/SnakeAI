import os
import shutil
import glob
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
import time 
import importlib

# Konfiguracja środowiska
ENV_MODULE_NAME = 'env.CNN.snake_env_v13'
NUM_ENVS = 16

class ApplesControlCallback(BaseCallback):
    def __init__(self, total_timesteps, num_envs, verbose=0, start_apples=1, end_apples=1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.num_envs = num_envs
        self.start_apples = start_apples
        self.end_apples = end_apples

    def _on_step(self) -> bool:
        progress = min(self.num_timesteps / (self.total_timesteps * 0.5), 1.0)
        current_apples = int(self.start_apples - progress * (self.start_apples - self.end_apples))
        current_apples = max(self.end_apples, current_apples)
        self.training_env.env_method("set_max_apples", current_apples)
        if self.verbose and self.num_timesteps % 10000 == 0:
            print(f"Aktualna liczba jabłek: {current_apples}")
        return True
    
class AdditionalMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.scores = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        infos = self.locals['infos']
        current_scores = [info['score'] for info in infos if 'score' in info and 'terminal_observation' in info]
        snake_lengths = [info['snake_length'] for info in infos if 'snake_length' in info and 'terminal_observation' in info]
        
        if current_scores:
            mean_score = sum(current_scores) / len(current_scores)
            self.logger.record("rollout/ep_score_mean", mean_score)

        if snake_lengths:
            mean_length = sum(snake_lengths) / len(snake_lengths)
            self.logger.record("rollout/ep_snake_length_mean", mean_length)


def main():
    # Definiowanie hiperparametrów
    LEARNING_RATE = 0.0001
    N_STEPS = 2048
    BATCH_SIZE = 512
    N_EPOCHS = 10
    TOTAL_TIMESTEPS = 200000000

    # Tworzenie uproszczonych nazw folderów
    base_dir = 'models'
    run_number = len(glob.glob(os.path.join(base_dir, 'PPO-*'))) + 1
    model_name = f'PPO-{run_number}'
    models_dir = os.path.join(base_dir, model_name)
    logdir = os.path.join('logs', model_name)

    # Tworzenie katalogów, jeśli nie istnieją
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)
    
    # Ścieżki do plików, które mają zostać skopiowane
    current_script_path = os.path.abspath(__file__)
    env_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ENV_MODULE_NAME.replace('.', os.sep) + '.py')
    
    # Tworzenie katalogu na skrypty wewnątrz katalogu z modelem
    scripts_dir = os.path.join(models_dir, 'scripts')
    os.makedirs(scripts_dir, exist_ok=True)

    # Kopiowanie skryptów
    try:
        shutil.copy(current_script_path, scripts_dir)
        print(f"Skopiowano skrypt treningowy do {scripts_dir}")
        shutil.copy(env_script_path, scripts_dir)
        print(f"Skopiowano skrypt środowiska do {scripts_dir}")
    except FileNotFoundError as e:
        print(f"Błąd kopiowania plików: {e}. Upewnij się, że pliki istnieją w podanych ścieżkach.")
        
    try:
        env_module = importlib.import_module(ENV_MODULE_NAME)
        EnvClass = getattr(env_module, 'SnakeEnv')
    except (ImportError, AttributeError) as e:
        print(f"Error importing environment: {e}")
        return

    def make_env():
        return lambda: Monitor(EnvClass(max_apples=5, initial_length=8), info_keywords=("score", "snake_length"))

    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])
    eval_env = Monitor(EnvClass())

    apples_callback = ApplesControlCallback(
        total_timesteps=TOTAL_TIMESTEPS,
        num_envs=NUM_ENVS,
        verbose=1,
        start_apples=40,
        end_apples=1
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=TOTAL_TIMESTEPS,
        save_path=models_dir,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(models_dir, "best_model"),
        log_path=logdir,
        eval_freq=50000,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    additional_metrics_callback = AdditionalMetricsCallback()

    model = PPO('MultiInputPolicy',
                env,
                verbose=1, 
                learning_rate=LEARNING_RATE,
                n_steps=N_STEPS,
                batch_size=BATCH_SIZE,
                n_epochs=N_EPOCHS,
                tensorboard_log=logdir,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name='log', callback=[checkpoint_callback, eval_callback, additional_metrics_callback])

if __name__ == '__main__':
    main()