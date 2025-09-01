from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.logger import HParam, configure
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import time 
import argparse
import importlib
from stable_baselines3.common.vec_env import SubprocVecEnv

ENV_MODULE_NAME = 'env.CNN.snake_env_v12'
NUM_ENVS = 16

class ApplesControlCallback(BaseCallback):
    def __init__(self, total_timesteps, num_envs, verbose=0, start_apples=20, end_apples=1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.num_envs = num_envs
        self.start_apples = start_apples
        self.end_apples = end_apples

    def _on_step(self) -> bool:
        progress = min(self.num_timesteps / (self.total_timesteps * 0.5), 1.0)
        current_apples = int(self.start_apples - progress * (self.start_apples - self.end_apples))
        current_apples = max(self.end_apples, current_apples)
        # Ustaw max_apples w każdym środowisku
        self.training_env.env_method("set_max_apples", current_apples)
        if self.verbose and self.num_timesteps % 10000 == 0:
            print(f"Aktualna liczba jabłek: {current_apples}")
        return True
    
# NOWA KLASA CALLBACK, KTÓRA LOGUJE DODATKOWE METRYKI
class AdditionalMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.scores = []

    def _on_step(self) -> bool:
        # Pamiętaj, aby _on_step zwracał True, aby kontynuować trening
        return True

    def _on_rollout_end(self) -> None:
        # Zbieraj niestandardowe metryki ze słownika 'info'
        infos = self.locals['infos']
        current_scores = [info['score'] for info in infos if 'score' in info and 'terminal_observation' in info]
        
        if current_scores:
            mean_score = sum(current_scores) / len(current_scores)
            # Loguj średni wynik do TensorBoard
            self.logger.record("rollout/ep_score_mean", mean_score)

def main():
    parser = argparse.ArgumentParser(description='Train a RL model')
    parser.add_argument('-lc', '--location', type=str, default='')
    parser.add_argument('-ts', '--timestamps', type=int, default=200000000)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)
    parser.add_argument('-bs', '--batch_size', type=int, default=512)
    parser.add_argument('-es', '--exploration_start', type=float, default=1.0)
    parser.add_argument('-ee', '--exploration_end', type=float, default=0.005)
    parser.add_argument('-ef', '--exploration_fraction', type=float, default=0.1)
    args = parser.parse_args()

    try:
        env_module = importlib.import_module(ENV_MODULE_NAME)
        EnvClass = getattr(env_module, 'SnakeEnv')
    except (ImportError, AttributeError) as e:
        print(f"Error importing environment: {e}")
        return

    # Uaktualniona funkcja make_env, która przekazuje niestandardowe metryki do logowania
    def make_env():
        return lambda: Monitor(EnvClass(), info_keywords=("score", "snake_length"))

    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])
    eval_env = Monitor(EnvClass())

    TIMESTAMPS = args.timestamps

    arg_info = f'ts_{args.timestamps}-lr_{args.learning_rate:.5f}-bs_{args.batch_size}-exp_{args.exploration_start:.2f}_{args.exploration_end:.2f}_{args.exploration_fraction:.2f}'

    models_dir = f'{args.location}models/DQN-{arg_info}-{(time.strftime("%Y%m%d-%H%M"))}'
    logdir = f'{args.location}logs/DQN-{arg_info}-{(time.strftime("%Y%m%d-%H%M"))}'

    apples_callback = ApplesControlCallback(
        total_timesteps=TIMESTAMPS,
        num_envs=NUM_ENVS,
        verbose=1,
        start_apples=40,
        end_apples=1
    )


    checkpoint_callback = CheckpointCallback(
        save_freq=TIMESTAMPS,
        save_path= models_dir,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{models_dir}/best_model",
        log_path=logdir,
        eval_freq=50000,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Utwórz instancję nowego callbacku
    additional_metrics_callback = AdditionalMetricsCallback()

    model = DQN('MultiInputPolicy',
                env,
                verbose=1, 
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                tensorboard_log=logdir,
                exploration_initial_eps=args.exploration_start,  
                exploration_final_eps=args.exploration_end,  
                exploration_fraction=args.exploration_fraction      
    )

    
    model.learn(total_timesteps=TIMESTAMPS, tb_log_name='log', callback=[checkpoint_callback, eval_callback, apples_callback, additional_metrics_callback])


if __name__ == '__main__':
    main()
