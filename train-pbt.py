import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
import os
import importlib
import ray.train.session as session
from ray.tune.experiment import Trial


ENV_MODULE_NAME = 'env.CNN.snake_env_v7'
ENV_CLASS_NAME = 'SnakeEnv'

def train_dqn_pbt(config):
    try:
        env_module = importlib.import_module(ENV_MODULE_NAME)
        EnvClass = getattr(env_module, ENV_CLASS_NAME)
    except (ImportError, AttributeError) as e:
        raise RuntimeError(f"Błąd importowania środowiska: {e}")

    env = EnvClass(warmup_episodes=config["timesteps_per_iteration"], max_length=20)
    eval_env = EnvClass(warmup_episodes=config["timesteps_per_iteration"], max_length=20)

    model = DQN(
        'CnnPolicy',
        env,
        verbose=0,
        learning_rate=config["learning_rate"],
        exploration_initial_eps=config["exploration_start"],
        exploration_final_eps=config["exploration_final_eps"],
        exploration_fraction=config["exploration_fraction"],
    )

    trial_dir = ray.tune.get_context().get_trial_dir()

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(trial_dir, "best_model"),
        log_path=trial_dir,
        eval_freq=10000,
        deterministic=True,
        render=False,
        verbose=0
    )

    model.learn(
        total_timesteps=config["timesteps_per_iteration"], 
        callback=eval_callback
    )

    ray.tune.report(metrics={"mean_reward": eval_callback.best_mean_reward})


    model.save(os.path.join(trial_dir, "checkpoint"))


def main_pbt():
    ray.init()

    config = {
        "learning_rate": tune.loguniform(1e-5, 1e-3),
        "exploration_start": tune.uniform(0.1, 0.5),
        "exploration_fraction": tune.uniform(0.5, 0.9),
        "exploration_final_eps": 0.005, 
        "timesteps_per_iteration": 250000,
    }

    scheduler = PopulationBasedTraining(
        time_attr="training_iteration", 
        metric="mean_reward",          
        mode="max",                   
        perturbation_interval=1,       
        hyperparam_mutations={         
            "learning_rate": lambda: tune.loguniform(1e-5, 1e-3).sample(),
            "exploration_start": lambda: tune.uniform(0.1, 0.5).sample(),
            "exploration_fraction": lambda: tune.uniform(0.5, 0.9).sample(),
        }
    )


    def custom_trial_dirname_creator(trial: Trial):
        return f"trial_{trial.trial_id}"

    analysis = tune.run(
        train_dqn_pbt,
        config=config,
        scheduler=scheduler,
        num_samples=10, 
        stop={"training_iteration": 10},
        storage_path=os.path.abspath("./ray_results"),
        name="dqn_pbt_snake",
        verbose=1,
        trial_dirname_creator=custom_trial_dirname_creator
    )

    print("\n--- Najlepszy agent PBT ---")
    best_trial = analysis.best_trial(metric="mean_reward", mode="max")
    print(f"  Najlepsza średnia nagroda: {best_trial.last_result['mean_reward']}")
    print("  Parametry najlepszego agenta:")
    for key, value in best_trial.config.items():
        print(f"    {key}: {value}")

    ray.shutdown()

if __name__ == '__main__':
    main_pbt()