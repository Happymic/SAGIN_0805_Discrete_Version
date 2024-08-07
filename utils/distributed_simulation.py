import multiprocessing as mp
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DistributedSimulation:
    def __init__(self, num_processes):
        self.num_processes = num_processes
        self.pool = mp.Pool(processes=num_processes)
        logger.info(f"Initialized DistributedSimulation with {num_processes} processes")

    def run_parallel_episodes(self, num_episodes, config, hierarchical_maddpg):
        episodes_per_process = num_episodes // self.num_processes
        remainder = num_episodes % self.num_processes

        args = [(config, hierarchical_maddpg, episodes_per_process + (1 if i < remainder else 0))
                for i in range(self.num_processes)]

        try:
            results = self.pool.starmap(self.run_episodes_batch, args)
            flattened_results = [item for sublist in results for item in sublist]
            logger.info(f"Completed {len(flattened_results)} episodes across {self.num_processes} processes")
            return flattened_results
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt detected. Terminating processes...")
            self.pool.terminate()
            self.pool.join()
        finally:
            self.pool.close()

    @staticmethod
    def run_episodes_batch(config, hierarchical_maddpg, num_episodes):
        # Lazy import to avoid circular dependency
        from environment.sagin_env import SAGINEnv

        env = SAGINEnv(config)
        episode_rewards = []

        for _ in range(num_episodes):
            episode_reward = DistributedSimulation.run_episode(env, hierarchical_maddpg)
            episode_rewards.append(episode_reward)

        return episode_rewards

    @staticmethod
    def run_episode(env, hierarchical_maddpg):
        try:
            state = env.reset()
            if not isinstance(state, np.ndarray) or state.size == 0:
                logger.error(f"Invalid state returned from reset: {state}")
                return 0

            episode_reward = 0
            done = False

            while not done:
                action = hierarchical_maddpg.select_action(state)
                next_state, reward, done, _ = env.step(action)

                if not isinstance(next_state, np.ndarray) or next_state.size == 0:
                    logger.error(f"Invalid next_state returned from step: {next_state}")
                    break

                hierarchical_maddpg.store_transition(state, action, reward, next_state, done)
                state = next_state
                episode_reward += sum(reward)

            return episode_reward
        except Exception as e:
            logger.error(f"Error in run_episode: {str(e)}")
            return 0

    def close(self):
        logger.info("Closing DistributedSimulation and terminating processes")
        self.pool.close()
        self.pool.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        if exc_type is not None:
            logger.error(f"An error occurred: {exc_type}, {exc_val}")
            return False
        return True