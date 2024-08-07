import multiprocessing as mp
import numpy as np
import logging
import time
from multiprocessing import Pool, TimeoutError

logger = logging.getLogger(__name__)


class DistributedSimulation:
    def __init__(self, num_processes):
        self.num_processes = num_processes
        self.pool = None
        logger.info(f"Initialized DistributedSimulation with {num_processes} processes")

    def run_parallel_episodes(self, num_episodes, config, hierarchical_maddpg):
        episodes_per_process = num_episodes // self.num_processes
        remainder = num_episodes % self.num_processes

        args = [(config, hierarchical_maddpg, episodes_per_process + (1 if i < remainder else 0))
                for i in range(self.num_processes)]

        try:
            with Pool(processes=self.num_processes) as pool:
                self.pool = pool
                results = pool.starmap_async(self.run_episodes_batch, args)

                # 设置一个超时时间,防止无限等待
                timeout = max(300, num_episodes * 10)  # 假设每个episode最多需要10秒
                try:
                    flattened_results = [item for sublist in results.get(timeout=timeout) for item in sublist]
                except TimeoutError:
                    logger.error(f"Parallel execution timed out after {timeout} seconds")
                    pool.terminate()
                    return None

            logger.info(f"Completed {len(flattened_results)} episodes across {self.num_processes} processes")
            return flattened_results
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt detected. Terminating processes...")
            if self.pool:
                self.pool.terminate()
                self.pool.join()
        except Exception as e:
            logger.error(f"Error in parallel execution: {str(e)}")
        finally:
            if self.pool:
                self.pool.close()
                self.pool.join()

    @staticmethod
    def run_episodes_batch(config, hierarchical_maddpg, num_episodes):
        from environment.sagin_env import SAGINEnv

        env = SAGINEnv(config)
        episode_rewards = []

        for _ in range(num_episodes):
            try:
                episode_reward = DistributedSimulation.run_episode(env, hierarchical_maddpg)
                episode_rewards.append(episode_reward)
            except Exception as e:
                logger.error(f"Error in episode: {str(e)}")

        env.close()  # 确保环境资源被正确释放
        return episode_rewards

    @staticmethod
    def run_episode(env, hierarchical_maddpg):
        state = env.reset()
        if not isinstance(state, np.ndarray) or state.size == 0:
            logger.error(f"Invalid state returned from reset: {state}")
            return 0

        episode_reward = 0
        done = False
        step_count = 0
        max_steps = 1000  # 防止无限循环

        while not done and step_count < max_steps:
            try:
                action = hierarchical_maddpg.select_action(state)
                next_state, reward, done, _ = env.step(action)

                if not isinstance(next_state, np.ndarray) or next_state.size == 0:
                    logger.error(f"Invalid next_state returned from step: {next_state}")
                    break

                hierarchical_maddpg.store_transition(state, action, reward, next_state, done)
                state = next_state
                episode_reward += sum(reward)
                step_count += 1

            except Exception as e:
                logger.error(f"Error in episode step: {str(e)}")
                break

        if step_count >= max_steps:
            logger.warning("Episode reached maximum step limit")

        return episode_reward

    def close(self):
        if self.pool:
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