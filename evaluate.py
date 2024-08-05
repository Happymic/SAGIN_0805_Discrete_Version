from config import load_config
from environment import SAGINEnv
from agents import create_agents
from trainers import HybridTrainer
from visualization import PygameVisualizer

def evaluate():
    config = load_config("config/base_config.yaml")
    env = SAGINEnv(config)
    agents = create_agents(config, env)
    trainer = HybridTrainer(env, agents, config)
    visualizer = PygameVisualizer(env, config)

    trainer.load_models()
    trainer.evaluate(config.num_eval_episodes, visualizer)
    visualizer.close()

if __name__ == "__main__":
    evaluate()