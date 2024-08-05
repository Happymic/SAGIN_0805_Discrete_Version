from .base_trainer import BaseTrainer

class DistributedTrainer(BaseTrainer):
    def update_models(self, states, actions, rewards, next_states, dones):
        for i, model in enumerate(self.models):
            model.update(states[i], actions[i], rewards[i], next_states[i], dones[i])