from .base_trainer import BaseTrainer

class IntegratedTrainer(BaseTrainer):
    def update_models(self, states, actions, rewards, next_states, dones):
        self.models[0].update(states, actions, rewards, next_states, dones)