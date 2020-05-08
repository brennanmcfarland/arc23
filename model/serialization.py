import torch
from torch.nn import Module

from model.execution import Trainer, TrainState


# TODO: any other vars that would need saving? may want an optional additional_state arg
# saves the train state of the model st it can be used to initialize a new model to resume future training, picking up
# where it left off
def save_train_state(out_path: str):
    def _bind(model: Module, trainer: Trainer, train_state: TrainState):
        def _apply():
            torch.save({
                'model_state_dict': model.state_dict(),
                'trainer': trainer,
                'train_state': train_state,
                'optimizer_state_dict': trainer.optimizer.state_dict(),
            }, out_path)
        return _apply
    return _bind


# loads the train state of the model so it can be picked up where it left off
# NOTE: still requires a model, trainer and train state to be passed in
def load_train_state(in_path: str):
    def _bind(model: Module, trainer: Trainer, train_state: TrainState):
        def _apply():
            save_dict = torch.load(in_path)
            model.load_state_dict(save_dict['model_state_dict'])
            trainer = save_dict['trainer']
            train_state = save_dict['train_state']
            trainer.optimizer.load_state_dict(save_dict['optimizer_state_dict'])
            return model, trainer, train_state
        return _apply
    return _bind
