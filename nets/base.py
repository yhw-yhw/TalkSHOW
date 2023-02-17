import torch
import torch.nn as nn
import torch.optim as optim

class TrainWrapperBaseClass():
    def __init__(self, args, config) -> None:
        self.init_optimizer()

    def init_optimizer(self) -> None:
        print('using Adam')
        self.generator_optimizer = optim.Adam(
            self.generator.parameters(),
            lr = self.config.Train.learning_rate.generator_learning_rate,
            betas=[0.9, 0.999]
        )
        if self.discriminator is not None:
            self.discriminator_optimizer = optim.Adam(
                self.discriminator.parameters(),
                lr = self.config.Train.learning_rate.discriminator_learning_rate,
                betas=[0.9, 0.999]
            )

    def __call__(self, bat):
        raise NotImplementedError

    def get_loss(self, **kwargs):
        raise NotImplementedError

    def state_dict(self):
        model_state = {
            'generator': self.generator.state_dict(),
            'generator_optim': self.generator_optimizer.state_dict(),
            'discriminator': self.discriminator.state_dict() if self.discriminator is not None else None,
            'discriminator_optim': self.discriminator_optimizer.state_dict() if self.discriminator is not None else None
        }
        return model_state

    def parameters(self):
        return self.generator.parameters()

    def load_state_dict(self, state_dict):
        if 'generator' in state_dict:
            self.generator.load_state_dict(state_dict['generator'])
        else:
            self.generator.load_state_dict(state_dict)

        if 'generator_optim' in state_dict and self.generator_optimizer is not None:
            self.generator_optimizer.load_state_dict(state_dict['generator_optim'])

        if self.discriminator is not None:
            self.discriminator.load_state_dict(state_dict['discriminator'])

            if 'discriminator_optim' in state_dict and self.discriminator_optimizer is not None:
                self.discriminator_optimizer.load_state_dict(state_dict['discriminator_optim'])

    def infer_on_audio(self, aud_fn, initial_pose=None, norm_stats=None, **kwargs):
        raise NotImplementedError