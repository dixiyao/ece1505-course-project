"""Need to log down G_i"""
import torch

from plato.trainers import basic

class Trainer(basic.Trainer):
    """Log down G_i"""

    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)
        self.gbound = 0

    def perform_forward_and_backward_passes(self, config, examples, labels):
        self.optimizer.zero_grad()

        outputs = self.model(examples)

        loss = self._loss_criterion(outputs, labels)
        self._loss_tracker.update(loss, labels.size(0))

        if "create_graph" in config:
            loss.backward(create_graph=config["create_graph"])
        else:
            loss.backward()

        for param in self.model.parameters():
            grad = torch.abs(torch.norm(param.grad, p=2))
            self.gbound = max(self.gbound, grad.item())

        self.optimizer.step()

        return loss