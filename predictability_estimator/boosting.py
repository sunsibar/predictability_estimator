

import torch
import torch.nn as nn

class BoostedModel(nn.Module):
    def __init__(self, n_models=5, models=None):
        super(BoostedModel, self).__init__()
        self.models = nn.ModuleList() #[] # no ModuleList, we dont need to optimize further
        if models is not None:
            self.models = models
        self.weights = torch.zeros((n_models, 1), requires_grad=False)
        self.n_models = max(n_models, len(self.models))

    @property
    def i(self):
        '''Position of next model to be added.'''
        return len(self.models)
    
    def add_model(self, model, weight):
        self.weights[self.i] = weight
        self.models.append(model)
        if self.i > self.n_models:
            self.n_models = self.i


    def forward(self, x):
        y = None
        for i, model in enumerate(self.models):
            model_result = self.weights[i] * model(x)
            if y is None:
                y = model_result
            else:
                y += model_result
        return y 
    




