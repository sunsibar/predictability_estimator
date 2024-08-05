import unittest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
# import torch.optim as optim
# from torch.utils.data import DataLoader, random_split
import optuna
# import pandas as pd
# from collections import defaultdict
# import numpy as np
# import json
# import time

from predictability_estimator.hyperparameter_optimizer import HyperparameterOptimizer

class TestHyperparameterOptimizer(unittest.TestCase):

    def setUp(self):
        X = torch.rand(100, 10)
        y = torch.rand(100, 2)
        self.dataset = TensorDataset(X, y)
        self.metric = 'variance_explained'
        self.hyperparameter_ranges = {'hidden_size': (32, 512), 'num_layers': (1, 5), 'dropout': (0.0, 0.5)}
        self.optimizer = HyperparameterOptimizer(self.dataset, self.metric, 10, 2, self.hyperparameter_ranges)

    def test_init(self):
        self.assertEqual(self.optimizer.metric, 'variance_explained')
        self.assertEqual(self.optimizer.loss_type, 'MSE')
        self.assertEqual(self.optimizer.algorithm, 'TPE')
        self.assertEqual(self.optimizer.log_file, 'results.csv')
        self.assertEqual(self.optimizer.patience, 10)
        self.assertEqual(self.optimizer.max_epochs, 100)
        self.assertFalse(self.optimizer.stop_when_overfitting)
        self.assertIsInstance(self.optimizer.study, optuna.study.Study)
        self.assertEqual(self.optimizer.results, [])

    def test_create_model(self):
        trial = MagicMock()
        trial.suggest_int.side_effect = [64, 3]
        trial.suggest_float.return_value = 0.2
        model = self.optimizer._create_model(trial)
        self.assertIsInstance(model, nn.Sequential)
        self.assertEqual(len(model), 7)  # 3 layers + 2 ReLU + 2 Dropout
    def test_metric(self):
        y_true = torch.tensor([1.0, 2.0, 3.0])
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        result = self.optimizer._metric(y_true, y_pred)
        self.assertEqual(result, 1.0)

    # @patch('optuna.trial.Trial') 
    # def test_objective(self, MockTrial):
    #     trial = MockTrial()
    #     trial.suggest_float.return_value = 0.5 

    #     trial.params = {
    #         'batch_size': 32,  # Set a valid batch size for testing
    #         'hidden_size': 128,
    #         'num_layers': 2,
    #         'dropout': 0.5,
    #         'optimizer': 'SGD',
    #         'regularization': 'L1',
    #         'weight_decay': 0.01,
    #         'lr': 0.01,
    #         'sgd_momentum': 0.9
    #     }
          

    #     result = self.optimizer._objective(trial)
    #     self.assertIsInstance(result, float)

    # @patch('optuna.trial.Trial')
    # def test_optimize(self, MockTrial):
    #     trial = MockTrial()
    #     trial.suggest_int.side_effect = [64, 3, 32, 32]
    #     trial.suggest_float.return_value = 0.2
    #     trial.suggest_categorical.side_effect = ['Adam', 'L2']
    #     trial.suggest_loguniform.side_effect = [1e-4, 1e-3]
    #     trial.suggest_uniform.return_value = 0.9

    #     with patch.object(self.optimizer.study, 'optimize') as mock_optimize:
    #         mock_optimize.return_value = None
    #         best_trial = self.optimizer.optimize(n_trials=1)
    #         self.assertIsNotNone(best_trial)
    #         self.assertTrue(mock_optimize.called)

    def test_linear_dependency_with_noise(self):
        # Create linear data with noise
        X = torch.rand(100, 10)
        true_weights = torch.rand(10, 3)
        noise = torch.randn(100, 3) * 0.1
        y = X @ true_weights + noise 
        dataset = TensorDataset(X, y)
        optimizer = HyperparameterOptimizer(dataset, self.metric, 10, 3, self.hyperparameter_ranges)
        best_trial = optimizer.optimize(n_trials=5)
        
        self.assertIsNotNone(best_trial)
        self.assertGreaterEqual(best_trial.value, 0.5)  # Expect high R^2  ; ca 0.1**2 possible (?)

    def test_hyperparam_use(self):
        # Create linear data with noise
        X = torch.rand(100, 10)
        true_weights = torch.rand(10, 3)
        noise = torch.randn(100, 3) * 0.1
        y = X @ true_weights + noise 
        dataset = TensorDataset(X, y)
        
        # Define hyperparameter ranges to only include L1 loss and 3 layers
        hyperparameter_ranges = {
            'regularization': ['L1'],
            'num_layers': (3,3)
        }
        
        optimizer = HyperparameterOptimizer(dataset, self.metric, 10, 3, hyperparameter_ranges)
        best_trial = optimizer.optimize(n_trials=3)
        
        self.assertIsNotNone(best_trial)
        # self.assertGreaterEqual(best_trial.value, 0.5)  # Expect high R^2  ; ca 0.1**2 possible (?)
        self.assertEqual(best_trial.params['num_layers'], 3)  # Assert the best trial has 3 layers

    def test_random_mlp_outputs(self):
        # Create random MLP outputs as data
        class RandomMLP(nn.Module):
            def __init__(self):
                super(RandomMLP, self).__init__()
                self.fc1 = nn.Linear(10, 50)
                self.fc2 = nn.Linear(50, 5)
            
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        mlp = RandomMLP()


        # Disable gradient calculations for all parameters
        for param in mlp.parameters():
            param.requires_grad = False

        X = torch.rand(100, 10)
        y = mlp(X)
        dataset = TensorDataset(X, y) 
        
        optimizer = HyperparameterOptimizer(dataset, self.metric, 10, 5, self.hyperparameter_ranges)
        best_trial = optimizer.optimize(n_trials=10)
        
        self.assertIsNotNone(best_trial)
        self.assertGreaterEqual(best_trial.value, 0.5)  # Expect high R^2

if __name__ == '__main__':
    unittest.main( )