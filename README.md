# Predictability Estimator

A tiny package to fit MLP models from 1D features to 1D features. 

    Example usage:
    --------------
        X = torch.rand(100, 10) 
        true_weights = torch.rand(10, 3)
        noise = torch.randn(100, 3) * 0.1
        y = X @ true_weights + noise 
        dataset = TensorDataset(X, y)
        optimizer = HyperparameterOptimizer(dataset, self.metric, 10, 3, hyperparameter_ranges={})
        best_trial = optimizer.optimize(n_trials=3)

        Hyperparameters to chose ranges for:
        -----------------------------------
        - hidden_size: Tuple[int, int] = (32, 512)
        - num_layers: Tuple[int, int] = (1, 5)
        - dropout: Tuple[float, float] = (0.0, 0.5)
        - optimizer: List[str] = ['Adam', 'SGD', 'RMSProp']
        - regularization: List[str] = ['L1', 'L2']
        - weight_decay: Tuple[float, float] = (1e-6, 1e-2)
        - lr: Tuple[float, float] = (1e-5, 1e-1)
        - batch_size: Tuple[int, int] = (16, 1024)
        - sgd_momentum: Tuple[float, float] = (0.0, 0.999)