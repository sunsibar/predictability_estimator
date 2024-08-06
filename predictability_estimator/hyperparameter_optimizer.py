'''
Prompt: (in case I need major adjustments)

Do you know optuna well? Can you write me a class that uses optuna internally, or another, better suited package or multiple of them, and does the following: 
Inputs: - a dataset yielding X and y, batches of them, both two-dimensional: batch_size x feature_size_{X or y}. So y is multi-dimensional. - a metric to select 
by, for example variance explained - optionally, a dict with hyperparameter ranges (endpoints) , plus lists for other parameters to vary (for example, the loss 
to optimize in training) - an algorithm (str) to use for hyperparameter search.

And then it does automated hyperparameter search, with early stopping of unpromising runs (it could eg first optimize the learning rate to be as high as 
possible without getting nans, and early-stop at any nan). It should also allow hyperparameters like learning algorithm (Adam, SGD, RMSProp etc), batch size, 
regularization type (L1 or L2 or similar), and allow for a choice of loss to optimize, either MSE or cross-entropy (then y must contain integer class values). 
It should also store, either in a csv file or via another, not very memory intensive method, intermediate loss values + corresponding epochs (+ once the list of 
hyperparameters) for each run, to later be able to plot the training curves (maybe optuna has built-in methods for that?). It should also store a) whether 
training converged (no further decrease of loss in 10 or so epochs), whether it early-stopped because of nans or because of exploding loss values, best train 
and val loss. It should return after hyperparameter search the best model found + its hyperparameters + loss curves. The code should be documented well, so that 
I can later describe it to you again easily to build on that code, to for example create plots for visualizing a hyperparameter search. For example, I want to 
be able to plot train cuves based on the stored data, as well as 2d colormaps visualizing which parameters gave what loss values. The models to optimize should 
be MLPs, with same-width layers (except first and last), where the width and number layers are hyperparameters, as well as dropout, plus all the other typical 
hyperparameters. These are a lot of specifications, so please re-read it before you finalize your answer and make sure you do not forget anything.
'''
from collections import defaultdict
import json
import numpy as np
import optuna
from optuna.samplers import TPESampler, RandomSampler, GridSampler
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader , random_split 
import pandas as pd


def overfitting_metric(train_loss_curve, eval_loss_curve, epoch, maximize=False):
    '''
    Checks whether a) the typical difference between train and eval loss is larger than the
                        standard deviation of the train loss in the last 10 epochs/how much larger and 
                   b) the eval loss is increasing while the train loss is decreasing
    '''
    RECENT_EPOCHS = 10
    assert epoch >= RECENT_EPOCHS

    result = {}
    if maximize:
        result["eval_loss_end_minus_best"] = eval_loss_curve[epoch] - torch.max(eval_loss_curve[:epoch])
        result["train_loss_end_minus_best"] = train_loss_curve[epoch] - torch.max(train_loss_curve[:epoch])
    else:
        result["eval_loss_end_minus_best"] = eval_loss_curve[epoch] - torch.min(eval_loss_curve[:epoch])
        result["train_loss_end_minus_best"] = train_loss_curve[epoch] - torch.min(train_loss_curve[:epoch])


    train_loss_curve = train_loss_curve[epoch-RECENT_EPOCHS:epoch]
    eval_loss_curve = eval_loss_curve[epoch-RECENT_EPOCHS:epoch]
    loss_diffs = eval_loss_curve - train_loss_curve
    if maximize:
        loss_diffs *= (-1)
    std_train_loss = torch.std(train_loss_curve ) 
    if std_train_loss == 0:
        raise ValueError("Standard deviation of train loss is zero, I cant compute the overfitting metric")
        std_train_loss += 1e-6
    mean_loss_diff = torch.mean(loss_diffs )
    if mean_loss_diff <= 0:
        result['overfitting_metric'] =  0.0 
    else:
        result['overfitting_metric'] = mean_loss_diff / (std_train_loss ) 
    result["mean_loss_diff_at_end"] = mean_loss_diff 
    # is eval loss increasing? # calculate mean step direction across last 10 epochs
    eval_loss_diffs =  (eval_loss_curve[1:] - eval_loss_curve[:-1]) 
    train_loss_diffs =  (train_loss_curve[1:] - train_loss_curve[:-1])
    if maximize:
        eval_loss_diffs *= (-1)
        train_loss_diffs *= (-1)
    # is eval loss increasing? # calculate mean step direction across last 10 epochs 
    eval_loss_increase = torch.mean((eval_loss_diffs)) / std_train_loss
    train_loss_increase = torch.sum((train_loss_diffs)) / std_train_loss   
    result["eval_loss_increase_end"] = eval_loss_increase
    result["train_loss_increase_end"] = train_loss_increase
    if (train_loss_increase <= 0 ) and (eval_loss_increase > 0.1): # 0.1: Somewhat arbitrary; can reconstruct what "divergence" means from eval/train loss increase values
        result["train_eval_curves_diverge"] = True
    else:
        result["train_eval_curves_diverge"] = False 
    return result

def _model_size(trial):
    return (trial.params['num_layers'], trial.params['hidden_size'])


class HyperparameterOptimizer:
    '''
    Example usage:
    --------------
        X = torch.rand(100, 10)
        true_weights = torch.rand(10, 3)
        noise = torch.randn(100, 3) * 0.1
        y = X @ true_weights + noise 
        dataset = TensorDataset(X, y)
        optimizer = HyperparameterOptimizer(dataset, self.metric, 10, 3, self.hyperparameter_ranges)
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

    '''
    def __init__(self, dataset, metric, in_features, out_features, hyperparameter_ranges=None, 
                 algorithm='TPE', log_file='results.csv', 
                 loss_type='MSE', patience=10, max_epochs=100, stop_when_overfitting=False):
        self.in_features = in_features
        self.out_features = out_features 
        self.dataset = dataset
        if metric == "R2":
            metric = "variance_explained"
        self.metric = metric 
        assert metric in ["variance_explained",  "train_loss", "eval_loss"]
        self.loss_type = loss_type
        self.hyperparameter_ranges = hyperparameter_ranges or {}
        for k, v  in self.hyperparameter_ranges.items(): 
            assert k in ['hidden_size', 'num_layers', 'dropout', 'optimizer', 'regularization', 'weight_decay', 'lr', 'batch_size', 'sgd_momentum']
            assert type(v) in [list, tuple], f"Hyperparameter ranges must be lists or tuples; found: {type(v)}"
            assert (len(v) == 2) or k in ['optimizer', 'regularization'], "Hyperparameter ranges must consist of two elements except for optimizer and regularization"
        self.algorithm = algorithm
        if algorithm =="TPE":
            sampler = TPESampler()
        elif algorithm == "Random":
            sampler = RandomSampler()
        elif algorithm == "Grid":
            sampler = GridSampler()
        else:
            raise ValueError("Algorithm not implemented, feel free to add it above: ", algorithm)
        self.log_file = log_file
        self.patience = patience
        self.max_epochs = max_epochs
        self.max_batch_sizes = defaultdict(lambda: int(np.floor(max((1024, len(self.dataset)*0.8)))))
        self.stop_when_overfitting = stop_when_overfitting
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.study = optuna.create_study(direction='maximize' if metric == 'variance_explained' else 'minimize',
                                         sampler=sampler)
        self.results = []

    def _range(self, key, default_range):
        '''abbreviation'''
        return self.hyperparameter_ranges.get(key, default_range)
    
    def _create_model(self, trial):
  
        hidden_size = trial.suggest_int('hidden_size', *self._range("hidden_size", (32, 512))) 
        num_layers = trial.suggest_int('num_layers', *self._range("num_layers", (1, 5)))
        dropout = trial.suggest_float('dropout', *self._range("dropout", (0.0, 0.5)))
        
        layers = []
        layers.append(nn.Linear(self.in_features, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(1, num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_size, self.out_features))
        
        return nn.Sequential(*layers)
    

    def _metric(self, y_true, y_pred):
        if self.metric == 'variance_explained':
                ss_total = torch.sum((y_true - torch.mean(y_true)) ** 2)
                ss_residual = torch.sum((y_true - y_pred) ** 2)
                r2_score = 1 - ss_residual / ss_total
                return r2_score
        else: 
            raise ValueError("Invalid metric", self.metric)
            
    def get_data_loaders(self, batch_size):
        '''For convenience'''
        # Split dataset into training and evaluation sets
        train_size = int(0.8 * len(self.dataset))
        eval_size = len(self.dataset) - train_size
        train_dataset, eval_dataset = random_split(self.dataset, [train_size, eval_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, eval_loader
        
    def retrain_model(self, trial):
        score, results, model = self._objective(trial, return_model=True)
        return score, results, model 

    def _objective(self, trial, return_model=False):     
        
        model = self._create_model(trial)

        optimizer_name      = trial.suggest_categorical('optimizer', self._range("optimizer", ['Adam', 'SGD', 'RMSProp']))
        regularization_type = trial.suggest_categorical('regularization', self._range("regularization", ['L1', 'L2']))
        weight_decay        = trial.suggest_float('weight_decay', *self._range("weight_decay", (1e-6, 1e-2)), log=True)
        lr                  = trial.suggest_float('lr', *self._range("lr", (1e-5, 1e-1)), log=True)

        if regularization_type == 'L1': # Suggested by Copilot. Looks neat, let's see whether this works and isn't too slow! --nope, does not work.
            # for param in model.parameters():
            #     param.register_hook(lambda grad, param=param: grad + weight_decay * torch.sign(param))
            weight_decay_ = 0
        else:
            weight_decay_ = weight_decay

        bs_range = self._range("batch_size", (16, self.max_batch_sizes[_model_size(trial)]))
        batch_size = trial.suggest_int('batch_size', *bs_range, log=True)


        # loss_type = trial.suggest_categorical('loss', ['MSE', 'CrossEntropy'])
        if self.loss_type == 'MSE':
            criterion = nn.MSELoss()
        elif self.loss_type == "CrossEntropy":
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError("Invalid loss type", self.loss_type)   

        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay_,)
        elif optimizer_name == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr,  weight_decay=weight_decay_,
                                       momentum=trial.suggest_uniform('sgd_momentum', *self._range("sgd_momentum", (0.0, 0.999))))
        else:
            optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay_)
        del weight_decay_
        
        # Split dataset into training and evaluation sets
        # train_size = int(0.8 * len(self.dataset))
        # eval_size = len(self.dataset) - train_size
        # train_dataset, eval_dataset = random_split(self.dataset, [train_size, eval_size])

        # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
        train_loader, eval_loader = self.get_data_loaders(batch_size)
        
        
        epochs = self.max_epochs
        best_loss_train = float('inf')
        best_loss_eval = float('inf')
        patience = self.patience
        patience_counter = 0
        counter_no_decrease_eval_loss = 0 
        diverged = False
        oom = False
        train_loss_curve = torch.zeros(epochs)
        eval_loss_curve = torch.zeros(epochs)
        metric_curve = torch.zeros(epochs)
        t0 = time.time()
        try:
        # if True:
            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    if regularization_type == 'L1':
                        l1_regularization = torch.tensor(0.0)
                        for param in model.parameters():
                            l1_regularization += torch.norm(param, 1)
                        loss += weight_decay * l1_regularization
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                
                train_loss = running_loss / len(train_loader)
                train_loss_curve[epoch] = (train_loss)

                # Evaluation
                model.eval()
                eval_loss = 0.0
                y_true = []
                y_pred = []
                with torch.no_grad():
                    for X_batch, y_batch in eval_loader:
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                        eval_loss += loss.item()
                        if self.metric == 'variance_explained':
                            y_true.append(y_batch)
                            y_pred.append(outputs)
                eval_loss /= len(eval_loader)
                eval_loss_curve[epoch] = eval_loss
                if self.metric == "variance_explained":
                    y_true = torch.cat(y_true, dim=0)
                    y_pred = torch.cat(y_pred, dim=0)
                    metric_curve[epoch] = self._metric(y_true, y_pred)

                if eval_loss < best_loss_eval:
                    best_loss_eval = eval_loss
                    counter_no_decrease_eval_loss = 0
                else:
                    counter_no_decrease_eval_loss += 1

                if train_loss < best_loss_train:
                    best_loss_train = train_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    break
                
                if np.isnan(train_loss):
                    diverged = True 
                    break
                    # return float('inf')
                
                if self.stop_when_overfitting:
                    if counter_no_decrease_eval_loss >= 10:
                        overfitting_metric_dict =  overfitting_metric(train_loss_curve, eval_loss_curve, epoch, maximize=self.study.direction == 'maximize')
                        if overfitting_metric_dict['train_eval_curves_diverge']: 
                            break

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"Out of memory error: {e}")
                self.max_batch_sizes[_model_size(trial)] = trial.params["batch_size"] // 2
                del model
                del optimizer
                torch.cuda.empty_cache()
                oom = True
            else:
                raise e  

        # metric_curve = metric_curve[:epoch]
        # train_loss_curve = train_loss_curve[:epoch]
        # eval_loss_curve = eval_loss_curve[:epoch]

        loss_cuve_slice = np.linspace(0, epoch, 20).astype(int)
        overfitting_metrics_ = overfitting_metric(train_loss_curve, eval_loss_curve, epoch, maximize=self.study.direction == 'maximize')

        self.results.append({
            'trial': trial.number,
            'diverged': diverged,
            'oom': oom,
            'time_taken_s': (time.time() - t0),
            'epoch': epoch,
            'max_epochs': self.max_epochs,
            'metric': self.metric,
            'train_loss_curve': json.dumps(train_loss_curve[loss_cuve_slice].numpy().tolist()),
            'eval_loss_curve': json.dumps(eval_loss_curve[loss_cuve_slice].numpy().tolist()),
            'metric_curve': json.dumps(metric_curve[loss_cuve_slice].numpy().tolist()),
            'epochs_loss_curves': json.dumps(loss_cuve_slice.tolist()),
            'hyperparameters': trial.params,
            'converged': patience_counter >= patience,
            'train_loss': train_loss,
            'eval_loss': eval_loss, 
            'best_train_loss': best_loss_train,
            'best_eval_loss': best_loss_eval, 
            **{'overfitting_'+k: v for k, v in overfitting_metrics_.items() }
        })
        if self.metric == "variance_explained":
            score = self._metric(y_true, y_pred)
        elif self.metric == "train_loss":
            score = best_loss_train
        elif self.metric == "eval_loss":
            score = best_loss_eval
        else:
            raise ValueError("Invalid metric", self.metric)

        if return_model:
            return score, self.results[-1], model
        return score


    def optimize(self, n_trials=100, verbose=True, hyperparameters=None):
        if hyperparameters:
            # Update hyperparameter ranges
            for k, v in hyperparameters.items():
                self.hyperparameter_ranges[k] = v

        self.study.optimize(self._objective, n_trials=n_trials)
        best_trial = self.study.best_trial
        
        # Save results to CSV
        df = pd.DataFrame(self.results)
        df.to_csv(self.log_file, index=False)

        if verbose:
            # Print for the best trial whether it converged, whether it diverged, whether it ran out of memory, and the overfitting metrics.
            print(f"Best trial: {best_trial.number}")
            row = df[df['trial'] == best_trial.number].iloc[0]
            print(f"Converged: {row['converged']}")
            print(f"Diverged: {row['diverged']}")
            print(f"Out of memory: {row['oom']}")
            overfitting_metrics = {k: row[k] for k in df.columns if 'overfitting' in k}
            print(f"Overfitting metrics: ")
            for k, v in overfitting_metrics.items():
            # for k, v in row["overfitting"].items():
                print(f"  {k}: {v}")
            try: 
                import matplotlib.pyplot as plt
                from datetime import datetime
                from pathlib import Path
                plt.figure()
                # add subplots, 2 to plot metric separately
                plt.subplot(2, 1, 1)
                x = json.loads(row['epochs_loss_curves'])
                plt.plot(x, json.loads(row['train_loss_curve']), label='Train loss')
                plt.plot(x, json.loads(row['eval_loss_curve']), label='Eval loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                if self.metric not in ['train_loss', 'eval_loss']:
                    plt.subplot(2, 1, 2)
                    plt.plot(x, json.loads(row['metric_curve']), label=self.metric)
                    plt.xlabel('Epoch')
                    plt.ylabel(self.metric)
                plt.legend() 
                plt_path = Path(self.log_file).parent / f'loss_curves_best_trial_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                plt.savefig(plt_path)
                print(f"HyperparameterOptimizer: Saved loss curves to {plt_path}") 
            except BaseException as e:
                pass



        
        return best_trial

