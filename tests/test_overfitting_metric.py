import torch
from predictability_estimator.hyperparameter_optimizer import overfitting_metric

def test_overfitting_metric():
    # Test case 1: No overfitting
    eval_loss_curve = torch.tensor([0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01])
    train_loss_curve = torch.tensor([0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1])
    epoch = 10
    result = overfitting_metric(train_loss_curve, eval_loss_curve, epoch)
    assert result['overfitting_metric'] <= 0.1/0.15
    assert result['train_eval_curves_diverge'] == False 

    # Test case 1: No overfitting
    train_loss_curve = torch.tensor([0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01])
    eval_loss_curve = torch.tensor([0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1])
    epoch = 10
    result = overfitting_metric(train_loss_curve, eval_loss_curve, epoch)
    assert result['overfitting_metric'] <= 0.1 / 0.15 
    assert result['train_eval_curves_diverge'] == False

    # Test case 2: Overfitting
    train_loss_curve = torch.tensor([0.1, 0.09, 0.11, 0.09, 0.1, 0.1, 0.09, 0.11, 0.1, 0.1])
    eval_loss_curve = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    epoch = 10
    result = overfitting_metric(train_loss_curve, eval_loss_curve, epoch)
    assert result['overfitting_metric'] > 0.5
    assert result['train_eval_curves_diverge'] == False

    # Test case 2: Overfitting
    train_loss_curve = torch.tensor([0.1, 0.09, 0.11, 0.09, 0.1, 0.1, 0.09, 0.11, 0.1, 0.1])
    eval_loss_curve = torch.tensor([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    epoch = 10
    result = overfitting_metric(train_loss_curve, eval_loss_curve, epoch)
    assert result['overfitting_metric'] > 1
    assert result['train_eval_curves_diverge'] == True
    epoch = 10
    result = overfitting_metric(train_loss_curve, eval_loss_curve, epoch, maximize=True)
    assert result['overfitting_metric'] == 0
    assert result['train_eval_curves_diverge'] == False

    # Test case 3: Maximize=True
    eval_loss_curve = torch.tensor([0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01])
    train_loss_curve = torch.tensor([0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1])
    epoch = 10
    result = overfitting_metric(train_loss_curve, eval_loss_curve, epoch, maximize=True)
    assert result['overfitting_metric'] > 0.1
    assert result['overfitting_metric'] < 0.1/0.15
    assert result['train_eval_curves_diverge'] == False # both are becoming worse actually

if __name__ == '__main__':  
    test_overfitting_metric()
    print("overfitting_metric test passed")