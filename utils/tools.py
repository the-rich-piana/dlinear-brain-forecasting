import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
import time

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.jpg', model_name=None, pred_len=None, seq_len=None):
    """
    Results visualization with a clearer history/future split and ground truth mean.
    """
    plt.figure(figsize=(12, 6))
    
    # Define the point where the prediction begins
    prediction_start_index = len(true) - pred_len
    true_mean = np.mean(true)

    # 1. Plot the historical data that the model used as input
    # --- MODIFIED: Added the mean to the label ---
    plt.plot(range(0, prediction_start_index), 
             true[:prediction_start_index], 
             label=f'Ground Truth (History) | Mean: {true_mean:.3f}', 
             linewidth=2, 
             color='green')

    # 2. Plot the future ground truth that the model is trying to predict
    plt.plot(range(prediction_start_index - 1, len(true)), 
             true[prediction_start_index-1:], 
             label='Ground Truth (Future)', 
             linewidth=2, 
             color='mediumseagreen',
             linestyle='--')

    # 3. Plot the prediction, also starting from the last historical point
    if preds is not None:
        pred_x_range = range(prediction_start_index - 1, len(true))
        pred_y_values = [true[prediction_start_index - 1]] + list(preds[-pred_len:])
        plt.plot(pred_x_range, pred_y_values, label='Prediction', linewidth=2, color='red')
    
    # 4. The vertical line now correctly marks the start of the future/prediction period
    plt.axvline(x=prediction_start_index, color='gray', linestyle=':', alpha=0.7, label='Prediction Start')

    # --- Title and other settings remain the same ---
    title_parts = []
    if model_name:
        title_parts.append(f'Model: {model_name}')
    if seq_len:
        title_parts.append(f'Seq Len: {seq_len}')
    if pred_len:
        title_parts.append(f'Pred Len: {pred_len}')
    
    if title_parts:
        plt.title(' | '.join(title_parts))
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(name, bbox_inches='tight', dpi=150)
    plt.close()


def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def output_results(args, setting, mse, mae, rmse, mape, mspe, rse, corr):
    """
    Outputs experiment results to CSV file with structured format
    
    Args:
        args: experiment arguments containing model, seq_len, pred_len, label_len
        setting: experiment setting string
        mse, mae, rmse, mape, mspe, rse, corr: metric values
    """
    # Create results dataframe
    result_data = {
        'model': [args.model],
        'setting': [setting],
        'experiment_name': [getattr(args, 'experiment_name', None)],
        'seq_len': [args.seq_len],
        'pred_len': [args.pred_len], 
        'label_len': [args.label_len],
        'mse': [mse],
        'mae': [mae],
        'rmse': [rmse],
        'mape': [mape],
        'mspe': [mspe],
        'rse': [rse],
        'corr': [corr]
    }
    
    df = pd.DataFrame(result_data)
    
    # Use experiment name for CSV filename, fallback to results.csv
    experiment_name = getattr(args, 'experiment_name', None)
    csv_filename = f"{experiment_name}.csv" if experiment_name else "results.csv"
    
    # Append to CSV file (create header if file doesn't exist)
    if os.path.exists(csv_filename):
        df.to_csv(csv_filename, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_filename, mode='w', header=True, index=False)