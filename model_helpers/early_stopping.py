import numpy as np
import torch


class EarlyStopping:
    """
    Early stop the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, log_path, patience=7, verbose=False, delta=0):
        """
        log_path - absolute path, where the model (or parameter) is stored.
        patience - int, how long to wait after last improved validation loss. Default: 7
        verbose  - bool, if True, prints a message for each validation loss improvement. Default: False
        delta    - float, minimum change in the monitored quantity to qualify as an improvement. Default: 0
        """

        log_path = log_path.strip()      
        log_path = log_path.rstrip("/") 

        self.log_path = log_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Save model when validation loss decrease.
        """

        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        torch.save(model.state_dict(), self.log_path + '/' + 'checkpoint_param.pkl')
        self.val_loss_min = val_loss



