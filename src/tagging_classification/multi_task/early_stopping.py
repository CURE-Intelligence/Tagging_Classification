from transformers import TrainerCallback

class EarlyStopping(TrainerCallback):
    def __init__(self, patience):
        self.patience = patience
        self.wait = 0
        self.best_loss = float('inf')

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        eval_loss = metrics.get('eval_loss', float('inf'))
        if eval_loss < self.best_loss:
            self.best_loss = eval_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                control.should_early_stop = True
                print("Early stopping triggered")