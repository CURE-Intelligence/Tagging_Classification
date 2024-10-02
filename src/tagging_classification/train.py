from src.tagging_classification.multi_task.weighted_trainer import WeightedLossTrainer
from src.tagging_classification.single_task.weighted_trainer import WeightedLossTrainerSingle


def init_trainer(task = 'multi', **kwargs):

    #early_stopping = EarlyStopping(patience=2)

    if task == 'multi':

        return WeightedLossTrainer(
            model=kwargs['model'],
            args=kwargs['training_arguments'],
            train_dataset=kwargs['train_dataset'],
            eval_dataset=kwargs['eval_dataset'],
            compute_metrics=kwargs['compute_metrics'],
            tokenizer=kwargs['tokenizer'],
            data_collator=kwargs['data_collator'],
            #callbacks=[early_stopping],
            article_type_class_weights=kwargs['article_type_class_weights'],
            category_class_weights=kwargs['category_class_weights']
            )
    return WeightedLossTrainerSingle(
        model=kwargs['model'],
        args=kwargs['training_arguments'],
        train_dataset=kwargs['train_dataset'],
        eval_dataset=kwargs['eval_dataset'],
        compute_metrics=kwargs['compute_metrics'],
        tokenizer=kwargs['tokenizer'],
        data_collator=kwargs['data_collator'],
        class_weights=kwargs['class_weights']
    )

def train_trainer(trainer):

    """Training Process"""
    print("Starting training...")
    return trainer.train()


def evaluate_trainer(trainer):

    """Evaluation Process"""
    return trainer.evaluate()

def predict_trainer(trainer, test_data):

    """Predict Process"""
    return trainer.predict(test_data)