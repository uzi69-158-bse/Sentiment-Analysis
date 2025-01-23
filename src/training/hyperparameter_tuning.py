import yaml
import optuna

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.training.train import Trainer
from src.data.data_ingestion import DataIngestion

# Load the YAML configuration file
with open('config/config.yaml', 'r') as config_file:
    base_config = yaml.safe_load(config_file)

def objective(trial):
    # Update training-related configurations dynamically during hyperparameter tuning
    config = base_config.copy()
    config['training'].update({
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'scheduler_step_size': trial.suggest_int('scheduler_step_size', 1, 5),
        'scheduler_gamma': trial.suggest_float('scheduler_gamma', 0.8, 0.99),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
    })

    # Data Ingestion
    data_ingestion = DataIngestion(config['data'])
    train_data, val_data, _ = data_ingestion.process_data()

    # Training
    trainer = Trainer(config)
    train_loader = trainer.prepare_dataloader(train_data, config['training']['batch_size'])
    val_loader = trainer.prepare_dataloader(val_data, config['training']['batch_size'])

    trainer.train(train_loader, val_loader, config['training']['num_epochs'])

    # Evaluation
    _, val_metrics = trainer.evaluate(val_loader)
    return val_metrics['accuracy']

def run_hyperparameter_tuning():
    tuning_config = base_config['hyperparameter_tuning']
    study = optuna.create_study(direction='maximize', study_name=tuning_config['study_name'])
    study.optimize(objective, n_trials=tuning_config['n_trials'])

    print("Best trial:")
    trial = study.best_trial
    print(f"Value: {trial.value}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    run_hyperparameter_tuning()
