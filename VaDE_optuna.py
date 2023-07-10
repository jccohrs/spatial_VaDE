# pylint: disable=invalid-name

''' Script to perform hyperparameter tuning for the VaDE segmentation framework using optuna '''

import os
import optuna
import yaml
# from numpy import linspace
import argparse
from train_vade import read_args
from optuna.trial import TrialState
from yaml.loader import SafeLoader
from train_vade import train_vade
import numpy as np
from numpy import random

def hyper_tuning_SA(trial):
    ''' function performing hyperparameter tuning '''
    args = read_args()  
    ## Example overwrite for hyperparameter fields
    ## args['reg_par'] = trial.suggest_float('reg_par', 0.05, 0.25)
    seed = args['master_seed']
    np.random.seed(seed)
    np.random.randint(1, 1000, size=1)


    return train_vade(args)


def main():
    ''' main function '''
    # load MySQL optuna configuration from yaml file
    with open(os.path.expandvars('${OPTUNA_CONF}'), encoding='utf-8') as y:
        conf = yaml.load(y, Loader=SafeLoader)
        storage = f"mysql://{conf['user']}:{conf['password']}@{conf['host']}/optuna?unix_socket=/var/run/mysqld/mysqld.sock"

    # Salinas
    study = optuna.create_study(storage=storage,
                                study_name='VaDE_SA_raw_spat0', load_if_exists=True)
    study.optimize(hyper_tuning_SA, n_trials=100, show_progress_bar=False)


    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
