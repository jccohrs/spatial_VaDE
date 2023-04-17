# pylint: disable=invalid-name

''' Script to perform hyperparameter tuning for the VaDE segmentation framework using optuna '''

import os
import optuna
import yaml
# from numpy import linspace
from optuna.trial import TrialState
from yaml.loader import SafeLoader
from train_vade import train_vade

def hyper_tuning_SA(trial):
    ''' function performing hyperparameter tuning '''
    image = 'Salinas'
    args = {
        # TODO: change it to create dictionary for VaDE arguments
        'k': 16,
        'reg_par': trial.suggest_float('reg_par', 0.05, 0.25),
        'ind_eps': trial.suggest_float('ind_eps', 0.1, 0.15),
        'reduce_dimensionality': True,
        'n_features': 7,
        'ind_func': 'non-squared-anisotropic-eps-inverse',
        'config_path': None,
        'filename': os.path.expandvars('${DATA_DIR}') + f'/segmentation/Salinas/{image}.nc',
        'gt_file': os.path.expandvars('${DATA_DIR}') + f'/segmentation/Salinas/{image}_gt.png',
        'irreg': False,
        'r': 1,
        'irreg_eps': 1,
        'kernel': 'direct-summation',
        'ind_params': [20],
        'convexification': 'strongly-conv',
        'pdhgtype': 2,
        'precision': 'float',
        'max_iter': trial.suggest_int('pdhg_it', 10, 2000, 10),
        'outer_iter': 20,
        'stop_eps': 1e-06,
        'outer_stop_eps': 1e-06,
        'band_selection': 'full',
        'band_select_param': 0.0001,
        'dim_red_method': 'mnf',
        'init': 'kmeans',
        'reduce_dim_init': False,
        'means': 'arithmetic',
        'variances': 'sdm',
        'components': 'pca',
        'trim_proportion': 0.5,
        'seed': 42,
        'ignore_label': 0,
        'ignore_pixels': False,
        'ignore_pixels_data_term': False,
        'solve_empty_segment': False,
        'save_intermediate_segs': False,
        'use_gpu': True
    }

    return train_vade(args)


def main():
    ''' main function '''
    # load MySQL optuna configuration from yaml file
    with open(os.path.expandvars('${OPTUNA_CONF}'), encoding='utf-8') as y:
        conf = yaml.load(y, Loader=SafeLoader)
        storage = f"mysql://{conf['user']}:{conf['password']}@{conf['host']}/optuna"

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
