from botorch.exceptions.errors import InputDataError
from ax.service.ax_client import AxClient

import utils 
from utils import *
from deepred import DeepRed
from data import load_data

import numpy as np

import torch

import os
    
device = torch.device("cpu")


class ParamSearch:
    
    def __init__(self, args):
        self.args = args
        self._build_search_space()
        self.args.nbr_size = self.args.nbr_size[0]
        self.train_dataset, self.dev_dataset = load_data(args=self.args)
        
    def _build_search_space(self):
        self.search_space = []
        self.param_position = {}
        pos = 0
        for arg in vars(args):
            val = getattr(args, arg)
            if isinstance(val, list):
                
                if len(val) == 1:
                    log(f'Fixed argument {arg} and value is {val[0]}')
                    self.search_space.append({'name': arg, 'type': 'fixed', 'value': val[0]})
                else:
                    log(f'Range argument {arg} between {val[0]} and {val[1]}')
                    self.search_space.append({'name': arg, 'type': 'range', 'bounds': val})
                self.param_position[arg] = pos
                pos += 1
    
    def fit_eval(self, params):
        global device
        args = self.args
        deepred = DeepRed(num_nodes=self.train_dataset.num_nodes, dim=params['dim'], dropout=params['dropout'],
                    temporal=args.temporal, lr=params['lr'], reg_cof=params['reg_cof'], 
                    epochs=params['epochs'], root_dir=self.args.root_dir)
        deepred.fit(self.train_dataset)
        results = deepred.validate(self.dev_dataset, scoring_fun=compute_interaction_scores, eval_fun=AUC)
        #return results['loss'] if self.args.minimize else results['pred']
        #return {'AUC': (results['pred'], 0.0)}
        return {'MSE': (results['loss'], 0.0), 'AUC': (results['pred'], 0.0)}

    def search(self):
        def rebuild_neighborhood(dataset):
            dataset.neighborhood_matrix = np.zeros((dataset.num_nodes, next_parameters['nbr_size']))
            dataset.neighborhood_mask = get_neighborhood_mask(dataset.neighborhood_matrix)
            
        ax_client = AxClient(enforce_sequential_optimization=False)
        objective_name = 'MSE' if self.args.minimize else 'AUC'
        outcome_constraints=["AUC >= 99"] if self.args.minimize else None
        ax_client.create_experiment(
            name='Tuning', parameters=self.search_space, objective_name=objective_name, 
            outcome_constraints=outcome_constraints)
            
        for _ in range(self.args.trials):
            try:
                next_parameters, trial_index = ax_client.get_next_trial()
                if self.search_space[self.param_position['nbr_size']]['type'] != 'fixed':
                    rebuild_neighborhood(self.train_dataset)
                    rebuild_neighborhood(self.dev_dataset)

                ax_client.complete_trial(trial_index=trial_index, raw_data=self.fit_eval(next_parameters))
                
            except InputDataError:
                ax_client.log_trial_failure(trial_index=trial_index)
        best_parameters, metrics = ax_client.get_best_parameters()
        return best_parameters, ax_client.get_trials_data_frame().sort_values(objective_name)
        
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, default='./data/matador/train.txt')
    parser.add_argument('--nbr-size', type=int, nargs='+', action='store', default=[100])
    parser.add_argument('--dim', type=int, nargs='+', action='store', default=[128])
    parser.add_argument('--lr', type=float, nargs='+', action='store', default=[0.0001, 0.1])
    parser.add_argument('--dropout', nargs='+', type=float, action='store', default=[0.1, 0.9])
    parser.add_argument('--reg-cof', nargs='+', type=float, action='store', default=[0., 1.])
    parser.add_argument('--epochs', type=int, nargs='+', action='store', default=[2])
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--temporal', dest='temporal', action='store_true')
    parser.add_argument('--static', dest='temporal', action='store_false')
    parser.add_argument('--min', dest='minimize', action='store_true')
    parser.add_argument('--log-level', type=int, default=utils.PROGRESS_LOG)
    parser.set_defaults(binary=False)
    parser.set_defaults(temporal=False)
    parser.set_defaults(minimize=False)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    path = os.path.join(args.root_dir, OUTPUT_DIR, 'best_pars.txt')
    utils.LOG_LEVEL = args.log_level
    ps = ParamSearch(args)
    best_pars, trials = ps.search()
    log(f'Saving best hyper parameters to {path}')
    with open(path, 'w') as f:
        for k, v in best_pars.items():
            f.write(f'{k} {v}\n')
            
    path = os.path.join(args.root_dir, OUTPUT_DIR, 'trials.txt')
    trials.to_csv(path, index=False, sep=' ')