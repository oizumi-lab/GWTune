import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import numpy as np
import ot

from src.gw_alignment import GW_Alignment, InitMatrix, load_optimizer

class Optimization_Config:
    def __init__(self, 
                 data_name = "THINGS",
                 delete_study = True, 
                 device = 'cpu',
                 to_types = 'numpy',
                 n_jobs = 4,
                 init_plans_list = ['random'],
                 num_trial = 4,
                 n_iter = 1,
                 max_iter = 200,
                 sampler_name = 'tpe',
                 eps_list = [1, 10],
                 eps_log = True,
                 pruner_name = 'hyperband',
                 pruner_params = {'n_startup_trials': 1, 'n_warmup_steps': 2, 'min_resource': 2, 'reduction_factor' : 3}
                 ) -> None:
        self.data_name = data_name
        self.delete_study = delete_study
        self.device = device
        self.to_types = to_types
        self.n_jobs = n_jobs
        self.init_plans_list = init_plans_list
        self.num_trial = num_trial
        self.n_iter = n_iter
        self.max_iter = max_iter
        self.sampler_name = sampler_name
        self.eps_list = eps_list
        self.eps_log = eps_log
        self.pruner_name = pruner_name
        self.pruner_params = pruner_params


def run_main_gw(config : Optimization_Config,
                RDM_source, 
                RDM_target, 
                results_dir,
                filename,
                load_OT = False):

        sql_name = 'sqlite'
        storage = "sqlite:///" + results_dir +  '/' + filename + '.db'
        save_path = results_dir + filename
        # distribution in the source space, and target space
        p = ot.unif(len(RDM_source))
        q = ot.unif(len(RDM_target))

        # generate instance solves gw_alignment　
        test_gw = GW_Alignment(RDM_source, RDM_target, p, q, save_path, max_iter = config.max_iter, n_iter = config.n_iter, to_types = config.to_types)

        # generate instance optimize gw_alignment　
        opt = load_optimizer(save_path,
                                n_jobs = config.n_jobs,
                                num_trial = config.num_trial,
                                to_types = config.device,
                                method = 'optuna',
                                sampler_name = config.sampler_name,
                                pruner_name = config.pruner_name,
                                pruner_params = config.pruner_params,
                                n_iter = config.n_iter,
                                filename = filename,
                                sql_name = sql_name,
                                storage = storage,
                                delete_study = config.delete_study
        )

        ### optimization
        # 1. 初期値の選択。実装済みの初期値条件の抽出をgw_optimizer.pyからinit_matrix.pyに移動しました。
        init_plans = InitMatrix().implemented_init_plans(config.init_plans_list)

        # used only in grid search sampler below the two lines
        eps_space = opt.define_eps_space(config.eps_list, config.eps_log, config.num_trial)
        search_space = {"eps": eps_space, "initialize": init_plans}
        if not load_OT:
            # 2. run optimzation
            study = opt.run_study(test_gw, config.device, init_plans_list = init_plans, eps_list = config.eps_list, eps_log = config.eps_log, search_space = search_space)
            best_trial = study.best_trial
            OT = np.load(save_path+f'/{config.init_plans_list[0]}/gw_{best_trial.number}.npy')
        else:
            study = opt.run_study(test_gw, config.device, init_plans_list = init_plans, eps_list = config.eps_list, eps_log = config.eps_log, search_space = search_space, forced_run = False)
            best_trial = study.best_trial
            OT = np.load(save_path+f'/{config.init_plans_list[0]}/gw_{best_trial.number}.npy')
        return OT