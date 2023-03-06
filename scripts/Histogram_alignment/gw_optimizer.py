# %%
import os, gc
import numpy as np
import torch
import ot
import matplotlib.pyplot as plt
import optuna
from joblib import parallel_backend
import matplotlib.style as mplstyle
import seaborn as sns



class GW_Alignment():
    def __init__(self, pred_dist, target_dist, p, q, filename, device, plans = True, random_test = True, gpu_queue = None):
        self.pred_dist = pred_dist
        self.target_dist = target_dist
        self.p = p
        self.q = q
        self.device = device
        self.train_size = len(pred_dist)
        self.plans = plans
        
        self.gpu_queue = gpu_queue
        
        if plans and random_test:
            self.filename = filename + '_with_init_diag_plans'
        elif plans and not random_test:
            self.filename = filename + '_with_init_random_plans'
        else:
            self.filename = filename 
            
        self.random_test = random_test
        
        self.save_path = './GWD/Test/' + self.filename
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
    
    def entropic_gw_jax(self, epsilon, device, T = None, max_iter = 1000, tol = 1e-9):
        log = True
        verbose = False

        C1, C2, p, q = ot.utils.list_to_array(self.pred_dist.to(device),
                                              self.target_dist.to(device),
                                              self.p.to(device),
                                              self.q.to(device))
        
        nx = ot.backend.get_backend(C1, C2, p, q)
        
        # add T as an input
        if T is None:
            T = nx.outer(p, q)
        
        constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, loss_fun = "square_loss")
        cpt = 0
        err = 1
        
        if log:
            log = {'err': []}
        
        while (err > tol and cpt < max_iter):
            Tprev = T
            # compute the gradient
            tens = ot.gromov.gwggrad(constC, hC1, hC2, T)
            T = ot.bregman.sinkhorn(p, q, tens, epsilon, method = 'sinkhorn')
            
            if cpt % 10 == 0:
                # we can speed up the process by checking for the error only all the 10th iterations
                err = nx.norm(T - Tprev)
                if log:
                    log['err'].append(err)
                if verbose:
                    if cpt % 200 == 0:
                        print('{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                    print('{:5d}|{:8e}|'.format(cpt, err))
            cpt += 1
        
        if log:
            log['gw_dist'] = ot.gromov.gwloss(constC, hC1, hC2, T)
            return T, log
        
        else:
            return T
        
    def my_entropic_gromov_wasserstein(self, epsilon, device, T = None, max_iter = 1000, tol = 1e-9):
        log = True
        verbose = False

        C1, C2, p, q = ot.utils.list_to_array(self.pred_dist.to(device),
                                              self.target_dist.to(device),
                                              self.p.to(device),
                                              self.q.to(device))
        
        nx = ot.backend.get_backend(C1, C2, p, q)
        
        # add T as an input
        if T is None:
            T = nx.outer(p, q)
        
        constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, loss_fun = "square_loss")
        cpt = 0
        err = 1
        
        if log:
            log = {'err': []}
        
        while (err > tol and cpt < max_iter):
            Tprev = T
            # compute the gradient
            tens = ot.gromov.gwggrad(constC, hC1, hC2, T)
            T = ot.bregman.sinkhorn(p, q, tens, epsilon, method = 'sinkhorn')
            
            if cpt % 10 == 0:
                # we can speed up the process by checking for the error only all the 10th iterations
                err = nx.norm(T - Tprev)
                if log:
                    log['err'].append(err)
                if verbose:
                    if cpt % 200 == 0:
                        print('{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                    print('{:5d}|{:8e}|'.format(cpt, err))
            cpt += 1
        
        if log:
            log['gw_dist'] = ot.gromov.gwloss(constC, hC1, hC2, T)
            return T, log
        
        else:
            return T

    def rand_mat(self, n_init_plan):
        """
        ここでの乱数作成を上手にしないといけない。
        GWDの輸送行列Tの初期値を作成するメソッド。

        Args:
            n_init_plan (int): 乱数の個数

        Returns:
            rand_mat (n_init_plan * train_size * train_size) : 輸送行列Tの初期値
        """
        # Function for creating initial plans
        # T = np.random.uniform(low = 0, high = 1, size = (self.train_size, self.train_size))
        # T = T / np.sum(T)
        # return Ts
        rand_mat = np.random.rand(n_init_plan, self.train_size, self.train_size)
        row_sums = rand_mat.sum(axis = 2)
        rand_mat /= row_sums[:, :, np.newaxis] * self.train_size
        return rand_mat

    def __call__(self, trial):   
        eps = trial.suggest_float("eps", 1e-4, 1e-2, log = True)
        
        if self.gpu_queue is None:
            device = self.device
        
        else:
            gpu_id = self.gpu_queue.get()
            device = 'cuda:' + str(gpu_id) 
        
        if self.plans:
            init_plans = self.rand_mat(10)
            
            if self.random_test:
                init_plans[0] = np.diag(ot.unif(self.train_size))
                init_index = 0
            
            else:
                init_index = trial.suggest_int("init_index", 0, len(init_plans) - 1)
            
            init_plans = torch.from_numpy(init_plans).float().to(device)
        
        gw, logv = self.my_entropic_gromov_wasserstein(eps, device, T = init_plans[init_index] if self.plans is True else None)
        
        gw_loss = logv['gw_dist'].item()
        
        if torch.count_nonzero(gw).item() != 0:
            gw_loss = logv['gw_dist'].item()
            
            _, pred = torch.max(gw, 1)
            acc = pred.eq(torch.arange(len(gw)).to(device)).sum() / len(gw)

            if self.plans:
                torch.save(gw, self.save_path + '/GW({} pictures, epsilon={}, init_plan={}).pt'.format(gw.shape[0], round(eps, 6), init_index))
            else:
                torch.save(gw, self.save_path + '/GW({} pictures, epsilon={}).pt'.format(gw.shape[0], round(eps, 6)))
            
        else:
            gw_loss = 1e6
            acc = -1
        
        del gw, logv
        torch.cuda.empty_cache()
        gc.collect()
        
        if self.gpu_queue is not None:
            self.gpu_queue.put(gpu_id)

        return gw_loss, acc
    
    def run_study(self, filename, n_gpu = 10, num_trial = 50):
        if self.random_test:
            plan_name = "test"
        else:
            plan_name = "init_plans"
            
        save_file_name = plan_name + ", GWD (" + filename + ").db"
        
        if not os.path.exists(self.save_path + "/" + save_file_name):
            study = optuna.create_study(directions = ["minimize", "maximize"],
                                        study_name = 'GWD_' + plan_name,
                                        storage = "sqlite://" + self.save_path.split('.')[1] + "/" + save_file_name,
                                        load_if_exists = True)
            
            with parallel_backend("multiprocessing", n_jobs = n_gpu):
                study.optimize(self, n_trials = num_trial, n_jobs = n_gpu)

        else:
            study = optuna.create_study(directions = ["minimize", "maximize"],
                                        study_name = 'GWD_' + plan_name,
                                        storage = "sqlite://" + self.save_path.split('.')[1] + "/"  + save_file_name,
                                        load_if_exists = True)
        
        return study
    
    def load_graph(self, study):
        best_trial = study.best_trials[0]
        eps = best_trial.params['eps']
        acc = best_trial.values[1]
        
        if self.random_test:
            plans = 0
        elif not self.random_test and self.plans:
            plans = best_trial.params['init_index']
        
        if self.plans:
            gw = torch.load(self.save_path + '/GW({} pictures, epsilon={}, init_plan={}).pt'.format(self.train_size, round(eps, 6), plans))
        else:
            gw = torch.load(self.save_path + '/GW({} pictures, epsilon={}).pt'.format(self.train_size, round(eps, 6)))
        
        self.plot_coupling(gw, eps, acc)

    def make_eval_graph(self, study):
        df_test = study.trials_dataframe()
        success_test = df_test[df_test['values_1'] != -1]
        success_test = success_test[success_test['values_0'] > 1e-7]
        
        plt.figure()
        plt.title('The evaluation of GW results for random pictures')
        plt.scatter(success_test['values_1'], np.log(success_test['values_0']), label = 'init diag plan ('+str(self.train_size)+')', c = 'C0')
        plt.xlabel('accuracy')
        plt.ylabel('log(GWD)')
        plt.legend()
        plt.show()
    
    def plot_coupling(self, T, epsilon, acc):
        mplstyle.use('fast')
        N = T.shape[0]
        plt.figure(figsize=(8,6))
        sns.heatmap(T.to('cpu').numpy())
    
        plt.title('GW results ({} pictures, eps={}, acc.= {})'.format(N, round(epsilon, 6), round(acc, 4)))
        plt.tight_layout()
        plt.show()
        
# %%
if __name__ == '__main__':
    test = GWD_Dataset(train_size = 2000, random_dataset = True, device = 'cuda:3')
    model1, model2, p, q, filename = test.extract(sort_dataset=True)
    
    # %%
    tt = GW_Alignment(model1, model2, p, q, filename, plans = test.random_dataset, random_test = False)
    tt.my_entropic_gromov_wasserstein(1e-3, tt.device)
    # %%