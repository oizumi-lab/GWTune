#%%
import gc
import math
import os
import time

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
import optuna
import ot
import seaborn as sns
import torch
from tqdm.auto import tqdm

# warnings.simplefilter("ignore")
from utils.backend import Backend
from utils.init_matrix import InitMatrix

# nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv


# %%
class GW_Alignment:
    def __init__(
        self,
        pred_dist,
        target_dist,
        p,
        q,
        save_path,
        max_iter=1000,
        numItermax=1000,
        n_iter=100,
        to_types="torch",
        sinkhorn_method="sinkhorn",
    ):
        """
        2023/3/6 大泉先生

        1. epsilonに関して
        epsilon: １つ
        epsilonの範囲を決める：サーチ方法 optuna, 単純なgrid (samplerの種類, optuna)

        2. 初期値に関して
        初期値1つ固定: diagonal, uniform outer(p,q), 乱数
        初期値ランダムで複数: 乱数
        """
        self.to_types = to_types
        self.sinkhorn_method = sinkhorn_method
        self.size = len(p)

        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.n_iter = n_iter

        self.main_compute = MainGromovWasserstainComputation(
            pred_dist, target_dist, p, q, self.to_types, max_iter=max_iter, numItermax=numItermax, n_iter=n_iter
        )

    def define_eps_range(self, trial, eps_list, eps_log):
        """
        2023.3.16 佐々木 作成

        epsの範囲を指定する関数。
        """
        if len(eps_list) == 2:
            ep_lower, ep_upper = eps_list
            eps = trial.suggest_float("eps", ep_lower, ep_upper, log=eps_log)
        elif len(eps_list) == 3:
            ep_lower, ep_upper, ep_step = eps_list
            eps = trial.suggest_float("eps", ep_lower, ep_upper, step=ep_step)
        else:
            raise ValueError("The eps_list doesn't match.")

        return trial, eps

    def __call__(self, trial, device, init_mat_plan, eps_list, eps_log=True):
        if self.to_types == "numpy":
            assert device == "cpu", "numpy does not run in CUDA."

        """
        1.  define hyperparameter (eps, T)
        """

        trial, eps = self.define_eps_range(trial, eps_list, eps_log)

        # init_mat_plan = trial.suggest_categorical(
        #     "initialize", init_plans_list
        # )  # init_matをdeviceに上げる作業はentropic_gw中で行うことにしました。(2023/3/14 阿部)

        trial.set_user_attr("size", self.size)

        file_path = self.save_path + "/" + init_mat_plan  # ここのパス設定はoptimizer.py側でも使う可能性があるので、変更の可能性あり。

        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok=True)

        """
        2.  Compute GW alignment with hyperparameters defined above.
        """
        gw, logv, gw_loss, init_mat, trial = self.main_compute.compute_GW_with_init_plans(
            trial,
            eps,
            init_mat_plan,
            device,
            sinkhorn_method = self.sinkhorn_method
        )

        """
        3.  count the accuracy of alignment and save the results if computation was finished in the right way.
            If not, set the result of accuracy and gw_loss as float('nan'), respectively. This will be used as a handy marker as bad results to be removed in the evaluation analysis.
        """

        self.main_compute.back_end.save_computed_results(gw, init_mat, file_path, trial.number)

        """
        4. delete unnecessary memory for next computation. If not, memory error would happen especially when using CUDA.
        """

        del gw, logv
        torch.cuda.empty_cache()
        gc.collect()

        return gw_loss


class MainGromovWasserstainComputation:
    def __init__(self, pred_dist, target_dist, p, q, to_types, max_iter=1000, numItermax=1000, n_iter=100) -> None:

        self.to_types = to_types

        self.pred_dist, self.target_dist, self.p, self.q = pred_dist, target_dist, p, q
        self.size = len(p)

        # hyperparameter
        self.init_mat_builder = InitMatrix(matrix_size=self.size)  # 基本的に初期値はnumpyで作成するようにしておく。

        # gw alignmentに関わるparameter
        self.max_iter = max_iter

        # sinkhornに関わるparameter
        self.numItermax = numItermax

        # 初期値のiteration回数, かつ hyperbandのparameter
        self.n_iter = n_iter

        self.back_end = Backend("cpu", self.to_types)  # 並列計算をしない場合は、こちらにおいた方がはやい。(2023.4.19 佐々木)

    def entropic_gw(
        self,
        device,
        epsilon,
        T,
        max_iter=1000,
        numItermax=1000,
        tol=1e-9,
        trial=None,
        sinkhorn_method="sinkhorn",
        log=True,
        verbose=False
    ):
        """
        2023.3.16 佐々木
        backendに実装した "change_device" で、全型を想定した変数のdevice切り替えを行う。
        numpyに関しては、CPUの指定をしていないとエラーが返ってくるようにしているだけ。
        torch, jaxに関しては、GPUボードの番号指定が、これでできる。

        2023.4.11 佐々木
        multiprocessingで動かすために、backendでdeviceの変更を行うのを、このmethod内のみにした。

        multiprocessingが動かなかった原因は、各変数をGPUに上げる作業(args.to("cuda")が、一回の計算につき
        一回しかできないのに、途中で(安全のため、同じものであっても)何回もdeviceの切り替えを行っていたことが原因だった。
        """

        # ここで、全ての変数をto_typesのdeviceに変更している。
        self.back_end.device = device
        C1, C2, p, q, T = self.back_end(self.pred_dist, self.target_dist, self.p, self.q, T)

        constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, loss_fun="square_loss")

        cpt = 0
        err = 1

        if log:
            log = {"err": []}

        while err > tol and cpt < max_iter:
            Tprev = T
            # compute the gradient
            tens = ot.gromov.gwggrad(constC, hC1, hC2, T)
            T = ot.bregman.sinkhorn(p, q, tens, epsilon, method=sinkhorn_method, numItermax=numItermax)

            if cpt % 10 == 0:
                # err_prev = copy.copy(err)　#ここは使われていないようなので、一旦コメントアウトしました (2023.3.16 佐々木)
                err = self.back_end.nx.norm(T - Tprev)
                if log:
                    log["err"].append(err)
                if verbose:
                    if cpt % 200 == 0:
                        print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                    print("{:5d}|{:8e}|".format(cpt, err))
            cpt += 1

        if log:
            log["gw_dist"] = ot.gromov.gwloss(constC, hC1, hC2, T)
            return T, log

        else:
            return T

    def gw_alignment_computation(
        self,
        init_mat_plan,
        eps,
        max_iter,
        numItermax,
        device,
        trial=None,
        seed=42,
        sinkhorn_method="sinkhorn",
    ):
        """
        2023.3.17 佐々木
        gw_alignmentの計算を行う。ここのメソッドは変更しない方がいいと思う。
        外部で、特定のhyper parametersでのgw_alignmentの計算結果だけを抽出したい時にも使えるため。
        """

        init_mat = self.init_mat_builder.make_initial_T(init_mat_plan, seed)

        gw, logv = self.entropic_gw(
            device,
            eps,
            init_mat,
            max_iter=max_iter,
            numItermax=numItermax,
            trial=trial,
            sinkhorn_method=sinkhorn_method,
        )

        gw_loss = logv["gw_dist"]

        if self.back_end.check_zeros(gw):
            gw_loss = float("nan")
            acc = float("nan")

        else:
            pred = self.back_end.nx.argmax(gw, 1)
            correct = (pred == self.back_end.nx.arange(len(gw), type_as=gw)).sum()
            acc = correct / len(gw)

        return gw, logv, gw_loss, acc, init_mat

    def _save_results(self, gw_loss, acc, trial, init_mat_plan, num_iter=None, seed=None):

        gw_loss, acc = self.back_end.get_item_from_torch_or_jax(gw_loss, acc)

        trial.set_user_attr("best_acc", acc)
        if init_mat_plan in ["random", "permutation"]:
            trial.set_user_attr("best_iter", num_iter)
            trial.set_user_attr("best_seed", int(seed))  # ここはint型に変換しないと、謎のエラーが出る (2023.3.18 佐々木)。

        return trial

    def _check_pruner_should_work(self, gw_loss, trial, init_mat_plan, eps, num_iter=None):
        """
        2023.3.28 佐々木
        全条件において、prunerを動かすメソッド。

        Args:
            gw_loss (_type_): _description_
            trial (_type_): _description_
            init_mat_plan (_type_): _description_
            eps (_type_): _description_
            num_iter (_type_, optional): _description_. Defaults to None.
            gpu_id (_type_, optional): _description_. Defaults to None.

        Raises:
            optuna.TrialPruned: _description_
        """

        if init_mat_plan in ["uniform", "diag"] and math.isnan(gw_loss):  # math.isnan()を使わないとnanの判定ができない。
            # このifブロックがなくても、diag, uniformのprunerは正しく動作する。
            # ただ、tutorialの挙動を見ていると、これがあった方が良さそう。(2023.3.28 佐々木)
            raise optuna.TrialPruned(f"Trial for '{init_mat_plan}' was pruned with parameters: {{'eps': {eps}}}")

        if num_iter is None:  # uniform, diagにおいて、nanにならなかったがprunerが動くときのためのifブロック。
            num_iter = self.n_iter

        trial.report(gw_loss, num_iter)

        if trial.should_prune():
            raise optuna.TrialPruned(
                f"Trial was pruned at iteration {num_iter} with parameters: {{'eps': {eps}, 'initialize': '{init_mat_plan}'}}"
            )

    def compute_GW_with_init_plans(
        self,
        trial,
        eps,
        init_mat_plan,
        device,
        sinkhorn_method = "sinkhorn"
    ):
        """
        2023.3.17 佐々木
        uniform, diagでも、prunerを使うこともできるが、いまのところはコメントアウトしている。
        どちらにも使えるようにする場合は、ある程度の手直しが必要。

        2023.3.28 佐々木
        全条件において、正しくprunerを動かすメソッドを作成。
        各条件ごとへの拡張性を考慮すると、prunerの挙動は一本化しておく方が絶対にいい。

        2023.4.18 佐々木
        並行・並列計算による高速化は、Numpy環境だと全く意味がない。
        CUDAであっても、高速化は高々20%弱しか速くならず、よくわからないエラーも出るので、中止にします。
        """

        if init_mat_plan in ["uniform", "diag"]:
            gw, logv, gw_loss, acc, init_mat = self.gw_alignment_computation(
                init_mat_plan,
                eps,
                self.max_iter,
                self.numItermax,
                device,
                sinkhorn_method=sinkhorn_method,
            )
            trial = self._save_results(gw_loss, acc, trial, init_mat_plan)
            self._check_pruner_should_work(gw_loss, trial, init_mat_plan, eps)
            return gw, logv, gw_loss, init_mat, trial

        elif init_mat_plan in ["random", "permutation"]:
            best_gw_loss = float("inf")

            pbar = tqdm(np.random.randint(0, 100000, self.n_iter))
            pbar.set_description("trial: " + str(trial.number) + ", eps:" + str(round(eps, 3)))

            for i, seed in enumerate(pbar):
                (
                    c_gw, 
                    c_logv, 
                    c_gw_loss, 
                    c_acc, 
                    c_init_mat
                ) = self.gw_alignment_computation(
                    init_mat_plan,
                    eps,
                    self.max_iter,
                    self.numItermax,
                    device,
                    seed=seed,
                    sinkhorn_method=sinkhorn_method,
                )

                if c_gw_loss < best_gw_loss:
                    (
                        best_gw, 
                        best_logv, 
                        best_gw_loss, 
                        best_acc, 
                        best_init_mat, 
                    ) = (
                        c_gw,
                        c_logv,
                        c_gw_loss,
                        c_acc,
                        c_init_mat,
                    )

                    trial = self._save_results(best_gw_loss, best_acc, trial, init_mat_plan, num_iter=i, seed=seed)

                self._check_pruner_should_work(c_gw_loss, trial, init_mat_plan, eps, num_iter=i)

            if best_gw_loss == float("inf"):
                raise optuna.TrialPruned(
                    f"All iteration was failed with parameters: {{'eps': {eps}, 'initialize': '{init_mat_plan}'}}"
                )

            else:
                return best_gw, best_logv, best_gw_loss, best_init_mat, trial

        else:
            raise ValueError("Not defined initialize matrix.")
