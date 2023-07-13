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
from .utils.backend import Backend
from .utils.init_matrix import InitMatrix
# nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# %%
class GW_Alignment:
    def __init__(
        self,
        source_dist,
        target_dist,
        data_path,
        max_iter=1000,
        numItermax=1000,
        n_iter=20,
        to_types="torch",
        data_type="double",
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
        self.data_type = data_type
        self.sinkhorn_method = sinkhorn_method

        # distribution in the source space, and target space
        self.source_size = len(source_dist)
        self.target_size = len(target_dist)

        self.data_path = data_path
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path, exist_ok=True)

        self.n_iter = n_iter

        self.main_compute = MainGromovWasserstainComputation(
            source_dist,
            target_dist,
            self.to_types,
            data_type=self.data_type,
            max_iter=max_iter,
            numItermax=numItermax,
            n_iter=n_iter,
        )

    def define_eps_range(self, trial, eps_list, eps_log):
        """_summary_

        Args:
            trial (_type_): _description_
            eps_list (_type_): _description_
            eps_log (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if len(eps_list) == 2:
            ep_lower, ep_upper = eps_list
            eps = trial.suggest_float("eps", ep_lower, ep_upper, log=eps_log)
        elif len(eps_list) == 3:
            ep_lower, ep_upper, ep_step = eps_list
            eps = trial.suggest_float("eps", ep_lower, ep_upper, step=ep_step)
        else:
            raise ValueError("The eps_list and/or eps_log doesn't match.")

        return trial, eps

    def __call__(self, trial, device, init_mat_plan, eps_list, eps_log=True):
        if self.to_types == "numpy":
            assert device == "cpu", "numpy does not run in CUDA."

        """
        1.  define hyperparameter (eps, T)
        """

        trial, eps = self.define_eps_range(trial, eps_list, eps_log)
        trial.set_user_attr("source_size", self.source_size)
        trial.set_user_attr("target_size", self.target_size)

        """
        2.  Compute GW alignment with hyperparameters defined above.
        """
        logv, init_mat, trial = self.main_compute.compute_GW_with_init_plans(
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
        gw = logv["ot"]
        gw_loss = logv["gw_dist"]
        self.main_compute.back_end.save_computed_results(gw, init_mat, self.data_path, trial.number)

        """
        4. delete unnecessary memory for next computation. If not, memory error would happen especially when using CUDA.
        """

        del gw, logv
        torch.cuda.empty_cache()
        gc.collect()

        return gw_loss


class MainGromovWasserstainComputation:
    def __init__(
        self,
        source_dist,
        target_dist,
        to_types,
        data_type='double',
        max_iter=1000,
        numItermax=1000,
        n_iter=100
    ) -> None:

        self.to_types = to_types
        self.data_type = data_type

        p = ot.unif(len(source_dist))
        q = ot.unif(len(target_dist))

        self.source_dist, self.target_dist, self.p, self.q = source_dist, target_dist, p, q

        self.source_size = len(source_dist)
        self.target_size = len(target_dist)

        # hyperparameter
        self.init_mat_builder = InitMatrix(self.source_size, self.target_size)  # 基本的に初期値はnumpyで作成するようにしておく。

        # gw alignmentに関わるparameter
        self.max_iter = max_iter

        # sinkhornに関わるparameter
        self.numItermax = numItermax

        # 初期値のiteration回数, かつ hyperbandのparameter
        self.n_iter = n_iter

        self.back_end = Backend("cpu", self.to_types, self.data_type)  # 並列計算をしない場合は、こちらにおいた方がはやい。(2023.4.19 佐々木)

    def entropic_gw(
        self,
        device,
        epsilon,
        T,
        tol=1e-9,
        trial=None,
        sinkhorn_method="sinkhorn",
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
        C1, C2, p, q, T = self.back_end(self.source_dist, self.target_dist, self.p, self.q, T)

        constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, loss_fun="square_loss")

        cpt = 0
        err = 1
        log = {"err": []}

        while err > tol and cpt < self.max_iter:
            Tprev = T
            tens = ot.gromov.gwggrad(constC, hC1, hC2, T)
            T = ot.bregman.sinkhorn(p, q, tens, epsilon, method=sinkhorn_method, numItermax=self.numItermax)

            if cpt % 10 == 0:
                err = self.back_end.nx.norm(T - Tprev)
                if log:
                    log["err"].append(err)
                if verbose:
                    if cpt % 200 == 0:
                        print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                    print("{:5d}|{:8e}|".format(cpt, err))
            cpt += 1

        log["gw_dist"] = ot.gromov.gwloss(constC, hC1, hC2, T)
        log["ot"] = T
        return log

    def gw_alignment_computation(
        self,
        init_mat,
        eps,
        device,
        trial=None,
        sinkhorn_method="sinkhorn",
    ):
        """
        2023.3.17 佐々木
        gw_alignmentの計算を行う。ここのメソッドは変更しない方がいいと思う。
        外部で、特定のhyper parametersでのgw_alignmentの計算結果だけを抽出したい時にも使えるため。
        """

        logv = self.entropic_gw(
            device,
            eps,
            init_mat,
            trial=trial,
            sinkhorn_method=sinkhorn_method,
        )

        if self.back_end.check_zeros(logv["ot"]):
            logv["gw_dist"] = float("nan")
            logv["acc"] = float("nan")

        else:
            pred = self.back_end.nx.argmax(logv["ot"], 1)
            correct = (pred == self.back_end.nx.arange(len(logv["ot"]), type_as=logv["ot"])).sum()
            logv["acc"] = correct / len(logv["ot"])

        return logv

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
        
        if num_iter is None:  # uniform, diagにおいて、nanにならなかったがprunerが動くときのためのifブロック。
            num_iter = self.n_iter

        if math.isinf(gw_loss) or gw_loss <= 0.0:
            raise optuna.TrialPruned(f"Trial for '{init_mat_plan}' was pruned with parameters: {{'eps': {eps:.5e}, 'gw_loss': '{gw_loss:.5e}'}}")

        trial.report(gw_loss, num_iter)

        if trial.should_prune():
            raise optuna.TrialPruned(
                f"Trial for '{init_mat_plan}' was pruned at iteration {num_iter} with parameters: {{'eps': {eps:.5e}, 'gw_loss': '{gw_loss:.5e}'}}"
            )

    def _compute_GW_with_init_plans(
        self,
        trial,
        init_mat_plan,
        eps,
        device,
        sinkhorn_method,
        num_iter=None,
        seed=None
    ):

        if init_mat_plan == "user_define":
            init_mat = seed
        else:
            init_mat = self.init_mat_builder.make_initial_T(init_mat_plan, seed)

        logv = self.gw_alignment_computation(
            init_mat,
            eps,
            device,
            sinkhorn_method=sinkhorn_method,
        )

        if init_mat_plan in ["uniform", "diag"]:
            best_flag = None
            trial = self._save_results(
                logv["gw_dist"],
                logv["acc"],
                trial,
                init_mat_plan,
            )

        elif init_mat_plan in ["random", "permutation", "user_define"]:
            if logv["gw_dist"] < self.best_gw_loss:
                best_flag = True
                self.best_gw_loss = logv["gw_dist"]

                trial = self._save_results(
                    logv["gw_dist"],
                    logv["acc"],
                    trial,
                    init_mat_plan,
                    num_iter=num_iter,
                    seed=seed,
                )

            else:
                best_flag = False

        self._check_pruner_should_work(
            logv["gw_dist"],
            trial,
            init_mat_plan,
            eps,
            num_iter=num_iter,
        )

        return logv, init_mat, trial, best_flag


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
            logv, init_mat, trial, _ = self._compute_GW_with_init_plans(
                trial,
                init_mat_plan,
                eps,
                device,
                sinkhorn_method,
            )
            return logv, init_mat, trial

        elif init_mat_plan in ["random", "permutation", "user_define"]:
            self.best_gw_loss = float("inf")

            if init_mat_plan in ["random", "permutation"]:
                pbar = tqdm(np.random.randint(0, 100000, self.n_iter))

            if init_mat_plan == "user_define":
                pbar = tqdm(self.init_mat_builder.user_define_init_mat_list)

            pbar.set_description(f"Trial No.{trial.number}, eps:{eps:.3e}")

            for i, seed in enumerate(pbar):
                logv, init_mat, trial, best_flag = self._compute_GW_with_init_plans(
                    trial,
                    init_mat_plan,
                    eps,
                    device,
                    sinkhorn_method,
                    num_iter=i,
                    seed=seed,
                )

                if best_flag:
                    best_logv, best_init_mat = logv, init_mat

            if self.best_gw_loss == float("inf"):
                raise optuna.TrialPruned(
                    f"All iteration was failed with parameters: {{'eps': {eps}, 'initialize': '{init_mat_plan}'}}"
                )

            else:
                return best_logv, best_init_mat, trial

        else:
            raise ValueError("Not defined initialize matrix.")
