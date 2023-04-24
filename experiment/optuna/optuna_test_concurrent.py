# %%
import optuna
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import torch
import numpy as np
import torch.multiprocessing as mp

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

test_arr = torch.randn(10).to('cuda')
print(test_arr)

print('---------')

def objective(trial, test_arr):
    x = trial.suggest_float("x", -10, 10)
    return sum(test_arr * x)
 
study = optuna.create_study(study_name="my_study", storage = "sqlite:///cuda_test.db", load_if_exists = True)

def multi_run(seed, i, n_trials):
    device = 'cuda:0'#+str(i)
    test_arr = torch.randn(10).to(device)
    # sampler = optuna.samplers.RandomSampler(seed = seed + i)
    sampler = optuna.samplers.GridSampler(search_space, seed = seed + i)
    loaded_study = optuna.load_study(sampler = sampler, study_name="my_study", storage="sqlite:///cuda_test.db")
    
    loaded_study.optimize(lambda trial: objective(trial, test_arr), n_trials = n_trials)

processes = []
n_trials = 20

search_space = {'x':np.logspace(0, 1, num = n_trials)}

n_jobs = 4
seed = 42

with ThreadPoolExecutor(n_jobs) as pool:
    for i in range(n_jobs):
        pool.submit(multi_run, seed, i, n_trials // n_jobs)
        

processes = []
for i in range(n_jobs):
    p = mp.Process(target=multi_run, args=(seed, i, n_trials // n_jobs))
    processes.append(p)

# プロセスを実行する
for p in processes:
    p.start()

# # プロセスが終了するのを待つ
for p in processes:
    p.join()
# %%
    """
[32m[I 2023-03-31 18:47:37,426][0m A new study created in RDB with name: my_study[0m
[32m[I 2023-03-31 18:47:37,868][0m Trial 1 finished with value: 4.718250274658203 and parameters: {'x': -2.50919762305275}. Best is trial 2 with value: -12.592623710632324.[0m
[32m[I 2023-03-31 18:47:37,901][0m Trial 2 finished with value: -12.592623710632324 and parameters: {'x': 6.6968429733129895}. Best is trial 2 with value: -12.592623710632324.[0m
[32m[I 2023-03-31 18:47:37,914][0m Trial 0 finished with value: 14.476884841918945 and parameters: {'x': -7.698908672204421}. Best is trial 2 with value: -12.592623710632324.[0m
[32m[I 2023-03-31 18:47:37,943][0m Trial 3 finished with value: -18.39056396484375 and parameters: {'x': 9.780230269512003}. Best is trial 3 with value: -18.39056396484375.[0m
[32m[I 2023-03-31 18:47:37,982][0m Trial 5 finished with value: -4.101734161376953 and parameters: {'x': 2.181330785589628}. Best is trial 3 with value: -18.39056396484375.[0m
[32m[I 2023-03-31 18:47:38,054][0m Trial 4 finished with value: -16.950302124023438 and parameters: {'x': 9.014286128198322}. Best is trial 3 with value: -18.39056396484375.[0m
[32m[I 2023-03-31 18:47:38,068][0m Trial 6 finished with value: 13.787300109863281 and parameters: {'x': -7.332180716280234}. Best is trial 3 with value: -18.39056396484375.[0m
[32m[I 2023-03-31 18:47:38,085][0m Trial 8 finished with value: 14.862686157226562 and parameters: {'x': -7.904077912602605}. Best is trial 3 with value: -18.39056396484375.[0m
[32m[I 2023-03-31 18:47:38,110][0m Trial 7 finished with value: -1.8632605075836182 and parameters: {'x': 0.9908945368945083}. Best is trial 3 with value: -18.39056396484375.[0m
[32m[I 2023-03-31 18:47:38,182][0m Trial 9 finished with value: -8.72474479675293 and parameters: {'x': 4.639878836228101}. Best is trial 3 with value: -18.39056396484375.[0m
[32m[I 2023-03-31 18:47:38,187][0m Trial 11 finished with value: 9.75581169128418 and parameters: {'x': -5.1882076006930244}. Best is trial 3 with value: -18.39056396484375.[0m
[32m[I 2023-03-31 18:47:38,201][0m Trial 10 finished with value: -9.200352668762207 and parameters: {'x': 4.892809633590357}. Best is trial 3 with value: -18.39056396484375.[0m
[32m[I 2023-03-31 18:47:38,280][0m Trial 12 finished with value: 8.219252586364746 and parameters: {'x': -4.3710539608987435}. Best is trial 3 with value: -18.39056396484375.[0m
[32m[I 2023-03-31 18:47:38,284][0m Trial 14 finished with value: 6.50089168548584 and parameters: {'x': -3.4572188837772044}. Best is trial 3 with value: -18.39056396484375.[0m
[32m[I 2023-03-31 18:47:38,313][0m Trial 15 finished with value: 5.24623441696167 and parameters: {'x': -2.7899832748742863}. Best is trial 3 with value: -18.39056396484375.[0m
[32m[I 2023-03-31 18:47:38,332][0m Trial 13 finished with value: -3.710313081741333 and parameters: {'x': 1.973169683940732}. Best is trial 3 with value: -18.39056396484375.[0m
[32m[I 2023-03-31 18:47:38,387][0m Trial 16 finished with value: 5.29098653793335 and parameters: {'x': -2.813783243838561}. Best is trial 3 with value: -18.39056396484375.[0m
[32m[I 2023-03-31 18:47:38,392][0m Trial 18 finished with value: 12.936327934265137 and parameters: {'x': -6.87962719115127}. Best is trial 3 with value: -18.39056396484375.[0m
[32m[I 2023-03-31 18:47:38,420][0m Trial 17 finished with value: 15.89714241027832 and parameters: {'x': -8.454208678416546}. Best is trial 3 with value: -18.39056396484375.[0m
[32m[I 2023-03-31 18:47:38,498][0m Trial 19 finished with value: -4.108198165893555 and parameters: {'x': 2.184767612363034}. Best is trial 3 with value: -18.39056396484375.[0m
[32m[I 2023-03-31 18:47:38,523][0m Trial 20 finished with value: -13.506311416625977 and parameters: {'x': 7.182749818971953}. Best is trial 3 with value: -18.39056396484375.[0m
[32m[I 2023-03-31 18:47:38,562][0m Trial 21 finished with value: 2.0883712768554688 and parameters: {'x': -1.110610080563939}. Best is trial 3 with value: -18.39056396484375.[0m
[32m[I 2023-03-31 18:47:38,573][0m Trial 22 finished with value: 12.937235832214355 and parameters: {'x': -6.880109593275947}. Best is trial 3 with value: -18.39056396484375.[0m
[32m[I 2023-03-31 18:47:38,599][0m Trial 23 finished with value: 3.9946999549865723 and parameters: {'x': -2.124408978234973}. Best is trial 3 with value: -18.39056396484375.[0m
[32m[I 2023-03-31 18:47:38,625][0m Trial 24 finished with value: -6.246262073516846 and parameters: {'x': 3.3218042619605157}. Best is trial 3 with value: -18.39056396484375.[0m
[32m[I 2023-03-31 18:47:38,673][0m Trial 26 finished with value: 16.61943244934082 and parameters: {'x': -8.83832775663601}. Best is trial 3 with value: -18.39056396484375.[0m
[32m[I 2023-03-31 18:47:38,747][0m Trial 25 finished with value: 1.0226280689239502 and parameters: {'x': -0.5438406048902937}. Best is trial 3 with value: -18.39056396484375.[0m
[32m[I 2023-03-31 18:47:38,770][0m Trial 29 finished with value: -1.5480135679244995 and parameters: {'x': 0.8232442456680751}. Best is trial 3 with value: -18.39056396484375.[0m
[32m[I 2023-03-31 18:47:38,771][0m Trial 28 finished with value: -13.77101993560791 and parameters: {'x': 7.323522915498703}. Best is trial 3 with value: -18.39056396484375.[0m
[32m[I 2023-03-31 18:47:38,840][0m Trial 30 finished with value: 16.979022979736328 and parameters: {'x': -9.029559981871083}. Best is trial 3 with value: -18.39056396484375.[0m
[32m[I 2023-03-31 18:47:38,842][0m Trial 32 finished with value: 17.71268081665039 and parameters: {'x': -9.419723511512792}. Best is trial 3 with value: -18.39056396484375.[0m
[32m[I 2023-03-31 18:47:38,845][0m Trial 31 finished with value: -3.802696466445923 and parameters: {'x': 2.0223002348641756}. Best is trial 3 with value: -18.39056396484375.[0m
[32m[I 2023-03-31 18:47:38,850][0m Trial 27 finished with value: 3.4195644855499268 and parameters: {'x': -1.8185478027820263}. Best is trial 3 with value: -18.39056396484375.[0m
[32m[I 2023-03-31 18:47:38,920][0m Trial 34 finished with value: 12.661572456359863 and parameters: {'x': -6.733511035834665}. Best is trial 3 with value: -18.39056396484375.[0m
[32m[I 2023-03-31 18:47:38,955][0m Trial 36 finished with value: -0.3724062144756317 and parameters: {'x': 0.19804819179212885}. Best is trial 3 with value: -18.39056396484375.[0m
[32m[I 2023-03-31 18:47:39,001][0m Trial 33 finished with value: -8.790719985961914 and parameters: {'x': 4.674965925605658}. Best is trial 3 with value: -18.39056396484375.[0m
[32m[I 2023-03-31 18:47:39,005][0m Trial 35 finished with value: -7.825117588043213 and parameters: {'x': 4.161451555920909}. Best is trial 3 with value: -18.39056396484375.[0m
[32m[I 2023-03-31 18:47:39,009][0m Trial 37 finished with value: 14.443187713623047 and parameters: {'x': -7.6809857700411115}. Best is trial 3 with value: -18.39056396484375.[0m
[32m[I 2023-03-31 18:47:39,042][0m Trial 38 finished with value: -7.903170108795166 and parameters: {'x': 4.20295986654147}. Best is trial 3 with value: -18.39056396484375.[0m
[32m[I 2023-03-31 18:47:39,068][0m Trial 39 finished with value: -4.790901184082031 and parameters: {'x': 2.547833654088521}. Best is trial 3 with value: -18.39056396484375.[0m

    """
    
"""
[32m[I 2023-03-31 18:48:14,534][0m A new study created in RDB with name: my_study[0m
[32m[I 2023-03-31 18:48:14,732][0m Trial 1 finished with value: 12.65576457977295 and parameters: {'x': 6.6968429733129895}. Best is trial 1 with value: 12.65576457977295.[0m
[32m[I 2023-03-31 18:48:14,902][0m Trial 3 finished with value: -14.937210083007812 and parameters: {'x': -7.904077912602605}. Best is trial 3 with value: -14.937210083007812.[0m
[32m[I 2023-03-31 18:48:14,928][0m Trial 0 finished with value: -4.741909027099609 and parameters: {'x': -2.50919762305275}. Best is trial 0 with value: -4.741909027099609.[0m
[32m[I 2023-03-31 18:48:14,948][0m Trial 2 finished with value: 18.482786178588867 and parameters: {'x': 9.780230269512003}. Best is trial 3 with value: -14.937210083007812.[0m
[32m[I 2023-03-31 18:48:15,003][0m Trial 5 finished with value: 9.246484756469727 and parameters: {'x': 4.892809633590357}. Best is trial 3 with value: -14.937210083007812.[0m
[32m[I 2023-03-31 18:48:15,046][0m Trial 7 finished with value: 1.872603178024292 and parameters: {'x': 0.9908945368945083}. Best is trial 3 with value: -14.937210083007812.[0m
[32m[I 2023-03-31 18:48:15,092][0m Trial 6 finished with value: 17.035293579101562 and parameters: {'x': 9.014286128198322}. Best is trial 3 with value: -14.937210083007812.[0m
[32m[I 2023-03-31 18:48:15,097][0m Trial 4 finished with value: -14.549480438232422 and parameters: {'x': -7.698908672204421}. Best is trial 3 with value: -14.937210083007812.[0m
[32m[I 2023-03-31 18:48:15,100][0m Trial 8 finished with value: -5.27254056930542 and parameters: {'x': -2.7899832748742863}. Best is trial 3 with value: -14.937210083007812.[0m
[32m[I 2023-03-31 18:48:15,141][0m Trial 9 finished with value: -5.3175177574157715 and parameters: {'x': -2.813783243838561}. Best is trial 3 with value: -14.937210083007812.[0m
[32m[I 2023-03-31 18:48:15,179][0m Trial 11 finished with value: 8.768492698669434 and parameters: {'x': 4.639878836228101}. Best is trial 3 with value: -14.937210083007812.[0m
[32m[I 2023-03-31 18:48:15,210][0m Trial 10 finished with value: 4.12879753112793 and parameters: {'x': 2.184767612363034}. Best is trial 3 with value: -14.937210083007812.[0m
[32m[I 2023-03-31 18:48:15,240][0m Trial 13 finished with value: -8.260465621948242 and parameters: {'x': -4.3710539608987435}. Best is trial 3 with value: -14.937210083007812.[0m
[32m[I 2023-03-31 18:48:15,253][0m Trial 12 finished with value: 3.728917121887207 and parameters: {'x': 1.973169683940732}. Best is trial 3 with value: -14.937210083007812.[0m
[32m[I 2023-03-31 18:48:15,284][0m Trial 14 finished with value: -4.014730930328369 and parameters: {'x': -2.124408978234973}. Best is trial 3 with value: -14.937210083007812.[0m
[32m[I 2023-03-31 18:48:15,333][0m Trial 16 finished with value: -15.976852416992188 and parameters: {'x': -8.454208678416546}. Best is trial 16 with value: -15.976852416992188.[0m
[32m[I 2023-03-31 18:48:15,346][0m Trial 15 finished with value: -3.436711311340332 and parameters: {'x': -1.8185478027820263}. Best is trial 3 with value: -14.937210083007812.[0m
[32m[I 2023-03-31 18:48:15,371][0m Trial 17 finished with value: -13.001192092895508 and parameters: {'x': -6.87962719115127}. Best is trial 16 with value: -15.976852416992188.[0m
[32m[I 2023-03-31 18:48:15,435][0m Trial 20 finished with value: -13.002105712890625 and parameters: {'x': -6.880109593275947}. Best is trial 16 with value: -15.976852416992188.[0m
[32m[I 2023-03-31 18:48:15,484][0m Trial 21 finished with value: -16.70276641845703 and parameters: {'x': -8.83832775663601}. Best is trial 21 with value: -16.70276641845703.[0m
[32m[I 2023-03-31 18:48:15,491][0m Trial 18 finished with value: -2.0988430976867676 and parameters: {'x': -1.110610080563939}. Best is trial 16 with value: -15.976852416992188.[0m
[32m[I 2023-03-31 18:48:15,501][0m Trial 22 finished with value: 4.122302055358887 and parameters: {'x': 2.181330785589628}. Best is trial 16 with value: -15.976852416992188.[0m
[32m[I 2023-03-31 18:48:15,534][0m Trial 23 finished with value: -1.0277557373046875 and parameters: {'x': -0.5438406048902937}. Best is trial 21 with value: -16.70276641845703.[0m
[32m[I 2023-03-31 18:48:15,579][0m Trial 24 finished with value: 13.840071678161621 and parameters: {'x': 7.323522915498703}. Best is trial 21 with value: -16.70276641845703.[0m
[32m[I 2023-03-31 18:48:15,591][0m Trial 19 finished with value: 0.3742735981941223 and parameters: {'x': 0.19804819179212885}. Best is trial 21 with value: -16.70276641845703.[0m
[32m[I 2023-03-31 18:48:15,665][0m Trial 27 finished with value: 7.942798137664795 and parameters: {'x': 4.20295986654147}. Best is trial 21 with value: -16.70276641845703.[0m
[32m[I 2023-03-31 18:48:15,727][0m Trial 26 finished with value: -17.064159393310547 and parameters: {'x': -9.029559981871083}. Best is trial 26 with value: -17.064159393310547.[0m
[32m[I 2023-03-31 18:48:15,732][0m Trial 25 finished with value: -13.856433868408203 and parameters: {'x': -7.332180716280234}. Best is trial 26 with value: -17.064159393310547.[0m
[32m[I 2023-03-31 18:48:15,735][0m Trial 28 finished with value: 3.8217649459838867 and parameters: {'x': 2.0223002348641756}. Best is trial 26 with value: -17.064159393310547.[0m
[32m[I 2023-03-31 18:48:15,802][0m Trial 29 finished with value: -12.72506046295166 and parameters: {'x': -6.733511035834665}. Best is trial 26 with value: -17.064159393310547.[0m
[32m[I 2023-03-31 18:48:15,811][0m Trial 30 finished with value: -9.804730415344238 and parameters: {'x': -5.1882076006930244}. Best is trial 26 with value: -17.064159393310547.[0m
[32m[I 2023-03-31 18:48:15,855][0m Trial 31 finished with value: 7.86435604095459 and parameters: {'x': 4.161451555920909}. Best is trial 26 with value: -17.064159393310547.[0m
[32m[I 2023-03-31 18:48:15,903][0m Trial 33 finished with value: -6.533489227294922 and parameters: {'x': -3.4572188837772044}. Best is trial 26 with value: -17.064159393310547.[0m
[32m[I 2023-03-31 18:48:15,906][0m Trial 32 finished with value: -14.515609741210938 and parameters: {'x': -7.6809857700411115}. Best is trial 26 with value: -17.064159393310547.[0m
[32m[I 2023-03-31 18:48:15,964][0m Trial 34 finished with value: 4.814923286437988 and parameters: {'x': 2.547833654088521}. Best is trial 26 with value: -17.064159393310547.[0m
[32m[I 2023-03-31 18:48:15,970][0m Trial 35 finished with value: 13.574039459228516 and parameters: {'x': 7.182749818971953}. Best is trial 26 with value: -17.064159393310547.[0m
[32m[I 2023-03-31 18:48:15,996][0m Trial 36 finished with value: 6.277581214904785 and parameters: {'x': 3.3218042619605157}. Best is trial 26 with value: -17.064159393310547.[0m
[32m[I 2023-03-31 18:48:16,022][0m Trial 37 finished with value: 1.5557758808135986 and parameters: {'x': 0.8232442456680751}. Best is trial 26 with value: -17.064159393310547.[0m
[32m[I 2023-03-31 18:48:16,056][0m Trial 38 finished with value: -17.801494598388672 and parameters: {'x': -9.419723511512792}. Best is trial 38 with value: -17.801494598388672.[0m
[32m[I 2023-03-31 18:48:16,078][0m Trial 39 finished with value: 8.834800720214844 and parameters: {'x': 4.674965925605658}. Best is trial 38 with value: -17.801494598388672.[0m

"""

"""
SamplerをTPEにした場合。
[32m[I 2023-04-04 17:29:15,426][0m A new study created in RDB with name: my_study[0m
[32m[I 2023-04-04 17:29:15,955][0m Trial 0 finished with value: -14.784391403198242 and parameters: {'x': -7.698908672204421}. Best is trial 0 with value: -14.784391403198242.[0m
[32m[I 2023-04-04 17:29:15,971][0m Trial 1 finished with value: 18.78120231628418 and parameters: {'x': 9.780230269512003}. Best is trial 0 with value: -14.784391403198242.[0m
[32m[I 2023-04-04 17:29:16,040][0m Trial 3 finished with value: -4.818470001220703 and parameters: {'x': -2.50919762305275}. Best is trial 0 with value: -14.784391403198242.[0m
[32m[I 2023-04-04 17:29:16,058][0m Trial 2 finished with value: 12.860101699829102 and parameters: {'x': 6.6968429733129895}. Best is trial 0 with value: -14.784391403198242.[0m
[32m[I 2023-04-04 17:29:16,139][0m Trial 4 finished with value: 4.188859939575195 and parameters: {'x': 2.181330785589628}. Best is trial 0 with value: -14.784391403198242.[0m
[32m[I 2023-04-04 17:29:16,141][0m Trial 5 finished with value: 1.9028377532958984 and parameters: {'x': 0.9908945368945083}. Best is trial 0 with value: -14.784391403198242.[0m
[32m[I 2023-04-04 17:29:16,227][0m Trial 6 finished with value: 17.310340881347656 and parameters: {'x': 9.014286128198322}. Best is trial 7 with value: -15.178382873535156.[0m
[32m[I 2023-04-04 17:29:16,271][0m Trial 7 finished with value: -15.178382873535156 and parameters: {'x': -7.904077912602605}. Best is trial 7 with value: -15.178382873535156.[0m
[32m[I 2023-04-04 17:29:16,300][0m Trial 8 finished with value: -14.080155372619629 and parameters: {'x': -7.332180716280234}. Best is trial 7 with value: -15.178382873535156.[0m
[32m[I 2023-04-04 17:29:16,335][0m Trial 9 finished with value: -8.393835067749023 and parameters: {'x': -4.3710539608987435}. Best is trial 7 with value: -15.178382873535156.[0m
[32m[I 2023-04-04 17:29:16,462][0m Trial 13 finished with value: -17.484174728393555 and parameters: {'x': -9.10480868726457}. Best is trial 10 with value: -18.40508460998535.[0m
[32m[I 2023-04-04 17:29:16,490][0m Trial 10 finished with value: -18.40508460998535 and parameters: {'x': -9.584368990623158}. Best is trial 10 with value: -18.40508460998535.[0m
[32m[I 2023-04-04 17:29:16,504][0m Trial 12 finished with value: -18.9892520904541 and parameters: {'x': -9.888571378201192}. Best is trial 12 with value: -18.9892520904541.[0m
[32m[I 2023-04-04 17:29:16,505][0m Trial 11 finished with value: -16.813541412353516 and parameters: {'x': -8.755579717761465}. Best is trial 12 with value: -18.9892520904541.[0m
[32m[I 2023-04-04 17:29:16,652][0m Trial 14 finished with value: -9.206493377685547 and parameters: {'x': -4.794242265355829}. Best is trial 12 with value: -18.9892520904541.[0m
[32m[I 2023-04-04 17:29:16,669][0m Trial 16 finished with value: -19.14096450805664 and parameters: {'x': -9.967575141461143}. Best is trial 16 with value: -19.14096450805664.[0m
[32m[I 2023-04-04 17:29:16,710][0m Trial 17 finished with value: -19.135587692260742 and parameters: {'x': -9.964776126801826}. Best is trial 16 with value: -19.14096450805664.[0m
[32m[I 2023-04-04 17:29:16,750][0m Trial 15 finished with value: -18.784616470336914 and parameters: {'x': -9.78200852801233}. Best is trial 16 with value: -19.14096450805664.[0m
[32m[I 2023-04-04 17:29:16,872][0m Trial 18 finished with value: -11.030925750732422 and parameters: {'x': -5.744307312842218}. Best is trial 16 with value: -19.14096450805664.[0m
[32m[I 2023-04-04 17:29:16,893][0m Trial 19 finished with value: -11.30757999420166 and parameters: {'x': -5.888374211078641}. Best is trial 16 with value: -19.14096450805664.[0m
[32m[I 2023-04-04 17:29:16,907][0m Trial 20 finished with value: -10.573952674865723 and parameters: {'x': -5.506340218851047}. Best is trial 16 with value: -19.14096450805664.[0m
[32m[I 2023-04-04 17:29:16,973][0m Trial 21 finished with value: -12.499029159545898 and parameters: {'x': -6.508815610988999}. Best is trial 16 with value: -19.14096450805664.[0m
[32m[I 2023-04-04 17:29:17,046][0m Trial 22 finished with value: -18.88240623474121 and parameters: {'x': -9.83293336006993}. Best is trial 16 with value: -19.14096450805664.[0m
[32m[I 2023-04-04 17:29:17,076][0m Trial 24 finished with value: -19.04241371154785 and parameters: {'x': -9.916255082727362}. Best is trial 16 with value: -19.14096450805664.[0m
[32m[I 2023-04-04 17:29:17,115][0m Trial 23 finished with value: -18.769529342651367 and parameters: {'x': -9.774151918029865}. Best is trial 16 with value: -19.14096450805664.[0m
[32m[I 2023-04-04 17:29:17,118][0m Trial 25 finished with value: -15.093816757202148 and parameters: {'x': -7.860041705341492}. Best is trial 16 with value: -19.14096450805664.[0m
[32m[I 2023-04-04 17:29:17,255][0m Trial 28 finished with value: -6.4926910400390625 and parameters: {'x': -3.381040773757192}. Best is trial 16 with value: -19.14096450805664.[0m
[32m[I 2023-04-04 17:29:17,259][0m Trial 27 finished with value: -7.292077541351318 and parameters: {'x': -3.7973186185572523}. Best is trial 16 with value: -19.14096450805664.[0m
[32m[I 2023-04-04 17:29:17,280][0m Trial 26 finished with value: -6.466975212097168 and parameters: {'x': -3.3676497110484735}. Best is trial 16 with value: -19.14096450805664.[0m
[32m[I 2023-04-04 17:29:17,300][0m Trial 29 finished with value: -6.2885050773620605 and parameters: {'x': -3.2747121616681367}. Best is trial 16 with value: -19.14096450805664.[0m
[32m[I 2023-04-04 17:29:17,447][0m Trial 30 finished with value: -14.241615295410156 and parameters: {'x': -7.416260287497812}. Best is trial 16 with value: -19.14096450805664.[0m
[32m[I 2023-04-04 17:29:17,456][0m Trial 31 finished with value: -13.607038497924805 and parameters: {'x': -7.085807117678533}. Best is trial 16 with value: -19.14096450805664.[0m
[32m[I 2023-04-04 17:29:17,485][0m Trial 32 finished with value: -14.127785682678223 and parameters: {'x': -7.356983957954218}. Best is trial 16 with value: -19.14096450805664.[0m
[32m[I 2023-04-04 17:29:17,489][0m Trial 33 finished with value: -13.858644485473633 and parameters: {'x': -7.21683017540972}. Best is trial 16 with value: -19.14096450805664.[0m
[32m[I 2023-04-04 17:29:17,629][0m Trial 34 finished with value: -17.205699920654297 and parameters: {'x': -8.959795095082113}. Best is trial 16 with value: -19.14096450805664.[0m
[32m[I 2023-04-04 17:29:17,659][0m Trial 35 finished with value: -16.281858444213867 and parameters: {'x': -8.478707189171878}. Best is trial 16 with value: -19.14096450805664.[0m
[32m[I 2023-04-04 17:29:17,672][0m Trial 37 finished with value: -15.697824478149414 and parameters: {'x': -8.17457484064348}. Best is trial 16 with value: -19.14096450805664.[0m
[32m[I 2023-04-04 17:29:17,690][0m Trial 36 finished with value: -16.226194381713867 and parameters: {'x': -8.449721767760654}. Best is trial 16 with value: -19.14096450805664.[0m
[32m[I 2023-04-04 17:29:17,760][0m Trial 38 finished with value: -3.42380690574646 and parameters: {'x': -1.7829327945157853}. Best is trial 16 with value: -19.14096450805664.[0m
[32m[I 2023-04-04 17:29:17,764][0m Trial 39 finished with value: -4.1305036544799805 and parameters: {'x': -2.150942157947353}. Best is trial 16 with value: -19.14096450805664.[0m


[32m[I 2023-04-04 17:30:09,055][0m A new study created in RDB with name: my_study[0m
[32m[I 2023-04-04 17:30:09,601][0m Trial 1 finished with value: 18.78120231628418 and parameters: {'x': 9.780230269512003}. Best is trial 0 with value: -14.784391403198242.[0m
[32m[I 2023-04-04 17:30:09,608][0m Trial 0 finished with value: -14.784391403198242 and parameters: {'x': -7.698908672204421}. Best is trial 0 with value: -14.784391403198242.[0m
[32m[I 2023-04-04 17:30:09,644][0m Trial 2 finished with value: -4.818470001220703 and parameters: {'x': -2.50919762305275}. Best is trial 0 with value: -14.784391403198242.[0m
[32m[I 2023-04-04 17:30:09,670][0m Trial 3 finished with value: 12.860101699829102 and parameters: {'x': 6.6968429733129895}. Best is trial 0 with value: -14.784391403198242.[0m
[32m[I 2023-04-04 17:30:09,770][0m Trial 4 finished with value: 1.9028377532958984 and parameters: {'x': 0.9908945368945083}. Best is trial 0 with value: -14.784391403198242.[0m
[32m[I 2023-04-04 17:30:09,791][0m Trial 5 finished with value: 4.188859939575195 and parameters: {'x': 2.181330785589628}. Best is trial 0 with value: -14.784391403198242.[0m
[32m[I 2023-04-04 17:30:09,863][0m Trial 7 finished with value: -15.178382873535156 and parameters: {'x': -7.904077912602605}. Best is trial 7 with value: -15.178382873535156.[0m
[32m[I 2023-04-04 17:30:09,903][0m Trial 8 finished with value: -8.393835067749023 and parameters: {'x': -4.3710539608987435}. Best is trial 7 with value: -15.178382873535156.[0m
[32m[I 2023-04-04 17:30:09,931][0m Trial 9 finished with value: -14.080155372619629 and parameters: {'x': -7.332180716280234}. Best is trial 7 with value: -15.178382873535156.[0m
[32m[I 2023-04-04 17:30:09,952][0m Trial 6 finished with value: 17.310340881347656 and parameters: {'x': 9.014286128198322}. Best is trial 7 with value: -15.178382873535156.[0m
[32m[I 2023-04-04 17:30:10,088][0m Trial 11 finished with value: -17.484174728393555 and parameters: {'x': -9.10480868726457}. Best is trial 11 with value: -17.484174728393555.[0m
[32m[I 2023-04-04 17:30:10,109][0m Trial 10 finished with value: -16.813541412353516 and parameters: {'x': -8.755579717761465}. Best is trial 11 with value: -17.484174728393555.[0m
[32m[I 2023-04-04 17:30:10,165][0m Trial 12 finished with value: -18.9892520904541 and parameters: {'x': -9.888571378201192}. Best is trial 12 with value: -18.9892520904541.[0m
[32m[I 2023-04-04 17:30:10,179][0m Trial 13 finished with value: -18.72496223449707 and parameters: {'x': -9.750944188037604}. Best is trial 12 with value: -18.9892520904541.[0m
[32m[I 2023-04-04 17:30:10,271][0m Trial 14 finished with value: -18.72040557861328 and parameters: {'x': -9.748570561997303}. Best is trial 12 with value: -18.9892520904541.[0m
[32m[I 2023-04-04 17:30:10,334][0m Trial 15 finished with value: -9.53722858428955 and parameters: {'x': -4.966471800062036}. Best is trial 12 with value: -18.9892520904541.[0m
[32m[I 2023-04-04 17:30:10,365][0m Trial 17 finished with value: -10.620771408081055 and parameters: {'x': -5.530721247071516}. Best is trial 12 with value: -18.9892520904541.[0m
[32m[I 2023-04-04 17:30:10,404][0m Trial 16 finished with value: -10.833944320678711 and parameters: {'x': -5.641730123296563}. Best is trial 12 with value: -18.9892520904541.[0m
[32m[I 2023-04-04 17:30:10,500][0m Trial 18 finished with value: -4.376008033752441 and parameters: {'x': -2.278787309974457}. Best is trial 12 with value: -18.9892520904541.[0m
[32m[I 2023-04-04 17:30:10,538][0m Trial 19 finished with value: -6.688595294952393 and parameters: {'x': -3.4830576655943934}. Best is trial 12 with value: -18.9892520904541.[0m
[32m[I 2023-04-04 17:30:10,547][0m Trial 20 finished with value: -18.91336441040039 and parameters: {'x': -9.84905361367414}. Best is trial 12 with value: -18.9892520904541.[0m
[32m[I 2023-04-04 17:30:10,624][0m Trial 21 finished with value: -19.11369514465332 and parameters: {'x': -9.953375900121488}. Best is trial 21 with value: -19.11369514465332.[0m
[32m[I 2023-04-04 17:30:10,744][0m Trial 24 finished with value: -13.1964693069458 and parameters: {'x': -6.872005487905397}. Best is trial 21 with value: -19.11369514465332.[0m
[32m[I 2023-04-04 17:30:10,753][0m Trial 23 finished with value: -13.874442100524902 and parameters: {'x': -7.225056684635605}. Best is trial 21 with value: -19.11369514465332.[0m
[32m[I 2023-04-04 17:30:10,765][0m Trial 22 finished with value: -13.367324829101562 and parameters: {'x': -6.96097693991328}. Best is trial 21 with value: -19.11369514465332.[0m
[32m[I 2023-04-04 17:30:10,849][0m Trial 26 finished with value: -18.963909149169922 and parameters: {'x': -9.875374324433638}. Best is trial 21 with value: -19.11369514465332.[0m
[32m[I 2023-04-04 17:30:10,875][0m Trial 25 finished with value: -18.175926208496094 and parameters: {'x': -9.465035275042029}. Best is trial 21 with value: -19.11369514465332.[0m
[32m[I 2023-04-04 17:30:10,927][0m Trial 28 finished with value: -18.679645538330078 and parameters: {'x': -9.727345205332561}. Best is trial 21 with value: -19.11369514465332.[0m
[32m[I 2023-04-04 17:30:10,937][0m Trial 29 finished with value: -11.983743667602539 and parameters: {'x': -6.240482955060525}. Best is trial 21 with value: -19.11369514465332.[0m
[32m[I 2023-04-04 17:30:10,938][0m Trial 27 finished with value: -17.97249412536621 and parameters: {'x': -9.359100169212267}. Best is trial 21 with value: -19.11369514465332.[0m
[32m[I 2023-04-04 17:30:11,088][0m Trial 32 finished with value: -16.400169372558594 and parameters: {'x': -8.54031710204144}. Best is trial 21 with value: -19.11369514465332.[0m
[32m[I 2023-04-04 17:30:11,130][0m Trial 30 finished with value: -14.686637878417969 and parameters: {'x': -7.648003753576977}. Best is trial 21 with value: -19.11369514465332.[0m
[32m[I 2023-04-04 17:30:11,142][0m Trial 31 finished with value: -14.502996444702148 and parameters: {'x': -7.552372228586541}. Best is trial 21 with value: -19.11369514465332.[0m
[32m[I 2023-04-04 17:30:11,144][0m Trial 33 finished with value: -15.638079643249512 and parameters: {'x': -8.143463666786555}. Best is trial 21 with value: -19.11369514465332.[0m
[32m[I 2023-04-04 17:30:11,304][0m Trial 34 finished with value: -19.05310821533203 and parameters: {'x': -9.921824175014882}. Best is trial 21 with value: -19.11369514465332.[0m
[32m[I 2023-04-04 17:30:11,317][0m Trial 37 finished with value: -15.965682983398438 and parameters: {'x': -8.314060924635374}. Best is trial 21 with value: -19.11369514465332.[0m
[32m[I 2023-04-04 17:30:11,318][0m Trial 35 finished with value: -18.975284576416016 and parameters: {'x': -9.88129821851258}. Best is trial 21 with value: -19.11369514465332.[0m
[32m[I 2023-04-04 17:30:11,345][0m Trial 36 finished with value: -19.14128875732422 and parameters: {'x': -9.96774511626159}. Best is trial 36 with value: -19.14128875732422.[0m
[32m[I 2023-04-04 17:30:11,401][0m Trial 38 finished with value: -12.2225980758667 and parameters: {'x': -6.36486586803369}. Best is trial 36 with value: -19.14128875732422.[0m
[32m[I 2023-04-04 17:30:11,416][0m Trial 39 finished with value: -12.320055961608887 and parameters: {'x': -6.415616767079081}. Best is trial 36 with value: -19.14128875732422.[0m

"""