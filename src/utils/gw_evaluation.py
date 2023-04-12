
# def load_graph(self, study):
#     best_trial = study.best_trial
#     eps = best_trial.params['eps']
#     acc = best_trial.user_attrs['acc']
#     size = best_trial.user_attrs['size']
#     number = best_trial.number

#     if self.to_types == 'torch':
#         gw = torch.load(self.save_path +f'/gw_{best_trial.number}.pt')
#     elif self.to_types == 'numpy':
#         gw = np.load(self.save_path +f'/gw_{best_trial.number}.npy')
#     # gw = torch.load(self.file_path + '/GW({} pictures, epsilon = {}, trial = {}).pt'.format(size, round(eps, 6), number))
#     self.plot_coupling(gw, eps, acc)

# def make_eval_graph(self, study):
#     df_test = study.trials_dataframe()
#     success_test = df_test[df_test['values_0'] != float('nan')]

#     plt.figure()
#     plt.title('The evaluation of GW results for random pictures')
#     plt.scatter(success_test['values_1'], np.log(success_test['values_0']), label = 'init diag plan ('+str(self.train_size)+')', c = 'C0')
#     plt.xlabel('accuracy')
#     plt.ylabel('log(GWD)')
#     plt.legend()
#     plt.show()

# def plot_coupling(self, T, epsilon, acc):
#     mplstyle.use('fast')
#     N = T.shape[0]
#     plt.figure(figsize=(8,6))
#     if self.to_types == 'torch':
#         T = T.to('cpu').numpy()
#     sns.heatmap(T)

#     plt.title('GW results ({} pictures, eps={}, acc.= {})'.format(N, round(epsilon, 6), round(acc, 4)))
#     plt.tight_layout()
#     plt.show()

# # %%
