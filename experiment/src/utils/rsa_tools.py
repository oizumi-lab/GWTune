#%%
# Standard Library
import itertools
import os

# Third Party Library
import numpy as np
import rsatoolbox as rsa

#%%
class RSA_Tools:
    def __init__(self, rdm_dict: dict):
        self.rdm_dict = rdm_dict

    def compare_rdms(self, method):
        n = len(self.rdm_dict)

        sms = []
        for i, rdm1 in enumerate(self.rdm_dict.values()):
            for j, rdm2 in enumerate(self.rdm_dict.values()):
                if j > i:
                    sm = rsa.rdm.compare(rdm1, rdm2)
                    sms =


        ids = itertools.combinations(self.rdm_dict.keys(), 2)


        for id1, id2 in ids:
            print(f'{id1}, {id2}')
        # sm = rsa.rdm.compare(method)
        return 0


#%%
dic = {}
for i in np.arange(10):
    dic[str(i)] = np.random.normal(0,1,(10,10))

a = RSA_Tools(dic)
a.compare_rdms('a')
# %%
