import sys
from itertools import product
from copy import deepcopy
from environment import Env, HomoEnv, VirtEnv
from accesspoints import Cluster, MonoCluster, Ap
from agent import Agent, HomoAgent
import numpy as np
import pickle as pk
from datetime import datetime


if __name__ == "__main__":
    argv = [int(v) for v in sys.argv[1:]]
    N, E, R, cap, num_save, num_sample, run_num = argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6]
    Slist = [1 for i in range(E)]
    c = np.array([[cap for i in range(R-1)]])
    c_b = cap

    sum_util_ks_rand, sum_util_ks_max, sum_util_mnw_eq, sum_util_ks_eq, sum_util_eq, sum_util_virtual_ks, sum_util_virtual_norm_utilitarian,\
         sum_norm_util_ks_rand, sum_norm_util_ks_max, sum_norm_util_mnw_eq, sum_norm_util_ks_eq, sum_norm_util_eq, sum_norm_util_virtual_ks, sum_norm_util_virtual_norm_utilitarian,\
             e_ks_rand, e_ks_max, e_mnw_eq, e_ks_eq, e_eq = ([0 for i in range(num_sample)] for j in range(19))
    res_util_ks_rand, res_util_ks_max, res_util_mnw_eq, res_util_ks_eq, res_util_eq  = ([[0, 0, 0] for i in range(num_sample)] for j in range(5))
    res_util_virtual_ks, res_util_virtual_norm_utilitarian = ([0 for i in range(num_sample)] for j in range(2))

    # Start the routine
    for save_iter in range(num_save):
        print(f"save iter: {save_iter}")
        for sample_iter in range(num_sample):
            print(f"sample iter: {sample_iter}")
            # Making cluster, ap, and agents
            clusters = []
            aps = []
            for i in range(E):
                clusters.append(MonoCluster(i, c[::]))
                aps.append(Ap(i, clusters[i], c_b))
            agents = []
            for i in range(N):
                agents.append(HomoAgent(i,
                                        np.random.randint(low=1, high=cap, size=(1, R)),
                                        aps))
            # Creating various copies for each environment
            env_ks_rand, env_ks_max, env_mnw_eq, env_ks_eq, env_eq, env_pool_ks, env_pool_norm_utilitarian = (HomoEnv(deepcopy(agents), deepcopy(aps)) for i in range(7))

            # MAGIKS WITH PICKING ONE UNHAPPY AGENT RANDOMLY
            print(f"starting ks rand at {datetime.now().hour}:{datetime.now().minute}")
            env_ks_rand.initialize(how='balanced', opt='ks')
            env_ks_rand.append_agents_pref(opt='ks')
            env_ks_rand.find_eq(opt='ks', agent_how='rand', ap_how='rand')
            env_ks_rand.set_homo_total_utilization_percent()
            res_util_ks_rand[sample_iter] = env_ks_rand.util_percent
            sum_util_ks_rand[sample_iter] = np.sum([ag.x for ag in env_ks_rand.agents])
            sum_norm_util_ks_rand[sample_iter] = np.sum([ag.x/np.sum(list(ag.max_task.values())) for ag in env_ks_rand.agents])
            e_ks_rand[sample_iter] = env_ks_rand.max_intra_envy_ratio()
            
            # MAGIKS WITH PICKING ONE UNHAPPY AGENT WITH MAXIMUM IMPROVEMENT
            print(f"starting ks max at {datetime.now().hour}:{datetime.now().minute}")
            env_ks_max.initialize(how='balanced', opt='ks')
            env_ks_max.append_agents_pref(opt='ks')
            env_ks_max.find_eq(opt='ks', agent_how='max_ratio', ap_how='rand')
            env_ks_max.set_homo_total_utilization_percent()
            res_util_ks_max[sample_iter] = env_ks_max.util_percent
            sum_util_ks_max[sample_iter] = np.sum([ag.x for ag in env_ks_max.agents])
            sum_norm_util_ks_max[sample_iter] = np.sum([ag.x/np.sum(list(ag.max_task.values())) for ag in env_ks_max.agents])
            e_ks_max[sample_iter] = env_ks_max.max_intra_envy_ratio()
            
            # MAGIKS WITH PICKING ONE UNHAPPY AGENT WITH MINIMUM IMPROVEMENT
            print(f"starting ks_eq rand at {datetime.now().hour}:{datetime.now().minute}")
            env_ks_eq.initialize(how='balanced', opt='ks')
            env_ks_eq.set_homo_total_utilization_percent()
            res_util_ks_eq[sample_iter] = env_ks_eq.util_percent
            sum_util_ks_eq[sample_iter] = np.sum([ag.x for ag in env_ks_eq.agents])
            sum_norm_util_ks_eq[sample_iter] = np.sum([ag.x/np.sum(list(ag.max_task.values())) for ag in env_ks_eq.agents])
            e_ks_eq[sample_iter] = env_ks_eq.max_intra_envy_ratio()
            
            # MNW WITH PICKING ONE UNHAPPY AGENT RANDOMLY
            print(f"starting mnw_eq rand at {datetime.now().hour}:{datetime.now().minute}")
            env_mnw_eq.initialize(how='balanced', opt='mnw')
            env_mnw_eq.set_homo_total_utilization_percent()
            res_util_mnw_eq[sample_iter] = env_mnw_eq.util_percent
            sum_util_mnw_eq[sample_iter] = np.sum([ag.x for ag in env_mnw_eq.agents])
            sum_norm_util_mnw_eq[sample_iter] = np.sum([ag.x/np.sum(list(ag.max_task.values())) for ag in env_mnw_eq.agents])
            e_mnw_eq[sample_iter] = env_mnw_eq.max_intra_envy_ratio()
            
            # EQUAL DinISION
            print(f"starting eq at {datetime.now().hour}:{datetime.now().minute}")
            env_eq.initialize(how='balanced', opt='EQ')
            env_eq.set_homo_total_utilization_percent()
            res_util_eq[sample_iter] = env_eq.util_percent
            sum_util_eq[sample_iter] = np.sum([ag.x for ag in env_eq.agents])
            sum_norm_util_eq[sample_iter] = np.sum([ag.x/np.sum(list(ag.max_task.values())) for ag in env_eq.agents])
            e_eq[sample_iter] = env_eq.max_intra_envy_ratio()
            
            # KS WITH VIRTUALLY POOLED SERVERS
            print(f"starting pool ks at {datetime.now().hour}:{datetime.now().minute}")
            env_pool_ks.initialize(how='given', opt='ks', init_loc=[0 for i in range(N)])
            env_pool_ks.aps[0].set_mets()
            env_pool_ks.set_homo_total_utilization_percent()
            res_util_virtual_ks[sample_iter] = env_pool_ks.util_percent
            sum_util_virtual_ks[sample_iter] = np.sum([ag.x for ag in env_pool_ks.agents])
            sum_norm_util_virtual_ks[sample_iter] = np.sum([ag.x/np.sum(list(ag.max_task.values())) for ag in env_pool_ks.agents])
            
            # MAXIMIZING SUM OF NORMALIZED UTILITY WITH VIRTUALLY POOLED SERVERS
            print(f"starting pool opt at {datetime.now().hour}:{datetime.now().minute}")
            env_pool_norm_utilitarian.initialize(how='given', opt='mnw', init_loc=[0 for i in range(N)])
            env_pool_norm_utilitarian.aps[0].norm_utilitarian()
            env_pool_norm_utilitarian.set_homo_total_utilization_percent()
            res_util_virtual_norm_utilitarian[sample_iter] = env_pool_norm_utilitarian.util_percent
            sum_util_virtual_norm_utilitarian[sample_iter] = np.sum([ag.x for ag in env_pool_norm_utilitarian.agents])
            sum_norm_util_virtual_norm_utilitarian[sample_iter] = np.sum([ag.x/np.sum(list(ag.max_task.values())) for ag in env_pool_norm_utilitarian.agents])

        name = f"./simulation_result/{N}_{E}_{R}_{cap}_{save_iter}_{run_num}_no_mnw.pickle"
        save_dic = {'res_util':{'ks_rand': res_util_ks_rand,
                                'ks_max': res_util_ks_max,
                                'ks_eq': res_util_ks_eq,
                                'mnw_eq': res_util_mnw_eq,
                                'eq': res_util_eq,
                                'pool_opt': res_util_virtual_norm_utilitarian,
                                'pool_ks': res_util_virtual_ks},
                    'max_intra_e': {'ks_rand': e_ks_rand,
                                    'ks_max': e_ks_max,
                                    'ks_eq': e_ks_eq,
                                    'mnw_eq': e_mnw_eq,
                                    'eq': e_eq},
                    'sum_util':{'ks_rand': sum_util_ks_rand,
                                'ks_max': sum_util_ks_max,
                                'ks_eq': sum_util_ks_eq,
                                'mnw_eq': sum_util_mnw_eq,
                                'eq': sum_util_eq,
                                'pool_opt': sum_util_virtual_norm_utilitarian,
                                'pool_ks': sum_util_virtual_ks},
                    'sum_norm_util':{'ks_rand': sum_norm_util_ks_rand,
                                     'ks_max': sum_norm_util_ks_max,
                                     'ks_eq': sum_norm_util_ks_eq,
                                     'mnw_eq': sum_norm_util_mnw_eq,
                                     'eq': sum_norm_util_eq,
                                     'pool_ks': sum_norm_util_virtual_ks,
                                     'pool_opt': sum_norm_util_virtual_norm_utilitarian}}
        pk.dump(save_dic, open(name, 'wb'))
