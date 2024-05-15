import random
from math import floor
from accesspoints import MonoCluster, Ap
from agent import HomoAgent
from helper import get_keyOfMaxVal, get_keyOfMinVal, agent_pref_ap, agent_x,\
                   slice_row
import numpy as np
MEI_REL_IMP = 1+1e-6
REL_ERROR = 1e-6


class Env:
    def __init__(self, agents, aps):
        self.agents = agents
        self.aps = aps
        self.N = len(agents)
        self.E = len(aps)
        self.R = aps[0].R
        self.unallocateds = agents.copy()
        self.prefs_appended = False
        self.moves_to_eq = 0
        self.util_percent = [0 for i in range(aps[0].R)]
        self.found_eq = False

    def __repr__(self):
        out = ""
        out += "=" * 10 + f" print env " + "=" * 10 + "\n"
        out += "++++++ agents:\n"
        for agent in self.agents:
            out += f"agent {agent.num:03d} at ap {agent.loc_num:02d} "
            if np.sum(agent.x) >= agent.mms_task:
                out += f"with x={np.sum(agent.x):.3f} and mms=True\n"
            else:
                out += f"with x={np.sum(agent.x):.3f} and mms={agent.mms_task / np.sum(agent.x)}\n"
        out += "------- APs:\n"
        for ap in self.aps:
            out += f"AP {ap.num:03d} with mets={ap.mets:.3f}\n"
        out += "=" * 31
        return out

    def initialize(self, how='rand_with_mets', opt='ks', init_loc=[]):
        if how == 'rand_with_mets':
            for agent in self.agents:
                agent.loc_num = np.random.randint(0, self.E)
                self.aps[agent.loc_num].agents.append(agent)
                self.unallocateds.remove(agent)
            for ap in self.aps:
                if opt in ['ks', 'KS']:
                    ap.set_mets()
                elif opt in ['lks', 'LKS']:
                    ap.set_lmets()
                elif opt in ['DRFH', 'drfh']:
                    ap.set_medrs()
                elif opt in ['MNW', 'mnw']:
                    ap.set_MNW()
                elif opt in ['EQ', 'eq']:
                    ap.set_eq()
        elif how == 'all_zero':
            for agent in self.agents:
                agent.loc_num = 0
                self.aps[0].agents.append(agent)
                self.unallocateds.remove(agent)
            for ap in self.aps:
                if opt in ['ks', 'KS']:
                    ap.set_mets()
                elif opt in ['lks', 'LKS']:
                    ap.set_lmets()
                elif opt in ['DRFH', 'drfh']:
                    ap.set_medrs()
                elif opt in ['MNW', 'mnw']:
                    ap.set_MNW()
                elif opt in ['EQ', 'eq']:
                    ap.set_eq()
        elif how == 'given':
            for indx, agent in enumerate(self.agents):
                agent.loc_num = init_loc[indx]
                self.aps[init_loc[indx]].agents.append(agent)
                self.unallocateds.remove(agent)
            for ap in self.aps:
                if opt in ['ks', 'KS']:
                    ap.set_mets()
                elif opt in ['lks', 'LKS']:
                    ap.set_lmets()
                elif opt in ['DRFH', 'drfh']:
                    ap.set_medrs()
                elif opt in ['MNW', 'mnw']:
                    ap.set_MNW()
                elif opt in ['EQ', 'eq']:
                    ap.set_eq()
        elif how == 'unallocate':
            for agent in self.agents:
                if not np.isnan(agent.loc_num):
                    self.aps[agent.loc_num].agents.remove(agent)
                    agent.loc_num = np.nan
                    agent.x = np.nan
                    self.pref = []
                    self.unallocateds.append(agent)
            for ap in self.aps:
                if opt in ['ks', 'KS']:
                    ap.set_mets()
                elif opt in ['lks', 'LKS']:
                    ap.set_lmets()
                elif opt in ['DRFH', 'drfh']:
                    ap.set_medrs()
                elif opt in ['MNW', 'mnw']:
                    ap.set_MNW()
                elif opt in ['EQ', 'eq']:
                    ap.set_eq()
        elif how == 'balanced':
            N, E = self.N, self.E
            s = 0
            for k in range(0, N, floor(N/E)):
                for i in range(k, min(N - (N % E), k + floor(N/E))):
                    self.agents[i].loc_num = s
                    self.aps[s].agents.append(self.agents[i])
                    self.unallocateds.remove(self.agents[i])
                s += 1
            # Adding one agent to each busy aps
            for k in range(N - (N % E), N):
                self.agents[k].loc_num = (k % E)
                self.aps[k % E].agents.append(self.agents[k])
                self.unallocateds.remove(self.agents[k])
            for ap in self.aps:
                if opt in ['ks', 'KS']:
                    ap.set_mets()
                elif opt in ['lks', 'LKS']:
                    ap.set_lmets()
                elif opt in ['DRFH', 'drfh']:
                    ap.set_medrs()
                elif opt in ['MNW', 'mnw']:
                    ap.set_MNW()
                elif opt in ['EQ', 'eq']:
                    ap.set_eq()
        elif how == 'add_order_min_x_inv':
            self.initialize(how='unallocate', opt=opt)
            for agent in self.agents:
                x_inv_dict = {}
                x_inv_dict = {ap.num: 1/agent_x(agent, ap, opt=opt) for ap in self.aps}
                apt_num = get_keyOfMinVal(x_inv_dict)
                agent.move(np.nan, self.aps[apt_num], opt='ks')
                self.unallocateds.remove(agent)

    def set_agents_MMS(self):
        psi_value_mat = np.zeros(shape=(self.N, self.E))
        mms_bundle_mat = np.zeros(shape=(self.N, self.E))
        for agent in self.agents:
            psi_value_mat[agent.num, :] = list(agent.max_task.values())
            mms_bundle_mat[agent.num, :] = [agent.mms_task_conf[k]['num_user'] for k in agent.mms_task_conf]
        self.psi_value_mat = psi_value_mat
        self.mms_bundle_mat = mms_bundle_mat

    def find_MMS_association(self, server_order_lst=[]):
        unalloc_user_lst = [i for i in range(self.N)]
        assoc_loc_lst = [0 for i in range(self.N)]
        idx = 0
        while idx < self.E and unalloc_user_lst:
            s = server_order_lst[idx]
            sorted_user = sorted(unalloc_user_lst,
                                 key=lambda k: self.mms_bundle_mat[:, s][k],
                                 reverse=True)
            num_assoc_users = 0
            for user_num in sorted_user:
                user_bundle_on_server = self.mms_bundle_mat[user_num, s]
                if user_bundle_on_server >= num_assoc_users+1:
                    assoc_loc_lst[user_num] = s
                    unalloc_user_lst.remove(user_num)
                    num_assoc_users += 1
                else:
                    break
            idx += 1
        assert not unalloc_user_lst
        return assoc_loc_lst

    def max_intra_envy_ratio(self):
        max_ratio = 0
        for from_agent in self.agents:
            from_agent_x = np.sum(from_agent.x)
            for to_agent in self.agents:
                if from_agent.loc_num == to_agent.loc_num:
                    continue
                e = np.min(np.sum(to_agent.x) * slice_row(to_agent.d, to_agent.loc_num, 0, self.aps[0].R) /
                           slice_row(from_agent.d, to_agent.loc_num, 0, self.aps[0].R)) / from_agent_x
                if e > max_ratio:
                    max_ratio = e
        return max_ratio

    def is_REF(self, opt='ks'):
        bad_agents = {i:[] for i in range(self.E)}
        # Find minimum load server index and value
        max_mets = -1
        for a in self.aps:
            if max_mets < a.mets:
                max_mets = a.mets
                max_loc = a.num
        # Remove each agent and check the new load
        for leaving_agent in self.agents:
            this_ap = self.aps[leaving_agent.loc_num]
            if this_ap.num != max_loc and len(this_ap.agents) > 1:
                sum_d_b = np.sum(this_ap.get_d_times_max_task(),
                                 axis=0, keepdims=True)
                agent_d = slice_row(leaving_agent.d, this_ap.num, 0, this_ap.R)
                agent_d_b = agent_d * leaving_agent.max_task[this_ap.num]
                sum_d_b -= agent_d_b
                # new_mets
                new_mets = min(np.min(this_ap.cluster.capacity/
                                      sum_d_b[:, 0:this_ap.R-1]),
                               this_ap.bw_cap/sum_d_b[0, -1])
                if (new_mets - max_mets)/max_mets < -1*REL_ERROR:
                    bad_agents[this_ap.num].append(leaving_agent)
        output = {'is_REFX': True, 'is_REF1': True}
        # check for REFX(1)
        for ap_num in bad_agents:
            if len(bad_agents[ap_num]) > 0:
                output['is_REFX'] = False
            if len(bad_agents[ap_num]) == len(self.aps[ap_num].agents):
                output['is_REF1'] = False
        return output

    def is_eq(self, opt='ks'):
        if not self.prefs_appended:
            self.append_agents_pref(opt=opt)
        for agent in self.agents:
            if len(agent.prefs[-1]) > 0:
                return False
        return True

    def is_mms(self):
        for agent in self.agents:
            if (np.sum(agent.x) - agent.mms_task)/agent.mms_task < -1e-4:
                return False
        return True

    def append_agents_pref(self, opt='ks'):
        if opt in ['ks', 'KS', 'lks', 'LKS', 'MNW', 'mnw', 'EQ', 'eq', 'DRFH', 'drfh']:
            for agent in self.agents:
                pref_dict = {}
                for ap in self.aps:
                    is_pref, sum_x_ratio = agent_pref_ap(agent, ap, opt=opt)
                    if is_pref:
                        pref_dict[ap.num] = sum_x_ratio
                agent.prefs.append(pref_dict)
            self.prefs_appended = True

    def find_eq(self, opt='ks', agent_how='rand', ap_how='rand'):
        moves = 0
        if opt in ['MNW', 'mnw', 'EQ', 'eq', 'lks', 'LKS']:
            check_if_continue = lambda is_not_eq, m: (m <= 300) and is_not_eq
        else:
            check_if_continue = lambda is_not_eq, m: is_not_eq
        while check_if_continue(not self.is_eq(opt=opt), moves):
            moves += 1
            self.move_one_agent(agent_how=agent_how, ap_how=ap_how, opt=opt)
            self.append_agents_pref(opt=opt)
#             print(f"***** move {moves} *****")
#             for ap in self.aps:
#                 print([agent.num for agent in ap.agents])
        self.move_to_eq = moves
        self.found_eq = (moves <= 300) if opt in ['MNW', 'mnw', 'EQ', 'eq', 'lks', 'LKS'] else True

    def get_who_can_move(self, opt='ks'):
        if not self.prefs_appended:
            self.append_agents_pref(opt=opt)
        return [agent.num for agent in self.agents if len(agent.prefs[-1]) > 0]

    def move_one_agent(self, agent_how='rand', ap_how='rand', opt='ks'):
        if agent_how == 'rand':
            agent_num = np.random.choice(self.get_who_can_move(opt=opt))
            agent = self.agents[agent_num]
            apf = self.aps[agent.loc_num]
            if ap_how == 'rand':
                apt = self.aps[np.random.choice(list(agent.prefs[-1].keys()))]
            elif ap_how == 'max_task':
                apt_num = get_keyOfMaxVal(agent.prefs[-1])
                apt = self.aps[apt_num]
            elif ap_how == 'min_task':
                apt_num = get_keyOfMinVal(agent.prefs[-1])
                apt = self.aps[apt_num]
            agent.move(apf, apt, opt=opt)
            self.prefs_appended = False
        elif agent_how == 'max_ratio':
            max_ratio = 0
            for ag_num in self.get_who_can_move(opt=opt):
                k = get_keyOfMaxVal(self.agents[ag_num].prefs[-1])
                if self.agents[ag_num].prefs[-1][k] > max_ratio:
                    max_ap_num = k
                    max_agent_num = ag_num
                    max_ratio = self.agents[ag_num].prefs[-1][k]
            agent = self.agents[max_agent_num]
            apf = self.aps[agent.loc_num]
            apt = self.aps[max_ap_num]
            agent.move(apf, apt, opt=opt)
            self.prefs_appended = False
        elif agent_how == 'min_ratio':
            min_ratio = np.inf
            for ag_num in self.get_who_can_move(opt=opt):
                k = get_keyOfMinVal(self.agents[ag_num].prefs[-1])
                if self.agents[ag_num].prefs[-1][k] < min_ratio:
                    min_ap_num = k
                    min_agent_num = ag_num
                    min_ratio = self.agents[ag_num].prefs[-1][k]
            agent = self.agents[min_agent_num]
            apf = self.aps[agent.loc_num]
            apt = self.aps[min_ap_num]
            agent.move(apf, apt, opt=opt)
            self.prefs_appended = False

    def set_homo_total_utilization_percent(self):
        # combine all servers
        R = self.aps[0].cluster.R_comp + 1
        total_c = np.zeros(shape=(1, R))
        N = len(self.agents)
        E = len(self.aps)
        for i in range(E):
            total_c[0:1, 0:-1] += np.sum(self.aps[i].cluster.capacity, axis=0, keepdims=True)
            total_c[0, -1] += self.aps[i].bw_cap
        # combine all users utilization
        consumption = np.zeros(shape=(1, R))
        for i in range(N):
            consumption += np.sum(self.agents[i].x) * self.agents[i].d[0:1, :]
        util_percent = 100 * consumption / total_c
        for r in range(R):
            self.util_percent[r] = util_percent[0][r]


class HomoEnv(Env):
    def __init__(self, agents, aps):
        for ap in aps[1:]:
            if not np.array_equal(aps[0].cluster.capacity, ap.cluster.capacity) or aps[0].bw_cap != ap.bw_cap:
                raise Exception("APs must be exactly the same in HomoEnv.")
        super().__init__(agents, aps)


class VirtEnv(Env):
    def __init__(self, agents, aps):
        virt_agents = []
        R_comp = aps[0].cluster.R_comp
        N, E = len(agents), len(aps)
        virt_c, virt_c_b = np.zeros(shape=(1, R_comp)), 0
        for i in range(E):
            virt_c += aps[i].cluster.capacity
            virt_c_b += aps[i].bw_cap
        virt_clusters = [MonoCluster(0, virt_c)]
        virt_aps = [Ap(0, virt_clusters[0], virt_c_b)]
        for i in range(N):
            virt_agents.append(HomoAgent(i, agents[i].d[0:1, :], virt_aps))
        super().__init__(virt_agents, virt_aps)
