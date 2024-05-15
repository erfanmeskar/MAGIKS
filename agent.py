from helper import slice_row, slice_col, reset_np_print, find_mms_task, min_of_rest
import numpy as np
import environment as ev
MIN_REL_IMP = 1+1e-6


class Agent:
    def __init__(self, number, d, aps, N):
        self.num = number
        self.d = d
        self.loc_num = np.nan
        self.x = np.nan
        # Create self.max_task: A dictionary containing
        # maximum executable task on each AP
        self.make_max_task_dict(aps, d)
        self.tot_max_task = np.sum(list(self.max_task.values()))
        self.max_task_agg = self.find_max_task_agg(aps, d)
        self.prefs = []
        # Find MMS configuration: update self.mms_task_conf
        # Find the number of mmstask: update self.mms_task
        self.find_mms_task(N)
        # print(f"{self.num}: {self.mms_task}")

    def __repr__(self):
        out = ""
        np.set_printoptions(precision=3, floatmode='fixed')
        out += f"===== print agent {self.num:03d} =====\n"
        out += f"demand:\n{self.d}\n"
        out += f"on ap: {self.loc_num:02f}\n"
        out += f"x:\n{self.x}\n"
        out += f"max_task on each ap:\n{self.max_task}\n"
        print(self.prefs)
        out += f"prefs:\n{self.prefs[-1]}\n"
        out += "=" * 22 + "\n"
        reset_np_print()
        return out

    def make_max_task_dict(self, aps, d):
        self.max_task = {}
        for ap in aps:
            max_comp = ap.cluster.max_task(slice_row(d, ap.num, 0, ap.R - 1))
            max_bw = ap.bw_cap/d[ap.num, -1]
            self.max_task[ap.num] = min(max_comp, max_bw)

    def find_max_task_agg(self, aps, d):
        agg_c = np.zeros(shape=np.sum(aps[0].cluster.capacity, axis=0, keepdims=True).shape)
        agg_bw_cap = 0
        for ap in aps:
            agg_c += np.sum(ap.cluster.capacity, axis=0, keepdims=True)
            agg_bw_cap += ap.bw_cap
        return min(np.min(agg_c / slice_row(d, 0, 0, ap.R - 1)), agg_bw_cap/d[0, -1])

    def find_mms_task(self, N):
        E = len(self.max_task)
        self.mms_task_conf = {k: {'num_user': 0, 'max_task': self.max_task[k]} for k in self.max_task}
        mms_lst = [np.inf for k in self.max_task]
        mms = np.inf
        for i in range(N):
            mms = min(min_of_rest(mms_lst, 0), self.mms_task_conf[0]['max_task']/(self.mms_task_conf[0]['num_user']+1))
            best_e = 0
            for k in range(1, E):
                tmp = min(min_of_rest(mms_lst, k), self.mms_task_conf[k]['max_task']/(self.mms_task_conf[k]['num_user']+1))
                if tmp > mms:
                    mms = tmp
                    best_e = k
            mms_lst[best_e] = mms
            self.mms_task_conf[best_e]['num_user'] += 1
        for k in range(E):
            self.mms_task_conf[k]['mms'] = mms_lst[k]
        self.mms_task = np.min(mms_lst)

    def move(self, apf, apt, opt='ks'):
        self.loc_num = apt.num
        apf.rm_agent(self)
        apt.add_agent(self)
        if opt in ['ks', 'KS']:
            apf.set_mets()
            apt.set_mets()
        elif opt in ['lks', 'LKS']:
            apf.set_lmets()
            apt.set_lmets()
        elif opt in ['DRFH', 'drfh']:
            apf.set_medrs()
            apt.set_medrs()
        elif opt in ['MNW', 'mnw']:
            apf.set_MNW()
            apt.set_MNW()
        elif opt in ['EQ', 'eq']:
            apf.set_eq()
            apt.set_eq()

class HomoAgent(Agent):
    def __init__(self, number, d, aps, N):
        if d.shape[0] > 1:
            raise Exception("Your demand is not fixed! HomoAgent accepts fixed demands only.")
        super().__init__(number, np.ones(shape=(len(aps), 1)) * d, aps, N)
