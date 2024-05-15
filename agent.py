from helper import slice_row, slice_col, reset_np_print
import numpy as np
import environment as ev
MIN_REL_IMP = 1+1e-6


class Agent:
    def __init__(self, number, d, aps):
        self.num = number
        self.d = d
        self.loc_num = np.nan
        self.x = np.nan
        self.max_task = {}
        for ap in aps:
            max_comp = ap.cluster.max_task(slice_row(d, ap.num, 0, ap.R - 1))
            max_bw = ap.bw_cap/d[ap.num, -1]
            self.max_task[ap.num] = min(max_comp, max_bw)
        self.prefs = []

    def __repr__(self):
        out = ""
        np.set_printoptions(precision=3, floatmode='fixed')
        out += f"===== print agent {self.num:03d} =====\n"
        out += f"demand:\n{self.d}\n"
        out += f"on ap: {self.loc_num:02d}\n"
        out += f"x:\n{self.x}\n"
        out += f"max_task on each ap:\n{self.max_task}\n"
        out += f"prefs:\n{self.prefs[-1]}\n"
        out += "=" * 22 + "\n"
        reset_np_print()
        return out

    def move(self, apf, apt, opt='ks'):
        self.loc_num = apt.num
        apf.rm_agent(self)
        apt.add_agent(self)
        if opt == 'ks':
            apf.set_mets()
            apt.set_mets()
        if opt == 'MNW':
            apf.set_MNW()
            apt.set_MNW()

class HomoAgent(Agent):
    def __init__(self, number, d, aps):
        if d.shape[0] > 1:
            raise Exception("Your demand is not fixed! HomoAgent accepts fixed demands only.")
        d = np.ones(shape=(len(aps), 1)) * d
        self.d = d
        self.num = number
        self.loc_num = np.nan
        self.x = np.nan
        self.max_task = {}
        for ap in aps:
            max_comp = ap.cluster.max_task(slice_row(d, ap.num, 0, ap.R - 1))
            max_bw = ap.bw_cap/d[ap.num, -1]
            self.max_task[ap.num] = min(max_comp, max_bw)
        self.prefs = []
