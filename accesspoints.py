from helper import slice_row, slice_col, reset_np_print
import numpy as np
import cvxpy as cpy


class Cluster:
    def __init__(self, num, c):
        self.num = num
        self.S = c.shape[0]
        self.R_comp = c.shape[1]
        self.capacity = c

    def __repr__(self):
        out = ""
        np.set_printoptions(precision=3, floatmode='fixed')
        out += f"===== print cluster {self.num:03d} =====\n"
        out += f"capacity:\n{self.capacity}\n"
        out += f"S: {self.S:02d}\n"
        out += f"R_comp: {self.R_comp}\n"
        out += "=" * 24
        reset_np_print()
        return out

    def max_task(self, d_comp):
        return np.sum(np.min(self.capacity/d_comp, axis=1, keepdims=True))


class MonoCluster(Cluster):
    def __init__(self, num, c):
        if c.shape[0] != 1:
            raise Exception("There are more than 1 servers in your 'c'. You should have 1 server with MonoCluster class.}")
        super().__init__(num, c)


class Ap:
    def __init__(self, number, cluster, bw_cap):
        self.num = number
        self.cluster = cluster
        self.bw_cap = bw_cap
        self.S = cluster.S
        self.R = cluster.R_comp + 1
        self.agents = []
        self.mets = np.nan # maximum equalized task share (mets)
        self.MNW = np.nan

    def __repr__(self):
        out = ""
        out += f"++++++++++++ print AP {self.num:03d} ++++++++++++\n"
        out += self.cluster.__repr__()
        np.set_printoptions(precision=3, floatmode='fixed')
        out += f"\nbw cap: {self.bw_cap}\n"
        out += f"R: {self.R}\n"
        out += "++++++++++++++++++++++++++++++++++++++"
        reset_np_print()
        return out

    def add_agent(self, agent):
        self.agents.append(agent)

    def rm_agent(self, agent):
        self.agents.remove(agent)

    def get_d(self):
        N = len(self.agents)
        d = np.zeros(shape=(N, self.R))
        for idx, agent in enumerate(self.agents):
            d[idx, :] = slice_row(agent.d, self.num, 0, self.R)
        return d

    def get_d_times_max_task(self):
        N = len(self.agents)
        dbar = np.zeros(shape=(N, self.R))
        for idx, agent in enumerate(self.agents):
            agent_d = slice_row(agent.d, self.num, 0, self.R)
            dbar[idx, :] = agent_d * agent.tot_max_task
        return dbar

    def get_d_times_local_max_task(self):
        N = len(self.agents)
        dbar = np.zeros(shape=(N, self.R))
        for idx, agent in enumerate(self.agents):
            agent_d = slice_row(agent.d, self.num, 0, self.R)
            dbar[idx, :] = agent_d * agent.max_task[self.num]
        return dbar

    def get_d_times_max_task_agg(self):
        N = len(self.agents)
        dbar = np.zeros(shape=(N, self.R))
        for idx, agent in enumerate(self.agents):
            agent_d = slice_row(agent.d, self.num, 0, self.R)
            dbar[idx, :] = agent_d * agent.max_task_agg
        return dbar

    def get_diag_max_task_inv(self):
        N = len(self.agents)
        max_task_inv = np.zeros(shape=(N, N))
        for idx, agent in enumerate(self.agents):
            max_task_inv[idx, idx] = 1/agent.tot_max_task
        return max_task_inv

    def get_diag_local_max_task_inv(self):
        N = len(self.agents)
        max_task_inv = np.zeros(shape=(N, N))
        for idx, agent in enumerate(self.agents):
            max_task_inv[idx, idx] = 1/agent.max_task[self.num]
        return max_task_inv

    def get_diag_max_task_agg_inv(self):
        N = len(self.agents)
        max_task_inv = np.zeros(shape=(N, N))
        for idx, agent in enumerate(self.agents):
            max_task_inv[idx, idx] = 1/agent.max_task_agg
        return max_task_inv

    # Equal Division
    def set_eq(self):
        local_N = len(self.agents)
        S = self.S
        if local_N > 0:
            d = self.get_d()
            c = self.cluster.capacity
            bw = self.bw_cap
            for agent in self.agents:
                x_comm = np.min(c/agent.d[self.num, 0:-1], axis=1,
                               keepdims=True).T/local_N
                x_comp = (bw/agent.d[self.num, -1])/local_N
                if sum(x_comm) <= x_comp:
                    agent.x = x_comm
                else:
                    agent.x = x_comm * x_comp / sum(x_comm)
    
    # Maximum Equalized Task Share 
    def set_mets(self):
        if len(self.agents) > 1 and self.S == 1:
            sum_d_b = np.sum(self.get_d_times_max_task(), axis=0, keepdims=True)
            # update ap.mets and agent.x
            self.mets = min(np.min(self.cluster.capacity/sum_d_b[:, 0:self.R-1]),
                            self.bw_cap/sum_d_b[0, -1])
            for agent in self.agents:
                agent.x = np.array([[agent.tot_max_task * self.mets]])
        elif len(self.agents) > 1 and self.S > 1:
            # make matrix d and diagonal matrix max_task_inv_diag
            local_N = len(self.agents)
            one_N = np.ones(shape=(local_N, 1))
            one_S = np.ones(shape=(self.cluster.S, 1))
            demand = self.get_d()
            max_task_inv = self.get_diag_max_task_inv()
            # create optimization problem
            mets = cpy.Variable(nonneg=True)
            x = cpy.Variable(shape=(local_N, self.cluster.S), nonneg=True)
            # constraints and objective declaration
            constr = []
            constr += [x.T @ demand[:, 0:self.cluster.R_comp] <=
                        self.cluster.capacity]
            constr += [slice_col(demand, 0, local_N, self.R-1).T @ x @ one_S <=
                        self.bw_cap]
            constr += [max_task_inv @ x @ one_S == mets * one_N]
            obj = cpy.Maximize(mets)
            prob = cpy.Problem(obj, constr)
            prob.solve(solver=cpy.MOSEK, verbose=False)
            # update ap.mets and agent.x
            self.mets = mets.value
            for idx, agent in enumerate(self.agents):
                agent.x = slice_row(x.value, idx, 0, self.cluster.S)
        elif len(self.agents) == 1:
            self.mets = 1
            dcomp = slice_row(self.agents[0].d, self.num, 0, self.R-1)
            x_comp = np.min(self.cluster.capacity/dcomp, axis=1, keepdims=True).T
            x_comp_sum = np.sum(x_comp)
            x_comm_sum = self.bw_cap/self.agents[0].d[self.num, -1]
            self.agents[0].x = x_comp * min(1, x_comm_sum/x_comp_sum)
        elif len(self.agents) == 0:
            self.mets = 0

    # Local Maximum Equalized Task Share 
    def set_lmets(self):
        if len(self.agents) > 1 and self.S == 1:
            sum_d_b = np.sum(self.get_d_times_local_max_task(), axis=0, keepdims=True)
            # update ap.mets and agent.x
            self.lmets = min(np.min(self.cluster.capacity/sum_d_b[:, 0:self.R-1]),
                             self.bw_cap/sum_d_b[0, -1])
            for agent in self.agents:
                agent.x = np.array([[agent.max_task[self.num] * self.lmets]])
        elif len(self.agents) > 1 and self.S > 1:
            # make matrix d and diagonal matrix max_task_inv_diag
            local_N = len(self.agents)
            one_N = np.ones(shape=(local_N, 1))
            one_S = np.ones(shape=(self.cluster.S, 1))
            demand = self.get_d()
            max_task_inv = self.get_diag_local_max_task_inv()
            # create optimization problem
            lmets = cpy.Variable(nonneg=True)
            x = cpy.Variable(shape=(local_N, self.cluster.S), nonneg=True)
            # constraints and objective declaration
            constr = []
            constr += [x.T @ demand[:, 0:self.cluster.R_comp] <=
                        self.cluster.capacity]
            constr += [slice_col(demand, 0, local_N, self.R-1).T @ x @ one_S <=
                        self.bw_cap]
            constr += [max_task_inv @ x @ one_S == lmets * one_N]
            obj = cpy.Maximize(lmets)
            prob = cpy.Problem(obj, constr)
            prob.solve(solver=cpy.MOSEK, verbose=False)
            # update ap.mets and agent.x
            self.lmets = lmets.value
            for idx, agent in enumerate(self.agents):
                agent.x = slice_row(x.value, idx, 0, self.cluster.S)
        elif len(self.agents) == 1:
            self.lmets = 1
            dcomp = slice_row(self.agents[0].d, self.num, 0, self.R-1)
            x_comp = np.min(self.cluster.capacity/dcomp, axis=1, keepdims=True).T
            x_comp_sum = np.sum(x_comp)
            x_comm_sum = self.bw_cap/self.agents[0].d[self.num, -1]
            self.agents[0].x = x_comp * min(1, x_comm_sum/x_comp_sum)
        elif len(self.agents) == 0:
            self.lmets = 0

    # Maximum Equalized Dominant Share 
    def set_medrs(self):
        if len(self.agents) > 1 and self.S == 1:
            sum_d_b = np.sum(self.get_d_times_max_task_agg(), axis=0, keepdims=True)
            # update ap.mets and agent.x
            self.medrs = min(np.min(self.cluster.capacity/sum_d_b[:, 0:self.R-1]),
                            self.bw_cap/sum_d_b[0, -1])
            for agent in self.agents:
                agent.x = np.array([[agent.max_task_agg * self.medrs]])
        elif len(self.agents) > 1 and self.S > 1:
            # make matrix d and diagonal matrix max_task_inv_diag
            local_N = len(self.agents)
            one_N = np.ones(shape=(local_N, 1))
            one_S = np.ones(shape=(self.cluster.S, 1))
            demand = self.get_d()
            max_task_inv = self.get_diag_max_task_agg_inv()
            # create optimization problem
            medrs = cpy.Variable(nonneg=True)
            x = cpy.Variable(shape=(local_N, self.cluster.S), nonneg=True)
            # constraints and objective declaration
            constr = []
            constr += [x.T @ demand[:, 0:self.cluster.R_comp] <=
                        self.cluster.capacity]
            constr += [slice_col(demand, 0, local_N, self.R-1).T @ x @ one_S <=
                        self.bw_cap]
            constr += [max_task_inv @ x @ one_S == medrs * one_N]
            obj = cpy.Maximize(medrs)
            prob = cpy.Problem(obj, constr)
            prob.solve(solver=cpy.MOSEK, verbose=False)
            # update ap.mets and agent.x
            self.medrs = medrs.value
            for idx, agent in enumerate(self.agents):
                agent.x = slice_row(x.value, idx, 0, self.cluster.S)
        elif len(self.agents) == 1:
            self.medrs = 1
            dcomp = slice_row(self.agents[0].d, self.num, 0, self.R-1)
            x_comp = np.min(self.cluster.capacity/dcomp, axis=1, keepdims=True).T
            x_comp_sum = np.sum(x_comp)
            x_comm_sum = self.bw_cap/self.agents[0].d[self.num, -1]
            self.agents[0].x = x_comp * min(1, x_comm_sum/x_comp_sum)
        elif len(self.agents) == 0:
            self.medrs = 0

    # Maximum Nash Welfare
    def set_MNW(self):
        if len(self.agents) == 0:
            self.MNW = 0
        elif len(self.agents) == 1:
            dcomp = slice_row(self.agents[0].d, self.num, 0, self.R-1)
            x_comp = np.min(self.cluster.capacity/dcomp, axis=1, keepdims=True).T
            x_comp_sum = np.sum(x_comp)
            x_comm_sum = self.bw_cap/self.agents[0].d[self.num, -1]
            self.agents[0].x = x_comp * min(1, x_comm_sum/x_comp_sum)
            self.MNW = np.sum(self.agents[0].x)
        else:
            local_N = len(self.agents)
            one_N = np.ones(shape=(local_N, 1))
            one_S = np.ones(shape=(self.cluster.S, 1))
            demand = self.get_d()
            x = cpy.Variable(shape=(local_N, self.cluster.S), nonneg=True)
            constr = []
            constr += [x.T @ demand[:, 0:self.cluster.R_comp] <=
                        self.cluster.capacity]
            constr += [slice_col(demand, 0, local_N, self.R-1).T @ x @ one_S <=
                        self.bw_cap]
            obj = cpy.Maximize(cpy.sum(cpy.log(x @ one_S)))
            prob = cpy.Problem(obj, constr)
            try:
                prob.solve(solver=cpy.MOSEK, verbose=False)
            except:
                prob.solve(solver=cpy.SCS, verbose=False)
            self.MNW = np.exp(prob.value)
            for idx, agent in enumerate(self.agents):
                agent.x = slice_row(x.value, idx, 0, self.cluster.S)

    # Utilitarian (Maximum Sum of number of Tasks)
    def utilitarian(self):
        if len(self.agents) > 1:
            # make matrix d and diagonal matrix max_task_inv_diag
            local_N = len(self.agents)
            one_N = np.ones(shape=(local_N, 1))
            one_S = np.ones(shape=(self.cluster.S, 1))
            demand = self.get_d()
            # create optimization problem
            x = cpy.Variable(shape=(local_N, self.cluster.S), nonneg=True)
            # constraints and objective declaration
            constr = []
            constr += [x.T @ demand[:, 0:self.cluster.R_comp] <=
                        self.cluster.capacity]
            constr += [slice_col(demand, 0, local_N, self.R-1).T @ x @ one_S <=
                        self.bw_cap]
            obj = cpy.Maximize(np.ones(shape=(1, local_N)) @ x @
                               np.ones(shape=(self.cluster.S, 1)))
            prob = cpy.Problem(obj, constr)
            prob.solve(solver=cpy.MOSEK, verbose=False)
            # update ap.mets and agent.x
            for idx, agent in enumerate(self.agents):
                agent.x = slice_row(x.value, idx, 0, self.cluster.S)
        elif len(self.agents) == 1:
            dcomp = slice_row(self.agents[0].d, self.num, 0, self.R-1)
            x_comp = np.min(self.cluster.capacity/dcomp, axis=1, keepdims=True).T
            x_comp_sum = np.sum(x_comp)
            x_comm_sum = self.bw_cap/self.agents[0].d[self.num, -1]
            self.agents[0].x = x_comp * min(1, x_comm_sum/x_comp_sum)

    # Normalized Utilitarian (Maximum Sum of Normalized number of Tasks)
    def norm_utilitarian(self):
        if len(self.agents) > 1:
            # make matrix d and diagonal matrix max_task_inv_diag
            local_N = len(self.agents)
            one_N = np.ones(shape=(local_N, 1))
            one_S = np.ones(shape=(self.cluster.S, 1))
            demand = self.get_d()
            max_task_inv = self.get_diag_max_task_inv()
            # create optimization problem
            x = cpy.Variable(shape=(local_N, self.cluster.S), nonneg=True)
            # constraints and objective declaration
            constr = []
            constr += [x.T @ demand[:, 0:self.cluster.R_comp] <=
                        self.cluster.capacity]
            constr += [slice_col(demand, 0, local_N, self.R-1).T @ x @ one_S <=
                        self.bw_cap]
            obj = cpy.Maximize(np.ones(shape=(1, local_N)) @ max_task_inv @ x @
                               np.ones(shape=(self.cluster.S, 1)))
            prob = cpy.Problem(obj, constr)
            prob.solve(solver=cpy.MOSEK, verbose=False)
            # update ap.mets and agent.x
            for idx, agent in enumerate(self.agents):
                agent.x = slice_row(x.value, idx, 0, self.cluster.S)
        elif len(self.agents) == 1:
            dcomp = slice_row(self.agents[0].d, self.num, 0, self.R-1)
            x_comp = np.min(self.cluster.capacity/dcomp, axis=1, keepdims=True).T
            x_comp_sum = np.sum(x_comp)
            x_comm_sum = self.bw_cap/self.agents[0].d[self.num, -1]
            self.agents[0].x = x_comp * min(1, x_comm_sum/x_comp_sum)
