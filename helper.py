import numpy as np
import cvxpy as cpy
from copy import deepcopy
min_rel_imp = 1+1e-6


# slice a row of np.ndarray and keep dimension
def slice_row(nparray, row, colstart, colend):
    return nparray[row:row+1, colstart:colend]


# slice a column of np.ndarray and keep dimension
def slice_col(nparray, rowstart, rowend, col):
    return nparray[rowstart:rowend, col:col+1]


# reset np.set_printoptions
def reset_np_print():
    np.set_printoptions(precision=8, floatmode='maxprec_equal')


def randperm(*args):
    if len(args) == 0:
        raise Exception('arguments missing')
    elif len(args) == 1:
        l = args[0]
        if type(l) == list:
            return list(np.random.permutation(l))
        elif type(l) == int:
            return list(np.random.permutation(list(range(0, l))))
    elif len(args) == 2:
        if type(l[0]) == int and type(l[1]) == int:
            if l[0] < l[1]:
                return list(np.random.permutation(list(range(l[0], l[1]))))
            else:
                raise Exception('first arg should be less than the second')
        else:
            raise Exception('arguments are not of the same type int')
    elif len(args) > 2:
        raise Exception('more than 2 arguments')


def get_keyOfMaxVal(dic):
    max_val = np.NINF
    max_key = 0
    for k, v in dic.items():
        if v > max_val:
            max_key = k
            max_val = v
    return max_key


def get_keyOfMinVal(dic):
    min_val = np.inf
    Min_key = 0
    for k, v in dic.items():
        if v < min_val:
            min_key = k
            min_val = v
    return min_key


def agent_x(moving_agent, ap, opt='ks'):
    if moving_agent.loc_num == ap.num:
        return moving_agent.x
    if opt == 'ks':
        if len(ap.agents) > 0 and ap.S == 1:
            sum_d_b = np.sum(ap.get_d_times_max_task(), axis=0, keepdims=True)
            agent_d = slice_row(moving_agent.d, ap.num, 0, ap.R)
            agent_d_b = agent_d * moving_agent.tot_max_task
            sum_d_b += agent_d_b
            # new_mets and new_x
            new_mets = min(np.min(ap.cluster.capacity/sum_d_b[:, 0:ap.R-1]),
                            ap.bw_cap/sum_d_b[0, -1])
            new_x = np.array([[moving_agent.tot_max_task * new_mets]])
            return np.sum(new_x)
        elif len(ap.agents) > 0 and ap.S > 1:
            # make matrix d and diagonal matrix max_task_inv_diag
            local_N = len(ap.agents)+1
            one_N = np.ones(shape=(local_N, 1))
            one_S = np.ones(shape=(ap.cluster.S, 1))
            demand = np.vstack((ap.get_d(),
                                slice_row(moving_agent.d, ap.num, 0, ap.R)))
            max_task_inv = np.zeros(shape=(local_N, local_N))
            max_task_inv[0:-1, 0:-1] = ap.get_diag_max_task_inv()
            max_task_inv[-1, -1] = 1/moving_agent.tot_max_task
            # create optimization problem
            mets = cpy.Variable(nonneg=True)
            x = cpy.Variable(shape=(local_N, ap.cluster.S), nonneg=True)
            # constraints and objective declaration
            constr = []
            constr += [x.T @ demand[:, 0:ap.cluster.R_comp] <=
                        ap.cluster.capacity]
            constr += [slice_col(demand, 0, local_N, ap.R-1).T @ x @ one_S <=
                        ap.bw_cap]
            constr += [max_task_inv @ x @ one_S == mets * one_N]
            obj = cpy.Maximize(mets)
            prob = cpy.Problem(obj, constr)
            prob.solve(solver=cpy.MOSEK, verbose=False)
            new_sum_x = np.sum(slice_row(x.value, local_N-1, 0, ap.cluster.S))
            return  new_sum_x
        elif len(ap.agents) == 0:
            new_sum_x = moving_agent.max_task[ap.num]
            return new_sum_x
    if opt in ['lks', 'LKS']:
        if len(ap.agents) > 0 and ap.S == 1:
            sum_d_b = np.sum(ap.get_d_times_local_max_task(), axis=0, keepdims=True)
            agent_d = slice_row(moving_agent.d, ap.num, 0, ap.R)
            agent_d_b = agent_d * moving_agent.max_task[ap.num]
            sum_d_b += agent_d_b
            # new_mets and new_x
            new_lmets = min(np.min(ap.cluster.capacity/sum_d_b[:, 0:ap.R-1]),
                             ap.bw_cap/sum_d_b[0, -1])
            new_x = np.array([[moving_agent.max_task[ap.num] * new_lmets]])
            return np.sum(new_x)
        elif len(ap.agents) > 0 and ap.S > 1:
            # make matrix d and diagonal matrix max_task_inv_diag
            local_N = len(ap.agents)+1
            one_N = np.ones(shape=(local_N, 1))
            one_S = np.ones(shape=(ap.cluster.S, 1))
            demand = np.vstack((ap.get_d(),
                                slice_row(moving_agent.d, ap.num, 0, ap.R)))
            max_task_inv = np.zeros(shape=(local_N, local_N))
            max_task_inv[0:-1, 0:-1] = ap.get_diag_local_max_task_inv()
            max_task_inv[-1, -1] = 1/moving_agent.max_task[ap.num]
            # create optimization problem
            lmets = cpy.Variable(nonneg=True)
            x = cpy.Variable(shape=(local_N, ap.cluster.S), nonneg=True)
            # constraints and objective declaration
            constr = []
            constr += [x.T @ demand[:, 0:ap.cluster.R_comp] <=
                        ap.cluster.capacity]
            constr += [slice_col(demand, 0, local_N, ap.R-1).T @ x @ one_S <=
                        ap.bw_cap]
            constr += [max_task_inv @ x @ one_S == lmets * one_N]
            obj = cpy.Maximize(lmets)
            prob = cpy.Problem(obj, constr)
            prob.solve(solver=cpy.MOSEK, verbose=False)
            new_sum_x = np.sum(slice_row(x.value, local_N-1, 0, ap.cluster.S))
            return  new_sum_x
        elif len(ap.agents) == 0:
            new_sum_x = moving_agent.max_task[ap.num]
            return new_sum_x
    if opt == 'MNW':
        if len(ap.agents) == 0:
            new_sum_x = moving_agent.max_task[ap.num]
            return new_sum_x
        else:
            # make matrix d
            local_N = len(ap.agents)+1
            one_N = np.ones(shape=(local_N, 1))
            one_S = np.ones(shape=(ap.cluster.S, 1))
            demand = np.vstack((ap.get_d(),
                                slice_row(moving_agent.d, ap.num, 0, ap.R)))
            x = cpy.Variable(shape=(local_N, ap.cluster.S), nonneg=True)
            constr = []
            constr += [x.T @ demand[:, 0:ap.cluster.R_comp] <=
                       ap.cluster.capacity]
            constr += [slice_col(demand, 0, local_N, ap.R-1).T @ x @ one_S <=
                       ap.bw_cap]
            obj = cpy.Maximize(cpy.sum(cpy.log(x @ one_S)))
            prob = cpy.Problem(obj, constr)
            prob.solve(solver=cpy.SCS, verbose=False)
            new_sum_x = np.sum(slice_row(x.value, local_N-1, 0, ap.cluster.S))
            return new_sum_x


def agent_pref_ap(moving_agent, ap, opt='ks'):
    if moving_agent.loc_num == ap.num:
        return False, 1
    if opt == 'ks':
        if len(ap.agents) > 0 and ap.S == 1:
            sum_d_b = np.sum(ap.get_d_times_max_task(), axis=0, keepdims=True)
            agent_d = slice_row(moving_agent.d, ap.num, 0, ap.R)
            agent_d_b = agent_d * moving_agent.tot_max_task
            sum_d_b += agent_d_b
            # new_mets and new_x
            new_mets = min(np.min(ap.cluster.capacity/sum_d_b[:, 0:ap.R-1]),
                            ap.bw_cap/sum_d_b[0, -1])
            new_x = np.array([[moving_agent.tot_max_task * new_mets]])
            ratio = np.sum(new_x) / np.sum(moving_agent.x)
            return  ratio > min_rel_imp, ratio
        elif len(ap.agents) > 0 and ap.S > 1:
            # make matrix d and diagonal matrix max_task_inv_diag
            local_N = len(ap.agents)+1
            one_N = np.ones(shape=(local_N, 1))
            one_S = np.ones(shape=(ap.cluster.S, 1))
            demand = np.vstack((ap.get_d(),
                                slice_row(moving_agent.d, ap.num, 0, ap.R)))
            max_task_inv = np.zeros(shape=(local_N, local_N))
            max_task_inv[0:-1, 0:-1] = ap.get_diag_max_task_inv()
            max_task_inv[-1, -1] = 1/moving_agent.tot_max_task
            # create optimization problem
            mets = cpy.Variable(nonneg=True)
            x = cpy.Variable(shape=(local_N, ap.cluster.S), nonneg=True)
            # constraints and objective declaration
            constr = []
            constr += [x.T @ demand[:, 0:ap.cluster.R_comp] <=
                        ap.cluster.capacity]
            constr += [slice_col(demand, 0, local_N, ap.R-1).T @ x @ one_S <=
                        ap.bw_cap]
            constr += [max_task_inv @ x @ one_S == mets * one_N]
            obj = cpy.Maximize(mets)
            prob = cpy.Problem(obj, constr)
            prob.solve(solver=cpy.MOSEK, verbose=False)
            new_sum_x = np.sum(slice_row(x.value, local_N-1, 0, ap.cluster.S))
            ratio = new_sum_x / np.sum(moving_agent.x)
            return  ratio > min_rel_imp, ratio
        elif len(ap.agents) == 0:
            new_sum_x = moving_agent.max_task[ap.num]
            ratio = new_sum_x / np.sum(moving_agent.x)
            return ratio > min_rel_imp, ratio
    elif opt in ['lks', 'LKS']:
        if len(ap.agents) > 0 and ap.S == 1:
            sum_d_b = np.sum(ap.get_d_times_local_max_task(), axis=0, keepdims=True)
            agent_d = slice_row(moving_agent.d, ap.num, 0, ap.R)
            agent_d_b = agent_d * moving_agent.max_task[ap.num]
            sum_d_b += agent_d_b
            # new_mets and new_x
            new_lmets = min(np.min(ap.cluster.capacity/sum_d_b[:, 0:ap.R-1]),
                             ap.bw_cap/sum_d_b[0, -1])
            new_x = np.array([[moving_agent.max_task[ap.num] * new_lmets]])
            ratio = np.sum(new_x) / np.sum(moving_agent.x)
            return  ratio > min_rel_imp, ratio
        elif len(ap.agents) > 0 and ap.S > 1:
            # make matrix d and diagonal matrix max_task_inv_diag
            local_N = len(ap.agents)+1
            one_N = np.ones(shape=(local_N, 1))
            one_S = np.ones(shape=(ap.cluster.S, 1))
            demand = np.vstack((ap.get_d(),
                                slice_row(moving_agent.d, ap.num, 0, ap.R)))
            max_task_inv = np.zeros(shape=(local_N, local_N))
            max_task_inv[0:-1, 0:-1] = ap.get_diag_max_local_task_inv()
            max_task_inv[-1, -1] = 1/moving_agent.max_task[ap.num]
            # create optimization problem
            lmets = cpy.Variable(nonneg=True)
            x = cpy.Variable(shape=(local_N, ap.cluster.S), nonneg=True)
            # constraints and objective declaration
            constr = []
            constr += [x.T @ demand[:, 0:ap.cluster.R_comp] <=
                        ap.cluster.capacity]
            constr += [slice_col(demand, 0, local_N, ap.R-1).T @ x @ one_S <=
                        ap.bw_cap]
            constr += [max_task_inv @ x @ one_S == lmets * one_N]
            obj = cpy.Maximize(lmets)
            prob = cpy.Problem(obj, constr)
            prob.solve(solver=cpy.MOSEK, verbose=False)
            new_sum_x = np.sum(slice_row(x.value, local_N-1, 0, ap.cluster.S))
            ratio = new_sum_x / np.sum(moving_agent.x)
            return  ratio > min_rel_imp, ratio
        elif len(ap.agents) == 0:
            new_sum_x = moving_agent.max_task[ap.num]
            ratio = new_sum_x / np.sum(moving_agent.x)
            return ratio > min_rel_imp, ratio
    elif opt in ['DRFH', 'drfh']:
        if len(ap.agents) > 0 and ap.S == 1:
            sum_d_b = np.sum(ap.get_d_times_max_task_agg(), axis=0, keepdims=True)
            agent_d = slice_row(moving_agent.d, ap.num, 0, ap.R)
            agent_d_b = agent_d * moving_agent.max_task_agg
            sum_d_b += agent_d_b
            # new_mets and new_x
            new_medrs = min(np.min(ap.cluster.capacity/sum_d_b[:, 0:ap.R-1]),
                            ap.bw_cap/sum_d_b[0, -1])
            new_x = np.array([[moving_agent.max_task_agg * new_medrs]])
            ratio = np.sum(new_x) / np.sum(moving_agent.x)
        elif len(ap.agents) > 0 and ap.S > 1:
            # make matrix d and diagonal matrix max_task_inv_diag
            local_N = len(ap.agents)+1
            one_N = np.ones(shape=(local_N, 1))
            one_S = np.ones(shape=(ap.cluster.S, 1))
            demand = np.vstack((ap.get_d(),
                                slice_row(moving_agent.d, ap.num, 0, ap.R)))
            max_task_inv = np.zeros(shape=(local_N, local_N))
            max_task_inv[0:-1, 0:-1] = ap.get_diag_max_task_agg_inv()
            max_task_inv[-1, -1] = 1/moving_agent.max_task_agg
            # create optimization problem
            medrs = cpy.Variable(nonneg=True)
            x = cpy.Variable(shape=(local_N, ap.cluster.S), nonneg=True)
            # constraints and objective declaration
            constr = []
            constr += [x.T @ demand[:, 0:ap.cluster.R_comp] <=
                        ap.cluster.capacity]
            constr += [slice_col(demand, 0, local_N, ap.R-1).T @ x @ one_S <=
                        ap.bw_cap]
            constr += [max_task_inv @ x @ one_S == medrs * one_N]
            obj = cpy.Maximize(medrs)
            prob = cpy.Problem(obj, constr)
            prob.solve(solver=cpy.MOSEK, verbose=False)
            new_sum_x = np.sum(slice_row(x.value, local_N-1, 0, ap.cluster.S))
            ratio = new_sum_x / np.sum(moving_agent.x)
        elif len(ap.agents) == 0:
            new_sum_x = moving_agent.max_task[ap.num]
            ratio = new_sum_x / np.sum(moving_agent.x)
        return ratio > min_rel_imp, ratio
    elif opt in ['MNW', 'mnw']:
        if len(ap.agents) == 0:
            new_sum_x = moving_agent.max_task[ap.num]
            ratio = new_sum_x / np.sum(moving_agent.x)
            return ratio > min_rel_imp, ratio
        else:
            # make matrix d
            local_N = len(ap.agents)+1
            one_N = np.ones(shape=(local_N, 1))
            one_S = np.ones(shape=(ap.cluster.S, 1))
            demand = np.vstack((ap.get_d(),
                                slice_row(moving_agent.d, ap.num, 0, ap.R)))
            x = cpy.Variable(shape=(local_N, ap.cluster.S), nonneg=True)
            constr = []
            constr += [x.T @ demand[:, 0:ap.cluster.R_comp] <=
                       ap.cluster.capacity]
            constr += [slice_col(demand, 0, local_N, ap.R-1).T @ x @ one_S <=
                       ap.bw_cap]
            obj = cpy.Maximize(cpy.sum(cpy.log(x @ one_S)))
            prob = cpy.Problem(obj, constr)
            prob.solve(solver=cpy.SCS, verbose=False)
            new_sum_x = np.sum(slice_row(x.value, local_N-1, 0, ap.cluster.S))
            ratio = new_sum_x / np.sum(moving_agent.x)
            return  ratio > min_rel_imp, ratio
    elif opt in ['EQ', 'eq']:
        new_sum_x = moving_agent.max_task[ap.num]/(len(ap.agents)+1)
        ratio = new_sum_x / np.sum(moving_agent.x)
        return ratio > min_rel_imp, ratio


def find_mms_task(N, max_task_dct):
    E = len(max_task_dct)
    mms_dct = {k: {'num_user': 0, 'max_task': max_task_dct[k]} for k in max_task_dct}
    mms_lst = [np.inf for k in max_task_dct]
    mms = np.inf
    for i in range(N):
        mms = min(min_of_rest(mms_lst, 0), mms_dct[0]['max_task']/(mms_dct[0]['num_user']+1))
        best_e = 0
        for k in range(1, E):
            tmp = min(min_of_rest(mms_lst, k), mms_dct[k]['max_task']/(mms_dct[k]['num_user']+1))
            if tmp > mms:
                mms = tmp
                best_e = k
        mms_lst[best_e] = mms
        mms_dct[best_e]['num_user'] += 1
    for k in range(E):
        mms_dct[k]['mms'] = mms_lst[k]
    return mms_dct, np.min(mms_lst)


def min_of_rest(lst, i):
    if i == 0:
        return np.min(lst[1:])
    if i == (len(lst) - 1):
        return np.min(lst[:-1])
    return np.min(lst[0:i]+lst[i+1:])
