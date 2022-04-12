import numpy as np
from gurobipy import *

def random_hospital_groups_demands(meta_types, budget_random=False, limit=None):
    L = len(meta_types)
    valid=False
    while not valid:
        groups = []
        for Omega_l in meta_types:
            size = np.random.randint(0, len(Omega_l)+1)
            if size>0:
                valid = True
            groups.append(np.random.choice(Omega_l, size, replace=False))
    
    # Up to 5 times more demand for one resource than another
    demands = [np.random.randint(1, 10) if len(groups[l])>0 else 0 for l in range(L)]
    
    if budget_random:
        budget = np.random.randint(1, 10)
    else:
        budget = 1
    return groups, demands, meta_types, budget, limit

def social_welfare(agents, allocations, supplies):
    return sum([a.utility(alloc) for a, alloc in zip(agents, allocations)])

def nash_welfare(agents, allocations):
    return np.product([a.utility(alloc) for a, alloc in zip(agents, allocations)])

def market_clearing_error(resources, allocations):
    return sum(abs(resources-np.sum(allocations, axis=0)))

def MMS(supplies, agent, N):
    """Note that MMS does not take into account of budget"""
    m1 = Model("MMS")
    m1.setParam('OutputFlag', False)
    u = m1.addVar(lb=0)
    # N by M assignment variables
    x = np.array([[m1.addVar(vtype=GRB.INTEGER, lb=0, ub=s) for s in supplies] for _ in range(N)])
    
    for i in range(N):
        for g, d in zip(agent.groups, agent.demands):
            if d==0:
                continue
            m1.addConstr(u<= sum(x[i,g]/d))

    m1.setObjective(u)
    m1.modelSense = GRB.MAXIMIZE
    m1.optimize()
    return u.x

def min_MMS_ratio(agents, allocations, supplies):
    N = len(agents)
    return min([a.utility(l)/MMS(supplies, a, N) for a, l in zip(agents, allocations)])


def envy(a, b, l_a, l_b):
    """maximum weighted envy between agents a and b
    weighted envy is the difference between bang per buck
    """
    envy_a = (a.utility(l_b)/b.budget*a.budget - a.utility(l_a))/a.utility(l_a)
    envy_b = (b.utility(l_a)/a.budget*b.budget - b.utility(l_b))/b.utility(l_b)
    return max([envy_a, envy_b])

def max_envy(agents, allocations, supplies, supply_weighted = False):
    max_envy = 0
    N = len(agents)
    for i in range(N):
        for j in range(i+1, N):
            e = envy(agents[i], agents[j], allocations[i], allocations[j])
            if e > max_envy:
                max_envy = e
    return max_envy
