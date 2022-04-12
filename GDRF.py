from collections import defaultdict, Counter
import numpy as np
import math
from gurobipy import *
from agent import Agent
def find_eliminated_resources(m, M):
    """Returns a list of resources that have to be eliminated this round
    """
    shadow_price = m.getAttr(GRB.Attr.Pi)
    return set(np.arange(M)[np.array(shadow_price[:M])>0])

def _is_eliminated(agent, eliminated_resources):
    for g in agent.groups:
        if len(g)==0:
            continue
        if set(g).issubset(eliminated_resources):
            return True
    return False
        
def find_eliminated_agents(agents, eliminated_resources, remaining_agents):
    eliminated_agents = set({})
    for i, a in enumerate(agents):
        if i not in remaining_agents:
            continue
        if _is_eliminated(a, eliminated_resources):
            eliminated_agents.add(i)
    return eliminated_agents
    

# implementation of group drf
def gdrf(agents, resources, meta_types, rounding=True): # if rand_rounding, then use pure random, if not use envy-based
    M = len(resources)
    N = len(agents)

    # CHECK FOR ILL-CONDITIONED INPUTS
    scaling_factor = sum([sum(a.demands) for a in agents])/sum(resources)
    resources = resources * scaling_factor

    
    d = {} # d[(i,j)] is the d_i^j value for agent i, resource group j
    for i,a in enumerate(agents):
        max_d = 0
        for j,g in enumerate(a.groups): # j is the index of the group
            group_cap = sum(resources[meta_types[j]])
            d[(i,j)] = a.demands[j]/group_cap
            max_d = max(max_d, d[(i,j)])
        for j,_ in enumerate(a.groups):
            d[(i,j)] /= max_d
    #max_budget = max([a.budget for a in agents])
    budgets = {i:a.budget for i, a in enumerate(agents)}
    gammas = {}
    
    eliminated_resources = set({})
    remaining_agents = set(range(N))
    fractional_assignments = defaultdict(lambda: defaultdict(float)) # assignment[i] = [r1:v1,...]

    round_ind = 0
    while True: # while there are still remaining agents
        # solve the LP
        #print("Round %d"%round_ind)
        #print("Remaining agents:", remaining_agents)
        m = Model("LP_round[%d]"%round_ind)
        m.setParam( 'OutputFlag', False)
        y = m.addVar(obj=1,name="y[%d]"%round_ind)
        x = m.addVars(range(N), range(M), lb=0) # x[i][r] = units of resource r assigned to agent i in current round
        m.addConstrs((x.sum("*",r) <= resources[r] for r in range(M)), name="capacity[%d]"%round_ind)
        for i, a in enumerate(agents):
            if i in remaining_agents:
                for j,g in enumerate(a.groups):
                    if len(g)==0:
                        continue
                    constr = sum([x[i,r] for r in g]) == sum(resources[meta_types[j]]) * y * budgets[i]  * d[(i,j)]
                    m.addConstr(constr, name="equal_round_drf[%d,%d,%d]"%(round_ind,i,j))
            else:
                for j,g in enumerate(a.groups):
                    if len(g)==0:
                        continue
                    constr = sum([x[i,r] for r in g]) == sum(resources[meta_types[j]]) * gammas[i] * budgets[i] * d[(i,j)]
                    m.addConstr(constr, name="equal_round_drf[%d,%d,%d]"%(round_ind,i,j))
        m.modelSense = GRB.MAXIMIZE
        m.optimize()

        # update remaining resource
        new_elimination = find_eliminated_resources(m, M)
        if new_elimination.issubset(eliminated_resources):
            raise ValueError("No resource is being eliminated. Instance mightbe ill-conditioned. Try rescaling the inputs so that every thing is roughly on the same scale and not too large/small.")
        eliminated_resources.update(
            new_elimination
        )
        # update remaining agents
        eliminated_agents = find_eliminated_agents(agents, eliminated_resources, remaining_agents)
        for i in eliminated_agents:
            gammas[i] = y.x
        remaining_agents -= eliminated_agents
        
        if not remaining_agents:
            for (i,r) in x.keys():
                fractional_assignments[i][r] = x[(i,r)].x
            break

        round_ind += 1
    #print("GDRF took {} rounds".format(round_ind))
    # rounding

    assignments = np.array([[fractional_assignments[i][r] for r in range(M)] for i in range(N)]) / scaling_factor
    if rounding:
        return assignments.astype(int)
    else:
        return assignments
