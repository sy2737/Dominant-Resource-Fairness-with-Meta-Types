import numpy as np
import mosek.fusion as m
def log(M, t, x):
    # t <= log(x), x>=0
    M.constraint(m.Expr.hstack(x, 1, t), m.Domain.inPExpCone())

def mnw(agents, supplies, meta_types):
    # 
    allocations = None
    utilities = None
    N = len(agents)
    M = len(supplies)
    with m.Model("ECP") as Mod:
        x = Mod.variable([N, M], m.Domain.greaterThan(0))
        supply_c = Mod.constraint(m.Expr.mul([1]*N,x),m.Domain.lessThan(list(supplies)))
        u = Mod.variable('u', N, m.Domain.unbounded())
        utility_constraints = []
        for i, a in enumerate(agents):
            for g, d in zip(a.groups, a.demands):
                if d==0:
                    continue
                p = np.zeros(M)
                p[g] = 1/d
                c = Mod.constraint(m.Expr.sub(u.index(i), m.Expr.dot(x.slice([i,0], [i+1, M]), p)), m.Domain.lessThan(0))
                utility_constraints.append(c)
        
        t = Mod.variable(N)
        for i, a in enumerate(agents):
            log(Mod, m.Expr.mul(t.index(i),1/a.budget), u.index(i))
        
        
        Mod.objective(m.ObjectiveSense.Maximize, m.Expr.sum(t))
        Mod.solve()
        
        allocations = np.array(np.floor(x.level()).astype(int)).reshape(N, M)
        utilities = np.array(u.level())
    return allocations

def mnw_pre_rounding(agents, supplies, meta_types):
    # 
    allocations = None
    utilities = None
    N = len(agents)
    M = len(supplies)
    with m.Model("ECP") as Mod:
        x = Mod.variable([N, M], m.Domain.greaterThan(0))
        supply_c = Mod.constraint(m.Expr.mul([1]*N,x),m.Domain.lessThan(list(supplies)))
        u = Mod.variable('u', N, m.Domain.unbounded())
        utility_constraints = []
        for i, a in enumerate(agents):
            for g, d in zip(a.groups, a.demands):
                if d==0:
                    continue
                p = np.zeros(M)
                p[g] = 1/d
                c = Mod.constraint(m.Expr.sub(u.index(i), m.Expr.dot(x.slice([i,0], [i+1, M]), p)), m.Domain.lessThan(0))
                utility_constraints.append(c)
        
        t = Mod.variable(N)
        for i, a in enumerate(agents):
            log(Mod, m.Expr.mul(t.index(i),1/a.budget), u.index(i))
        
        
        Mod.objective(m.ObjectiveSense.Maximize, m.Expr.sum(t))
        Mod.solve()
        
        allocations = np.array(np.floor(x.level())).reshape(N, M)
        utilities = np.array(u.level())
    return allocations

def discrete_mnw(agents, supplies, meta_types=None):
    allocations = None
    utilities = None
    N = len(agents)
    M = len(supplies)
    with m.Model("MIECP") as Mod:
        x = Mod.variable([N, M], m.Domain.integral(m.Domain.greaterThan(0)))
        supply_c = Mod.constraint(m.Expr.mul([1]*N,x),m.Domain.lessThan(list(supplies)))
        u = Mod.variable('u', N, m.Domain.unbounded())
        utility_constraints = []
        for i, a in enumerate(agents):
            for g, d in zip(a.groups, a.demands):
                if d<= 0:
                    continue
                p = np.zeros(M)
                p[g] = 1/d
                c = Mod.constraint(m.Expr.sub(u.index(i), m.Expr.dot(x.slice([i,0], [i+1, M]), p)), m.Domain.lessThan(0))
                utility_constraints.append(c)
        
        t = Mod.variable(N)
        for i, a in enumerate(agents):
            log(Mod, m.Expr.mul(t.index(i),1/a.budget), u.index(i))
        
        
        Mod.objective(m.ObjectiveSense.Maximize, m.Expr.sum(t))
        Mod.solve()
        
        allocations = np.array(np.round(x.level()).astype(int)).reshape(N, M)
        utilities = np.array(u.level())
    return allocations