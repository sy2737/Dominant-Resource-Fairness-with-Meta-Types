import numpy as np



class Agent:
    """
    Hospital
    
    Args:
    - groups: list of k lists, each list contains the indices of the resources accepted in that group
    - demands: demand over groups. List of length k. 
    """
    def __init__(self, groups, demands, meta_types, budget=100, limit=None):
        if max(demands)==0:
            raise ValueError("Invalid Agent Demands")
        self.groups = groups
        self.demands = demands
        self.budget = budget
        self.meta_types = meta_types
        self.limit = limit
    def utility(self, allocation):
        """Leontief Utility"""
        return min([sum(allocation[self.groups[i]])/self.demands[i] for i in range(len(self.demands)) if self.demands[i]>0])
    