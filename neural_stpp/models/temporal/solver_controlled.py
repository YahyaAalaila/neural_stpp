import torch
from torchdiffeq import odeint

class TimeVariableODEControlled:
    def __init__(self, func, atol=1e-6, rtol=1e-6, method='dopri5'):
        self.func  = func
        self.atol  = atol
        self.rtol  = rtol
        self.method = method

    def integrate_interval(self, t0, t1, state):
        # t0,t1: (N,) or (1,) in repo; collapse to scalars per batch by broadcasting
        ts = torch.stack([t0, t1], dim=0) if t0.ndim else torch.tensor([t0, t1], device=state[0].device, dtype=state[0].dtype)
        sol = odeint(self.func, state, ts, atol=self.atol, rtol=self.rtol, method=self.method)
        return (sol[0][-1], sol[1][-1])
