import torch
import torch.nn as nn
import torch.nn.functional as F

class ControlledDrift(nn.Module):
    # dh/dt = f(h) + G(h) ⊙ Pz(z) + Pt(rho_t)   (rho_t optional)
    def __init__(self, hdim, zdim=0, tfeat_dim=0):
        super().__init__()
        self.f  = nn.Sequential(nn.Linear(hdim, hdim), nn.Tanh(), nn.Linear(hdim, hdim))
        self.G  = nn.Sequential(nn.Linear(hdim, hdim), nn.Tanh(), nn.Linear(hdim, hdim))
        self.Pz = nn.Linear(zdim, hdim, bias=False) if zdim > 0 else None
        self.Pt = nn.Linear(tfeat_dim, hdim, bias=False) if tfeat_dim > 0 else None

    def forward(self, t, h, z=None, rho_t=None):
        base = self.f(h)
        add_z = self.G(h) * self.Pz(z) if (self.Pz is not None and z is not None) else 0.0
        add_t = self.Pt(rho_t) if (self.Pt is not None and rho_t is not None) else 0.0
        return base + add_z + add_t

class IntensityHead(nn.Module):
    # log λ(t) = a(h) + βᵀ z   ⇒ λ = softplus(log λ)
    def __init__(self, hdim, zdim=0):
        super().__init__()
        self.a = nn.Linear(hdim, 1)
        self.beta = nn.Linear(zdim, 1, bias=False) if zdim > 0 else None
    def forward(self, h, z=None):
        loglam = self.a(h)
        if self.beta is not None and z is not None:
            loglam = loglam + self.beta(z)
        return F.softplus(loglam).squeeze(-1)

class IntensityODEFuncControlled(nn.Module):
    """
    Holds current control z for the active interval; solver sets it before each integrate.
    State is (Lambda, h).
    """
    def __init__(self, hdim, zdim=0, tfeat_dim=0):
        super().__init__()
        self.drift = ControlledDrift(hdim, zdim=zdim, tfeat_dim=tfeat_dim)
        self.head  = IntensityHead(hdim, zdim=zdim)
        self._z = None
        self._rho = None

    def set_control(self, z):  self._z = z
    def set_time_features(self, rho): self._rho = rho

    def forward(self, t, state):
        Lambda, h = state
        z = self._z
        rho = self._rho
        dh = self.drift(t, h, z=z, rho_t=rho)
        lam = self.head(h, z=z)
        dLambda = lam
        return (dLambda, dh)
