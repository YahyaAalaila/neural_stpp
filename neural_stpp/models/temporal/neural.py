# Copyright (c) Facebook, Inc. and its affiliates.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import dtype
from torchdiffeq import odeint_adjoint as odeint

from neural_stpp import diffeq_layers
from .basic import TemporalPointProcess


ACTFNS = {
    "softplus": lambda dim: diffeq_layers.diffeq_wrapper(nn.Softplus()),
    "swish": lambda dim: diffeq_layers.diffeq_wrapper(Swish(dim)),
    "celu": lambda dim: diffeq_layers.diffeq_wrapper(nn.CELU()),
    "relu": lambda dim: diffeq_layers.diffeq_wrapper(nn.ReLU(inplace=True))
}


def construct_diffeqnet(input_dim, hidden_dims, output_dim, time_dependent=False, actfn="softplus", zero_init=False, gated=False):

    linear_fn = diffeq_layers.IgnoreLinear if time_dependent else diffeq_layers.ConcatLinear_v2

    if gated:
        linear_fn = GatedLinear

    layers = []
    if len(hidden_dims) > 0:
        dims = [input_dim] + list(hidden_dims)
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(linear_fn(d_in, d_out))
            layers.append(ActNorm(d_out))
            if not gated:
                layers.append(ACTFNS[actfn](d_out))
        layers.append(linear_fn(hidden_dims[-1], output_dim))
    else:
        layers.append(linear_fn(input_dim, output_dim))

    # Initialize to zero.
    if zero_init:
        for m in layers[-1].modules():
            if isinstance(m, nn.Linear):
                m.weight.data.fill_(0)
                if m.bias is not None:
                    m.bias.data.fill_(0)
    diffeqnet = diffeq_layers.SequentialDiffEq(*layers)

    return diffeqnet


# class Sine(nn.Module):

#     def forward(self, x):
#         return torch.sin(x)


class Swish(nn.Module):

    def __init__(self, dim=1):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.5] * dim))

    def forward(self, x):
        return (x * torch.sigmoid_(x * F.softplus(self.beta)))

    def extra_repr(self):
        return f'{self.beta.nelement()}'


class GatedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.gate = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x) * torch.sigmoid(self.gate(x))



class ActNorm(nn.Module):

    def __init__(self, num_features, init_scale=1.0):
        super(ActNorm, self).__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.init_scale = init_scale
        self.register_buffer('initialized', torch.tensor(0))

    def forward(self, x):
        if not self.initialized:
            with torch.no_grad():
                # compute batch statistics
                x_ = x.reshape(-1, x.shape[-1])
                batch_mean = torch.mean(x_, dim=0)
                batch_var = torch.var(x_, dim=0)

                # for numerical issues
                batch_var = torch.max(batch_var, torch.tensor(0.2).to(batch_var))

                self.bias.data.copy_(-batch_mean)
                self.weight.data.copy_(-0.5 * torch.log(batch_var) + math.log(self.init_scale))
                self.initialized.fill_(1)

        bias = self.bias.expand_as(x)
        weight = self.weight.expand_as(x)

        # y = (x + bias) * torch.exp(weight)
        y = (x + bias) * F.softplus(weight)

        return y

    def __repr__(self):
        return ('{name}({num_features})'.format(name=self.__class__.__name__, **self.__dict__))


class IntensityODEFunc(nn.Module):
    def __init__(self, hdim, dstate_fn, intensity_fn):
        super().__init__()
        self.hdim = hdim
        self.dstate_fn = dstate_fn
        self.intensity_fn = intensity_fn

    # def forward(self, t, state):
    #     Lambda, tpp_state = state
    #     intensity = self.get_intensity(tpp_state).reshape(-1)
    #     return intensity, self.dstate_fn(t, tpp_state)
    
    # change for multi-mark extension
    # def forward(self, t, state):
    #     Lambda, tpp_state = state
    #     lam_k = self.get_intensity(tpp_state)         # [N,K]
    #     lam_sum = lam_k.sum(-1)                   # [N]
    #     dLambda = lam_sum
    #     dstate  = self.dstate_fn(t, tpp_state)
    #     return dLambda, dstate
    def forward(self, t, state):
        Lambda, h = state
        lam_k = self.get_intensity(h)            # [B, K]
        lam_sum = lam_k.sum(-1)                  # [B]
        if (lam_sum < 0).any() or torch.isnan(lam_sum).any():
            print(f"[odefunc] NEG/NAN lam_sum at t={t.detach().mean().item():.6f} "
              f"min={lam_sum.min().item():.3e} max={lam_sum.max().item():.3e}")

        dLambda = lam_sum                  # [B]
        dstate  = self.dstate_fn(t, h)
        return dLambda, dstate


    ############################################
    def get_intensity(self, tpp_state):
        x = tpp_state[..., :self.hdim]
        weight_dtype = next(self.intensity_fn.parameters()).dtype
        if x.dtype != weight_dtype:
            x = x.to(weight_dtype)
        return  torch.sigmoid(self.intensity_fn(x) - 2.0) * 50.0
    #F.softplus(self.intensity_fn(x)) + 1e-8 # change for multi-mark extension
        # return torch.sigmoid(self.intensity_fn(x) - 2.0) * 50 

class SplitHiddenStateODEFunc(nn.Module):
    def __init__(self, dstate_net, update_net):
        super().__init__()
        self.dstate_net = dstate_net
        self.update_net = update_net

    def forward(self, t, tpp_state):
        dstate = self.dstate_net(t, tpp_state)
        c, h = torch.split(tpp_state, tpp_state.shape[1] // 2, dim=1)
        dcdt, dhdt = torch.split(dstate, tpp_state.shape[1] // 2, dim=1)
        dcdt = dcdt - (dcdt * c).sum(dim=-1, keepdim=True) / (c * c).sum(dim=-1, keepdim=True) * c
        dhdt = -F.softplus(dhdt) * h
        return torch.cat([dcdt, dhdt], dim=1)

    def update_state(self, t, tpp_state, cond=None):
        inputs = torch.cat([tpp_state, cond], dim=1) if cond is not None else tpp_state
        upd_c, upd_h = torch.split(self.update_net(t, inputs), tpp_state.shape[1] // 2, dim=1)
        update = torch.cat([torch.zeros_like(upd_c), upd_h], dim=1)
        return tpp_state + update

class SimpleHiddenStateODEFunc(nn.Module):
    def __init__(self, dstate_net, update_net):
        super().__init__()
        self.dstate_net = dstate_net
        self.update_net = update_net

    def forward(self, t, tpp_state):
        return torch.tanh(self.dstate_net(t, tpp_state))

    def update_state(self, t, tpp_state, cond=None):
        inputs = torch.cat([tpp_state, cond], dim=1) if cond is not None else tpp_state
        return self.update_net(t, inputs)

class GRUHiddenStateODEFunc(nn.Module):
    def __init__(self, dstate_net, update_net):
        super().__init__()
        self.dstate_net = dstate_net
        self.update_net = update_net

    def forward(self, t, tpp_state):
        return self.dstate_net(t, tpp_state)

    def update_state(self, t, tpp_state, cond=None):
        cond = cond if cond is not None else torch.zeros(tpp_state.shape[0], 0, device=tpp_state.device)
        return self.update_net(cond, tpp_state)

class HiddenStateODEFuncList(nn.Module):
    def __init__(self, *odefuncs):
        super().__init__()
        self.odefuncs = nn.ModuleList(odefuncs)

    def forward(self, t, tpp_state):
        states = torch.split(tpp_state, tpp_state.shape[-1] // len(self.odefuncs), dim=-1)
        ds = [func(t, s) for s, func in zip(states, self.odefuncs)]
        return torch.cat(ds, dim=-1)

    def update_state(self, t, tpp_state, cond=None):
        states = torch.split(tpp_state, tpp_state.shape[-1] // len(self.odefuncs), dim=-1)
        upds = [func.update_state(t, s, cond) for s, func in zip(states, self.odefuncs)]
        return torch.cat(upds, dim=-1)

class NeuralPointProcess(TemporalPointProcess):
    dynamics_dict = {"split": SplitHiddenStateODEFunc, "simple": SimpleHiddenStateODEFunc, "gru": GRUHiddenStateODEFunc}
    def __init__(self, num_marks, cond_dim=0, hidden_dims=[64, 64, 64], cond=False, style="split", actfn="softplus", hdim=None, separate=1, tol=1e-6, otreg_strength=0.1):
        super().__init__()
        
        if not cond:
            cond_dim = 0
        # Changes for multi-mark extension
        self.K = num_marks
        #################################
        self.cond = cond
        self.cond_dim = cond_dim
        self.hdim = hidden_dims[0] if hdim is None else hdim
        assert self.hdim % 2 == 0
        self._init_state = nn.Parameter(torch.randn(hidden_dims[0]) / math.sqrt(hidden_dims[0]))

        dynamics = []
        for i in range(separate):
            dstate_net = construct_diffeqnet(hidden_dims[0] // separate, hidden_dims[1:], hidden_dims[0] // separate, time_dependent=False, actfn=actfn, zero_init=True)
            if style in ["split", "simple"]:
                update_net = construct_diffeqnet(hidden_dims[0] // separate + cond_dim, hidden_dims[1:], hidden_dims[0] // separate, time_dependent=False, actfn="celu", gated=True, zero_init=False)
            elif style in ["gru"]:
                update_net = nn.GRUCell(cond_dim, hidden_dims[0] // separate)
            dynamics.append(self.dynamics_dict[style](dstate_net, update_net))

        self.hidden_state_dynamics = HiddenStateODEFuncList(*dynamics)

        # intensity_net = nn.Sequential(nn.Linear(self.hdim, self.hdim * 4), nn.Softplus(), nn.Linear(self.hdim * 4, 1))
        # changes for multi-mark extension
        intensity_net = nn.Sequential(
            nn.Linear(self.hdim, self.hdim * 4), nn.Softplus(),
            nn.Linear(self.hdim * 4, self.K)     # CHANGED: K outputs
        )
        #################################
        intensity_odefunc = IntensityODEFunc(self.hdim, self.hidden_state_dynamics, intensity_net)
        self.ode_solver = TimeVariableODE(intensity_odefunc, atol=tol, rtol=tol, method="dopri5", energy_regularization=otreg_strength)

    def logprob(self, event_times, spatial_locations, input_mask, t0, t1, marks=None): # changed for multi-mark extension (marks=None)
        intensities, Lambda, _ = self.integrate_lambda(event_times, spatial_locations, input_mask, t0, t1)
        
        # changed for multi-mark extension
        if marks is None:
            # Backward-compat: single-mark path (K can be 1). Accept [N,T,K] or [N,T].
            if intensities.dim() == 3:
                lam = intensities.squeeze(-1)            # [N, T, 1] -> [N, T]
            else:
                lam = intensities                         # [N, T]
        else:
            # Multi-mark: pick λ for each event’s mark
            # marks: [N, T] (long), intensities: [N, T, K]
            lam = intensities.gather(-1, marks.unsqueeze(-1)).squeeze(-1)  # [N, T]

        # mask and sum log λ_m(t_i)
        log_lam = lam.clamp_min(1e-8).log()
        if input_mask is not None:
            log_lam = torch.where(input_mask.bool(), log_lam, torch.zeros_like(log_lam))

        return log_lam.sum(dim=1) - Lambda
        ########################################### (previous version is below)
        # log_intensities = torch.log(intensities + 1e-8)
        # log_intensities = torch.where(input_mask.bool(), log_intensities, torch.zeros_like(log_intensities))
        # return torch.sum(log_intensities, dim=1) - Lambda
    def get_intensity(self, state):
        return self.ode_solver.func.get_intensity(state)

    def integrate_lambda(self, event_times, spatial_location, input_mask, t0, t1, nlinspace=1):
        if not self.cond:
            spatial_location = None
        target_dtype = next(self.ode_solver.func.intensity_fn.parameters()).dtype
        device = event_times.device
        event_times      = event_times.to(dtype=target_dtype, device=device)
        if spatial_location is not None:
            spatial_location = spatial_location.to(dtype=target_dtype, device=device)
            
                # build float32 t0 / t1
        if not torch.is_tensor(t0):
            t0 = torch.tensor(t0, dtype=torch.float32, device=device)
        else:
            t0 = t0.to(dtype=torch.float32, device=device)
        t0 = t0.expand(event_times.size(0))

        if t1 is not None:
            if not torch.is_tensor(t1):
                t1 = torch.tensor(t1, dtype=torch.float32, device=device)
            else:
                t1 = t1.to(dtype=torch.float32, device=device)
            t1 = t1.expand(event_times.size(0))
            
        N, T = event_times.shape

        input_mask = input_mask.bool() if input_mask is not None else torch.ones_like(event_times, dtype=torch.bool)
        
     # ---------- DEBUG BLOCK 1: time monotonicity on *masked* steps ----------

        with torch.no_grad():
            # lengths per sequence
            L = input_mask.sum(dim=1)  # [N]
            # diffs only where both i and i+1 are valid
            valid_pairs = input_mask[:, 1:] & input_mask[:, :-1]
            dt = (event_times[:, 1:] - event_times[:, :-1])[valid_pairs]
            if dt.numel() > 0:
                bad = (dt < 0)
                if bad.any():
                    i = bad.nonzero(as_tuple=False)[0]
                    print(f"[temporal] NON-MONOTONE times: dt.min={dt.min().item():.6e}, "
                        f"example seq={i[0].item()} pos={i[1].item()}")
      # -----------------------------------------------------------------------
                  
                    
        state = (torch.zeros(N, device=self._init_state.device, dtype=self._init_state.dtype),
                      self._init_state[None].expand(N, -1))

        t0 = (torch.tensor(t0, device=event_times.device, dtype=event_times.dtype).expand(N)) if not torch.is_tensor(t0) else t0.expand(N).to(event_times)

        self.ode_solver.nfe = 0
        intensities = [] # will become [N, T, K]
        prejump_hidden_states = []
        prev_Lambda = torch.zeros(N, device=event_times.device, dtype=self._init_state.dtype)

        for i in range(T):

            # Set t1 = t0 if the input is masked out at time t1.
            t1_i = torch.where(input_mask[:, i], event_times[:, i], t0)
            
            # ---------- DEBUG BLOCK 2: step size and Λ monotonicity per step ----
            with torch.no_grad():
                r = (t1_i - t0)
                if r.numel():
                    r_min, r_max = r.min().item(), r.max().item()
                    if r_min < 0:
                        print(f"[temporal] NEGATIVE step at i={i}: min(dt)={r_min:.6e}, max(dt)={r_max:.6e}")
            # -------------------------------------------------------------------

            state_traj = self.ode_solver.integrate(t0, t1_i, state, nlinspace=nlinspace, method="dopri5" if self.training else "dopri5")
            # # --- probe before integrate ---
            # dt = (torch.where(input_mask[:, i], event_times[:, i], t0) - t0)
            # n_zero = (dt.abs() < 1e-12).sum().item()
            # n_pos  = (dt > 0).sum().item()
            # print(f"[integrate_lambda] step={i} n_zero={n_zero} n_pos={n_pos} "
            #     f"dt_min={dt.clamp_min(0).min().item():.3e} dt_max={dt.max().item():.3e}")
            # # ---------------------------------------------------------------
            dt_full = (t1_i - t0)  # [B]
            dt_min  = dt_full.min().item()
            dt_neg  = (dt_full < 0).sum().item()
            print(f"[probe/dt] i={i} min={dt_min:.3e} neg={dt_neg}")
            
            hiddens = state_traj[1]  # (1 + nlinspace, N, D)
            if i > 0:
                hiddens = hiddens[1:]
            # set hidden states to zero if input is masked out at the next time step.
            hiddens = torch.where(input_mask[:, i].reshape(1, -1, 1).expand_as(hiddens), hiddens, torch.zeros_like(hiddens))
            prejump_hidden_states.append(hiddens)

            state = tuple(s[-1] for s in state_traj)
            Lambda, tpp_state = state
            # intensities.append(self.get_intensity(tpp_state).reshape(-1))
            # ---------- DEBUG BLOCK 3: Λ should never decrease -------------------
            with torch.no_grad():
                dec = (Lambda < prev_Lambda - 1e-9)
                if dec.any():
                    j = dec.nonzero(as_tuple=False)[0,0].item()
                    print(f"[temporal] Λ DECREASE at i={i}, seq={j}: "
                        f"prev={prev_Lambda[j].item():.6e}, now={Lambda[j].item():.6e}, "
                        f"dt={(t1_i[j]-t0[j]).item():.6e}, t0={t0[j].item():.6e}, t1={t1_i[j].item():.6e}")
            prev_Lambda = Lambda.detach()
            # --------------------------------------------------------------------

            # changed for multi-mark extension: append per-mark intensities at t_i^- : shape [N, K]   ← CHANGED line
            intensities.append(self.get_intensity(tpp_state))     # no reshape!
            if i < T - 1 or t1 is not None:
                cond = spatial_location[:, i] if spatial_location is not None else None
                updated_tpp_state = self.hidden_state_dynamics.update_state(event_times[:, i], tpp_state, cond=cond)
                tpp_state = torch.where(input_mask[:, i].reshape(-1, 1).expand_as(tpp_state), updated_tpp_state, tpp_state)
                state = (Lambda, tpp_state)

            # Track t0 as the last valid event time.
            t0 = torch.where(input_mask[:, i], event_times[:, i], t0)

        if t1 is not None:
            # Integrate from last time sample to t1.
            t1 = t1 if torch.is_tensor(t1) else torch.tensor(t1)
            t1 = t1.expand(N).to(event_times)
            state_traj = self.ode_solver.integrate(t0, t1, state, nlinspace=nlinspace, method="dopri5" if self.training else "dopri5")
            with torch.no_grad():
                dt_tail = t1 - t0
                print(f"[tail] dt_tail min={dt_tail.min().item():.6e} "
                    f"max={dt_tail.max().item():.6e} "
                    f"neg_count={(dt_tail < 0).sum().item()}")
                if (dt_tail < 0).any():
                    bad = (dt_tail < 0).nonzero(as_tuple=False)[:5, 0]
                    for b in bad.tolist():
                        print(f"[tail] row={b} t0={float(t0[b]):.6f} t1={float(t1[b]):.6f}")
            # dt_full = (t1 - t0)  # [B]
            # dt_min  = dt_full.min().item()
            # dt_neg  = (dt_full < 0).sum().item()
            # print(f"[probe/dt] i={i} min={dt_min:.3e} neg={dt_neg}")
            
            hiddens = state_traj[1][1:]
            prejump_hidden_states.append(hiddens)

            state = tuple(s[-1] for s in state_traj)

        intensities = torch.stack(intensities, dim=1) # stack to [N, T, K]
        prejump_hidden_states = torch.cat(prejump_hidden_states, dim=0).transpose(0, 1)  # (N, T * nlinspace, D)
        #print("prejump_hidden_states shape:", prejump_hidden_states.shape)
        return intensities, state[0], prejump_hidden_states
    
    def get_last_hidden(self, event_times, spatial_locations, input_mask, t0, t1):
        _, _, hidden = self.integrate_lambda(event_times, spatial_locations, input_mask, t0, t1)
        
        #print(f"[DEBUG] integrate_lambda → hidden.shape: {hidden.shape}")
        #hidden = hidden.transpose(0, 1)  # supposed to give [B, T, D]
        
        #print(f"[DEBUG] transposed hidden.shape: {hidden.shape}")
        # B = hidden.shape[0]
        # T = hidden.shape[1]
        # lengths = input_mask.sum(dim=1).long().clamp_max(T-1) - 1
        # row_indices = torch.arange(B, device=hidden.device)
        # print(f"[DEBUG] row_indices: {row_indices.shape}, lengths: {lengths.shape}")
        # return hidden[row_indices, lengths]
        B, T, _ = hidden.shape
        # lengths = number of valid history steps (mask is the history mask you passed in)
        lengths = input_mask.sum(dim=1).long()           # in [1..T]
        idx = (lengths - 1).clamp(min=0)                 # last valid index in history
        row = torch.arange(B, device=hidden.device)
        return hidden[row, idx]  # [B, D]

    
    def intensity_per_mark(self, tau, hidden_state):
        """
        Evolves the hidden state over tau, and returns per-mark intensities: [B, K]
        """
        B, D = hidden_state.shape
        K = self.K

        # Evolve hidden state using the ODE
        t0 = torch.zeros_like(tau)
        t1 = tau
        state = (torch.zeros(B, device=tau.device), hidden_state)
        state_traj = self.ode_solver.integrate(t0, t1, state, method="dopri5", allow_grad=True)
        
        
        
        evolved_hidden = state_traj[1][-1]
        lambda_k = self.get_intensity(evolved_hidden)     # [B, K]
        lam_sum = lambda_k.sum(-1)                      # [B]
        
        return lambda_k
        # state_traj = self.ode_solver.integrate(t0, t1, state, method="dopri5")
        # evolved_hidden = state_traj[1][-1]  # [B, D]

        # lambda_k = self.get_intensity(evolved_hidden)  # [B, K]
        # return lambda_k
    
    def sample_mark(self, lambda_k):
        """
        Samples mark k ∈ {0,...,K-1} given λ_k, shape: [B]
        """
        probs = lambda_k / lambda_k.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return torch.distributions.Categorical(probs).sample()
  
    def intensity_and_score(self, tau, hidden_state):
        """
        tau: [B] leaf variable, requires_grad=True
        hidden_state: [B, H] treated as constant
        Returns:
        lambda_k_detached [B, K], score_detached [B]
        """

        # We only need grad wrt tau. Ensure it is a leaf with requires_grad.
        if not tau.requires_grad:
            tau = tau.detach().requires_grad_(True)
            #print("🧭 [intensity_and_score] forced tau.requires_grad_(True)")

        with torch.enable_grad():
            lambda_k = self.intensity_per_mark(tau, hidden_state)     # [B, K]
            log_lam_sum = torch.log(lambda_k.clamp_min(1e-8)).sum()   # scalar
            grad_tau = torch.autograd.grad(log_lam_sum, tau, create_graph=False, retain_graph=False)[0]  # [B]

        #_dbg("SCORE.OUT", lambda_k=lambda_k, grad_tau=grad_tau)
        return lambda_k.detach(), grad_tau.detach()


class TimeVariableODE(nn.Module):
    start_time = 0.0
    end_time = 1.0

    def __init__(self, func, atol=1e-6, rtol=1e-6, method="dopri5", energy_regularization=0.01):
        super().__init__()
        
        self.func = func
        self.atol = float(atol)
        self.rtol = float(rtol)
        self.method = method
        self.energy_regularization = energy_regularization
        self.nfe = 0
        self._allow_grad = True   # <— NEW

    def integrate(self, t0, t1, x0, nlinspace=1, method=None, allow_grad=True):
        self._allow_grad = allow_grad
        dtype = x0[0].dtype    # x0[0] is your t0 in the new dtype
        device = t0.device

        # recast t0/t1 right here
        t0 = t0.to(dtype=dtype, device=device)
        t1 = t1.to(dtype=dtype, device=device)

        # now build the rest as before
        t     = torch.linspace(self.start_time, self.end_time,
                            nlinspace + 1,
                            device=device, dtype=dtype)
        reg0  = torch.zeros(1, device=device, dtype=dtype)
        # rtol  = torch.tensor(self.rtol, device=device, dtype=dtype)
        # atol  = torch.tensor(self.atol, device=device, dtype=dtype)

        method = method or self.method
        # build time grid in float32 on correct device
        t = torch.linspace(self.start_time, self.end_time, nlinspace + 1,
                           device=t0.device, dtype=x0[0].dtype)
        # prepare regularization tensor
        reg0 = torch.zeros(1, device=t0.device, dtype=x0[0].dtype)
        # cast tolerances
        # self.rtol = torch.tensor(self.rtol, device=t0.device, dtype=x0[0].dtype)
        # self.atol = torch.tensor(self.atol, device=t0.device, dtype=x0[0].dtype)
        rtol_t = torch.as_tensor(self.rtol, device=device, dtype=dtype)
        atol_t = torch.as_tensor(self.atol, device=device, dtype=dtype)
        # solve
        solution = odeint(
            self,
            (t0, t1, reg0, *x0),
            t,
            # rtol=float(self.rtol),   # <<–– Python float
            # atol=float(self.atol),   # <<–– Python float
            rtol=float(rtol_t.item()),   # pass Python floats to the solver
            atol=float(atol_t.item()),
            method=method,
            options={                   # <-- force the solver to float32
        "dtype": torch.float32
    }
        )
        _, _, energy, *xs = solution
        Lambda_traj, hidden_traj = xs[0], xs[1]  # expects x0=(Λ,h)
        # Monotonic Λ checks on the raw trajectory:
        with torch.no_grad():
            dΛ = Lambda_traj[1:] - Lambda_traj[:-1]       # [S-1, B]
            dec = (dΛ < -1e-7)                            # tolerate tiny fp
            if dec.any():
                s, b = torch.nonzero(dec, as_tuple=True)
                s, b = int(s[0]), int(b[0])
                print(f"[solver] Λ decreased at step {s} for sample {b}: "
                    f"prev={Lambda_traj[s,b].item():.6e} "
                    f"now={Lambda_traj[s+1,b].item():.6e}")
            if (Lambda_traj.min() < -1e-5) or torch.isnan(Lambda_traj).any():
                print(f"[solver] Λ min={Lambda_traj.min().item():.3e} "
                    f"max={Lambda_traj.max().item():.3e} "
                    f"rtol={self.rtol} atol={self.atol} method={method}")
        
        reg = energy * self.energy_regularization
        return WrapRegularization.apply(reg, *xs)
    
    def forward(self, s, state):
        #print(f"⏱️ [TimeVariableODE.forward] grad_enabled={torch.is_grad_enabled()}")

        self.nfe += 1
        t0, t1, _, *x = state
        ratio = (t1 - t0) / (self.end_time - self.start_time)
        #print("⏱️ [TimeVariableODE.forward] ratio:", ratio)
        t = (s - self.start_time) * ratio + t0
        with torch.enable_grad():
            x = tuple(x_.requires_grad_(True) for x_ in x)
            dx = self.func(t, x)
            dx = tuple(dx_ * ratio.reshape(-1, *([1] * (dx_.ndim - 1))) for dx_ in dx)
            d_energy = sum(torch.sum(dx_ * dx_) for dx_ in dx) / sum(x_.numel() for x_ in x)

        # *** DO NOT detach when we need gradients during evaluation ***
        if (not self.training) and (not self._allow_grad):
            dx = tuple(dx_.detach() for dx_ in dx)

        return (torch.zeros_like(t0), torch.zeros_like(t1), d_energy, *dx)


class WrapRegularization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, reg, *x):
        ctx.save_for_backward(reg)
        return x

    @staticmethod
    def backward(ctx, *grad_x):
        reg, = ctx.saved_variables
        return (torch.ones_like(reg), *grad_x)

