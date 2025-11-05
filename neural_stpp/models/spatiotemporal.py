# Copyright (c) Facebook, Inc. and its affiliates.

from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
from .spatial import JumpCNF, SelfAttentiveCNF, ConditionalGMM
from .temporal import NeuralPointProcess


class SpatiotemporalModel(nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def forward(self, event_times, spatial_locations, input_mask, t0, t1):
        """
        Args:
            event_times: (N, T)
            spatial_locations: (N, T, D)
            input_mask: (N, T)
            t0: () or (N,)
            t1: () or (N,)
        """
        pass

    @abstractmethod
    def spatial_conditional_logprob_fn(self, t, event_times, spatial_locations, t0, t1):
        pass


class CombinedSpatiotemporalModel(SpatiotemporalModel):

    def __init__(self, spatial_model, temporal_model):
        super().__init__()
        self.spatial_model = spatial_model
        self.temporal_model = temporal_model

    def forward(self, event_times, spatial_locations, input_mask, t0, t1):
        space_loglik = self._spatial_logprob(event_times, spatial_locations, input_mask)
        time_loglik = self._temporal_logprob(event_times, spatial_locations, input_mask, t0, t1)
        return space_loglik, time_loglik

    def spatial_conditional_logprob_fn(self, t, event_times, spatial_locations, t0, t1):
        return self.spatial_model.spatial_conditional_logprob_fn(t, event_times, spatial_locations)

    def _spatial_logprob(self, event_times, spatial_locations, input_mask):
        return self.spatial_model.logprob(event_times, spatial_locations, input_mask)

    def _temporal_logprob(self, event_times, spatial_locations, input_mask, t0, t1):
        return self.temporal_model.logprob(event_times, spatial_locations, input_mask, t0, t1)


class SharedHiddenStateSpatiotemporalModel(SpatiotemporalModel, metaclass=ABCMeta):

    def __init__(self, dim=2, hidden_dims=[64, 64, 64], tpp_hidden_dims=[8, 20], tpp_cond=False, tpp_style="split",
                 actfn="softplus", tpp_actfn="softplus", zero_init=True, share_hidden=False, solve_reverse=False, tpp_otreg_strength=0.0, tol=1e-6, **kwargs):
        super().__init__()
        num_marks = kwargs.pop("num_marks", 1)
        tpp_hidden_dims = [h for h in tpp_hidden_dims]
        self.temporal_model = NeuralPointProcess(
            num_marks = num_marks, cond_dim=dim, hidden_dims=tpp_hidden_dims, cond=tpp_cond, style=tpp_style, actfn=tpp_actfn, hdim=tpp_hidden_dims[0] // 2,
            separate=2 if not share_hidden else 1, tol=tol, otreg_strength=tpp_otreg_strength)
        
        self._build_spatial_model(dim, hidden_dims, actfn, zero_init, aux_dim=tpp_hidden_dims[0] // 2,
                                  aux_odefunc=self.temporal_model.hidden_state_dynamics if solve_reverse else zero_diffeq,
                                  tol=tol, **kwargs)
        

    @abstractmethod
    def _build_spatial_model(self, dim, hidden_dims, actfn, zero_init, aux_dim, aux_odefunc, **kwargs):
        pass
    
    def forward(self, event_times, spatial_locations, input_mask, t0, t1, marks=None):
        intensities, Lambda, hidden_states = self.temporal_model.integrate_lambda(
            event_times, spatial_locations, input_mask, t0, t1
        )  # intensities: (N, T, K), Lambda: (N, K)
        # if not torch.isfinite(Lambda).all() or (Lambda < 0).any():
        #     m, M = Lambda.min().item(), Lambda.max().item()
        #     print(f"[temporal-probe] Λ finite={torch.isfinite(Lambda).all().item()} "
        #         f"min={m:.6e} max={M:.6e}")
        #print("[SharedHiddenStateSpatiotemporalModel.forward] uniques:", torch.unique(marks).tolist())

        # Select the intensity for the observed mark at each event
        if marks is not None:
            # marks: (N, T) long in [0, K-1]
            chosen = torch.gather(intensities, dim=2, index=marks.long().unsqueeze(-1)).squeeze(-1)  # (N, T)
        else:
            # single-mark case: intensities might be (N, T) or (N, T, 1)
            chosen = intensities if intensities.dim() == 2 else intensities[..., 0]

        logI = torch.log(chosen + 1e-8)  # (N, T)

        # Survival is sum over marks (scalar per sequence)
        Lambda_sum = Lambda.sum(dim=-1) if Lambda.dim() == 2 else Lambda      # (N,)
        time_loglik = (logI * input_mask).sum(dim=1) - Lambda_sum    
        
        # print("[temporal-probe]",
        #       "logI.mean=", float(logI.mean()),
        #       "logI.max=", float(logI.max()),
        #       "Lambda_sum.mean=", float(Lambda_sum.mean()),
        #       "Lambda_sum.min=", float(Lambda_sum.min()),
        #       "finite.intens=", torch.isfinite(intensities).all().item(),
        #       "finite.Lambda=", torch.isfinite(Lambda).all().item(),
        #       "negative.Lambda.count=", (Lambda < 0).sum().item())  # Count of negative entries in Lambda

        hidden_states = hidden_states[:, 1:-1]
        
        space_loglik = self.spatial_model.logprob(event_times, spatial_locations, input_mask, aux_state=hidden_states)
        return space_loglik, time_loglik
    


    def spatial_conditional_logprob_fn(self, t, event_times, spatial_locations, t0, t1):
        hidden_state_times = torch.cat([event_times, torch.tensor(t).reshape(-1).to(event_times)]).reshape(1, -1)
        _, _, hidden_states = self.temporal_model.integrate_lambda(hidden_state_times, spatial_locations[None], input_mask=None, t0=t0, t1=None)
        hidden_states = hidden_states[:, 1:]  # Remove first (t=t0) hidden state.
        return self.spatial_model.spatial_conditional_logprob_fn(t, event_times, spatial_locations, aux_state=hidden_states)

    def vector_field_fn(self, t, event_times, spatial_locations, t0, t1):
        hidden_state_times = torch.cat([event_times, torch.tensor(t).reshape(-1).to(event_times)]).reshape(1, -1)
        _, _, hidden_states = self.temporal_model.integrate_lambda(hidden_state_times, spatial_locations[None], input_mask=None, t0=t0, t1=None)
        hidden_states = hidden_states[0, 1:]  # Remove first (t=t0) hidden state.
        return self.spatial_model.vector_field_fn(t, event_times, spatial_locations, aux_state=hidden_states)

    def sample_spatial(self, nsamples, event_times, spatial_locations, input_mask, t0, t1):
        intensities, Lambda, hidden_states = self.temporal_model.integrate_lambda(event_times, spatial_locations, input_mask, t0, t1)
        hidden_states = hidden_states[:, 1:-1]  # Remove first (t=t0) and last (t=t1) hidden states.
        samples = self.spatial_model.sample_spatial(nsamples, event_times, spatial_locations, input_mask, aux_state=hidden_states)
        return samples
    def sample(self, nsamples, event_times, spatial_locations, input_mask, t0, t1, steps=50, step_size=0.05):
        """
        Samples the next spatiotemporal event (x, t, k) nsamples times given the history.
        
        Args:
            nsamples: int – number of samples to draw
            event_times: [B, T]
            spatial_locations: [B, T, D]
            input_mask: [B, T]
            t0: float or [B]
            t1: float or [B]
            steps: int – LD steps for time sampling
            step_size: float – LD step size
        
        Returns:
            next_times: [B, nsamples]
            next_marks: [B, nsamples]
            next_locations: [B, nsamples, D]
        """
        B = event_times.size(0)
        device = event_times.device

        # Get last hidden state per sequence
        hidden_state = self.temporal_model.get_last_hidden(event_times, spatial_locations, input_mask, t0, t1)

        # Initialize tau samples from uniform(0, 1)
        tau = torch.rand(B * nsamples, device=device, requires_grad=True) * 0.5 + 0.05

        # Duplicate hidden state nsamples times
        hidden_rep = hidden_state.repeat_interleave(nsamples, dim=0)

        # Langevin dynamics for time
        for _ in range(steps):
            lambda_k, score = self.temporal_model.intensity_and_score(tau, hidden_rep)
            tau = tau + 0.5 * step_size * score + torch.sqrt(torch.tensor(step_size)) * torch.randn_like(tau)

        # Final denoising
        lambda_k, score = self.temporal_model.intensity_and_score(tau, hidden_rep)
        tau = tau + 0.1 * score  # σ₁ = 0.1 per SMASH

        # Clamp to valid range (optional)
        tau = tau.clamp(min=1e-3, max=10.0)

        # Compute mark distribution and sample
        lambda_k = self.temporal_model.intensity_per_mark(tau, hidden_rep)  # [B * nsamples, K]
        marks = self.temporal_model.sample_mark(lambda_k)                   # [B * nsamples]

        # Sample spatial location
        samples_x = self.spatial_model.sample(cond=hidden_rep, tau=tau, mark=marks)  # [B * nsamples, D]

        # Reshape results
        next_times = tau.view(B, nsamples)
        next_marks = marks.view(B, nsamples)
        next_locations = samples_x.view(B, nsamples, -1)

        return next_times, next_marks, next_locations


class JumpCNFSpatiotemporalModel(SharedHiddenStateSpatiotemporalModel):

    def _build_spatial_model(self, dim, hidden_dims, actfn, zero_init, aux_dim, aux_odefunc, **kwargs):
        self.spatial_model = JumpCNF(
            dim=dim, hidden_dims=hidden_dims, actfn=actfn, zero_init=zero_init, aux_dim=aux_dim, aux_odefunc=aux_odefunc, **kwargs,
        )


class SelfAttentiveCNFSpatiotemporalModel(SharedHiddenStateSpatiotemporalModel):

    def _build_spatial_model(self, dim, hidden_dims, actfn, zero_init, aux_dim, aux_odefunc, **kwargs):
        self.spatial_model = SelfAttentiveCNF(dim=dim, hidden_dims=hidden_dims, actfn=actfn, zero_init=zero_init, aux_dim=aux_dim, **kwargs)


class JumpGMMSpatiotemporalModel(SharedHiddenStateSpatiotemporalModel):

    def _build_spatial_model(self, dim, hidden_dims, actfn, zero_init, aux_dim, aux_odefunc, n_mixtures=5, **kwargs):
        self.spatial_model = ConditionalGMM(dim=dim, hidden_dims=hidden_dims, actfn=actfn, aux_dim=aux_dim, n_mixtures=n_mixtures)


def zero_diffeq(t, h):
    return torch.zeros_like(h)
