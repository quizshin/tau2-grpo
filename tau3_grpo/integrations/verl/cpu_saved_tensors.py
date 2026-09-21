"""Opt-in native PyTorch saved-tensor offload for the legacy PPO actor.

Load explicitly through actor_rollout_ref.model.external_lib. This changes
tensor storage only: model precision, losses, minibatches and optimizer steps
remain owned by the existing actor. No changes are made to FSDP state dicts.
"""
from functools import wraps

import torch


def offloaded_update(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            return method(self, *args, **kwargs)
    return wrapped


def release_forward_cache(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        # Let CUDA allocate the FLA kernels' local-memory workspace rather
        # than retaining unused PyTorch blocks from the vocabulary projection.
        if torch.is_grad_enabled() and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result
    return wrapped


def install():
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    actor = DataParallelPPOActor
    if getattr(actor.update_policy, '_tau3_cpu_saved_tensors', False):
        return
    actor.update_policy = offloaded_update(actor.update_policy)
    actor.update_policy._tau3_cpu_saved_tensors = True
    actor._forward_micro_batch = release_forward_cache(actor._forward_micro_batch)


install()
