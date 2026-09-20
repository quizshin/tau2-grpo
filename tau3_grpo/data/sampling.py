"""Attach actual batch group identities before rollout workers split the batch."""
from __future__ import annotations

from collections import defaultdict


def sampling_identities(group_uids, *, data_seed, step, validation=False):
    """Trial numbers are zero-based within the actual estimator group UID.

    This records the configured data seed, not a fabricated per-request decoder
    seed. It neither reseeds generation nor changes algorithm group membership.
    """
    counts = defaultdict(int)
    identities = []
    for raw in group_uids:
        if raw is None or not str(raw):
            raise ValueError("Sampling group UID must come from the actual batch")
        uid = str(raw)
        trial = counts[uid]
        counts[uid] += 1
        identities.append({"sample_group_uid": uid, "trial": trial,
                           "seed": None if data_seed is None else int(data_seed),
                           "seed_semantics": "configured_data_seed",
                           "generation_request_seed": None,
                           "global_step": int(step), "validation": bool(validation)})
    return identities
