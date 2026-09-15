"""Check native Qwen3.5 exports before publishing or starting inference."""

import json
from pathlib import Path
import struct


def qwen35_tensor_manifest(directory):
    """Inspect safetensors headers without importing torch or allocating weights."""
    directory = Path(directory)
    tensors = {}
    owners = {}
    for shard in sorted(directory.glob("*.safetensors")):
        with shard.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError("Truncated safetensors header")
            size = struct.unpack("<Q", prefix)[0]
            if size > 10**7:
                raise ValueError("Unexpected safetensors header size")
            header = json.loads(stream.read(size))
        payload_bytes = shard.stat().st_size - 8 - size
        for key, metadata in header.items():
            if key == "__metadata__":
                continue
            if key in tensors:
                raise ValueError(f"Duplicate tensor across shards: {key}")
            if not key.startswith(("model.language_model.", "model.visual.", "lm_head.")) or any(
                part in key for part in ("language_model.language_model.", "language_model.visual.")
            ):
                raise ValueError(f"Non-native Qwen3.5 tensor name: {key}")
            start, end = metadata["data_offsets"]
            if not 0 <= start <= end <= payload_bytes:
                raise ValueError(f"Truncated tensor data: {key}")
            tensors[key] = {"shape": metadata["shape"], "dtype": metadata["dtype"]}
            owners[key] = shard.name
    if "model.language_model.embed_tokens.weight" not in tensors or "model.language_model.norm.weight" not in tensors:
        raise ValueError("Qwen3.5 export lacks native embedding/norm weights")
    index = directory / "model.safetensors.index.json"
    if index.exists() and json.loads(index.read_text())["weight_map"] != owners:
        raise ValueError("Safetensors index differs from saved tensors")
    return tensors


def verify_qwen35_export(directory, source, tied):
    """Compare every saved tensor to the reconstructed BF16 FSDP state."""
    import torch
    from safetensors import safe_open

    manifest = qwen35_tensor_manifest(directory)
    missing = set(source) - set(manifest)
    if set(manifest) - set(source):
        raise ValueError("Export contains tensors absent from the FSDP source")
    if missing:
        if missing != {"lm_head.weight"} or not tied or not torch.equal(
            source["lm_head.weight"], source["model.language_model.embed_tokens.weight"]
        ):
            raise ValueError(f"Export lost FSDP tensors: {sorted(missing)}")
    for shard in sorted(Path(directory).glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as saved:
            for key in saved.keys():
                value = saved.get_tensor(key)
                expected = source[key]
                if value.dtype != expected.dtype or value.shape != expected.shape or not torch.equal(value, expected):
                    raise ValueError(f"Export differs from reconstructed FSDP tensor: {key}")
    receipt = {"native_format": True, "verified_tensors": len(manifest),
               "source_tensors": len(source), "omitted_tied_aliases": sorted(missing),
               "comparison": "exact equality to reconstructed BF16 FSDP tensors"}
    (Path(directory) / "export-verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
