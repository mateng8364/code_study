# SPDX-License-Identifier: Apache-2.0
"""Offline, byte-preserving pruning for unpacked ModelSlim Kimi K3.1 MoE checkpoints.

Design reference: CuiZ2007/vllm-ascend, tools/kimi_k3/prune_experts.py on
test/kimi-k3-pruned-experts-accuracy. This implementation targets A3 W4A8_DYNAMIC.
Only the Python standard library is needed. No inference or quantization is performed.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import struct
from collections import Counter
from pathlib import Path

CHUNK_BYTES = 8 * 1024 * 1024
MAX_HEADER_BYTES = 512 * 1024 * 1024
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "BF16": 2,
    "F16": 2,
    "I16": 2,
    "U16": 2,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
}
EXPERT = re.compile(r"^(language_model\.model\.layers\.(\d+)\.block_sparse_moe\.experts\.)(\d+)\.(.+)$")
ROUTER = re.compile(r"^language_model\.model\.layers\.(\d+)\.block_sparse_moe\.gate\.(weight|e_score_correction_bias)$")
PARTS = ("w1", "w2", "w3")
SUFFIXES = ("weight", "weight_scale", "weight_offset", "scale_bias")
INDEX_NAMES = ("quant_model_weights.safetensors.index.json", "model.safetensors.index.json")
INCOMPLETE = "PRUNING_INCOMPLETE.json"
COMPLETE = "PRUNING_COMPLETE.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)


def write_json(path, value):
    # Exclusive creation: never silently replace an existing artifact.
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def source_file(root, name):
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts, f"Unsafe path: {name}")
    path = root / relative
    require(path.is_file() and not path.is_symlink(), f"Missing file or symlink: {path}")
    require(path.resolve().is_relative_to(root.resolve()), f"Path escapes source: {path}")
    return path


def read_header(path):
    with path.open("rb") as stream:
        prefix = stream.read(8)
        require(len(prefix) == 8, f"Truncated header: {path}")
        length = struct.unpack("<Q", prefix)[0]
        require(0 < length <= MAX_HEADER_BYTES, f"Invalid header length: {path}")
        raw = stream.read(length)
    require(len(raw) == length, f"Truncated header: {path}")
    header = json.loads(raw, object_pairs_hook=unique_pairs)
    spans = []
    for name, meta in header.items():
        if name == "__metadata__":
            require(isinstance(meta, dict), "Invalid safetensors metadata")
            require(all(isinstance(k, str) and isinstance(v, str) for k, v in meta.items()), "Invalid metadata")
            continue
        require(meta["dtype"] in DTYPE_BYTES, f"Unsupported dtype: {name}: {meta['dtype']}")
        require(all(type(d) is int and d >= 0 for d in meta["shape"]), f"Invalid shape: {name}")
        start, end = meta["data_offsets"]
        require(type(start) is int and type(end) is int, f"Invalid offsets: {name}")
        require(end - start == math.prod(meta["shape"]) * DTYPE_BYTES[meta["dtype"]], f"Invalid bytes: {name}")
        spans.append((start, end))
    cursor = 0
    for start, end in sorted(spans):
        require(start == cursor and end >= start, f"Gap/overlap in {path}")
        cursor = end
    require(8 + length + cursor == path.stat().st_size, f"Invalid file size: {path}")
    return 8 + length, header


def fingerprint(path):
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


def select_experts(total, keep, layers, selection):
    require(type(keep) is int and 0 < keep < total, "keep-experts must be positive and below source count")
    if selection is None:
        return {str(layer): [i * total // keep for i in range(keep)] for layer in layers}
    require(set(selection) == {str(layer) for layer in layers}, "Expert map must cover exactly every MoE layer")
    for layer, ids in selection.items():
        require(isinstance(ids, list) and len(ids) == keep, f"Wrong selection length: layer {layer}")
        require(all(type(i) is int and 0 <= i < total for i in ids), f"Invalid expert ID: layer {layer}")
        require(len(set(ids)) == keep, f"Duplicate expert IDs: layer {layer}")
    return selection


def rename_expert(name, maps):
    match = EXPERT.fullmatch(name)
    if not match:
        require(".experts." not in name, f"Unsupported expert naming/layout: {name}")
        return name
    prefix, layer, old, suffix = match.groups()
    new = maps[layer].get(int(old))
    return None if new is None else f"{prefix}{new}.{suffix}"


def validate_expert(name, meta, config, quant):
    match = EXPERT.fullmatch(name)
    part, suffix = match[4].split(".", 1)
    require(part in PARTS and suffix in SUFFIXES, f"Unsupported expert tensor: {name}")
    hidden, intermediate = config["routed_expert_hidden_size"], config["moe_intermediate_size"]
    output, input_size = (hidden, intermediate) if part == "w2" else (intermediate, hidden)
    if suffix == "weight":
        require(output % 2 == 0, "Packed INT4 output dimension must be even")
        require(meta["dtype"] == "I8" and meta["shape"] == [output // 2, input_size], f"Not packed A3 INT4: {name}")
        require(quant.get(name) == "W4A8_DYNAMIC", f"Expected W4A8_DYNAMIC label: {name}")
    else:
        shape = meta["shape"]
        valid = shape == [output, 1]
        if suffix == "scale_bias" and part == "w2":
            valid = len(shape) == 2 and shape[0] == output and shape[1] in (1, 2, 4, 8, 16)
        require(meta["dtype"] == "F32" and valid, f"Unsupported ModelSlim quantization metadata: {name}")


def inspect_checkpoint(src):
    """Describe a checkpoint without assuming the A3 packing schema or reading tensor payloads."""
    src = src.resolve()
    indices = [name for name in INDEX_NAMES if (src / name).is_file()]
    require(len(indices) == 1, "Expected exactly one supported safetensors index")
    index = read_json(source_file(src, indices[0]))
    quant = read_json(source_file(src, "quant_model_description.json"))
    dtypes, labels, examples = Counter(), Counter(), {}
    for value in quant.values():
        if isinstance(value, str):
            labels[value] += 1
    for filename in sorted(set(index["weight_map"].values())):
        _, header = read_header(source_file(src, filename))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            dtypes[meta["dtype"]] += meta["data_offsets"][1] - meta["data_offsets"][0]
            expert, router = EXPERT.fullmatch(name), ROUTER.fullmatch(name)
            if (expert and expert[2] == "1" and expert[3] == "0") or (router and router[1] == "1"):
                examples[name] = {"dtype": meta["dtype"], "shape": meta["shape"], "quant_label": quant.get(name)}
    return {
        "source": str(src),
        "config": read_json(source_file(src, "config.json")),
        "index_name": indices[0],
        "tensor_bytes_by_dtype": dict(dtypes),
        "quant_label_counts": dict(labels),
        "group_size": quant.get("group_size"),
        "examples": examples,
        "note": "Inventory only; not a pruning compatibility or loading validation.",
    }


def build_plan(src, keep, selection=None, *, expected_experts=896, expected_layers=93):
    """Read headers only for weights. Freeze source assets and tensor layout in a plan."""
    src = src.resolve()
    require(not (src / INCOMPLETE).exists(), "Source is an incomplete pruning output")
    config = read_json(source_file(src, "config.json"))
    text = config["text_config"]
    total = text["num_experts"]
    require(total == expected_experts and text["num_hidden_layers"] == expected_layers, "Unexpected architecture")
    require(
        text["num_experts_per_token"] == 16 and text["num_shared_experts"] == 2, "Expected top-16 and 2 shared experts"
    )
    require(text["num_experts_per_token"] <= keep, "Cannot retain fewer experts than routing top-k")
    require(
        not text.get("use_grouped_topk", False) or (text.get("num_expert_group") == 1 and text.get("topk_group") == 1),
        "Grouped routing with multiple groups needs a group-preserving pruning implementation",
    )
    require(text.get("num_expert_group", 1) in (None, 1), "Grouped experts are not supported")
    require(keep % 16 == 0, "For this 16-device candidate, keep-experts must be divisible by 16")
    layers = list(range(1, expected_layers))
    ids = select_experts(total, keep, layers, selection)
    maps = {layer: {old: new for new, old in enumerate(values)} for layer, values in ids.items()}
    indices = [name for name in INDEX_NAMES if (src / name).is_file()]
    require(len(indices) == 1, "Expected exactly one supported safetensors index")
    index_name = indices[0]
    index = read_json(source_file(src, index_name))
    quant = read_json(source_file(src, "quant_model_description.json"))
    require(quant.get("group_size", 0) in (0, -1, None), "Per-group quantization is unsupported")
    weight_map = index["weight_map"]
    shards, seen, experts, routers = [], set(), {}, set()
    source_bytes = output_bytes = 0
    for filename in sorted(set(weight_map.values())):
        path = source_file(src, filename)
        base, header = read_header(path)
        items = []
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            require(name not in seen and weight_map.get(name) == filename, f"Index mismatch: {name}")
            seen.add(name)
            source_bytes += meta["data_offsets"][1] - meta["data_offsets"][0]
            match = EXPERT.fullmatch(name)
            if match:
                layer, old, suffix = match[2], int(match[3]), match[4]
                require(layer in maps and 0 <= old < total, f"Unexpected layer/expert: {name}")
                validate_expert(name, meta, text, quant)
                experts.setdefault((layer, old), set()).add(suffix)
            router = ROUTER.fullmatch(name)
            if ".block_sparse_moe.gate." in name:
                require(router is not None, f"Unsupported router tensor: {name}")
            new_name = rename_expert(name, maps)
            if new_name is None:
                continue
            new_meta = copy.deepcopy(meta)
            rows = None
            if router:
                layer, suffix = router.groups()
                require(layer in maps, f"Unexpected router layer: {name}")
                shape = [total, text["hidden_size"]] if suffix == "weight" else [total]
                require(meta["shape"] == shape and meta["dtype"] in ("BF16", "F32"), f"Unsupported router: {name}")
                require(quant.get(name, "FLOAT") in ("FLOAT", "BF16", "FP32", "F32"), f"Quantized router: {name}")
                new_meta["shape"][0] = keep
                rows = ids[layer]
                routers.add((layer, suffix))
            size = math.prod(new_meta["shape"]) * DTYPE_BYTES[new_meta["dtype"]]
            output_bytes += size
            items.append({"source": name, "name": new_name, "meta": new_meta, "rows": rows, "bytes": size})
        shards.append(
            {
                "source": filename,
                "base": base,
                "fingerprint": fingerprint(path),
                "header_sha256": header_hash(path, base),
                "metadata": header.get("__metadata__", {}),
                "items": items,
            }
        )
    require(seen == set(weight_map), "Index contains missing tensors")
    required = {f"{part}.{suffix}" for part in PARTS for suffix in SUFFIXES}
    require(len(experts) == len(layers) * total, "Missing routed experts")
    require(all(parts == required for parts in experts.values()), "Missing expert weights/scales/offsets/biases")
    require(
        routers == {(str(layer), suffix) for layer in layers for suffix in ("weight", "e_score_correction_bias")},
        "Missing router weights/correction bias",
    )
    require(index.get("metadata", {}).get("total_size") == source_bytes, "Index total_size mismatch")
    for name in quant:
        if ".experts." in name:
            require(name in seen, f"Stale/unsupported quantization key: {name}")
    assets = {}
    for path in sorted(src.rglob("*")):
        require(not path.is_symlink(), f"Materialize symlinks before pruning: {path}")
        if not path.is_file():
            continue
        name = path.relative_to(src).as_posix()
        if any(part.startswith(".") for part in path.relative_to(src).parts):
            continue
        if name in weight_map.values() or name == index_name:
            continue
        require(not name.endswith(".safetensors.index.json"), f"Unexpected extra weight index: {name}")
        require(path.suffix not in (".safetensors", ".bin", ".pt", ".pth"), f"Unindexed weights: {name}")
        require(name not in (INCOMPLETE, COMPLETE), "Use an original checkpoint as source")
        assets[name] = file_hash(path)
    return {
        "schema_version": 1,
        "source": str(src),
        "keep_experts": keep,
        "expert_ids": ids,
        "selection": "uniform_capacity_only" if selection is None else "provided_per_layer_map",
        "index_name": index_name,
        "index_sha256": file_hash(src / index_name),
        "source_tensor_bytes": source_bytes,
        "output_tensor_bytes": output_bytes,
        "shards": shards,
        "assets": assets,
        "config": config,
        "quant": quant,
    }


def header_hash(path, base):
    with path.open("rb") as stream:
        return hashlib.sha256(stream.read(base)).hexdigest()


def check_source(plan):
    src = Path(plan["source"])
    require(file_hash(src / plan["index_name"]) == plan["index_sha256"], "Source index changed")
    for shard in plan["shards"]:
        path = source_file(src, shard["source"])
        require(fingerprint(path) == shard["fingerprint"], f"Source changed: {path}")
        require(header_hash(path, shard["base"]) == shard["header_sha256"], f"Source header changed: {path}")
    for name, digest in plan["assets"].items():
        require(file_hash(source_file(src, name)) == digest, f"Source asset changed: {name}")


def payload(stream, shard, item):
    start, end = item["meta"]["data_offsets"]
    if item["rows"] is None:
        spans = [(start, end - start)]
    else:
        row_bytes = item["bytes"] // len(item["rows"])
        spans = [(start + row * row_bytes, row_bytes) for row in item["rows"]]
    for offset, size in spans:
        stream.seek(shard["base"] + offset)
        while size:
            block = stream.read(min(CHUNK_BYTES, size))
            require(bool(block), f"Truncated tensor: {item['source']}")
            size -= len(block)
            yield block


def output_header(shard):
    header = {"__metadata__": shard["metadata"]}
    offset = 0
    for item in shard["items"]:
        meta = copy.deepcopy(item["meta"])
        meta["data_offsets"] = [offset, offset + item["bytes"]]
        header[item["name"]] = meta
        offset += item["bytes"]
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    return struct.pack("<Q", len(raw)) + raw


def disjoint(first, second):
    require(not first.is_relative_to(second) and not second.is_relative_to(first), "Paths must be disjoint")


def run(plan, dst):
    src, dst = Path(plan["source"]), dst.resolve()
    disjoint(src, dst)
    check_source(plan)
    # mkdir without exist_ok is also an exclusive claim against concurrent writers.
    dst.mkdir(parents=True)
    write_json(dst / INCOMPLETE, {"source": str(src), "keep_experts": plan["keep_experts"]})
    required_disk = plan["output_tensor_bytes"] + sum((src / name).stat().st_size for name in plan["assets"])
    require(shutil.disk_usage(dst).free > required_disk, "Insufficient free disk space (headers need extra space too)")
    manifest = {
        "schema_version": 1,
        "source": str(src),
        "keep_experts": plan["keep_experts"],
        "expert_ids": plan["expert_ids"],
        "selection": plan["selection"],
        "files": {},
        "script_sha256": file_hash(Path(__file__)),
        "source_index_sha256": plan["index_sha256"],
        "source_asset_sha256": plan["assets"],
        "source_shards": {
            s["source"]: {"fingerprint": s["fingerprint"], "header_sha256": s["header_sha256"]} for s in plan["shards"]
        },
        "validation": {"byte_integrity": "pending", "model_loading": "not_tested", "accuracy": "not_tested"},
    }
    maps = {layer: {old: new for new, old in enumerate(ids)} for layer, ids in plan["expert_ids"].items()}
    for name in plan["assets"]:
        target = dst / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name == "config.json":
            config = copy.deepcopy(plan["config"])
            config["text_config"]["num_experts"] = plan["keep_experts"]
            write_json(target, config)
        elif name == "quant_model_description.json":
            quant = {}
            for key, value in plan["quant"].items():
                new_key = rename_expert(key, maps)
                if new_key is not None:
                    quant[new_key] = value
            write_json(target, quant)
        else:
            shutil.copyfile(src / name, target)
            require(file_hash(target) == plan["assets"][name], f"Asset copy mismatch: {name}")
        manifest["files"][name] = file_hash(target)
    weight_map = {}
    for number, shard in enumerate(plan["shards"], 1):
        if not shard["items"]:
            continue
        name = f"pruned-e{plan['keep_experts']}-{number:05d}.safetensors"
        target = dst / name
        expected = hashlib.sha256()
        with (src / shard["source"]).open("rb") as source, target.open("xb") as output:
            raw = output_header(shard)
            output.write(raw)
            expected.update(raw)
            for item in shard["items"]:
                weight_map[item["name"]] = name
                for block in payload(source, shard, item):
                    output.write(block)
                    expected.update(block)
            output.flush()
            os.fsync(output.fileno())
        read_header(target)
        require(file_hash(target) == expected.hexdigest(), f"Output readback mismatch: {name}")
        manifest["files"][name] = expected.hexdigest()
        print(f"Verified shard {number}/{len(plan['shards'])}: {name}", flush=True)
    index = {"metadata": {"total_size": plan["output_tensor_bytes"]}, "weight_map": weight_map}
    write_json(dst / plan["index_name"], index)
    manifest["files"][plan["index_name"]] = file_hash(dst / plan["index_name"])
    check_source(plan)
    manifest["validation"]["byte_integrity"] = "passed"
    write_json(dst / COMPLETE, manifest)
    (dst / INCOMPLETE).unlink()
    return manifest


def verify(dst):
    require(not (dst / INCOMPLETE).exists(), "Incomplete output; do not deploy")
    manifest = read_json(dst / COMPLETE)
    require(manifest["validation"]["byte_integrity"] == "passed", "Incomplete validation")
    actual = {path.relative_to(dst).as_posix() for path in dst.rglob("*") if path.is_file()}
    require(actual == set(manifest["files"]) | {COMPLETE}, "Unexpected or missing files in output directory")
    for name, digest in manifest["files"].items():
        path = source_file(dst, name)
        require(file_hash(path) == digest, f"Checksum mismatch: {name}")
        if path.suffix == ".safetensors":
            read_header(path)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inspect", "plan", "run", "verify"), required=True)
    parser.add_argument("--src", type=Path)
    parser.add_argument("--dst", type=Path)
    parser.add_argument("--report", type=Path, help="New plan JSON outside the source and destination")
    parser.add_argument("--keep-experts", type=int, default=256)
    parser.add_argument("--expert-map", type=Path, help='JSON: {"1": [old IDs in new order], ..., "92": [...]}')
    args = parser.parse_args()
    if args.mode == "verify":
        require(args.dst is not None, "verify requires --dst")
        verify(args.dst)
        print("File integrity passed; model loading and accuracy are separate checks.")
        return
    require(args.src is not None, "plan/run requires --src")
    if args.mode == "inspect":
        require(args.report is not None, "inspect requires --report")
        require(not args.report.resolve().is_relative_to(args.src.resolve()), "Report must be outside source")
        write_json(args.report, inspect_checkpoint(args.src))
        print(f"Checkpoint inventory written to {args.report}")
        return
    selection = read_json(args.expert_map) if args.expert_map else None
    plan = build_plan(args.src, args.keep_experts, selection)
    if args.report:
        require(not args.report.resolve().is_relative_to(args.src.resolve()), "Report must be outside source")
        if args.dst:
            require(not args.report.resolve().is_relative_to(args.dst.resolve()), "Report must be outside destination")
        write_json(args.report, plan)
    if args.mode == "plan":
        require(args.report is not None, "plan requires --report")
    else:
        require(args.dst is not None, "run requires --dst")
        run(plan, args.dst)
    print(
        json.dumps(
            {
                "keep_experts": plan["keep_experts"],
                "selection": plan["selection"],
                "output_tensor_bytes": plan["output_tensor_bytes"],
                "output_tensor_GiB": plan["output_tensor_bytes"] / 1024**3,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
