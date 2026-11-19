# SPDX-License-Identifier: Apache-2.0
"""CPU-only synthetic checkpoint tests; no torch, vLLM, or NPU required."""

import json
import struct
import tempfile
import unittest
from pathlib import Path

from tools.kimi_k31 import prune_experts as prune

try:
    from safetensors import safe_open
except ImportError:
    safe_open = None


def make_checkpoint(root):
    root.mkdir()
    text = {
        "num_experts": 32,
        "num_hidden_layers": 3,
        "num_experts_per_token": 16,
        "num_shared_experts": 2,
        "hidden_size": 4,
        "routed_expert_hidden_size": 4,
        "moe_intermediate_size": 8,
    }
    prune.write_json(root / "config.json", {"text_config": text, "vision_config": {"unchanged": True}})
    prune.write_json(root / "tokenizer_config.json", {"chat_template": "unchanged"})
    tensors, quant = {}, {"group_size": 0}
    for layer in (1, 2):
        prefix = f"language_model.model.layers.{layer}.block_sparse_moe."
        for expert in range(32):
            for part in prune.PARTS:
                output, input_size = (4, 8) if part == "w2" else (8, 4)
                for suffix in prune.SUFFIXES:
                    name = f"{prefix}experts.{expert}.{part}.{suffix}"
                    dtype, shape = ("I8", [output // 2, input_size]) if suffix == "weight" else ("F32", [output, 1])
                    size = prune.DTYPE_BYTES[dtype] * shape[0] * shape[1]
                    tensors[name] = (dtype, shape, bytes([layer * 64 + expert]) * size)
                    quant[name] = "W4A8_DYNAMIC" if suffix == "weight" else "FLOAT"
        for suffix, shape in (("weight", [32, 4]), ("e_score_correction_bias", [32])):
            # Distinct rows make reordered/nonuniform per-layer selections observable.
            row_bytes = 8 if suffix == "weight" else 2
            tensors[f"{prefix}gate.{suffix}"] = ("BF16", shape, b"".join(bytes([i]) * row_bytes for i in range(32)))
        tensors[f"{prefix}shared_experts.w1.weight"] = ("BF16", [2], b"keep")
    tensors["language_model.model.layers.0.mlp.w1.weight"] = ("BF16", [2], b"dense"[:4])
    filename = "weights.safetensors"
    header, data = {}, bytearray()
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(data), len(data) + len(raw)]}
        data.extend(raw)
    raw_header = json.dumps(header).encode()
    raw_header += b" " * (-len(raw_header) % 8)
    (root / filename).write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + data)
    prune.write_json(
        root / prune.INDEX_NAMES[0],
        {"metadata": {"total_size": len(data)}, "weight_map": dict.fromkeys(tensors, filename)},
    )
    prune.write_json(root / "quant_model_description.json", quant)
    return tensors


def tensor_bytes(root, name):
    index = prune.read_json(root / prune.INDEX_NAMES[0])
    path = root / index["weight_map"][name]
    base, header = prune.read_header(path)
    start, end = header[name]["data_offsets"]
    with path.open("rb") as stream:
        stream.seek(base + start)
        return stream.read(end - start)


class PruneTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.src, self.dst = self.root / "source", self.root / "output"
        self.tensors = make_checkpoint(self.src)

    def plan(self, selection=None):
        return prune.build_plan(self.src, 16, selection, expected_experts=32, expected_layers=3)

    def rewrite_json(self, name, mutate):
        path = self.src / name
        value = prune.read_json(path)
        mutate(value)
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_per_layer_mapping_preserves_every_tensor_and_metadata(self):
        selection = {"1": list(range(31, 15, -1)), "2": list(range(0, 32, 2))}
        plan = self.plan(selection)
        source_hash = prune.file_hash(self.src / "weights.safetensors")
        manifest = prune.run(plan, self.dst)
        prune.verify(self.dst)
        maps = {layer: {old: new for new, old in enumerate(ids)} for layer, ids in selection.items()}
        for name, (_, _, data) in self.tensors.items():
            new_name = prune.rename_expert(name, maps)
            if new_name is None:
                continue
            match = prune.ROUTER.fullmatch(name)
            if match:
                row_size = len(data) // 32
                data = b"".join(data[i * row_size : (i + 1) * row_size] for i in selection[match[1]])
            self.assertEqual(tensor_bytes(self.dst, new_name), data, name)
        self.assertEqual(source_hash, prune.file_hash(self.src / "weights.safetensors"))
        expected_config = prune.read_json(self.src / "config.json")
        expected_config["text_config"]["num_experts"] = 16
        self.assertEqual(prune.read_json(self.dst / "config.json"), expected_config)
        expected_quant = {
            prune.rename_expert(k, maps): v
            for k, v in plan["quant"].items()
            if prune.rename_expert(k, maps) is not None
        }
        self.assertEqual(prune.read_json(self.dst / "quant_model_description.json"), expected_quant)
        self.assertEqual(manifest["validation"]["accuracy"], "not_tested")

    def test_uniform_selection_and_no_plan_writes(self):
        before = set(self.src.rglob("*"))
        plan = self.plan()
        self.assertEqual(plan["expert_ids"]["1"], list(range(0, 32, 2)))
        self.assertEqual(set(self.src.rglob("*")), before)
        self.assertFalse(self.dst.exists())

    def test_inspection_reports_actual_shapes(self):
        inventory = prune.inspect_checkpoint(self.src)
        name = "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight"
        self.assertEqual(inventory["examples"][name]["shape"], [4, 4])
        self.assertEqual(inventory["quant_label_counts"]["W4A8_DYNAMIC"], 2 * 32 * 3)

    @unittest.skipIf(safe_open is None, "Optional independent safetensors reader is not installed")
    def test_output_accepted_by_safetensors(self):
        prune.run(self.plan(), self.dst)
        path = next(self.dst.glob("*.safetensors"))
        _, header = prune.read_header(path)
        with safe_open(path, framework="np") as reader:
            names = reader.keys()
            self.assertEqual(set(names), set(header) - {"__metadata__"})
            for name in names:
                self.assertEqual(reader.get_slice(name).get_shape(), header[name]["shape"])

    def test_reject_incorrect_packing_and_missing_quant_label(self):
        name = "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight"
        config = prune.read_json(self.src / "config.json")["text_config"]
        with self.assertRaisesRegex(ValueError, "Not packed A3 INT4"):
            prune.validate_expert(name, {"dtype": "I8", "shape": [8, 2]}, config, {name: "W4A8_DYNAMIC"})
        with self.assertRaisesRegex(ValueError, "W4A8_DYNAMIC"):
            prune.validate_expert(name, {"dtype": "I8", "shape": [4, 4]}, config, {})

    def test_reject_bad_maps(self):
        for selection in (
            {"1": list(range(16))},
            {"1": [0] * 16, "2": list(range(16))},
            {"1": list(range(16)), "2": list(range(17, 33))},
        ):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                self.plan(selection)
        with self.assertRaises(ValueError):
            prune.select_experts(32, 0, [1], None)

    def test_reject_mxfp(self):
        name = "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight"
        self.rewrite_json("quant_model_description.json", lambda q: q.update({name: "W4A8_MXFP"}))
        with self.assertRaisesRegex(ValueError, "W4A8_DYNAMIC"):
            self.plan()

    def test_reject_grouped_routing(self):
        self.rewrite_json("config.json", lambda c: c["text_config"].update({"use_grouped_topk": True}))
        with self.assertRaisesRegex(ValueError, "Grouped routing"):
            self.plan()

    def test_single_group_routing_preserves_config(self):
        routing = {"use_grouped_topk": True, "num_expert_group": 1, "topk_group": 1}
        self.rewrite_json("config.json", lambda c: c["text_config"].update(routing))
        plan = self.plan()
        prune.run(plan, self.dst)
        result = prune.read_json(self.dst / "config.json")["text_config"]
        for key, value in routing.items():
            self.assertEqual(result[key], value)

    def test_reject_multiple_routing_groups(self):
        routing = {"use_grouped_topk": True, "num_expert_group": 2, "topk_group": 1}
        self.rewrite_json("config.json", lambda c: c["text_config"].update(routing))
        with self.assertRaisesRegex(ValueError, "multiple groups"):
            self.plan()

    def test_reject_missing_tensor_index(self):
        self.rewrite_json(prune.INDEX_NAMES[0], lambda i: i["weight_map"].pop(next(iter(i["weight_map"]))))
        with self.assertRaisesRegex(ValueError, "Index mismatch"):
            self.plan()

    def test_reject_changed_source_after_plan(self):
        plan = self.plan()
        self.rewrite_json(prune.INDEX_NAMES[0], lambda i: i.update({"changed": True}))
        with self.assertRaisesRegex(ValueError, "Source index changed"):
            prune.run(plan, self.dst)
        self.assertFalse(self.dst.exists())

    def test_reject_overlapping_or_existing_destination(self):
        plan = self.plan()
        with self.assertRaisesRegex(ValueError, "disjoint"):
            prune.run(plan, self.src / "output")
        self.dst.mkdir()
        with self.assertRaises(FileExistsError):
            prune.run(plan, self.dst)

    def test_verify_detects_corruption_and_incomplete_marker(self):
        prune.run(self.plan(), self.dst)
        path = next(self.dst.glob("*.safetensors"))
        with path.open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"!")
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            prune.verify(self.dst)
        prune.write_json(self.dst / prune.INCOMPLETE, {})
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            prune.verify(self.dst)

    def test_header_rejects_truncation_and_duplicate_keys(self):
        path = self.root / "bad.safetensors"
        path.write_bytes(struct.pack("<Q", 100) + b"{}")
        with self.assertRaisesRegex(ValueError, "Truncated"):
            prune.read_header(path)
        raw = b'{"x": 1, "x": 2}'
        path.write_bytes(struct.pack("<Q", len(raw)) + raw)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            prune.read_header(path)

    def test_reject_unsupported_architecture_by_default(self):
        with self.assertRaisesRegex(ValueError, "Unexpected architecture"):
            prune.build_plan(self.src, 16)


if __name__ == "__main__":
    unittest.main()
