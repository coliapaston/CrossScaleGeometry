#!/usr/bin/env python3
"""
dump_hf_tensors.py

Download + export:
- input_embd  -> model.embed_tokens.weight (or best-match fallback)
- output_proj -> lm_head.weight (or best-match fallback)
- tokenizer   -> AutoTokenizer.save_pretrained()
- config.json -> AutoConfig.to_json_file()
- meta/resolved.json -> traceability record (resolved key, shard, shape, dtype, etc.)

Pythia: adapter placeholder + `--extra_tensors` hook (keys/patterns) reserved.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import shutil

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoConfig


# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# -----------------------------
# Adapter framework (Pythia reserved)
# -----------------------------

class ModelAdapter:
    """
    Adapter defines:
    - how to resolve logical spaces to tensor key candidates
    - index filename (if any)
    - weight file fallback candidates (if no index)
    """
    def __init__(self, model_name: str):
        self.model_name = model_name

    def index_filename(self) -> Optional[str]:
        return "model.safetensors.index.json"

    def weight_file_fallbacks(self) -> List[str]:
        # If index doesn't exist, try these in order.
        return ["model.safetensors"]

    def space_candidates(self, space_name: str) -> List[str]:
        raise NotImplementedError

    def extra_steps(self, ctx: Dict[str, Any]) -> None:
        # Reserved for Pythia “intermediate process”.
        return


class HFTransformerAdapter(ModelAdapter):
    """
    Default for common HF transformer repos (Mistral/Llama/etc).
    """
    def space_candidates(self, space_name: str) -> List[str]:
        if space_name == "input_embd":
            return [
                "model.embed_tokens.weight",
                "transformer.wte.weight",  # some GPT-like naming
                "wte.weight",
                "embed_tokens.weight",
            ]
        if space_name == "output_proj":
            return [
                "lm_head.weight",
                "model.lm_head.weight",
                "embed_out.weight",
                "output.weight",
            ]
        raise ValueError(f"Unknown space_name: {space_name}")


class PythiaAdapter(ModelAdapter):
    """
    Adapter for EleutherAI Pythia (GPT-NeoX) naming.

    Direction A (multi-step checkpoints) is handled in main()/driver by iterating revisions.
    This adapter only needs to:
      - resolve standard spaces to correct tensor key candidates for Pythia repos
      - keep hooks for future (no-op now)
    """

    def space_candidates(self, space_name: str) -> List[str]:
        if space_name == "input_embd":
            # GPT-NeoX / Pythia typical naming
            return [
                "gpt_neox.embed_in.weight",
                "model.embed_tokens.weight",  # fallback (some repos)
                "transformer.wte.weight",
                "wte.weight",
                "embed_tokens.weight",
            ]
        if space_name == "output_proj":
            # Pythia typical naming
            return [
                "embed_out.weight",          # common in GPT-NeoX / Pythia
                "lm_head.weight",            # fallback
                "model.lm_head.weight",      # fallback
                "output.weight",
            ]
        raise ValueError(f"Unknown space_name: {space_name}")

    def extra_steps(self, ctx: Dict[str, Any]) -> None:
        # Reserved for Direction B/C later (layer-wise intermediates, etc.)
        return


def make_adapter(model_name: str, family: str) -> ModelAdapter:
    fam = (family or "auto").lower()
    if fam == "pythia":
        return PythiaAdapter(model_name)
    if fam in ("auto", "hf", "transformer", "mistral", "llama"):
        return HFTransformerAdapter(model_name)
    raise ValueError(f"Unknown family='{family}'. Use auto|pythia.")


def make_adapter(model_name: str, family: str) -> ModelAdapter:
    fam = (family or "auto").lower()
    if fam == "pythia":
        return PythiaAdapter(model_name)
    if fam in ("auto", "hf", "transformer", "mistral", "llama"):
        return HFTransformerAdapter(model_name)
    raise ValueError(f"Unknown family='{family}'. Use auto|pythia.")


# -----------------------------
# Tensor resolution + extraction
# -----------------------------

def try_download(
    repo_id: str,
    filename: str,
    local_dir: str,
    cache_dir: str,
    token: Optional[str],
    revision: Optional[str],
) -> Optional[str]:
    """
    Download a single file if it exists; return local path, else None.
    """
    try:
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            cache_dir=cache_dir,
            token=token if token else None,
            revision=revision,
        )
    except Exception:
        return None


def load_index_weight_map(index_path: str) -> Dict[str, str]:
    idx = read_json(index_path)
    wm = idx.get("weight_map", {})
    if not isinstance(wm, dict) or not wm:
        raise ValueError("Index JSON has no valid weight_map")
    return wm


def pick_best_key_from_weight_map(
    weight_map: Dict[str, str],
    candidates: List[str],
) -> Optional[str]:
    # Exact match first
    for k in candidates:
        if k in weight_map:
            return k

    # Heuristic fallback: substring-ish match for robustness
    # (still conservative: must end with ".weight")
    cand_norm = [c.lower() for c in candidates]
    keys = list(weight_map.keys())
    for i, c in enumerate(cand_norm):
        for k in keys:
            kl = k.lower()
            if c in kl and kl.endswith(".weight"):
                return k
    return None


def list_relevant_keys(weight_map: Dict[str, str], patterns: List[str], limit: int = 50) -> List[str]:
    out = []
    keys = list(weight_map.keys())
    for k in keys:
        kl = k.lower()
        if any(p in kl for p in patterns):
            out.append(k)
        if len(out) >= limit:
            break
    return out


def resolve_weight_file_for_tensor(
    model_name: str,
    adapter: ModelAdapter,
    tensor_key_candidates: List[str],
    local_dir: str,
    cache_dir: str,
    token: Optional[str],
    revision: Optional[str],
) -> Tuple[str, str, Dict[str, str], Optional[str]]:
    """
    Returns:
      (resolved_tensor_key, weight_file_path, weight_map_or_empty, shard_filename_or_none)
    """
    # 1) Try index
    idx_name = adapter.index_filename()
    weight_map: Dict[str, str] = {}
    if idx_name:
        idx_path = try_download(
            repo_id=model_name,
            filename=idx_name,
            local_dir=local_dir,
            cache_dir=cache_dir,
            token=token,
            revision=revision,
        )
        if idx_path:
            weight_map = load_index_weight_map(idx_path)
            resolved_key = pick_best_key_from_weight_map(weight_map, tensor_key_candidates)
            if not resolved_key:
                # Provide a debuggable error
                hints = list_relevant_keys(weight_map, patterns=["embed", "lm_head", "wte", "output", "proj"])
                raise KeyError(
                    "Could not resolve tensor key from index weight_map.\n"
                    f"Candidates: {tensor_key_candidates}\n"
                    f"Hint keys (first {len(hints)}): {hints}"
                )
            shard_filename = weight_map[resolved_key]
            shard_path = try_download(
                repo_id=model_name,
                filename=shard_filename,
                local_dir=local_dir,
                cache_dir=cache_dir,
                token=token,
                revision=revision,
            )
            if not shard_path:
                raise FileNotFoundError(f"Index said shard '{shard_filename}' but download failed.")
            return resolved_key, shard_path, weight_map, shard_filename

    # 2) Fallback to single-file safetensors
    for wf in adapter.weight_file_fallbacks():
        wf_path = try_download(
            repo_id=model_name,
            filename=wf,
            local_dir=local_dir,
            cache_dir=cache_dir,
            token=token,
            revision=revision,
        )
        if not wf_path:
            continue

        # Need to discover actual key by loading keys (safetensors is lightweight)
        tdict = load_file(wf_path)
        keys = list(tdict.keys())

        # Exact match first
        for k in tensor_key_candidates:
            if k in tdict:
                return k, wf_path, {}, wf

        # Heuristic: pick a key that contains candidate substring
        for cand in tensor_key_candidates:
            c = cand.lower()
            for k in keys:
                kl = k.lower()
                if c in kl and kl.endswith(".weight"):
                    return k, wf_path, {}, wf

        # else try next wf
    raise FileNotFoundError(
        f"Could not locate weights for model '{model_name}'. "
        f"Tried index '{idx_name}' and fallbacks {adapter.weight_file_fallbacks()}."
    )


def parse_extra_tensors(extra: Optional[str]) -> Dict[str, Any]:
    """
    extra can be:
      - None
      - comma-separated tensor keys: "a,b,c"
      - a JSON file path:
          {"tensors":["a","b"], "patterns":["layers.0.", "mlp."]}
    Returns dict with fields: tensors(list[str]), patterns(list[str])
    """
    if not extra:
        return {"tensors": [], "patterns": []}

    extra = extra.strip()
    if os.path.isfile(extra) and extra.lower().endswith(".json"):
        obj = read_json(extra)
        tensors = obj.get("tensors", []) or []
        patterns = obj.get("patterns", []) or []
        if not isinstance(tensors, list) or not isinstance(patterns, list):
            raise ValueError("--extra_tensors JSON must have list fields: tensors, patterns")
        return {"tensors": tensors, "patterns": patterns}

    # Otherwise treat as comma-separated keys
    tensors = [x.strip() for x in extra.split(",") if x.strip()]
    return {"tensors": tensors, "patterns": []}


def expand_patterns_from_weight_map(weight_map: Dict[str, str], patterns: List[str]) -> List[str]:
    if not patterns:
        return []
    out = []
    pats = [p.lower() for p in patterns]
    for k in weight_map.keys():
        kl = k.lower()
        if any(p in kl for p in pats):
            out.append(k)
    return out



def clear_download_dir(dl_dir: str) -> None:
    """
    Remove all files/subdirs inside dl_dir, but keep dl_dir itself.
    """
    if not os.path.isdir(dl_dir):
        return

    for name in os.listdir(dl_dir):
        path = os.path.join(dl_dir, name)
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.unlink(path)
            else:
                shutil.rmtree(path)
        except Exception as e:
            print(f"[WARN] Failed to remove {path}: {e}")


# -----------------------------
# Main
# -----------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Export HF tensors + tokenizer/config using safetensors index logic."
    )
    ap.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HF repo id, e.g. mistralai/Mistral-7B-v0.1",
    )
    ap.add_argument("--out_dir", type=str, default="./artifacts", help="Output root directory")
    ap.add_argument("--cache_dir", type=str, default="./.hf_cache", help="HF cache directory")
    ap.add_argument(
        "--local_dir",
        type=str,
        default=None,
        help="Local download dir inside model folder (default: <out>/<model>/downloads)",
    )
    ap.add_argument("--revision", type=str, default=None, help="HF revision/commit/tag")
    ap.add_argument(
        "--token",
        type=str,
        default=None,
        help="HF token if gated repo (else rely on env/login)",
    )

    # Defaults are True; flags only used to disable.
    ap.add_argument(
        "--no_save_tokenizer",
        action="store_true",
        help="Disable saving tokenizer (default: enabled)",
    )
    ap.add_argument(
        "--no_save_config",
        action="store_true",
        help="Disable saving config.json (default: enabled)",
    )

    ap.add_argument(
        "--spaces",
        type=str,
        default="all",
        help="Which spaces to export: all | input_embd | output_proj | input_embd,output_proj",
    )
    ap.add_argument("--family", type=str, default="auto", help="auto|pythia")
    ap.add_argument(
        "--extra_tensors",
        type=str,
        default=None,
        help="Reserved: comma-separated tensor keys OR path to JSON with {tensors:[...],patterns:[...]}",
    )

    # Direction A: multi-step export (Pythia)
    ap.add_argument(
        "--steps",
        type=str,
        default=None,
        help='For family=pythia: "all" or comma-separated like "0,1000,143000". '
             "If set, export each step as a separate revision.",
    )
    ap.add_argument(
        "--step_prefix",
        type=str,
        default="step",
        help='Prefix used to form revision names and output subdirs (default: "step").',
    )

    args = ap.parse_args(argv)

    save_tokenizer = not args.no_save_tokenizer
    save_config = not args.no_save_config

    model_name = args.model_name.strip()
    base_root = os.path.join(args.out_dir, model_name)
    ensure_dir(base_root)
    ensure_dir(args.cache_dir)

    adapter = make_adapter(model_name, args.family)

    # If steps is provided, we do a multi-revision export.
    do_multi_steps = args.steps is not None and args.steps.strip() != ""

    # Tokenizer/config should NOT be saved per-step (your requirement).
    # We save them once at base_root (unless disabled).
    if save_tokenizer:
        out_tokenizer_dir = os.path.join(base_root, "tokenizer")
        ensure_dir(out_tokenizer_dir)
        tok = AutoTokenizer.from_pretrained(
            model_name,
            revision=args.revision,
            token=args.token if args.token else None,
        )
        tok.save_pretrained(out_tokenizer_dir)

    if save_config:
        out_config_dir = os.path.join(base_root, "config")
        ensure_dir(out_config_dir)
        cfg = AutoConfig.from_pretrained(
            model_name,
            revision=args.revision,
            token=args.token if args.token else None,
        )
        cfg_path = os.path.join(out_config_dir, "config.json")
        cfg.to_json_file(cfg_path)

    # Resolve which spaces
    spaces_raw = args.spaces.strip().lower()
    if spaces_raw == "all":
        spaces = ["input_embd", "output_proj"]
    else:
        spaces = [s.strip() for s in spaces_raw.split(",") if s.strip()]
        for s in spaces:
            if s not in ("input_embd", "output_proj"):
                raise ValueError(
                    f"Unsupported space '{s}'. Use all|input_embd|output_proj or comma-separated."
                )

    # Extra tensors (still supported; will be written under each step root)
    extra_spec = parse_extra_tensors(args.extra_tensors)
    extra_keys: List[str] = list(extra_spec["tensors"])
    extra_patterns: List[str] = list(extra_spec["patterns"])

    def parse_steps_arg(steps_arg: str) -> List[int]:
        s = steps_arg.strip().lower()
        if s == "all":
            # Discover refs via HF API
            from huggingface_hub import HfApi
            api = HfApi()
            refs = api.list_repo_refs(repo_id=model_name, token=args.token if args.token else None)
            found: List[int] = []
            pat = re.compile(rf"^{re.escape(args.step_prefix)}(\d+)$")
            # branches
            for b in getattr(refs, "branches", []) or []:
                m = pat.match(b.name)
                if m:
                    found.append(int(m.group(1)))
            # tags
            for t in getattr(refs, "tags", []) or []:
                m = pat.match(t.name)
                if m:
                    found.append(int(m.group(1)))
            found = sorted(set(found))
            if not found:
                raise RuntimeError(f'--steps all: no refs matched pattern "{args.step_prefix}<digits>"')
            return found

        # comma-separated list
        out: List[int] = []
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            out.append(int(part))
        if not out:
            raise ValueError("--steps provided but no valid step numbers parsed")
        return out

    def export_one(root: str, revision: Optional[str]) -> None:
        """
        Export tensors (+ extras) into `root`.
        Tokenizer/config already saved at base_root, not here.
        """
        dl_dir = args.local_dir or os.path.join(root, "downloads")
        ensure_dir(root)
        ensure_dir(dl_dir)

        out_tensors_dir = os.path.join(root, "tensors")
        out_extra_dir = os.path.join(root, "tensors_extra")
        out_meta_dir = os.path.join(root, "meta")

        ensure_dir(out_tensors_dir)
        ensure_dir(out_extra_dir)
        ensure_dir(out_meta_dir)

        resolved_records: List[Dict[str, Any]] = []
        run_meta: Dict[str, Any] = {
            "timestamp_utc": now_iso(),
            "model_name": model_name,
            "revision": revision,
            "family": args.family,
            "spaces_arg": args.spaces,
            "save_tokenizer": False,  # by design (saved at base_root)
            "save_config": False,     # by design (saved at base_root)
            "extra_tensors_arg": args.extra_tensors,
            "root": root,
            "outputs": [],
        }

        expanded_pattern_keys: List[str] = []

        # Export spaces (canonical filenames)
        for space in spaces:
            candidates = adapter.space_candidates(space)
            resolved_key, weight_path, weight_map, shard_filename = resolve_weight_file_for_tensor(
                model_name=model_name,
                adapter=adapter,
                tensor_key_candidates=candidates,
                local_dir=dl_dir,
                cache_dir=args.cache_dir,
                token=args.token,
                revision=revision,
            )

            if extra_patterns and weight_map and not expanded_pattern_keys:
                expanded_pattern_keys = expand_patterns_from_weight_map(weight_map, extra_patterns)

            tensor_dict = load_file(weight_path)
            if resolved_key not in tensor_dict:
                raise KeyError(f"Resolved key '{resolved_key}' not found in loaded file '{weight_path}'")
            t = tensor_dict[resolved_key]

            out_path = os.path.join(out_tensors_dir, f"{space}.pt")
            torch.save(t, out_path)

            rec = {
                "logical_name": space,
                "resolved_tensor_key": resolved_key,
                "weight_file_local": weight_path,
                "weight_file_name": shard_filename,
                "shape": list(t.shape),
                "dtype": str(t.dtype),
            }
            resolved_records.append(rec)
            run_meta["outputs"].append({"type": "tensor", "logical_name": space, "path": out_path})

        # Export extras (best-effort)
        extra_all: List[str] = []
        extra_all.extend(extra_keys)
        extra_all.extend(expanded_pattern_keys)

        seen = set()
        extra_all = [k for k in extra_all if not (k in seen or seen.add(k))]

        for i, tensor_key in enumerate(extra_all):
            try:
                resolved_key, weight_path, _wm, shard_filename = resolve_weight_file_for_tensor(
                    model_name=model_name,
                    adapter=adapter,
                    tensor_key_candidates=[tensor_key],
                    local_dir=dl_dir,
                    cache_dir=args.cache_dir,
                    token=args.token,
                    revision=revision,
                )
                tensor_dict = load_file(weight_path)
                if resolved_key not in tensor_dict:
                    raise KeyError(f"Resolved key '{resolved_key}' not found in '{weight_path}'")
                t = tensor_dict[resolved_key]

                safe_name = re.sub(r"[^A-Za-z0-9._\-]+", "_", resolved_key)
                out_path = os.path.join(out_extra_dir, f"{safe_name}.pt")
                torch.save(t, out_path)

                rec = {
                    "logical_name": f"extra[{i}]",
                    "requested_tensor_key": tensor_key,
                    "resolved_tensor_key": resolved_key,
                    "weight_file_local": weight_path,
                    "weight_file_name": shard_filename,
                    "shape": list(t.shape),
                    "dtype": str(t.dtype),
                }
                resolved_records.append(rec)
                run_meta["outputs"].append({"type": "tensor", "logical_name": rec["logical_name"], "path": out_path})
            except Exception as e:
                resolved_records.append(
                    {"logical_name": f"extra[{i}]", "requested_tensor_key": tensor_key, "error": repr(e)}
                )

        adapter.extra_steps({"args": vars(args), "root": root, "revision": revision})

        meta_path = os.path.join(out_meta_dir, "resolved.json")
        run_meta["resolved"] = resolved_records
        write_json(meta_path, run_meta)

        clear_download_dir(dl_dir)

        print(f"[OK] Export completed. Root: {root}")
        print(f"[OK] Meta: {meta_path}")
        print("[OK] downloads/ cleaned.")

    # ---- Execution modes ----
    if not do_multi_steps:
        # Single export (honor args.revision)
        export_one(root=base_root, revision=args.revision)
        return 0

    # Multi-step export: each step maps to revision "{step_prefix}{N}" and root "<base_root>/{step_prefix}{N}"
    steps = parse_steps_arg(args.steps)
    for n in steps:
        rev = f"{args.step_prefix}{n}"
        step_root = os.path.join(base_root, rev)
        export_one(root=step_root, revision=rev)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
