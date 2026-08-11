from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
from typing import Any, Optional, Sequence


DEFAULT_OUT_ROOT = "comp"
DEFAULT_SPACE_NAME = "output_proj"
SUMMARY_FILENAME = "summary.csv"
CLUSTER_CSV_PATH_COL = "cluster_csv_path"
TOKEN_ID_COL = "token_id"
CLUSTER_ID_COL = "cluster_id"
PROBABILITY_COL = "probability"


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _to_int(value: Any, *, field_name: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"Failed to parse {field_name} as int: {value!r}") from exc


def _decode_token_ids(tokenizer: Any, token_ids: Sequence[int]) -> list[str]:
    token_ids = [int(tid) for tid in token_ids]
    decoded_tokens: list[str] = []

    if hasattr(tokenizer, "convert_ids_to_tokens"):
        decoded = tokenizer.convert_ids_to_tokens(token_ids)
        if isinstance(decoded, list):
            decoded_tokens = [str(tok) for tok in decoded]

    if len(decoded_tokens) != len(token_ids) or any(tok in {"None", ""} for tok in decoded_tokens):
        decoded_tokens = []
        for tid in token_ids:
            decoded_tokens.append(
                str(tokenizer.decode([tid], add_special_tokens=False))
            )

    return decoded_tokens


def resolve_query_token_id(tokenizer: Any, query_token: Any) -> int:
    if isinstance(query_token, int):
        return int(query_token)

    if tokenizer is None:
        raise ValueError("tokenizer must not be None when query_token is not an int")

    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else None
    if vocab is not None and query_token in vocab:
        return _to_int(vocab[query_token], field_name="token_id")

    if hasattr(tokenizer, "convert_tokens_to_ids"):
        token_id = tokenizer.convert_tokens_to_ids(query_token)
        if token_id is not None:
            token_id = _to_int(token_id, field_name="token_id")
            unk_token_id = getattr(tokenizer, "unk_token_id", None)
            if unk_token_id is None or token_id != int(unk_token_id):
                return token_id
            if vocab is not None and query_token in vocab:
                return token_id

    if hasattr(tokenizer, "encode"):
        encoded = tokenizer.encode(str(query_token), add_special_tokens=False)
        if len(encoded) == 1:
            return _to_int(encoded[0], field_name="token_id")
        if len(encoded) == 0:
            raise ValueError(f"Token not found in tokenizer vocab: {query_token!r}")
        raise ValueError(
            "query_token does not map to a single token id under this tokenizer: "
            f"{query_token!r} -> {encoded}"
        )

    raise ValueError(
        "tokenizer does not support get_vocab(), convert_tokens_to_ids(), or encode()"
    )


def load_full_summary_rows(
    *,
    out_root: str | Path = DEFAULT_OUT_ROOT,
    model_name: str,
    space_name: str = DEFAULT_SPACE_NAME,
    summary_filename: str = SUMMARY_FILENAME,
) -> list[dict[str, str]]:
    summary_path = Path(out_root) / model_name / space_name / summary_filename
    return _read_csv_rows(summary_path)


def load_random_summary_rows(
    *,
    out_root: str | Path = DEFAULT_OUT_ROOT,
    model_name: str,
    space_name: str = DEFAULT_SPACE_NAME,
    seed: int,
    summary_filename: Optional[str] = None,
) -> list[dict[str, str]]:
    random_dir = Path(out_root) / model_name / space_name / "random" / f"seed_{int(seed)}"
    if summary_filename is None:
        summary_path = random_dir / f"summary_random_pca_seed_{int(seed)}.csv"
    else:
        summary_path = random_dir / summary_filename
    return _read_csv_rows(summary_path)


def _build_partition_record(
    *,
    partition_type: str,
    summary_row: dict[str, str],
    seed: Optional[int],
    query_token_id: int,
    tokenizer: Any,
) -> Optional[dict[str, Any]]:
    cluster_csv_path = summary_row.get(CLUSTER_CSV_PATH_COL)
    if not cluster_csv_path:
        raise KeyError(f"summary row missing required column: {CLUSTER_CSV_PATH_COL}")

    cluster_rows = _read_csv_rows(cluster_csv_path)
    query_cluster_id: Optional[int] = None
    matched_row_count = 0
    for row in cluster_rows:
        row_token_id = _to_int(row[TOKEN_ID_COL], field_name=TOKEN_ID_COL)
        if row_token_id == query_token_id:
            query_cluster_id = _to_int(row[CLUSTER_ID_COL], field_name=CLUSTER_ID_COL)
            matched_row_count += 1

    if matched_row_count > 1:
        raise ValueError(
            f"Token id {query_token_id} matched multiple rows in cluster CSV: {cluster_csv_path}"
        )

    if query_cluster_id is None:
        return None

    if query_cluster_id == -1:
        return {
            "partition_type": partition_type,
            "seed": seed,
            "pca_dim": _to_int(summary_row["pca_dim"], field_name="pca_dim"),
            "cluster_id": query_cluster_id,
            "cluster_size": 0,
            "cluster_csv_path": str(cluster_csv_path),
            "summary_row": summary_row,
            "cluster_content": [],
        }

    cluster_members = [
        row for row in cluster_rows
        if _to_int(row[CLUSTER_ID_COL], field_name=CLUSTER_ID_COL) == query_cluster_id
    ]
    member_token_ids = [_to_int(row[TOKEN_ID_COL], field_name=TOKEN_ID_COL) for row in cluster_members]
    decoded_tokens = _decode_token_ids(tokenizer, member_token_ids)

    cluster_content_rows: list[dict[str, Any]] = []
    for row, member_token, member_token_id in zip(cluster_members, decoded_tokens, member_token_ids):
        cluster_content_rows.append(
            {
                "token_id": member_token_id,
                "token": member_token,
                "cluster_id": query_cluster_id,
                "probability": row.get(PROBABILITY_COL),
            }
        )

    return {
        "partition_type": partition_type,
        "seed": seed,
        "pca_dim": _to_int(summary_row["pca_dim"], field_name="pca_dim"),
        "cluster_id": query_cluster_id,
        "cluster_size": len(cluster_content_rows),
        "cluster_csv_path": str(cluster_csv_path),
        "summary_row": summary_row,
        "cluster_content": cluster_content_rows,
    }


def collect_token_case_study(
    *,
    model_name: str,
    tokenizer: Any,
    query_token: str,
    seed_list: Sequence[int],
    out_root: str | Path = DEFAULT_OUT_ROOT,
    space_name: str = DEFAULT_SPACE_NAME,
) -> dict[str, Any]:
    query_token_id = resolve_query_token_id(tokenizer, query_token)

    matches: list[dict[str, Any]] = []

    for summary_row in load_full_summary_rows(
        out_root=out_root,
        model_name=model_name,
        space_name=space_name,
    ):
        record = _build_partition_record(
            partition_type="full",
            summary_row=summary_row,
            seed=None,
            query_token_id=query_token_id,
            tokenizer=tokenizer,
        )
        if record is not None:
            matches.append(record)

    for seed in seed_list:
        for summary_row in load_random_summary_rows(
            out_root=out_root,
            model_name=model_name,
            space_name=space_name,
            seed=int(seed),
        ):
            record = _build_partition_record(
                partition_type="random",
                summary_row=summary_row,
                seed=int(seed),
                query_token_id=query_token_id,
                tokenizer=tokenizer,
            )
            if record is not None:
                matches.append(record)

    matches.sort(
        key=lambda row: (
            row["partition_type"],
            -1 if row["seed"] is None else int(row["seed"]),
            int(row["pca_dim"]),
        )
    )

    summary_rows: list[dict[str, Any]] = []
    for match in matches:
        summary_rows.append(
            {
                "partition_type": match["partition_type"],
                "seed": match["seed"],
                "pca_dim": match["pca_dim"],
                "cluster_id": match["cluster_id"],
                "cluster_size": match["cluster_size"],
                "cluster_csv_path": match["cluster_csv_path"],
            }
        )

    return {
        "model_name": model_name,
        "query_token": query_token,
        "query_token_id": query_token_id,
        "summary": summary_rows,
        "matches": matches,
    }


def print_case_study_report(result: dict[str, Any]) -> None:
    print(f"model_name: {result['model_name']}")
    print(f"query_token: {result['query_token']}")
    print(f"query_token_id: {result['query_token_id']}")
    print(f"num_matches: {len(result['matches'])}")

    print("\n=== Summary ===")
    for row in result["summary"]:
        print(
            f"partition_type={row['partition_type']}, "
            f"seed={row['seed']}, "
            f"pca_dim={row['pca_dim']}, "
            f"cluster_id={row['cluster_id']}, "
            f"cluster_size={row['cluster_size']}"
        )

    for match in result["matches"]:
        print("\n" + "=" * 80)
        print(
            f"partition_type={match['partition_type']} | "
            f"seed={match['seed']} | "
            f"pca_dim={match['pca_dim']} | "
            f"cluster_id={match['cluster_id']} | "
            f"cluster_size={match['cluster_size']}"
        )
        print(match["cluster_csv_path"])
        if match["cluster_id"] == -1:
            print("cluster_label=-1")
            continue
        token_id_list = [row["token_id"] for row in match["cluster_content"]]
        token_list = [row["token"] for row in match["cluster_content"]]
        print(f"token_id_list={token_id_list}")
        print(f"token_list={token_list}")


def _select_summary_row_by_pca_dim(
    summary_rows: Sequence[dict[str, str]],
    *,
    pca_dim: int,
) -> dict[str, str]:
    matched = [
        row for row in summary_rows
        if _to_int(row["pca_dim"], field_name="pca_dim") == int(pca_dim)
    ]
    if len(matched) == 0:
        raise ValueError(f"No summary row found for pca_dim={pca_dim}")
    if len(matched) > 1:
        raise ValueError(
            f"Expected exactly one summary row for pca_dim={pca_dim}, got {len(matched)}"
        )
    return matched[0]


def _load_cluster_df(summary_row: dict[str, str]) -> pd.DataFrame:
    cluster_csv_path = summary_row.get(CLUSTER_CSV_PATH_COL)
    if not cluster_csv_path:
        raise KeyError(f"summary row missing required column: {CLUSTER_CSV_PATH_COL}")
    cluster_df = pd.read_csv(cluster_csv_path)
    required_cols = [TOKEN_ID_COL, CLUSTER_ID_COL]
    missing = [col for col in required_cols if col not in cluster_df.columns]
    if missing:
        raise KeyError(f"cluster CSV missing required columns: {missing}")
    cluster_df = cluster_df.copy()
    cluster_df[TOKEN_ID_COL] = pd.to_numeric(cluster_df[TOKEN_ID_COL], errors="raise").astype(int)
    cluster_df[CLUSTER_ID_COL] = pd.to_numeric(cluster_df[CLUSTER_ID_COL], errors="raise").astype(int)
    return cluster_df


def _collect_non_noise_token_ids_from_df(cluster_df: pd.DataFrame) -> set[int]:
    valid_df = cluster_df.loc[cluster_df[CLUSTER_ID_COL] != -1, [TOKEN_ID_COL]].copy()
    return set(valid_df[TOKEN_ID_COL].tolist())


def compare_dim_non_noise_overlap(
    *,
    model_name: str,
    tokenizer: Any,
    seed_list: Sequence[int],
    dim0: int,
    dim1: int,
    out_root: str | Path = DEFAULT_OUT_ROOT,
    space_name: str = DEFAULT_SPACE_NAME,
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []

    def _build_comparison(
        partition_type: str,
        seed: Optional[int],
        summary_rows: Sequence[dict[str, str]],
    ) -> dict[str, Any]:
        row0 = _select_summary_row_by_pca_dim(summary_rows, pca_dim=dim0)
        row1 = _select_summary_row_by_pca_dim(summary_rows, pca_dim=dim1)

        df0 = _load_cluster_df(row0)
        df1 = _load_cluster_df(row1)

        tokens0 = _collect_non_noise_token_ids_from_df(df0)
        tokens1 = _collect_non_noise_token_ids_from_df(df1)

        overlap_token_ids = sorted(tokens0 & tokens1)
        union_count = len(tokens0 | tokens1)
        overlap_ratio = 0.0 if union_count == 0 else len(overlap_token_ids) / union_count
        overlap_tokens = _decode_token_ids(tokenizer, overlap_token_ids)

        return {
            "partition_type": partition_type,
            "seed": seed,
            "dim0": int(dim0),
            "dim1": int(dim1),
            "dim0_token_count": len(tokens0),
            "dim1_token_count": len(tokens1),
            "overlap_count": len(overlap_token_ids),
            "overlap_ratio": overlap_ratio,
            "overlap_token_ids": overlap_token_ids,
            "overlap_tokens": overlap_tokens,
        }

    full_summary_rows = load_full_summary_rows(
        out_root=out_root,
        model_name=model_name,
        space_name=space_name,
    )
    comparisons.append(_build_comparison("full", None, full_summary_rows))

    for seed in seed_list:
        random_summary_rows = load_random_summary_rows(
            out_root=out_root,
            model_name=model_name,
            space_name=space_name,
            seed=int(seed),
        )
        comparisons.append(_build_comparison("random", int(seed), random_summary_rows))

    return {
        "model_name": model_name,
        "dim0": int(dim0),
        "dim1": int(dim1),
        "comparisons": comparisons,
    }


def print_dim_non_noise_overlap_report(result: dict[str, Any]) -> None:
    print(f"model_name: {result['model_name']}")
    print(f"dim0: {result['dim0']}")
    print(f"dim1: {result['dim1']}")

    for row in result["comparisons"]:
        print("\n" + "=" * 80)
        print(
            f"partition_type={row['partition_type']} | "
            f"seed={row['seed']} | "
            f"dim0_token_count={row['dim0_token_count']} | "
            f"dim1_token_count={row['dim1_token_count']} | "
            f"overlap_count={row['overlap_count']} | "
            f"overlap_ratio={row['overlap_ratio']:.6f}"
        )
        print(f"overlap_token_ids={row['overlap_token_ids']}")
        print(f"overlap_tokens={row['overlap_tokens']}")


def compare_dim_non_noise_disjoint_tokens(
    *,
    model_name: str,
    tokenizer: Any,
    seed_list: Sequence[int],
    dim0: int,
    dim1: int,
    out_root: str | Path = DEFAULT_OUT_ROOT,
    space_name: str = DEFAULT_SPACE_NAME,
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []

    def _build_comparison(
        partition_type: str,
        seed: Optional[int],
        summary_rows: Sequence[dict[str, str]],
    ) -> dict[str, Any]:
        row0 = _select_summary_row_by_pca_dim(summary_rows, pca_dim=dim0)
        row1 = _select_summary_row_by_pca_dim(summary_rows, pca_dim=dim1)

        df0 = _load_cluster_df(row0)
        df1 = _load_cluster_df(row1)

        tokens0 = _collect_non_noise_token_ids_from_df(df0)
        tokens1 = _collect_non_noise_token_ids_from_df(df1)

        disjoint_token_ids = sorted((tokens0 - tokens1) | (tokens1 - tokens0))
        disjoint_tokens = _decode_token_ids(tokenizer, disjoint_token_ids)

        return {
            "partition_type": partition_type,
            "seed": seed,
            "dim0": int(dim0),
            "dim1": int(dim1),
            "dim0_token_count": len(tokens0),
            "dim1_token_count": len(tokens1),
            "disjoint_count": len(disjoint_token_ids),
            "disjoint_token_ids": disjoint_token_ids,
            "disjoint_tokens": disjoint_tokens,
        }

    full_summary_rows = load_full_summary_rows(
        out_root=out_root,
        model_name=model_name,
        space_name=space_name,
    )
    comparisons.append(_build_comparison("full", None, full_summary_rows))

    for seed in seed_list:
        random_summary_rows = load_random_summary_rows(
            out_root=out_root,
            model_name=model_name,
            space_name=space_name,
            seed=int(seed),
        )
        comparisons.append(_build_comparison("random", int(seed), random_summary_rows))

    return {
        "model_name": model_name,
        "dim0": int(dim0),
        "dim1": int(dim1),
        "comparisons": comparisons,
    }


def print_dim_non_noise_disjoint_report(result: dict[str, Any]) -> None:
    print(f"model_name: {result['model_name']}")
    print(f"dim0: {result['dim0']}")
    print(f"dim1: {result['dim1']}")

    for row in result["comparisons"]:
        print("\n" + "=" * 80)
        print(
            f"partition_type={row['partition_type']} | "
            f"seed={row['seed']} | "
            f"dim0_token_count={row['dim0_token_count']} | "
            f"dim1_token_count={row['dim1_token_count']} | "
            f"disjoint_count={row['disjoint_count']}"
        )
        print(f"disjoint_token_ids={row['disjoint_token_ids']}")
        print(f"disjoint_tokens={row['disjoint_tokens']}")
