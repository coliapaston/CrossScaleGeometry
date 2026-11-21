"""Token-id to token-text conversion shared by every metric and control."""

from __future__ import annotations

from typing import Any, List, Sequence

import pandas as pd


def uses_decoded_token_text(model_name: str) -> bool:
    """Return whether a model's tokens must be decoded instead of read raw.

    Byte-level BPE vocabularies (GPT-oss) store symbols such as "Ġ" that only
    become readable text after decoding. SentencePiece vocabularies (Mistral,
    Mixtral) are read raw so that markers such as "▁" are kept. Only the last
    path component is checked, so control runs stored as, for example,
    "permutation/gpt-oss_rand" follow their source model.
    """

    return str(model_name).rsplit("/", 1)[-1].startswith("gpt")


def token_id_to_text(token_id: int, *, tokenizer: Any, model_name: str) -> Any:
    """Convert one token id to its text under the model's decoding rule."""

    tid = int(token_id)
    if uses_decoded_token_text(model_name):
        return tokenizer.decode([tid], clean_up_tokenization_spaces=False)
    return tokenizer.convert_ids_to_tokens(tid)


def token_ids_to_text(
    token_ids: Sequence[int],
    *,
    tokenizer: Any,
    model_name: str,
) -> List[str]:
    """Convert token ids to text under the model's decoding rule."""

    if tokenizer is None:
        raise ValueError("tokenizer must not be None when decoding token_id -> token_str")

    token_ids = list(token_ids)

    if uses_decoded_token_text(model_name):
        decoded: List[str] = []
        for tid in token_ids:
            try:
                decoded.append(
                    tokenizer.decode([int(tid)], clean_up_tokenization_spaces=False)
                )
            except Exception as exc:
                raise RuntimeError(f"Failed to decode token_id={tid}") from exc
        return decoded

    if hasattr(tokenizer, "convert_ids_to_tokens"):
        return tokenizer.convert_ids_to_tokens(token_ids)

    decoded = []
    for tid in token_ids:
        try:
            decoded.append(tokenizer.decode([int(tid)]))
        except Exception as exc:
            raise RuntimeError(f"Failed to decode token_id={tid}") from exc
    return decoded


def attach_token_str_from_tokenizer(
    cluster_df: pd.DataFrame,
    *,
    tokenizer,
    model_name: str,
    token_id_col: str = "token_id",
    token_str_col: str = "token_str",
) -> pd.DataFrame:
    """
    Use tokenizer to map token_id -> token_str.

    tokenizer must support:
        tokenizer.convert_ids_to_tokens(int)
    """
    if token_str_col in cluster_df.columns:
        return cluster_df

    if token_id_col not in cluster_df.columns:
        raise KeyError(
            f"cluster_df missing both '{token_str_col}' and '{token_id_col}'"
        )

    df = cluster_df.copy()

    df[token_str_col] = df[token_id_col].map(
        lambda x: token_id_to_text(x, tokenizer=tokenizer, model_name=model_name)
    )

    return df
