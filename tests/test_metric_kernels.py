"""Golden-value tests for the O-S and MS-E kernels.

The expected values were produced by the implementation that generated the
published paper results. They pin the kernels before they are moved or
rewritten during refactoring.
"""

import math
import unittest

import pandas as pd

from crossscale.metrics.orthographic import (
    compute_cluster_morphology_score,
    kondrak_trigram_similarity,
)
from crossscale.metrics.script_entropy import (
    decode_token_ids_to_strings,
    dominant_script,
    entropy_from_distribution,
    normalized_entropy,
)
from crossscale.partition.permutation import token_id_to_token_text
from crossscale.tokens import attach_token_str_from_tokenizer


class KondrakSimilarityTest(unittest.TestCase):
    GOLDEN_PAIRS = (
        ("running", "Running", 1.0),
        ("running", "runner", 0.6428571428571428),
        ("▁run", "run", 0.75),
        ("hello", "world", 0.19999999999999996),
        ("abc", "abd", 0.8333333333333334),
        ("a", "b", 0.0),
        ("ing", "▁running", 0.375),
        ("Привет", "привет", 1.0),
        ("中文", "中国", 0.75),
        ("(", ")", 0.0),
        ("abc", "", 0.0),
    )

    def test_golden_pairs(self) -> None:
        for left, right, expected in self.GOLDEN_PAIRS:
            with self.subTest(left=left, right=right):
                self.assertAlmostEqual(kondrak_trigram_similarity(left, right), expected, places=12)

    def test_similarity_is_symmetric(self) -> None:
        for left, right, _ in self.GOLDEN_PAIRS:
            with self.subTest(left=left, right=right):
                self.assertEqual(
                    kondrak_trigram_similarity(left, right),
                    kondrak_trigram_similarity(right, left),
                )

    def test_comparison_is_case_insensitive(self) -> None:
        self.assertEqual(kondrak_trigram_similarity("Token", "tOKEN"), 1.0)


class ClusterMorphologyScoreTest(unittest.TestCase):
    def test_small_cluster_uses_all_pairs(self) -> None:
        tokens = ["run", "running", "runner", "Run", "▁run"]
        pair_scores = [
            kondrak_trigram_similarity(tokens[i], tokens[j])
            for i in range(len(tokens))
            for j in range(i + 1, len(tokens))
        ]
        score = compute_cluster_morphology_score(tokens)
        self.assertAlmostEqual(score, sum(pair_scores) / len(pair_scores), places=12)
        self.assertAlmostEqual(score, 0.5619047619047619, places=12)

    def test_large_cluster_subsampling_is_deterministic(self) -> None:
        tokens = [f"tok{index:04d}" for index in range(600)]
        score = compute_cluster_morphology_score(tokens)
        self.assertAlmostEqual(score, 0.687654668166479, places=12)
        self.assertEqual(score, compute_cluster_morphology_score(list(reversed(tokens))))

    def test_singleton_cluster_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            compute_cluster_morphology_score(["only"])


class DominantScriptTest(unittest.TestCase):
    GOLDEN_LABELS = (
        ("hello", "Latin"),
        ("中文", "Han"),
        ("Привет", "Cyrillic"),
        ("アイ", "Katakana"),
        ("▁run", "Latin"),
        ("(a", "Latin"),
        ("a中", "Mixed"),
        ("ab中文", "Mixed"),
        ("123", "Other"),
        ("", "Other"),
        # Other wins outright here, so it is kept; the paper's prose would
        # reassign this token to Latin.
        ("((a", "Other"),
    )

    def test_golden_labels(self) -> None:
        for token, expected in self.GOLDEN_LABELS:
            with self.subTest(token=token):
                self.assertEqual(dominant_script(token), expected)


class ScriptEntropyTest(unittest.TestCase):
    def test_entropy_ignores_zero_probabilities(self) -> None:
        distribution = {"Latin": 0.5, "Han": 0.5, "Cyrillic": 0.0}
        self.assertAlmostEqual(entropy_from_distribution(distribution), math.log(2), places=12)

    def test_single_script_has_zero_entropy(self) -> None:
        self.assertEqual(entropy_from_distribution({"Latin": 1.0}), 0.0)

    def test_normalized_entropy_divides_by_log_script_set_size(self) -> None:
        self.assertAlmostEqual(
            normalized_entropy(math.log(2), global_script_set_size=4),
            0.5,
            places=12,
        )
        self.assertEqual(normalized_entropy(0.7, global_script_set_size=1), 0.0)


class FakeTokenizer:
    def decode(self, token_ids, clean_up_tokenization_spaces=True):
        return f"decoded-{token_ids[0]}"

    def convert_ids_to_tokens(self, token_ids):
        if isinstance(token_ids, list):
            return [f"raw-{token_id}" for token_id in token_ids]
        return f"raw-{token_ids}"


class TokenTextDecodingTest(unittest.TestCase):
    CASES = (
        ("gpt-oss", "decoded-7"),
        ("permutation/gpt-oss_rand", "decoded-7"),
        ("permutation/gpt-oss_length_bucket_rand", "decoded-7"),
        ("mistralai/Mistral-7B-v0.1", "raw-7"),
        ("permutation/mistralai/Mixtral-8x7B-v0.1_rand", "raw-7"),
    )

    def test_every_metric_path_decodes_byte_level_models(self) -> None:
        tokenizer = FakeTokenizer()
        for model_name, expected in self.CASES:
            with self.subTest(model_name=model_name):
                self.assertEqual(
                    decode_token_ids_to_strings([7], tokenizer, model_name=model_name),
                    [expected],
                )
                self.assertEqual(
                    token_id_to_token_text(7, tokenizer=tokenizer, model_name=model_name),
                    expected,
                )
                frame = attach_token_str_from_tokenizer(
                    pd.DataFrame({"token_id": [7]}),
                    tokenizer=tokenizer,
                    model_name=model_name,
                )
                self.assertEqual(frame["token_str"].tolist(), [expected])


if __name__ == "__main__":
    unittest.main()
