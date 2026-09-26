"""最少专用输入选址的必要测试。

运行：python -m unittest discover -s tests -v
独立参考见 tests/reference.py（枚举输入集合，不导入被测核心）。
"""

from __future__ import annotations

import itertools
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sparsestruct import (  # noqa: E402
    MAX_STATES,
    InputPlacement,
    InvalidStructureError,
    LimitExceededError,
    select_dedicated_inputs,
)

from reference import (  # noqa: E402
    reference_min_dedicated_inputs,
    reference_source_sccs,
)


def check_certificate(tc, edges, n, p):
    """对照独立参考核验证书的全部字段。"""
    tc.assertIsInstance(p, InputPlacement)
    edges = sorted(set(edges))
    k_ref, states_ref = reference_min_dedicated_inputs(edges, n)
    # 最少性与并列 tie-break（状态 ID 集合字典序最小）。
    tc.assertEqual(p.n_inputs, k_ref)
    tc.assertEqual(p.states, states_ref)
    tc.assertEqual(tuple(sorted(p.states)), p.states)
    # 匹配证据：未匹配行恰好是所选状态，匹配边合法且互逆。
    tc.assertEqual(p.unmatched_states, p.states)
    edge_set = set(edges)
    matched_rows = set()
    matched_cols = set()
    for r, c in enumerate(p.row_match):
        if c == -1:
            continue
        tc.assertIn((r, c), edge_set)
        tc.assertEqual(p.col_match[c], r)
        matched_rows.add(r)
        matched_cols.add(c)
    tc.assertEqual(matched_rows, set(range(n)) - set(p.states))
    # 源 SCC 与覆盖关系：每个源 SCC 恰由所列输入覆盖。
    tc.assertEqual(p.source_sccs, reference_source_sccs(edges, n))
    tc.assertEqual(len(p.scc_coverage), len(p.source_sccs))
    for comp, cover in zip(p.source_sccs, p.scc_coverage):
        tc.assertIn(cover, p.states)
        tc.assertIn(cover, comp)
        tc.assertEqual(cover, min(s for s in p.states if s in comp))


class TestCanonicalCases(unittest.TestCase):
    def test_zero_matrix(self):
        # 零矩阵：秩 0 且每个状态都是孤立源 SCC，需要全部 n 个输入。
        p = select_dedicated_inputs([], 4)
        self.assertEqual(p.states, (0, 1, 2, 3))
        self.assertEqual(p.structural_rank, 0)
        self.assertEqual(p.source_sccs, ((0,), (1,), (2,), (3,)))
        self.assertEqual(p.scc_coverage, (0, 1, 2, 3))
        check_certificate(self, [], 4, p)

    def test_single_cycle_full_rank(self):
        # 满结构秩单环：秩缺口为 0，但源 SCC 不可达，仍需 1 个输入。
        edges = [(1, 0), (2, 1), (0, 2)]
        p = select_dedicated_inputs(edges, 3)
        self.assertEqual(p.structural_rank, 3)
        self.assertEqual(p.states, (0,))  # 不能只数结构秩缺口（会给 0）
        check_certificate(self, edges, 3, p)

    def test_multiple_isolated_cycles_full_rank(self):
        # 满秩但两个孤立环：每个环一个输入，取各环最小状态 ID。
        edges = [(1, 0), (0, 1), (3, 2), (2, 3)]
        p = select_dedicated_inputs(edges, 4)
        self.assertEqual(p.structural_rank, 4)
        self.assertEqual(p.states, (0, 2))
        self.assertEqual(p.source_sccs, ((0, 1), (2, 3)))
        self.assertEqual(p.scc_coverage, (0, 2))
        check_certificate(self, edges, 4, p)

    def test_chain(self):
        # 链 0->1->2->3：秩缺口 1、源 SCC {0}，一个输入打在链头。
        edges = [(1, 0), (2, 1), (3, 2)]
        p = select_dedicated_inputs(edges, 4)
        self.assertEqual(p.states, (0,))
        self.assertEqual(p.rank_deficiency, 1)
        check_certificate(self, edges, 4, p)

    def test_diagonal_full_rank_all_sources(self):
        # 对角阵：满秩但每个状态都是源 SCC，需要 n 个输入。
        edges = [(i, i) for i in range(3)]
        p = select_dedicated_inputs(edges, 3)
        self.assertEqual(p.states, (0, 1, 2))
        check_certificate(self, edges, 3, p)

    def test_rank_deficiency_dominates(self):
        # 秩缺口大于源 SCC 数：下界由膨胀条件主导。
        edges = [(1, 0), (0, 1), (3, 2)]  # 环{0,1} + 链 2->3，秩 3
        p = select_dedicated_inputs(edges, 4)
        self.assertEqual(p.structural_rank, 3)
        self.assertEqual(p.n_inputs, 2)
        check_certificate(self, edges, 4, p)

    def test_empty_system(self):
        p = select_dedicated_inputs([], 0)
        self.assertEqual(p.states, ())
        self.assertEqual(p.source_sccs, ())
        self.assertEqual(p.n_inputs, 0)

    def test_inferred_dimension(self):
        p = select_dedicated_inputs([(2, 1)])  # 推断 n=3
        self.assertEqual(p.n_states, 3)
        check_certificate(self, [(2, 1)], 3, p)

    def test_duplicate_entries_and_order_irrelevant(self):
        edges = [(1, 0), (0, 1), (1, 0), (3, 2), (2, 3)]
        p1 = select_dedicated_inputs(edges, 4)
        p2 = select_dedicated_inputs(list(reversed(edges)), 4)
        self.assertEqual(p1, p2)


class TestExhaustiveCrossCheck(unittest.TestCase):
    """小图全枚举：核心的选址与独立枚举参考逐一相同。"""

    def test_all_digraphs_n3(self):
        n = 3
        cells = [(i, j) for i in range(n) for j in range(n)]
        for mask in range(1 << (n * n)):
            edges = [c for k, c in enumerate(cells) if mask >> k & 1]
            p = select_dedicated_inputs(edges, n)
            check_certificate(self, edges, n, p)

    def test_random_digraphs_n6(self):
        rng = random.Random(20260926)
        for _ in range(60):
            n = rng.randint(1, 6)
            edges = [
                (i, j)
                for i in range(n)
                for j in range(n)
                if rng.random() < 0.35
            ]
            p = select_dedicated_inputs(edges, n)
            check_certificate(self, edges, n, p)


class TestInputValidation(unittest.TestCase):
    def test_state_limit(self):
        with self.assertRaises(LimitExceededError):
            select_dedicated_inputs([], MAX_STATES + 1)
        with self.assertRaises(LimitExceededError):
            select_dedicated_inputs([(20, 0)])  # 推断出 21 个状态

    def test_bool_and_float_rejected(self):
        with self.assertRaises(TypeError):
            select_dedicated_inputs([(True, 0)], 2)
        with self.assertRaises(TypeError):
            select_dedicated_inputs([(0.5, 0)], 2)
        with self.assertRaises(InvalidStructureError):
            select_dedicated_inputs([(float("nan"), 0)], 2)
        with self.assertRaises(InvalidStructureError):
            select_dedicated_inputs([(0, float("inf"))], 2)
        with self.assertRaises(TypeError):
            select_dedicated_inputs([], True)

    def test_negative_and_out_of_range(self):
        with self.assertRaises(InvalidStructureError):
            select_dedicated_inputs([(-1, 0)], 2)
        with self.assertRaises(InvalidStructureError):
            select_dedicated_inputs([(2, 0)], 2)
        with self.assertRaises(InvalidStructureError):
            select_dedicated_inputs([(0, 1, 2)], 3)

    def test_bad_types(self):
        with self.assertRaises(TypeError):
            select_dedicated_inputs(None, 2)
        with self.assertRaises(TypeError):
            select_dedicated_inputs(["ab"], 2)

    def test_pure_function_on_failure(self):
        entries = [(1, 0), (0, 1)]
        with self.assertRaises(InvalidStructureError):
            select_dedicated_inputs(entries, 1)  # 越界
        # 失败后原输入仍可用于正常调用（无部分变更）。
        p = select_dedicated_inputs(entries, 2)
        self.assertEqual(p.states, (0,))


if __name__ == "__main__":
    unittest.main()
