"""最少专用输入选址的测试（独立参考对照 + 异常路径）。

独立参考（本文件内函数，不调用 sparsestruct）：
  * 可达性：从候选集合 S 在状态有向图（非零 (i,j) 即边 j->i）上 BFS，
    要求到达全部状态；
  * 增广结构秩：A 的状态列加上每个被选状态一个专用输入列，用回溯
    枚举求最大匹配，要求覆盖全部 n 行（无膨胀）。
按子集规模递增、字典序枚举全部输入集合，第一个同时满足两条件者即
参考答案，与被测库结果逐字段对照。
"""

from __future__ import annotations

import unittest
from itertools import combinations

from sparsestruct import (
    InputPlacement,
    InvalidStructureError,
    LimitExceededError,
    MAX_STATES,
    analyze,
    minimum_dedicated_inputs,
    verify,
)


# --------------------------------------------------------------------- #
# 独立参考实现（禁止导入 sparsestruct 的内部函数）
# --------------------------------------------------------------------- #


def _ref_reachable_all(edges, n, picked):
    """从 picked 出发沿状态图 BFS，返回是否到达全部状态。"""
    succ = [[] for _ in range(n)]
    for i, j in edges:
        succ[j].append(i)  # 非零 (i, j) 即边 j -> i
    seen = set(picked)
    stack = list(picked)
    while stack:
        v = stack.pop()
        for w in succ[v]:
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return seen == set(range(n))


def _ref_augmented_full_rank(edges, n, picked):
    """回溯枚举增广二部图匹配，返回是否存在覆盖全部 n 行的匹配。"""
    picked = sorted(picked)
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
    for idx, s in enumerate(picked):
        adj[s].append(n + idx)  # 专用输入列只连接自己的状态行
    cmatch = {}

    def rec(r):
        if r == n:
            return True
        for c in adj[r]:
            if c not in cmatch:
                cmatch[c] = r
                if rec(r + 1):
                    return True
                del cmatch[c]
        return False

    return rec(0)


def reference_min_inputs(edges, n):
    """枚举全部输入集合，返回 (最小规模, 字典序最小集合)。"""
    for size in range(0, n + 1):
        for combo in combinations(range(n), size):
            if _ref_reachable_all(edges, n, combo) and \
                    _ref_augmented_full_rank(edges, n, combo):
                return size, combo
    raise AssertionError("全集必然可行，参考实现不可能走到这里")


# --------------------------------------------------------------------- #
# 与独立参考对照
# --------------------------------------------------------------------- #


class TestAgainstIndependentReference(unittest.TestCase):
    def check(self, edges, n):
        edges = sorted(set(edges))
        size, combo = reference_min_inputs(edges, n)
        p = minimum_dedicated_inputs(edges, n)
        self.assertIsInstance(p, InputPlacement)
        self.assertEqual(len(p.selected_states), size)
        self.assertEqual(p.selected_states, combo)
        return p

    def test_full_rank_two_isolated_cycles(self):
        # 满结构秩，但两个孤立环互不可达：必须各加一个输入。
        edges = [(0, 1), (1, 0), (2, 3), (3, 2)]
        p = self.check(edges, 4)
        self.assertEqual(p.structural_rank, 4)
        self.assertEqual(p.selected_states, (0, 2))
        self.assertEqual(p.source_sccs, ((0, 1), (2, 3)))
        self.assertEqual(p.scc_inputs, (0, 2))

    def test_full_rank_three_isolated_cycles(self):
        edges = [(0, 0), (1, 2), (2, 1), (3, 4), (4, 5), (5, 3)]
        p = self.check(edges, 6)
        self.assertEqual(p.structural_rank, 6)
        self.assertEqual(p.selected_states, (0, 1, 3))

    def test_chain_single_input_at_source(self):
        # 链 0->1->2->3：结构秩缺口 1，源 SCC 为 {0}，一个输入即可。
        edges = [(1, 0), (2, 1), (3, 2)]
        p = self.check(edges, 4)
        self.assertEqual(p.structural_rank, 3)
        self.assertEqual(p.selected_states, (0,))
        self.assertEqual(p.source_sccs, ((0,),))
        self.assertEqual(p.scc_inputs, (0,))

    def test_chain_with_side_branch(self):
        # 0->1->2 且 0->3：源 SCC 只有 {0}，但结构秩缺口为 2，
        # 说明“只数源 SCC”或“只数秩缺口”都不够，必须同时满足两条件。
        edges = [(1, 0), (2, 1), (3, 0)]
        p = self.check(edges, 4)
        self.assertEqual(p.structural_rank, 2)
        self.assertEqual(p.selected_states, (0, 1))

    def test_zero_matrix(self):
        p = self.check([], 4)
        self.assertEqual(p.structural_rank, 0)
        self.assertEqual(p.rank_deficiency, 4)
        self.assertEqual(p.selected_states, (0, 1, 2, 3))
        self.assertEqual(p.source_sccs, ((0,), (1,), (2,), (3,)))
        self.assertEqual(p.scc_inputs, (0, 1, 2, 3))
        self.assertEqual(p.row_assign, (-1, -1, -1, -1))

    def test_empty_system(self):
        p = minimum_dedicated_inputs([], 0)
        self.assertEqual(p.n_states, 0)
        self.assertEqual(p.selected_states, ())
        self.assertEqual(p.source_sccs, ())
        self.assertEqual(p.scc_inputs, ())

    def test_mixed_cycles_chain_isolated(self):
        # 环 {0,1}、环 {2,3,4}、链 5->6->7、孤立状态 8。
        edges = [
            (0, 1), (1, 0),
            (2, 3), (3, 4), (4, 2),
            (6, 5), (7, 6),
        ]
        p = self.check(edges, 9)
        self.assertEqual(p.source_sccs, ((0, 1), (2, 3, 4), (5,), (8,)))
        self.assertEqual(p.selected_states, (0, 2, 5, 8))
        self.assertEqual(p.scc_inputs, (0, 2, 5, 8))

    def test_tie_break_picks_smallest_state_id(self):
        # 单个 2-环：{0} 与 {1} 都可行，选字典序最小的 {0}。
        p = minimum_dedicated_inputs([(0, 1), (1, 0)], 2)
        self.assertEqual(p.selected_states, (0,))

    def test_tie_break_inside_larger_scc(self):
        # 3-环 1<->2<->3<->1 加一个被驱动的状态 0：源 SCC 为 {1,2,3}。
        edges = [(0, 1), (1, 2), (2, 3), (3, 1), (2, 1), (3, 2), (1, 3)]
        p = self.check(edges, 4)
        self.assertEqual(p.selected_states, (1,))

    def test_deterministic_small_graphs_against_reference(self):
        # 确定性伪随机小图批量对照（不依赖 random 模块以外的内容）。
        import random

        rng = random.Random(20260926)
        for _ in range(30):
            n = rng.randint(1, 7)
            edges = [
                (i, j)
                for i in range(n)
                for j in range(n)
                if rng.random() < 0.3
            ]
            self.check(edges, n)


# --------------------------------------------------------------------- #
# 证书字段核验
# --------------------------------------------------------------------- #


class TestPlacementCertificate(unittest.TestCase):
    def test_row_assign_is_valid_augmented_matching(self):
        edges = [(0, 1), (1, 0), (2, 3), (3, 2), (4, 3), (5, 5), (6, 7)]
        n = 8
        p = minimum_dedicated_inputs(edges, n)
        used_cols = set()
        input_rows = set()
        for r, a in enumerate(p.row_assign):
            if a == -1:
                input_rows.add(r)
            else:
                self.assertIn((r, a), p.nonzeros)
                self.assertNotIn(a, used_cols)
                used_cols.add(a)
        self.assertEqual(len(p.row_assign), n)
        self.assertTrue(input_rows <= set(p.selected_states))

    def test_scc_inputs_cover_each_source_scc(self):
        edges = [(0, 1), (1, 0), (2, 3), (3, 4), (4, 2), (6, 5)]
        p = minimum_dedicated_inputs(edges, 7)
        selected = set(p.selected_states)
        self.assertEqual(len(p.source_sccs), len(p.scc_inputs))
        for comp, driver in zip(p.source_sccs, p.scc_inputs):
            self.assertIn(driver, comp)
            self.assertIn(driver, selected)

    def test_col_match_is_valid_matching(self):
        edges = [(1, 0), (2, 1), (0, 0), (2, 2)]
        p = minimum_dedicated_inputs(edges, 3)
        seen_rows = set()
        for c, r in enumerate(p.col_match):
            if r == -1:
                continue
            self.assertIn((r, c), p.nonzeros)
            self.assertNotIn(r, seen_rows)
            seen_rows.add(r)
        self.assertEqual(len(seen_rows), p.structural_rank)
        self.assertEqual(
            p.unmatched_states,
            tuple(c for c in range(3) if p.col_match[c] == -1),
        )
        self.assertEqual(p.rank_deficiency, 3 - p.structural_rank)

    def test_result_immutable_and_order_independent(self):
        edges = [(0, 1), (1, 0), (3, 2)]
        p1 = minimum_dedicated_inputs(edges, 4)
        p2 = minimum_dedicated_inputs(reversed(edges), 4)
        self.assertEqual(p1, p2)
        with self.assertRaises(Exception):
            p1.selected_states = (9,)

    def test_generator_input_accepted(self):
        # 对角矩阵 = 三个带自环的孤立状态，各自是源 SCC，需 3 个输入。
        p = minimum_dedicated_inputs(((i, i) for i in range(3)), 3)
        self.assertEqual(p.selected_states, (0, 1, 2))


# --------------------------------------------------------------------- #
# 异常路径
# --------------------------------------------------------------------- #


class TestPlacementErrors(unittest.TestCase):
    def test_too_many_states_explicit(self):
        with self.assertRaises(LimitExceededError):
            minimum_dedicated_inputs([], MAX_STATES + 1)

    def test_too_many_states_inferred(self):
        with self.assertRaises(LimitExceededError):
            minimum_dedicated_inputs([(MAX_STATES, MAX_STATES)])

    def test_non_square_inference_rejected(self):
        with self.assertRaises(InvalidStructureError):
            minimum_dedicated_inputs([(0, 3)])  # 推断为 1x4，非方阵

    def test_bool_index_rejected(self):
        with self.assertRaises(TypeError):
            minimum_dedicated_inputs([(True, 0)], 2)

    def test_bool_n_states_rejected(self):
        with self.assertRaises(TypeError):
            minimum_dedicated_inputs([], True)

    def test_nan_rejected(self):
        with self.assertRaises(InvalidStructureError):
            minimum_dedicated_inputs([(0, float("nan"))], 2)

    def test_float_index_rejected(self):
        with self.assertRaises(TypeError):
            minimum_dedicated_inputs([(0, 1.5)], 2)

    def test_negative_index_rejected(self):
        with self.assertRaises(InvalidStructureError):
            minimum_dedicated_inputs([(-1, 0)], 2)

    def test_index_out_of_explicit_dimension(self):
        with self.assertRaises(InvalidStructureError):
            minimum_dedicated_inputs([(3, 0)], 3)

    def test_none_entries_rejected(self):
        with self.assertRaises(TypeError):
            minimum_dedicated_inputs(None, 2)


# --------------------------------------------------------------------- #
# 回归：原矩阵分块输出不受新功能影响
# --------------------------------------------------------------------- #


class TestAnalyzeUnchanged(unittest.TestCase):
    def test_analyze_still_blocks_same_matrix(self):
        edges = [
            (0, 0), (1, 0), (1, 1), (2, 1),
            (3, 2), (3, 3), (3, 4), (4, 2), (4, 3), (4, 4),
            (5, 2), (5, 3), (5, 4),
        ]
        d = analyze(edges, 6, 5)
        self.assertTrue(verify(d))
        self.assertEqual(d.structural_rank, 5)
        self.assertEqual(d.block_regions, ("W", "O"))
        self.assertEqual([b.shape for b in d.iter_blocks()],
                         [(3, 3), (3, 2)])

    def test_placement_does_not_consume_or_mutate_input(self):
        edges = [(0, 1), (1, 0), (2, 2)]
        snapshot = list(edges)
        minimum_dedicated_inputs(edges, 3)
        self.assertEqual(edges, snapshot)
        d = analyze(edges, 3, 3)
        self.assertEqual(d.structural_rank, 3)


if __name__ == "__main__":
    unittest.main()
