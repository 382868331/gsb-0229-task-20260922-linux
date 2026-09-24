"""批量增删入口（SparsePattern）测试。

覆盖：
  * 正常：批量加/删/组合更新、删除匹配边后最大匹配与 DM 三区重算；
  * 全量重算一致性：每一步都与静态入口 analyze() 逐字段一致，
    结构秩/区域用 tests/reference.py 的回溯枚举独立核对；
  * 边界：空批次、删光（空行空列）、矩形矩阵（0×k / k×0 / 2×3）、
    推断维度扩展、显式维度越界、去重台账；
  * 失败：负下标 / NaN / bool / 错长度 / 非可迭代 / 超 100 上限，
    全部原子失败，不留半更新状态；
  * resize / recompute / 容器协议 / 确定性。

运行：python -m unittest discover -s tests -v
独立参考见 tests/reference.py（不导入被测核心）。
"""

from __future__ import annotations

import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sparsestruct import (  # noqa: E402
    MAX_COLS,
    MAX_ROWS,
    InvalidStructureError,
    LimitExceededError,
    SparsePattern,
    UpdateReport,
    analyze,
    verify,
)

from reference import enumerate_max_matching, reference_regions  # noqa: E402


def _snapshot(p):
    """可比较的完整状态快照（边集、维度、证书）。"""
    return sorted(p.nonzeros), p.n_rows, p.n_cols, p.result


def _assert_matches_reference(test, p, edges, n, m):
    """与独立枚举参考 + 静态入口 + verify 三方核对当前状态。"""
    d = p.result
    test.assertEqual((d.n_rows, d.n_cols), (n, m))
    static = analyze(edges, n, m)
    test.assertEqual(d, static)  # 全量重算结果必须与静态入口逐字段一致
    test.assertTrue(verify(d))
    rm, cm, enum_rank = enumerate_max_matching(edges, n, m)
    test.assertEqual(d.structural_rank, enum_rank)
    test.assertEqual(d.structural_rank, sum(1 for c in d.row_match if c != -1))
    ur, uc, wr, wc, orr, oc = reference_regions(
        list(set(edges)), rm, cm, n, m
    )
    test.assertEqual({r for r, t in enumerate(d.row_region) if t == "U"}, ur)
    test.assertEqual({r for r, t in enumerate(d.row_region) if t == "W"}, wr)
    test.assertEqual({r for r, t in enumerate(d.row_region) if t == "O"}, orr)
    test.assertEqual({c for c, t in enumerate(d.col_region) if t == "U"}, uc)
    test.assertEqual({c for c, t in enumerate(d.col_region) if t == "W"}, wc)
    test.assertEqual({c for c, t in enumerate(d.col_region) if t == "O"}, oc)


class BatchConsistencyTests(unittest.TestCase):
    """增删序列与独立枚举参考、静态入口逐拍核对。"""

    def test_hand_sequence_rank_region_and_blocks(self):
        # 手算序列（2x2 子图 + 第三行/列）：
        #   起步路径图 r0-c0,r0-c1,r1-c0：秩 2
        #   加 (1,1)：K2,2，秩仍 2（新增边不增秩）
        #   删匹配边 (0,0)：剩余 r0-c1,r1-c0,r1-c1，仍有完美匹配，秩 2
        #   再删 (0,1)：只剩 r1-c0,r1-c1，秩 1，行 0 过定、列之一欠定
        p = SparsePattern([(0, 0), (0, 1), (1, 0)], 2, 2)
        edges = [(0, 0), (0, 1), (1, 0)]
        _assert_matches_reference(self, p, edges, 2, 2)
        self.assertEqual(p.result.structural_rank, 2)

        rep = p.add([(1, 1)])
        edges.append((1, 1))
        self.assertEqual((rep.added, rep.removed), (((1, 1),), ()))
        self.assertEqual(rep.rank_before, rep.rank_after)  # 2 -> 2
        _assert_matches_reference(self, p, edges, 2, 2)

        rep = p.remove([(0, 0)])  # 删除当前匹配边：存在交替匹配，秩不变
        edges.remove((0, 0))
        self.assertEqual(rep.removed, ((0, 0),))
        self.assertEqual(rep.rank_after, 2)
        _assert_matches_reference(self, p, edges, 2, 2)

        rep = p.remove([(0, 1)])  # 再删一条：秩降到 1
        edges.remove((0, 1))
        self.assertEqual(rep.rank_before, 2)
        self.assertEqual(rep.rank_after, 1)
        _assert_matches_reference(self, p, edges, 2, 2)
        # 行 0 全空 => 过定；列中必有一个未匹配 => 欠定
        self.assertEqual(p.result.row_region[0], "O")
        self.assertIn("U", "".join(p.result.col_region))

    def test_random_add_remove_sequences(self):
        # 确定性 LCG 随机小例：每拍随机加/删若干边，逐拍三方核对。
        seed = 20260924

        def lcg():
            nonlocal seed
            seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
            return seed

        for trial in range(60):
            n, m = 1 + lcg() % 4, 1 + lcg() % 4
            present = set()
            p = SparsePattern([], n, m)
            for _ in range(12):
                if lcg() % 2 == 0:
                    batch = {
                        (lcg() % n, lcg() % m) for _ in range(1 + lcg() % 3)
                    }
                    p.add(batch)
                    present |= batch
                else:
                    cand = [
                        (lcg() % n, lcg() % m) for _ in range(1 + lcg() % 3)
                    ]
                    p.remove(cand)
                    present -= set(cand)
                with self.subTest(trial=trial, step=_):
                    _assert_matches_reference(
                        self, p, sorted(present), n, m
                    )

    def test_mixed_update_matches_static(self):
        p = SparsePattern([(0, 0), (1, 1), (2, 2)], 4, 4)
        rep = p.update(
            add=[(0, 1), (1, 0), (3, 3)],
            remove=[(0, 0), (1, 1), (2, 2)],
        )
        final = [(0, 1), (1, 0), (3, 3)]
        self.assertEqual(p.result, analyze(final, 4, 4))
        self.assertTrue(verify(p.result))
        self.assertEqual(rep.rank_before, 3)
        self.assertEqual(rep.rank_after, 3)
        self.assertEqual(set(rep.added), {(0, 1), (1, 0), (3, 3)})
        self.assertEqual(set(rep.removed), {(0, 0), (1, 1), (2, 2)})
        self.assertEqual(rep.n_nonzeros, 3)
        self.assertTrue(rep.rebuilt)
        self.assertIsInstance(rep, UpdateReport)

    def test_update_same_position_add_and_remove_keeps_it(self):
        p = SparsePattern([(0, 0)], 2, 2)
        # (0,0) 同时在两批：先删后增，最终保留，台账记 unchanged
        rep = p.update(add=[(0, 0), (1, 1)], remove=[(0, 0)])
        self.assertEqual(set(p.nonzeros), {(0, 0), (1, 1)})
        self.assertNotIn((0, 0), rep.removed)
        self.assertIn((0, 0), rep.unchanged)
        self.assertEqual(rep.added, ((1, 1),))
        self.assertEqual(p.result, analyze([(0, 0), (1, 1)], 2, 2))


class RemoveMatchedEdgeTests(unittest.TestCase):
    """删除匹配边：匹配重算、秩与区域变化。"""

    def test_remove_matched_edge_rank_drops_identity(self):
        # 3x3 对角：删 (1,1) => 秩 2；行 1 过定、列 1 欠定，其余良定。
        p = SparsePattern([(0, 0), (1, 1), (2, 2)], 3, 3)
        rep = p.remove([(1, 1)])
        self.assertEqual(rep.rank_before, 3)
        self.assertEqual(rep.rank_after, 2)
        self.assertEqual(p.result.unmatched_rows, (1,))
        self.assertEqual(p.result.unmatched_cols, (1,))
        self.assertEqual(p.result.row_region[1], "O")
        self.assertEqual(p.result.col_region[1], "U")
        self.assertEqual(p.result.row_region[0], "W")
        self.assertEqual(p.result.row_region[2], "W")
        self.assertEqual(p.result, analyze([(0, 0), (2, 2)], 3, 3))
        self.assertTrue(verify(p.result))
        shapes = [b.shape for b in p.result.iter_blocks()]
        # U(0x1) | W(1x1) | W(1x1) | O(1x0)
        self.assertEqual(shapes, [(0, 1), (1, 1), (1, 1), (1, 0)])

    def test_remove_matched_edge_rank_survives_via_alternating(self):
        # K2,2：Kuhn 增广后匹配为 (0,1),(1,0)（r0 先占 c0，r1 增广交换）。
        # 删除 r0 的匹配边 (0,1) 后，剩余 (0,0),(1,0),(1,1) 仍有完美匹配，
        # 秩保持 2——删除匹配边不必然降秩。
        p = SparsePattern(
            [(0, 0), (0, 1), (1, 0), (1, 1)], 2, 2
        )
        self.assertEqual(p.result.row_match, (1, 0))
        rep = p.remove([(0, 1)])  # 删除当前匹配边
        self.assertEqual(rep.removed, ((0, 1),))
        self.assertEqual(rep.rank_before, 2)
        self.assertEqual(rep.rank_after, 2)
        self.assertEqual(p.result, analyze([(0, 0), (1, 0), (1, 1)], 2, 2))
        self.assertTrue(verify(p.result))

    def test_remove_nonexistent_is_unchanged_and_no_rank_change(self):
        p = SparsePattern([(0, 0)], 3, 3)
        rep = p.remove([(2, 2), (2, 2)])
        self.assertEqual(rep.removed, ())
        self.assertEqual(rep.unchanged, ((2, 2),))  # 重复也只记一次
        self.assertEqual(rep.rank_before, rep.rank_after)
        self.assertEqual(set(p.nonzeros), {(0, 0)})

    def test_remove_all_edges_yields_empty_regions(self):
        p = SparsePattern([(0, 0), (0, 1), (1, 0)], 3, 3)
        p.remove([(0, 0), (0, 1), (1, 0)])
        self.assertEqual(p.result.structural_rank, 0)
        self.assertEqual(p.result, analyze([], 3, 3))
        self.assertEqual(p.result.block_regions, ("U", "O"))
        blocks = p.result.iter_blocks()
        self.assertEqual(blocks[0].shape, (0, 3))
        self.assertEqual(blocks[1].shape, (3, 0))
        self.assertTrue(verify(p.result))


class AddAndDeduplicationTests(unittest.TestCase):
    def test_report_added_vs_unchanged(self):
        p = SparsePattern([(0, 0), (1, 1)], 3, 3)
        rep = p.add([(0, 0), (2, 2), (2, 2)])
        self.assertEqual(rep.added, ((2, 2),))
        self.assertEqual(rep.unchanged, ((0, 0),))
        self.assertEqual(rep.removed, ())
        self.assertEqual((rep.rank_before, rep.rank_after), (2, 3))
        self.assertEqual(rep.n_nonzeros, 3)
        self.assertTrue(rep.rebuilt)

    def test_add_edges_merge_irreducible_block(self):
        # 对角 3x3 是三个 1x1 块；加循环边后合并为一个 3x3 不可约块。
        p = SparsePattern([(0, 0), (1, 1), (2, 2)], 3, 3)
        self.assertEqual(
            sum(1 for b in p.result.iter_blocks() if b.region == "W"), 3
        )
        p.add([(0, 1), (1, 2), (2, 0)])
        wblocks = [b for b in p.result.iter_blocks() if b.region == "W"]
        self.assertEqual(len(wblocks), 1)
        self.assertEqual(wblocks[0].shape, (3, 3))
        self.assertEqual(p.result.structural_rank, 3)
        self.assertTrue(verify(p.result))

    def test_inferred_dimensions_grow_on_add(self):
        p = SparsePattern([(0, 0)])  # 推断 1x1
        self.assertEqual((p.n_rows, p.n_cols), (1, 1))
        p.add([(3, 2)])
        self.assertEqual((p.n_rows, p.n_cols), (4, 3))
        self.assertEqual(
            p.result, analyze([(0, 0), (3, 2)], 4, 3)
        )

    def test_inferred_dimensions_do_not_shrink_after_remove(self):
        p = SparsePattern([(3, 3)])
        p.remove([(3, 3)])
        self.assertEqual((p.n_rows, p.n_cols), (4, 4))  # 维度只增不减
        self.assertEqual(p.result, analyze([], 4, 4))

    def test_empty_batches_are_noop_reports(self):
        p = SparsePattern([(0, 0)], 2, 2)
        for rep in (p.add(()), p.remove([]), p.update()):
            self.assertEqual(rep.added, ())
            self.assertEqual(rep.removed, ())
            self.assertEqual(rep.unchanged, ())
            self.assertEqual(rep.rank_before, rep.rank_after)
            self.assertEqual(rep.n_nonzeros, 1)
            self.assertTrue(rep.rebuilt)
        self.assertEqual(p.result, analyze([(0, 0)], 2, 2))

    def test_generators_accepted_and_consumed(self):
        p = SparsePattern()
        rep = p.add((r, r) for r in range(3))
        self.assertEqual(rep.rank_after, 3)
        rep = p.remove((r, r) for r in (0, 2))
        self.assertEqual(set(p.nonzeros), {(1, 1)})


class RectangularPatternTests(unittest.TestCase):
    """矩形矩阵上的批量增删。"""

    def test_wide_rectangle_add_remove(self):
        # 2x3：秩 2，整块欠定。
        edges = [(0, 0), (0, 2), (1, 1), (1, 2)]
        p = SparsePattern(edges, 2, 3)
        self.assertEqual(p.result.block_regions, ("U",))
        p.remove([(1, 2)])  # 剩余 (0,0),(0,2),(1,1)：仍有完美匹配，秩 2
        self.assertEqual(p.result.structural_rank, 2)
        p.remove([(0, 2)])  # 剩余 (0,0),(1,1)：仍是完美匹配，秩 2
        self.assertEqual(p.result.structural_rank, 2)
        self.assertEqual(p.result, analyze([(0, 0), (1, 1)], 2, 3))
        # 再删一条匹配边 => 秩降到 1
        rep = p.remove([(0, 0)])
        self.assertEqual(rep.rank_after, 1)
        p.add([(0, 0), (0, 2), (1, 2)])
        self.assertEqual(p.result, analyze(edges, 2, 3))
        self.assertTrue(verify(p.result))

    def test_zero_rows_add_is_out_of_bounds(self):
        p = SparsePattern([], 0, 3)
        self.assertEqual(p.result.block_regions, ("U",))
        with self.assertRaises(InvalidStructureError):
            p.add([(0, 0)])
        self.assertEqual(_snapshot(p)[0:3], ([], 0, 3))

    def test_zero_cols_add_is_out_of_bounds(self):
        p = SparsePattern([], 3, 0)
        self.assertEqual(p.result.block_regions, ("O",))
        with self.assertRaises(InvalidStructureError):
            p.add([(0, 0)])
        self.assertEqual(_snapshot(p)[0:3], ([], 3, 0))

    def test_explicit_dims_reject_out_of_bounds_add_atomically(self):
        p = SparsePattern([(0, 0)], 2, 2)
        snap = _snapshot(p)
        with self.assertRaises(InvalidStructureError):
            p.add([(1, 1), (2, 0)])  # 第二条越界，第一条也不得生效
        self.assertEqual(_snapshot(p), snap)
        with self.assertRaises(InvalidStructureError):
            p.update(add=[(0, 1), (0, 2)])  # 列 2 越界
        self.assertEqual(_snapshot(p), snap)

    def test_removing_in_rectangle_leaves_empty_row_and_col(self):
        # 3x4，只有列 0,1 有非零元：结构秩上界为 2。
        # 边 (0,0),(0,1),(1,1),(2,0) 的最大匹配秩为 2（三行争两列，
        # 一个未匹配行）；删 (2,0) 后行 2 全空成为未匹配行（O），
        # 列 2,3 本就孤立（U），秩仍为 2。
        edges = [(0, 0), (0, 1), (1, 1), (2, 0)]
        p = SparsePattern(edges, 3, 4)
        self.assertEqual(p.result.structural_rank, 2)
        rep = p.remove([(2, 0)])
        self.assertEqual(rep.rank_before, rep.rank_after)
        self.assertEqual(p.result.structural_rank, 2)
        self.assertEqual(p.result.unmatched_rows, (2,))
        self.assertEqual(p.result.row_region[2], "O")
        self.assertEqual(p.result.col_region[3], "U")
        self.assertEqual(p.result, analyze([(0, 0), (0, 1), (1, 1)], 3, 4))
        self.assertTrue(verify(p.result))


class InvalidUpdateAtomicityTests(unittest.TestCase):
    """非法更新：明确错误类型 + 原子性（无半更新）。"""

    def setUp(self):
        self.p = SparsePattern(
            [(0, 0), (0, 1), (1, 1), (2, 2)], 4, 4
        )

    def _assert_unchanged(self, snap):
        self.assertEqual(_snapshot(self.p), snap)
        self.assertTrue(verify(self.p.result))

    def test_negative_index_atomic(self):
        snap = _snapshot(self.p)
        with self.assertRaises(InvalidStructureError) as cm:
            self.p.add([(0, 2), (-1, 0)])
        self.assertIn("add 批次[1]", str(cm.exception))
        self._assert_unchanged(snap)
        with self.assertRaises(InvalidStructureError) as cm:
            self.p.remove([(0, 0), (0, -3)])
        self.assertIn("remove 批次[1]", str(cm.exception))
        self._assert_unchanged(snap)

    def test_nan_inf_atomic(self):
        snap = _snapshot(self.p)
        with self.assertRaises(InvalidStructureError):
            self.p.add([(0, float("nan"))])
        with self.assertRaises(InvalidStructureError):
            self.p.update(remove=[(float("inf"), 0)])
        self._assert_unchanged(snap)

    def test_bool_and_float_rejected_atomic(self):
        snap = _snapshot(self.p)
        with self.assertRaises(TypeError):
            self.p.add([(True, 0)])
        with self.assertRaises(TypeError):
            self.p.remove([(0, 1.0)])
        with self.assertRaises(TypeError):
            self.p.update(add=[False])
        self._assert_unchanged(snap)

    def test_bad_length_and_type_atomic(self):
        snap = _snapshot(self.p)
        with self.assertRaises(InvalidStructureError):
            self.p.add([(0, 0, 0)])
        with self.assertRaises(InvalidStructureError):
            self.p.remove([(1,)])
        with self.assertRaises(TypeError):
            self.p.add("(0,0)")
        with self.assertRaises(TypeError):
            self.p.add(None)
        with self.assertRaises(TypeError):
            self.p.remove(42)
        with self.assertRaises(TypeError):
            self.p.update(add=[b"xy"])
        self._assert_unchanged(snap)

    def test_limit_exceeded_atomic(self):
        snap = _snapshot(self.p)
        with self.assertRaises(LimitExceededError):
            self.p.add([(MAX_ROWS, 0)])
        with self.assertRaises(LimitExceededError):
            SparsePattern([(0, MAX_COLS)])
        self._assert_unchanged(snap)

    def test_input_container_not_mutated(self):
        good = [(0, 2), (3, 3)]
        good_copy = copy.deepcopy(good)
        self.p.add(good)
        self.assertEqual(good, good_copy)  # 调用方列表不被修改
        bad = [(0, 0), (-1, 0)]
        bad_copy = copy.deepcopy(bad)
        with self.assertRaises(InvalidStructureError):
            self.p.add(bad)
        self.assertEqual(bad, bad_copy)

    def test_failure_early_in_batch_nothing_applied(self):
        snap = _snapshot(self.p)
        # 批次第一万条合法、最后一条非法：前面的也不得生效。
        batch = [(r % 4, c % 4) for r in range(4) for c in range(4)] + [(0, -9)]
        with self.assertRaises(InvalidStructureError):
            self.p.add(batch)
        self._assert_unchanged(snap)

    def test_constructor_invalid_leaves_nothing(self):
        with self.assertRaises(InvalidStructureError):
            SparsePattern([(0, 0), (1, -1)])
        with self.assertRaises(TypeError):
            SparsePattern([(True, 0)])
        with self.assertRaises(LimitExceededError):
            SparsePattern([], MAX_ROWS + 1, 2)


class ResizeAndRecomputeTests(unittest.TestCase):
    def test_recompute_equals_current_result(self):
        p = SparsePattern([(r, r) for r in range(3)], 3, 3)
        p.add([(0, 1)])
        d = p.recompute()
        self.assertEqual(d, p.result)
        self.assertEqual(d, analyze([(0, 0), (0, 1), (1, 1), (2, 2)], 3, 3))

    def test_resize_grow(self):
        p = SparsePattern([(0, 0)], 2, 2)
        p.resize(4, 5)
        self.assertEqual((p.n_rows, p.n_cols), (4, 5))
        self.assertEqual(p.result, analyze([(0, 0)], 4, 5))
        p.resize(n_rows=5)
        self.assertEqual((p.n_rows, p.n_cols), (5, 5))
        self.assertEqual(p.result, analyze([(0, 0)], 5, 5))

    def test_resize_shrink_rejected_atomically(self):
        p = SparsePattern([(0, 0), (2, 2)], 3, 3)
        snap = _snapshot(p)
        with self.assertRaises(InvalidStructureError):
            p.resize(2, 3)
        with self.assertRaises(InvalidStructureError):
            p.resize(3, 2)
        self.assertEqual(_snapshot(p), snap)

    def test_resize_validation(self):
        p = SparsePattern([(0, 0)])
        with self.assertRaises(InvalidStructureError):
            p.resize(-1, 2)
        with self.assertRaises(TypeError):
            p.resize(True, 2)
        with self.assertRaises(LimitExceededError):
            p.resize(MAX_ROWS + 1)
        with self.assertRaises(TypeError):
            p.resize(1.5)

    def test_shrink_to_zero_after_clearing(self):
        # 推断维度扩展后不自动收缩；清空后可用 resize 显式收到 0x0。
        p = SparsePattern([(3, 3)])
        self.assertEqual((p.n_rows, p.n_cols), (4, 4))
        p.remove([(3, 3)])
        p.resize(0, 0)
        self.assertEqual((p.n_rows, p.n_cols), (0, 0))
        self.assertEqual(p.result, analyze([], 0, 0))
        self.assertTrue(verify(p.result))

    def test_mixed_explicit_dimension(self):
        # 只固定列数：行随下标增长，列越界拒绝。
        p = SparsePattern([(0, 0)], None, 2)
        p.add([(4, 1)])
        self.assertEqual((p.n_rows, p.n_cols), (5, 2))
        snap = _snapshot(p)
        with self.assertRaises(InvalidStructureError):
            p.add([(0, 2)])
        self.assertEqual(_snapshot(p), snap)
        self.assertEqual(p.result, analyze([(0, 0), (4, 1)], 5, 2))


class ContainerProtocolAndDeterminismTests(unittest.TestCase):
    def test_container_protocol(self):
        p = SparsePattern([(0, 0), (1, 2)], 3, 4)
        self.assertEqual(len(p), 2)
        self.assertTrue((0, 0) in p)
        self.assertFalse((0, 1) in p)
        self.assertEqual(list(p), [(0, 0), (1, 2)])
        self.assertEqual(sorted(p.nonzeros), [(0, 0), (1, 2)])
        self.assertTrue(p.contains((1, 2)))
        self.assertFalse(p.contains((2, 2)))
        # 非法 in 判定按容器协议返回 False（不抛异常）
        self.assertFalse((100, 0) in p)
        # 显式方法仍严格报错
        with self.assertRaises(InvalidStructureError):
            p.contains((-1, 0))

    def test_nonzeros_is_a_snapshot(self):
        p = SparsePattern([(0, 0)])
        snap = p.nonzeros
        self.assertIsInstance(snap, frozenset)
        p.add([(1, 1)])
        self.assertEqual(snap, frozenset({(0, 0)}))  # 旧快照不被污染

    def test_determinism_across_insertion_orders(self):
        edges = [(r, c) for r in range(5) for c in range(5) if (r + c) % 2 == 0]
        p1 = SparsePattern()
        for e in edges:
            p1.add([e])
        p2 = SparsePattern()
        for e in reversed(edges):
            p2.add([e])
        self.assertEqual(p1.result, p2.result)
        self.assertEqual(p1.result, analyze(edges, 5, 5))
        # 同样的删除序列后仍一致
        p1.remove([(0, 0), (2, 2)])
        p2.remove([(2, 2), (0, 0)])
        self.assertEqual(p1.result, p2.result)
        self.assertTrue(verify(p1.result))


if __name__ == "__main__":
    unittest.main(verbosity=2)
