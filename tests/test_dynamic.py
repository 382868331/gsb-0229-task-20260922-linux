"""有状态批量更新测试（StructuralModel / UpdateReport）。

运行：python -m unittest discover -s tests -v

期望值来源（不调用被测实现自身生成期望值）：
  * 手算例子的秩、未匹配行列、区域与块形状（注释里给出推导）；
  * tests/reference.py 的回溯枚举最大匹配与独立 DM 区域参考；
  * 解析关系：删边后秩只能下降或不变、加边只能上升或不变、
    空批次结果逐字段相同、同一边集无论更新顺序结果唯一；
  * 与静态 analyze 的逐字段一致仅作*补充*（两条路径共享同一实现），
    每次都同时用枚举参考独立核对秩与三区。
旧测试（test_sparsestruct.py）原样保留，本文件只覆盖新能力。
"""

from __future__ import annotations

import dataclasses
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sparsestruct import (  # noqa: E402
    MAX_COLS,
    MAX_ROWS,
    InvalidStructureError,
    LimitExceededError,
    SearchBudgetExhausted,
    StructuralModel,
    analyze,
    verify,
)

from reference import enumerate_max_matching, reference_regions  # noqa: E402


def _snapshot(model):
    """模型全部对外可见状态（用于“失败后无半更新”断言）。"""
    return (
        model.n_rows,
        model.n_cols,
        model.edges(),
        dataclasses.replace(model.decomposition),
    )


class BatchAddRemoveBasicTests(unittest.TestCase):
    """批量增删的正常路径与账目字段。"""

    def test_construct_matches_static_entry(self):
        edges = [(0, 0), (0, 1), (1, 0)]
        model = StructuralModel(edges, 2, 2)
        # 构造结果与静态入口逐字段一致（共享实现，补充性核对）。
        self.assertEqual(model.decomposition, analyze(edges, 2, 2))
        # 独立枚举核对秩。
        _, _, enum_rank = enumerate_max_matching(edges, 2, 2)
        self.assertEqual(model.decomposition.structural_rank, enum_rank)
        self.assertEqual(len(model), 3)
        self.assertIn((0, 1), model)
        self.assertNotIn((1, 1), model)
        self.assertTrue(verify(model.decomposition))

    def test_default_constructor_starts_empty_with_inferred_dimension(self):
        model = StructuralModel()
        self.assertEqual((model.n_rows, model.n_cols), (0, 0))
        self.assertEqual(model.decomposition.structural_rank, 0)
        rep = model.add_edges([(2, 1)])  # auto_dim：推断扩到 3x2
        self.assertEqual((model.n_rows, model.n_cols), (3, 2))
        self.assertEqual(rep.new_rank, 1)
        self.assertEqual(model.decomposition.unmatched_rows, (0, 1))
        self.assertEqual(model.decomposition.unmatched_cols, (0,))
        self.assertTrue(verify(model.decomposition))

    def test_batch_add_from_empty_with_duplicates(self):
        model = StructuralModel(n_rows=2, n_cols=2)
        self.assertEqual(len(model), 0)
        rep = model.add_edges([(0, 0), (1, 1), (0, 0), (1, 1)])
        # 手算：去重后只有 2 条边，对角矩阵秩 2。
        self.assertEqual(rep.added, ((0, 0), (1, 1)))
        self.assertEqual(rep.added_duplicates, ())
        self.assertEqual(rep.removed, ())
        self.assertEqual(rep.removed_missing, ())
        self.assertTrue(rep.changed)
        self.assertEqual((rep.old_rank, rep.new_rank), (0, 2))
        self.assertEqual(model.edges(), frozenset({(0, 0), (1, 1)}))
        self.assertTrue(verify(rep.decomposition))

    def test_add_existing_edges_counted_as_duplicates(self):
        model = StructuralModel([(0, 0)], 2, 2)
        rep = model.add_edges([(0, 0), (0, 1)])
        self.assertEqual(rep.added, ((0, 1),))
        self.assertEqual(rep.added_duplicates, ((0, 0),))
        self.assertTrue(rep.changed)  # 有一条真正新增
        # 行 1 仍无任何边，两行共列 => 秩仍为 1。
        self.assertEqual(rep.new_rank, 1)
        rep = model.add_edges([(1, 1)])
        self.assertEqual(rep.added, ((1, 1),))
        self.assertEqual(rep.new_rank, 2)  # (0,0),(1,1) 成对匹配

    def test_batch_remove_with_missing(self):
        model = StructuralModel([(0, 0), (0, 1), (1, 0)], 2, 2)
        rep = model.remove_edges([(0, 1), (3, 3), (0, 1)])
        # 只有 (0,1) 存在；(3,3) 缺失；重复删除只计一次。
        self.assertEqual(rep.removed, ((0, 1),))
        self.assertEqual(rep.removed_missing, ((3, 3),))
        self.assertEqual(rep.added, ())
        self.assertTrue(rep.changed)
        self.assertEqual(model.edges(), frozenset({(0, 0), (1, 0)}))
        # 删后剩两条行边共占列 0：秩 1。
        self.assertEqual(rep.new_rank, 1)
        _, _, enum_rank = enumerate_max_matching(
            sorted(model.edges()), 2, 2
        )
        self.assertEqual(rep.new_rank, enum_rank)
        self.assertTrue(verify(model.decomposition))

    def test_combined_add_and_remove_in_one_batch(self):
        model = StructuralModel([(0, 0), (1, 1)], 2, 2)
        rep = model.update(add=[(0, 1)], remove=[(0, 0)])
        self.assertEqual(rep.added, ((0, 1),))
        self.assertEqual(rep.removed, ((0, 0),))
        self.assertEqual(
            model.edges(), frozenset({(0, 1), (1, 1)})
        )  # 两行同争列 1
        self.assertEqual(rep.new_rank, 1)
        self.assertTrue(verify(model.decomposition))

    def test_empty_batch_is_identity(self):
        edges = [(0, 0), (0, 1), (1, 0)]
        model = StructuralModel(edges, 2, 2)
        before = model.decomposition
        rep = model.update()
        self.assertFalse(rep.changed)
        self.assertEqual(rep.added, ())
        self.assertEqual(rep.removed, ())
        self.assertEqual(rep.old_rank, rep.new_rank)
        self.assertEqual(rep.decomposition, before)
        # recompute 空批次同样给出逐字段相同的证书。
        again = model.recompute()
        self.assertEqual(again, before)


class MatchedEdgeDeletionTests(unittest.TestCase):
    """删除匹配边：秩下降与不降两种情形，及三区联动。"""

    def test_delete_only_edge_rank_drops_to_zero(self):
        model = StructuralModel([(0, 0)], 1, 1)
        rep = model.remove_edges([(0, 0)])
        self.assertEqual((rep.old_rank, rep.new_rank), (1, 0))
        self.assertEqual(model.edges(), frozenset())
        self.assertEqual(model.decomposition.unmatched_rows, (0,))
        self.assertEqual(model.decomposition.unmatched_cols, (0,))
        self.assertTrue(verify(model.decomposition))

    def test_delete_matched_diagonal_edge_splits_regions(self):
        # 3x3 对角阵秩 3；删掉中间唯一的匹配边 (1,1)：
        # 行 1、列 1 均孤立 => 行 1 属过定区 O，列 1 属欠定区 U；
        # 行/列 0、2 仍各自良定（两个 1x1 块）。秩 2。
        model = StructuralModel([(0, 0), (1, 1), (2, 2)], 3, 3)
        rep = model.remove_edges([(1, 1)])
        self.assertEqual((rep.old_rank, rep.new_rank), (3, 2))
        d = model.decomposition
        self.assertEqual(d.unmatched_rows, (1,))
        self.assertEqual(d.unmatched_cols, (1,))
        self.assertEqual(d.row_region, ("W", "O", "W"))
        self.assertEqual(d.col_region, ("W", "U", "W"))
        self.assertEqual(d.block_regions, ("U", "W", "W", "O"))
        self.assertEqual(
            [b.shape for b in d.iter_blocks()],
            [(0, 1), (1, 1), (1, 1), (1, 0)],
        )
        self.assertTrue(verify(d))

    def test_delete_matched_edge_rank_can_stay_maximum(self):
        # K2,2 有 2 个完美匹配；删掉 (0,0) 后剩
        # {(0,1),(1,0),(1,1)}，仍有完美匹配（0->1,1->0），秩不降。
        edges = [(0, 0), (0, 1), (1, 0), (1, 1)]
        model = StructuralModel(edges, 2, 2)
        rep = model.remove_edges([(0, 0)])
        self.assertEqual(rep.removed, ((0, 0),))
        self.assertEqual((rep.old_rank, rep.new_rank), (2, 2))
        self.assertEqual(model.edges(), frozenset({(0, 1), (1, 0), (1, 1)}))
        _, _, enum_rank = enumerate_max_matching(
            sorted(model.edges()), 2, 2
        )
        self.assertEqual(rep.new_rank, enum_rank)
        self.assertTrue(verify(model.decomposition))

    def test_remove_then_readd_restores_certificate(self):
        edges = [(0, 0), (0, 1), (1, 0), (1, 1)]
        model = StructuralModel(edges, 2, 2)
        original = model.decomposition
        model.remove_edges([(0, 0)])
        rep = model.add_edges([(0, 0)])
        self.assertEqual(model.edges(), frozenset(edges))
        # 同一边集 => 确定性的同一证书（匹配选择也恢复一致）。
        self.assertEqual(model.decomposition, original)
        self.assertEqual(rep.new_rank, original.structural_rank)

    def test_mixed_regions_repartition_after_matched_deletion(self):
        # 8x8 三区例（O 3行2列 / W 3x3 循环 / U 2行3列，互不相连）。
        edges = [
            (0, 0), (1, 0), (1, 1), (2, 1),
            (3, 2), (3, 3), (4, 3), (4, 4), (5, 4), (5, 2),
            (6, 5), (6, 7), (7, 6), (7, 7),
        ]
        model = StructuralModel(edges, 8, 8)
        self.assertEqual(model.decomposition.structural_rank, 7)
        # 删掉良定循环中的匹配边 (3,2)：循环变路径链，秩仍 3（链仍满秩），
        # 但结构要整体重算；再删 O 区两条边 (0,0),(1,0) 使该区最大匹配
        # 从 2 降为 1（行 0 无边成为孤立 O 行，列 0 仅余行 1 的边），秩 6。
        model.remove_edges([(3, 2)])
        self.assertEqual(model.decomposition.structural_rank, 7)
        self.assertTrue(verify(model.decomposition))
        rep = model.remove_edges([(0, 0), (1, 0)])
        self.assertEqual(rep.new_rank, 6)
        # 独立参考：枚举新边集的最大匹配与 DM 三区逐一比对。
        cur = sorted(model.edges())
        rm, cm, enum_rank = enumerate_max_matching(cur, 8, 8)
        self.assertEqual(rep.new_rank, enum_rank)
        ur, uc, wr, wc, orr, oc = reference_regions(cur, rm, cm, 8, 8)
        d = model.decomposition
        self.assertEqual({r for r, t in enumerate(d.row_region) if t == "U"}, ur)
        self.assertEqual({r for r, t in enumerate(d.row_region) if t == "W"}, wr)
        self.assertEqual({r for r, t in enumerate(d.row_region) if t == "O"}, orr)
        self.assertEqual({c for c, t in enumerate(d.col_region) if t == "U"}, uc)
        self.assertEqual({c for c, t in enumerate(d.col_region) if t == "W"}, wc)
        self.assertEqual({c for c, t in enumerate(d.col_region) if t == "O"}, oc)
        self.assertTrue(verify(d))

    def test_delete_all_edges_keeps_explicit_empty_dimensions(self):
        model = StructuralModel([(0, 0), (1, 1)], 3, 4)
        rep = model.update(remove=[(0, 0), (1, 1)])
        self.assertEqual(rep.new_rank, 0)
        # 显式维度不收缩：空行空列保留。
        self.assertEqual((model.n_rows, model.n_cols), (3, 4))
        d = model.decomposition
        self.assertEqual(d.unmatched_rows, (0, 1, 2))
        self.assertEqual(d.unmatched_cols, (0, 1, 2, 3))
        self.assertEqual(d.block_regions, ("U", "O"))
        self.assertEqual([b.shape for b in d.iter_blocks()], [(0, 4), (3, 0)])
        self.assertTrue(verify(d))


class RectangleAndEmptyLineTests(unittest.TestCase):
    """矩形矩阵、空行空列在动态更新中的保留与重分区。"""

    def test_wide_rectangle_remove_keeps_underdetermined(self):
        # 2x4：两行各占两列，秩 2，整块欠定。
        edges = [(0, 0), (0, 2), (1, 1), (1, 2)]
        model = StructuralModel(edges, 2, 4)
        rep = model.remove_edges([(0, 0)])
        # 剩 (0,2),(1,1),(1,2)：0->2 与 1->1 仍可匹配，秩 2。
        self.assertEqual(rep.new_rank, 2)
        self.assertEqual((model.n_rows, model.n_cols), (2, 4))
        self.assertTrue(verify(model.decomposition))
        rep = model.remove_edges([(1, 1)])
        # 剩两行都只接列 2：秩降为 1，3 个未匹配列。
        self.assertEqual(rep.new_rank, 1)
        self.assertEqual(len(model.decomposition.unmatched_cols), 3)
        self.assertTrue(verify(model.decomposition))

    def test_tall_rectangle_empty_rows_stay_overdetermined(self):
        # 4x2：前两行 K2,2 良定，行 2,3 孤立过定。
        edges = [(0, 0), (0, 1), (1, 0), (1, 1)]
        model = StructuralModel(edges, 4, 2)
        self.assertEqual(
            [b.shape for b in model.decomposition.iter_blocks()],
            [(2, 2), (2, 0)],
        )
        # 删除最后一条涉及行 1 的边不会让维度收缩：仍是 4 行。
        model.update(remove=[(1, 0), (1, 1), (0, 0), (0, 1)])
        self.assertEqual((model.n_rows, model.n_cols), (4, 2))
        self.assertEqual(model.decomposition.structural_rank, 0)
        self.assertTrue(verify(model.decomposition))

    def test_internal_empty_row_and_col_emerge_on_removal(self):
        # 3x4 满（秩 3，列 3 孤立为 U）；删 (1,1),(2,2) 后行 2 变空（O），
        # 列 1,2 变空（U）。
        edges = [(0, 0), (1, 1), (2, 2)]
        model = StructuralModel(edges, 3, 4)
        rep = model.update(remove=[(1, 1), (2, 2)])
        self.assertEqual(rep.new_rank, 1)
        d = model.decomposition
        self.assertEqual(d.unmatched_rows, (1, 2))
        self.assertEqual(set(d.unmatched_cols), {1, 2, 3})
        for r in d.unmatched_rows:
            self.assertEqual(d.row_region[r], "O")
        for c in d.unmatched_cols:
            self.assertEqual(d.col_region[c], "U")
        self.assertTrue(verify(d))

    def test_rectangle_then_fill_to_well_determined(self):
        # 2x2 中只有行 0 接两列：行 0/列0,1 为 U 块(1x2)，空行 1 为 O 块(1x0)。
        model = StructuralModel([(0, 0), (0, 1)], 2, 2)
        self.assertEqual(
            [b.shape for b in model.decomposition.iter_blocks()],
            [(1, 2), (1, 0)],
        )
        # 补 (1,0) 后满秩，整体良定（两个 1x1 块）。
        rep = model.add_edges([(1, 0)])
        self.assertEqual(rep.new_rank, 2)
        self.assertTrue(all(t == "W" for t in model.decomposition.row_region))
        self.assertTrue(all(t == "W" for t in model.decomposition.col_region))
        self.assertTrue(verify(model.decomposition))
        # 再补成 K2,2：单一 2x2 不可约块。
        model.add_edges([(1, 1)])
        wblocks = [b for b in model.decomposition.iter_blocks() if b.region == "W"]
        self.assertEqual(len(wblocks), 1)
        self.assertEqual(wblocks[0].shape, (2, 2))
        self.assertTrue(verify(model.decomposition))


class DimensionTests(unittest.TestCase):
    """维度自动扩展 / 固定边界 / 删除不收缩。"""

    def test_auto_dimension_grows_on_add(self):
        # 推断维度 1x1；加一条远处的边，维度自动扩到 6x6，中间留空行空列。
        model = StructuralModel([(0, 0)])
        self.assertEqual((model.n_rows, model.n_cols), (1, 1))
        rep = model.add_edges([(5, 5)])
        self.assertEqual((model.n_rows, model.n_cols), (6, 6))
        self.assertEqual(rep.new_rank, 2)
        d = model.decomposition
        # 行 1..4、列 1..4 均孤立：O 行、U 列。
        self.assertEqual(len(d.unmatched_rows), 4)
        self.assertEqual(len(d.unmatched_cols), 4)
        self.assertTrue(verify(d))

    def test_removal_never_shrinks_dimension(self):
        model = StructuralModel([(5, 5)], 6, 6)
        model.remove_edges([(5, 5)])
        self.assertEqual((model.n_rows, model.n_cols), (6, 6))

    def test_fixed_dimension_rejects_out_of_range_add(self):
        model = StructuralModel([(0, 0)], 2, 2, auto_dim=False)
        with self.assertRaises(InvalidStructureError) as cm:
            model.add_edges([(2, 0)])
        self.assertIn("(2, 0)", str(cm.exception))
        self.assertEqual(model.edges(), frozenset({(0, 0)}))
        self.assertEqual((model.n_rows, model.n_cols), (2, 2))
        with self.assertRaises(InvalidStructureError):
            model.update(add=[(0, 2)])
        # 边界上的合法添加成功。
        model.add_edges([(1, 1)])
        self.assertEqual(model.decomposition.structural_rank, 2)

    def test_hard_limit_enforced_on_add(self):
        model = StructuralModel([], MAX_ROWS, MAX_COLS, auto_dim=False)
        with self.assertRaises(LimitExceededError):
            model.add_edges([(MAX_ROWS, 0)])
        with self.assertRaises(LimitExceededError):
            model.add_edges([(0, MAX_COLS)])
        model.add_edges([(MAX_ROWS - 1, MAX_COLS - 1)])
        self.assertEqual(model.decomposition.structural_rank, 1)

    def test_auto_dim_false_requires_bool(self):
        with self.assertRaises(TypeError):
            StructuralModel([], 1, 1, auto_dim=1)  # type: ignore[arg-type]


class DeterminismAndReferenceTests(unittest.TestCase):
    """随机小例的批量更新序列：枚举参考 + 确定性 + 静态入口一致性。"""

    @staticmethod
    def _lcg(seed):
        def next_int():
            nonlocal seed
            seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
            return seed
        return next_int

    def test_random_update_walk_against_enumeration(self):
        n, m = 4, 5
        rng = self._lcg(20260924)
        model = StructuralModel([], n, m, auto_dim=False)
        truth: set[tuple[int, int]] = set()
        for step in range(160):
            mode = rng() % 3
            add_batch = []
            rem_batch = []
            if mode in (0, 2):
                add_batch = [(rng() % n, rng() % m) for _ in range(rng() % 4)]
            if mode in (1, 2):
                present = sorted(truth) or [(0, 0)]
                rem_batch = [
                    present[rng() % len(present)] for _ in range(rng() % 3)
                ]
                rem_batch.append((rng() % n, rng() % m))  # 可能缺失的删除
            # 同位置既增又删会被拒绝；为保持随机游走合法，剔除冲突。
            rem_batch = [p for p in rem_batch if p not in set(add_batch)]
            rep = model.update(add=add_batch, remove=rem_batch)
            truth = (truth | set(add_batch)) - set(rem_batch)
            with self.subTest(step=step):
                # 1) 模型边集必须与独立维护的真值一致；
                self.assertEqual(model.edges(), frozenset(truth))
                # 2) 秩与枚举参考一致（不使用被测实现造期望值）；
                rm, cm, enum_rank = enumerate_max_matching(sorted(truth), n, m)
                d = model.decomposition
                self.assertEqual(d.structural_rank, enum_rank)
                # 3) DM 三区与独立参考一致；
                ur, uc, wr, wc, orr, oc = reference_regions(
                    sorted(truth), rm, cm, n, m
                )
                self.assertEqual(
                    {r for r, t in enumerate(d.row_region) if t == "U"}, ur
                )
                self.assertEqual(
                    {r for r, t in enumerate(d.row_region) if t == "W"}, wr
                )
                self.assertEqual(
                    {r for r, t in enumerate(d.row_region) if t == "O"}, orr
                )
                self.assertEqual(
                    {c for c, t in enumerate(d.col_region) if t == "U"}, uc
                )
                self.assertEqual(
                    {c for c, t in enumerate(d.col_region) if t == "W"}, wc
                )
                self.assertEqual(
                    {c for c, t in enumerate(d.col_region) if t == "O"}, oc
                )
                # 4) 证书完整可核验；
                self.assertTrue(verify(d))
                # 5) 与静态入口逐字段一致（共享实现的补充核对）。
                self.assertEqual(d, analyze(sorted(truth), n, m))
                # 6) 秩的单调性账目：本次增/删方向与秩变化相容。
                if rep.added and not rep.removed:
                    self.assertGreaterEqual(rep.new_rank, rep.old_rank)
                if rep.removed and not rep.added:
                    self.assertLessEqual(rep.new_rank, rep.old_rank)

    def test_order_independent_batches_give_same_result(self):
        # 两个模型以不同批次顺序、不同重复率达到同一边集。
        target = [
            (0, 0), (0, 2), (1, 1), (1, 2), (2, 0), (3, 3), (3, 4)
        ]
        m1 = StructuralModel([], 4, 5, auto_dim=False)
        m1.add_edges(target[:3])
        m1.add_edges(target[3:])
        m1.add_edges([target[0], target[-1]])  # 纯重复批次

        m2 = StructuralModel([], 4, 5, auto_dim=False)
        m2.add_edges(list(reversed(target)))
        m2.remove_edges([(2, 0)])
        m2.add_edges([(2, 0)])

        self.assertEqual(m1.edges(), m2.edges())
        self.assertEqual(m1.decomposition, m2.decomposition)
        self.assertEqual(m1.decomposition, analyze(target, 4, 5))
        self.assertTrue(verify(m2.decomposition))


class IllegalUpdateTests(unittest.TestCase):
    """非法更新：明确错误 + 不留半更新状态。"""

    def _assert_state_untouched(self, model, snapshot_before, callable_, *args):
        with self.assertRaises(Exception):
            callable_(*args)
        self.assertEqual(_snapshot(model), snapshot_before)

    def test_illegal_add_inputs_leave_no_trace(self):
        model = StructuralModel([(0, 0)], 2, 2)
        bad_batches = [
            ([(-1, 0)], InvalidStructureError),
            ([(0, float("nan"))], InvalidStructureError),
            ([(float("inf"), 0)], InvalidStructureError),
            ([(True, 0)], TypeError),
            ([(1.0, 0)], TypeError),
            ([(0, 0, 0)], InvalidStructureError),
            (["ab"], TypeError),
        ]
        for batch, exc in bad_batches:
            with self.subTest(batch=batch):
                snap = _snapshot(model)
                with self.assertRaises(exc):
                    model.add_edges(batch)
                self.assertEqual(_snapshot(model), snap)

    def test_illegal_remove_localizable_and_atomic(self):
        model = StructuralModel([(0, 0), (1, 1)], 3, 3)
        snap = _snapshot(model)
        with self.assertRaises(InvalidStructureError) as cm:
            model.remove_edges([(0, 0), (-2, 1)])
        self.assertIn("remove[1]", str(cm.exception))
        # (0,0) 已通过校验也不能被删掉：整批回滚。
        self.assertEqual(_snapshot(model), snap)

        with self.assertRaises(LimitExceededError) as cm2:
            model.remove_edges([(MAX_ROWS, 0)])
        self.assertIn("remove[0]", str(cm2.exception))
        self.assertEqual(_snapshot(model), snap)

    def test_same_edge_added_and_removed_is_conflict(self):
        model = StructuralModel([(0, 0)], 2, 2)
        snap = _snapshot(model)
        with self.assertRaises(InvalidStructureError) as cm:
            model.update(add=[(1, 1), (0, 0)], remove=[(0, 0)])
        self.assertIn("(0, 0)", str(cm.exception))
        self.assertIn("既被添加又被删除", str(cm.exception))
        self.assertEqual(_snapshot(model), snap)

    def test_non_iterable_batch_rejected(self):
        model = StructuralModel([(0, 0)], 2, 2)
        snap = _snapshot(model)
        with self.assertRaises(TypeError):
            model.update(add=42)
        with self.assertRaises(TypeError):
            model.update(remove=None)
        self.assertEqual(_snapshot(model), snap)

    def test_budget_exhaustion_is_atomic(self):
        dense = [(r, c) for r in range(4) for c in range(4)]
        model = StructuralModel(dense, 4, 4)
        snap = _snapshot(model)
        with self.assertRaises(SearchBudgetExhausted):
            model.add_edges([(0, 0)], budget=0)  # 重复添加，但重算仍耗尽预算
        self.assertEqual(_snapshot(model), snap)
        # 边集里没有 (2,2) 之外的变化；预算给足后正常更新。
        rep = model.add_edges([], budget=None)
        self.assertEqual(rep.new_rank, 4)
        self.assertEqual(model.edges(), frozenset(dense))

    def test_empty_batch_with_zero_budget_raises_but_keeps_state(self):
        # 全量重算语义：即使边集不变，budget=0 在非空矩阵上仍会耗尽，
        # 旧证书保留。
        model = StructuralModel([(0, 0)], 1, 1)
        snap = _snapshot(model)
        with self.assertRaises(SearchBudgetExhausted):
            model.recompute(budget=0)
        self.assertEqual(_snapshot(model), snap)

    def test_set_budget_validates(self):
        model = StructuralModel([(0, 0)], 1, 1)
        with self.assertRaises(InvalidStructureError):
            model.set_budget(-1)
        with self.assertRaises(TypeError):
            model.set_budget(1.5)
        model.set_budget(None)
        self.assertIsNone(model.budget)
        model.set_budget(7)
        self.assertEqual(model.budget, 7)

    def test_constructor_rejects_invalid_initial_input(self):
        with self.assertRaises(InvalidStructureError):
            StructuralModel([(-1, 0)])
        with self.assertRaises(LimitExceededError):
            StructuralModel([], MAX_ROWS + 1, 1)
        with self.assertRaises(InvalidStructureError):
            StructuralModel([(0, 0)], budget=-1)

    def test_generator_batch_is_materialized_before_commit(self):
        model = StructuralModel([(0, 0)], 2, 2)
        snap = _snapshot(model)

        def bad_gen():
            yield (1, 1)
            raise RuntimeError("生成器中途失败")

        with self.assertRaises(RuntimeError):
            model.add_edges(bad_gen())
        # 生成器里已产出的 (1,1) 也不得入库：提交发生在全部物化之后。
        self.assertEqual(_snapshot(model), snap)


class SnapshotImmutabilityTests(unittest.TestCase):
    """对外视图不可被调用方污染。"""

    def test_edges_snapshot_is_frozenset(self):
        model = StructuralModel([(0, 0)], 2, 2)
        snap = model.edges()
        self.assertIsInstance(snap, frozenset)
        model.add_edges([(1, 1)])
        # 旧快照不随后续更新变化。
        self.assertEqual(snap, frozenset({(0, 0)}))

    def test_report_decomposition_is_current_after_later_updates(self):
        model = StructuralModel([(0, 0)], 2, 2)
        rep1 = model.add_edges([(1, 1)])
        rep2 = model.remove_edges([(0, 0)])
        # 历史报告是不可变快照；rep1 的秩不被后续更新改写。
        self.assertEqual(rep1.new_rank, 2)
        self.assertEqual(rep2.old_rank, 2)
        self.assertEqual(rep2.new_rank, 1)
        self.assertEqual(rep1.decomposition.structural_rank, 2)
        self.assertTrue(verify(rep1.decomposition))


if __name__ == "__main__":
    unittest.main(verbosity=2)
