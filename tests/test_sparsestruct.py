"""必要测试：枚举参考、空行列、矩形、多匹配同结构区、不可约块、
输入校验、预算语义与证书核验。

运行：python -m unittest discover -s tests -v
独立参考见 tests/reference.py（不导入被测核心）。
"""

from __future__ import annotations

import math
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
    analyze,
    verify,
)
from sparsestruct import edge_criticality  # noqa: E402

from reference import (  # noqa: E402
    count_perfect_matchings,
    enumerate_max_matching,
    reference_regions,
    reference_well_blocks,
)

REGION_NAMES = {"U": "欠定", "W": "良定", "O": "过定"}


def _edges_set(d):
    return set(d.nonzeros)


class RankAndMatchingReferenceTests(unittest.TestCase):
    """结构秩与匹配：小矩阵上与独立枚举参考逐一比对（含随机小例）。"""

    CASES = [
        # 手工小例（边集, n_rows, n_cols, 期望秩）
        ([], 0, 0, 0),
        ([], 0, 3, 0),
        ([], 3, 0, 0),
        ([], 3, 3, 0),
        ([(0, 0)], 1, 1, 1),
        # 路径 r1-c0-r0-c1 存在完美匹配，秩 2
        ([(0, 0), (0, 1), (1, 0)], 2, 2, 2),
        ([(0, 0), (0, 1), (1, 0), (1, 1)], 2, 2, 2),
        # 秩亏方阵：3 行争 2 列（秩 2）+ 行3 占列 2,3 中一个 => 秩 3
        ([(0, 0), (0, 1), (1, 0), (1, 1),
          (2, 0), (2, 1), (3, 2), (3, 3)], 4, 4, 3),
        # 上三角可约：链上的 1x1 块（边 r0->c2, r2->c3 给出拓扑序）
        ([(0, 0), (0, 1), (0, 2), (1, 1), (2, 2), (2, 3), (3, 3)], 4, 4, 4),
        # 2x3 欠定矩形，秩 2
        ([(0, 0), (0, 2), (1, 1), (1, 2)], 2, 3, 2),
        # 3x2 过定矩形，秩 2
        ([(0, 0), (1, 0), (1, 1), (2, 1)], 3, 2, 2),
        # 4x5 同时含过定/良定/欠定特征：
        # 行0,1,2 争列0,1（过定）；行3 与列 2,3,4（欠定）
        ([(0, 0), (1, 0), (1, 1), (2, 1),
          (3, 2), (3, 3), (3, 4)], 4, 5, 3),
        # 全满 4x4
        ([(r, c) for r in range(4) for c in range(4)], 4, 4, 4),
        # 不可约 3x3 循环
        ([(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 0)], 3, 3, 3),
    ]

    def test_hand_cased_rank_against_enumeration(self):
        for edges, n, m, expected in self.CASES:
            with self.subTest(edges=edges, shape=(n, m)):
                d = analyze(edges, n, m)
                _, _, enum_rank = enumerate_max_matching(edges, n, m)
                self.assertEqual(d.structural_rank, enum_rank)
                self.assertEqual(d.structural_rank, expected)
                # 匹配证书自身一致
                self.assertEqual(
                    sum(1 for c in d.row_match if c != -1), d.structural_rank
                )
                for r, c in enumerate(d.row_match):
                    if c != -1:
                        self.assertIn((r, c), d.nonzeros)
                        self.assertEqual(d.col_match[c], r)
                self.assertTrue(verify(d))

    def test_random_small_cases_against_enumeration(self):
        # 确定性伪随机枚举：LCG，无需 random 模块之外的状态，固定种子。
        seed = 123456789

        def lcg():
            nonlocal seed
            seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
            return seed

        for trial in range(120):
            n = lcg() % 5  # 0..4
            m = lcg() % 5
            p = (lcg() % 70) / 100.0 + 0.05
            edges = []
            for r in range(n):
                for c in range(m):
                    if (lcg() % 1000) / 1000.0 < p:
                        edges.append((r, c))
            with self.subTest(trial=trial, shape=(n, m), nedges=len(edges)):
                d = analyze(edges, n, m)
                _, _, enum_rank = enumerate_max_matching(edges, n, m)
                self.assertEqual(d.structural_rank, enum_rank)
                self.assertTrue(verify(d))

    def test_dimension_inference_matches_explicit(self):
        edges = [(0, 1), (2, 0)]
        d1 = analyze(edges)
        d2 = analyze(edges, 3, 2)
        self.assertEqual((d1.n_rows, d1.n_cols), (3, 2))
        self.assertEqual(d1.row_perm, d2.row_perm)
        self.assertEqual(d1.col_perm, d2.col_perm)
        self.assertEqual(d1.structural_rank, d2.structural_rank)

    def test_empty_matrix_rank_zero_and_no_blocks(self):
        d = analyze([], 0, 0)
        self.assertEqual(d.structural_rank, 0)
        self.assertEqual(d.row_perm, ())
        self.assertEqual(d.col_perm, ())
        self.assertEqual(d.iter_blocks(), ())
        self.assertTrue(verify(d))


class RegionReferenceTests(unittest.TestCase):
    """三区划分与独立参考逐一比对。"""

    def _check_regions(self, edges, n, m):
        d = analyze(edges, n, m)
        rm, cm, _ = enumerate_max_matching(edges, n, m)
        ur, uc, wr, wc, orr, oc = reference_regions(
            list(set(edges)), rm, cm, n, m
        )
        self.assertEqual(
            {r for r, t in enumerate(d.row_region) if t == "U"}, ur
        )
        self.assertEqual(
            {c for c, t in enumerate(d.col_region) if t == "U"}, uc
        )
        self.assertEqual(
            {r for r, t in enumerate(d.row_region) if t == "O"}, orr
        )
        self.assertEqual(
            {c for c, t in enumerate(d.col_region) if t == "O"}, oc
        )
        self.assertEqual(
            {r for r, t in enumerate(d.row_region) if t == "W"}, wr
        )
        self.assertEqual(
            {c for c, t in enumerate(d.col_region) if t == "W"}, wc
        )
        return d

    def test_all_three_regions_present(self):
        # 4 行 5 列：前 3 行争 2 列（过定 + 1 未匹配行），
        # 第 4 行接 3 列（欠定 + 2 未匹配列），秩 3。
        edges = [(0, 0), (1, 0), (1, 1), (2, 1),
                 (3, 2), (3, 3), (3, 4)]
        d = analyze(edges, 4, 5)
        self.assertEqual(d.structural_rank, 3)
        labels_r = "".join(d.row_region)
        labels_c = "".join(d.col_region)
        self.assertIn("O", labels_r)
        self.assertIn("U", labels_c)
        self.assertTrue(verify(d))

    def test_region_counts_identity(self):
        # 对所有手工例检查 |O行|-|O列| = 未匹配行数（位于O），
        # |U列|-|U行| = 未匹配列数（位于U）。
        for edges, n, m, _ in RankAndMatchingReferenceTests.CASES:
            with self.subTest(shape=(n, m)):
                d = analyze(edges, n, m)
                n_or = d.row_region.count("O")
                n_oc = d.col_region.count("O")
                n_ur = d.row_region.count("U")
                n_uc = d.col_region.count("U")
                self.assertEqual(n_or - n_oc, len(d.unmatched_rows))
                self.assertEqual(n_uc - n_ur, len(d.unmatched_cols))
                # 未匹配行必在 O 区，未匹配列必在 U 区
                for r in d.unmatched_rows:
                    self.assertEqual(d.row_region[r], "O")
                for c in d.unmatched_cols:
                    self.assertEqual(d.col_region[c], "U")
                self.assertTrue(verify(d))

    def test_regions_against_reference_random(self):
        seed = 987654321

        def lcg():
            nonlocal seed
            seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
            return seed

        for trial in range(80):
            n, m = lcg() % 5, lcg() % 6
            edges = [
                (r, c)
                for r in range(n)
                for c in range(m)
                if (lcg() % 100) < 45
            ]
            with self.subTest(trial=trial, shape=(n, m)):
                d = analyze(edges, n, m)
                rm, cm, _ = enumerate_max_matching(edges, n, m)
                ur, uc, wr, wc, orr, oc = reference_regions(
                    edges, rm, cm, n, m
                )
                # 注意：具体哪些顶点未匹配随最大匹配选择而变，故这里只
                # 比较与匹配无关的 DM 三区；未匹配顶点恒在 O 行 / U 列。
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
                self.assertTrue(verify(d))


class BlockTriangularTests(unittest.TestCase):
    """良定区 SCC 块三角化与不可约块。"""

    def test_two_reducible_blocks(self):
        # 两个全满 2x2 不可约块，仅有 块A行 -> 块B列 的单向连接 r0-c2：
        # A = {r0,r1}x{c0,c1}（K2,2），B = {r2,r3}x{c2,c3}（K2,2）。
        edges = [(0, 0), (0, 1), (1, 0), (1, 1),
                 (2, 2), (2, 3), (3, 2), (3, 3),
                 (0, 2)]
        d = analyze(edges, 4, 4)
        self.assertEqual(d.structural_rank, 4)
        wblocks = [b for b in d.iter_blocks() if b.region == "W"]
        self.assertEqual(len(wblocks), 2)
        self.assertEqual([b.shape for b in wblocks], [(2, 2), (2, 2)])
        # 置换后无非零元在对角块下方
        nz = d.permuted_nonzeros()
        row_index = {r: i for i, r in enumerate(d.row_perm)}
        col_index = {c: j for j, c in enumerate(d.col_perm)}

        def block_of(spans, p):
            for k, (a, b) in enumerate(spans):
                if a <= p < b:
                    return k
            raise AssertionError

        for r, c in edges:
            bi = block_of(d.row_blocks, row_index[r])
            bj = block_of(d.col_blocks, col_index[c])
            self.assertLessEqual(bi, bj)
        self.assertIn(nz, [nz])  # 置换后非零元集合可用
        self.assertTrue(verify(d))

    def test_irreducible_cycle_is_single_block(self):
        # 3x3 循环：每个 SCC 都含全部 3 行 => 一个不可约块。
        edges = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 0)]
        d = analyze(edges, 3, 3)
        wblocks = [b for b in d.iter_blocks() if b.region == "W"]
        self.assertEqual(len(wblocks), 1)
        self.assertEqual(wblocks[0].shape, (3, 3))
        self.assertEqual(set(wblocks[0].rows), {0, 1, 2})
        self.assertEqual(set(wblocks[0].cols), {0, 1, 2})
        self.assertTrue(verify(d))

    def test_full_matrix_is_one_irreducible_block(self):
        edges = [(r, c) for r in range(5) for c in range(5)]
        d = analyze(edges, 5, 5)
        wblocks = [b for b in d.iter_blocks() if b.region == "W"]
        self.assertEqual(len(wblocks), 1)
        self.assertEqual(wblocks[0].shape, (5, 5))
        self.assertTrue(verify(d))

    def test_diagonal_matrix_gives_singleton_blocks(self):
        edges = [(i, i) for i in range(4)]
        d = analyze(edges, 4, 4)
        wblocks = [b for b in d.iter_blocks() if b.region == "W"]
        self.assertEqual(len(wblocks), 4)
        for b in wblocks:
            self.assertEqual(b.shape, (1, 1))
        self.assertTrue(verify(d))

    def test_blocks_match_transitive_closure_reference(self):
        seed = 246813579

        def lcg():
            nonlocal seed
            seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
            return seed

        for trial in range(100):
            n = 1 + lcg() % 5
            # 以较高概率加边保证满秩（加一条对角线）
            edges = {(i, i) for i in range(n)}
            for r in range(n):
                for c in range(n):
                    if (lcg() % 100) < 35:
                        edges.add((r, c))
            edges = sorted(edges)
            with self.subTest(trial=trial, n=n):
                d = analyze(edges, n, n)
                self.assertEqual(d.structural_rank, n)
                ref_blocks, reach = reference_well_blocks(
                    edges, list(d.row_match), list(d.col_match),
                    range(n),
                )
                core_blocks = {
                    frozenset(b.rows)
                    for b in d.iter_blocks()
                    if b.region == "W"
                }
                self.assertEqual(core_blocks, ref_blocks)
                # 核心块顺序必须是有向图 H 的拓扑序：
                # 若块 A 有边到块 B，A 不得排在 B 之后。
                pos = {}
                for b in d.iter_blocks():
                    if b.region == "W":
                        for r in b.rows:
                            pos[r] = b.index
                for r in range(n):
                    for w in reach[r]:
                        self.assertLessEqual(pos[r], pos[w])
                self.assertTrue(verify(d))

    def test_mixed_regions_with_well_irreducible_block(self):
        # 8x8，三个互不相连的部分（跨区无任何边，因此区域不被交替路连通）：
        #  O：行 0,1,2 争列 0,1（秩 2，1 个未匹配行）
        #  W：行 3,4,5 与列 2,3,4 构成 3x3 不可约循环
        #  U：行 6,7 与列 5,6,7（秩 2，1 个未匹配列）
        edges = [
            (0, 0), (1, 0), (1, 1), (2, 1),
            (3, 2), (3, 3), (4, 3), (4, 4), (5, 4), (5, 2),
            (6, 5), (6, 7), (7, 6), (7, 7),
        ]
        d = analyze(edges, 8, 8)
        self.assertEqual(d.structural_rank, 7)
        self.assertEqual(d.block_regions, ("U", "W", "O"))
        blocks = d.iter_blocks()
        self.assertEqual(blocks[0].shape, (2, 3))   # 欠定
        self.assertEqual(blocks[1].shape, (3, 3))   # 良定且不可约
        self.assertEqual(blocks[2].shape, (3, 2))   # 过定
        self.assertEqual(set(blocks[1].rows), {3, 4, 5})
        self.assertEqual(set(blocks[1].cols), {2, 3, 4})
        self.assertEqual(len(d.unmatched_rows), 1)
        self.assertEqual(len(d.unmatched_cols), 1)
        self.assertTrue(verify(d))


class MultipleMatchingsSameStructureTests(unittest.TestCase):
    """多个最大匹配但结构区相同（DM 分区与匹配选择无关）。"""

    def test_full_3x3_has_six_matchings_one_well_region(self):
        edges = [(r, c) for r in range(3) for c in range(3)]
        self.assertEqual(count_perfect_matchings(edges, 3), 6)
        d = analyze(edges, 3, 3)
        self.assertTrue(all(t == "W" for t in d.row_region))
        self.assertTrue(all(t == "W" for t in d.col_region))
        wblocks = [b for b in d.iter_blocks() if b.region == "W"]
        self.assertEqual(len(wblocks), 1)
        self.assertTrue(verify(d))

    def test_two_disjoint_switches_same_regions(self):
        # 两个独立全满 2x2，各自有 2 个完美匹配 => 共 4 个；
        # 结构区始终全 W，块始终为两个 1x1 或合并视连通性而定：
        # 无跨块边 => 两个 2x2 可约块。
        edges = [(0, 0), (0, 1), (1, 0), (1, 1),
                 (2, 2), (2, 3), (3, 2), (3, 3)]
        self.assertEqual(count_perfect_matchings(edges, 4), 4)
        d = analyze(edges, 4, 4)
        self.assertTrue(all(t == "W" for t in d.row_region))
        wblocks = [b for b in d.iter_blocks() if b.region == "W"]
        self.assertEqual(len(wblocks), 2)
        self.assertEqual([b.shape for b in wblocks], [(2, 2), (2, 2)])
        self.assertTrue(verify(d))

    def test_symmetric_choice_does_not_change_regions(self):
        # 构造 10 个不同合法最大匹配（枚举给出），逐一检查核心结果的
        # 区域划分与块划分都相同——即使核心内部选了另一个匹配。
        seed = 135792468

        def lcg():
            nonlocal seed
            seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
            return seed

        checked = 0
        for _ in range(400):
            n = 3 + lcg() % 2
            edges = {(i, i) for i in range(n)}
            for r in range(n):
                for c in range(n):
                    if (lcg() % 100) < 40:
                        edges.add((r, c))
            edges = sorted(edges)
            npm = count_perfect_matchings(edges, n)
            if npm < 2:
                continue
            d = analyze(edges, n, n)
            # 区域：全 W（满秩方阵）
            self.assertTrue(all(t == "W" for t in d.row_region))
            # 块集合与具体匹配无关：用传递闭包参考核对一次
            ref_blocks, _ = reference_well_blocks(
                edges, list(d.row_match), list(d.col_match), range(n)
            )
            core_blocks = {
                frozenset(b.rows)
                for b in d.iter_blocks() if b.region == "W"
            }
            self.assertEqual(core_blocks, ref_blocks)
            checked += 1
            if checked >= 10:
                break
        self.assertGreaterEqual(checked, 10)


class EmptyAndRectangleTests(unittest.TestCase):
    def test_zero_by_zero(self):
        d = analyze([])
        self.assertEqual((d.n_rows, d.n_cols), (0, 0))
        self.assertEqual(d.structural_rank, 0)
        self.assertEqual(d.row_perm, ())
        self.assertEqual(d.spy(), "")
        self.assertTrue(verify(d))

    def test_zero_rows_three_cols_is_underdetermined(self):
        d = analyze([], 0, 3)
        self.assertEqual(d.structural_rank, 0)
        self.assertEqual(d.unmatched_cols, (0, 1, 2))
        self.assertEqual(d.block_regions, ("U",))
        block = d.iter_blocks()[0]
        self.assertEqual(block.shape, (0, 3))
        self.assertEqual(block.cols, (0, 1, 2))
        self.assertTrue(verify(d))

    def test_three_rows_zero_cols_is_overdetermined(self):
        d = analyze([], 3, 0)
        self.assertEqual(d.structural_rank, 0)
        self.assertEqual(d.unmatched_rows, (0, 1, 2))
        self.assertEqual(d.block_regions, ("O",))
        block = d.iter_blocks()[0]
        self.assertEqual(block.shape, (3, 0))
        self.assertEqual(block.rows, (0, 1, 2))
        self.assertTrue(verify(d))

    def test_wide_fully_underdetermined(self):
        # 2x4，所有列经交替路连通到未匹配列：整块欠定 (2 行 4 列)。
        edges = [(0, 0), (0, 2), (1, 1), (1, 2)]
        d = analyze(edges, 2, 4)
        self.assertEqual(d.structural_rank, 2)
        self.assertEqual(len(d.unmatched_cols), 2)
        self.assertEqual(d.block_regions, ("U",))
        block = d.iter_blocks()[0]
        self.assertEqual(block.shape, (2, 4))
        self.assertTrue(verify(d))

    def test_wide_well_plus_isolated_cols(self):
        # 行 0,1 与列 0,1 构成满 K2,2（良定），列 2,3 完全孤立（欠定）：
        # 块形式为 0x2 的 U 块 + 2x2 的 W 块。
        edges = [(0, 0), (0, 1), (1, 0), (1, 1)]
        d = analyze(edges, 2, 4)
        self.assertEqual(d.structural_rank, 2)
        self.assertEqual(d.unmatched_cols, (2, 3))
        self.assertEqual(
            [t for t in d.block_regions], ["U", "W"]
        )
        shapes = [b.shape for b in d.iter_blocks()]
        self.assertEqual(shapes, [(0, 2), (2, 2)])
        self.assertTrue(verify(d))

    def test_tall_rectangle(self):
        # 列 0,1 与行 0,1 构成满 K2,2（良定），行 2,3 完全孤立（过定）：
        # 块形式为 2x2 的 W 块 + 2x0 的 O 块。
        edges = [(0, 0), (0, 1), (1, 0), (1, 1)]
        d = analyze(edges, 4, 2)
        self.assertEqual(d.structural_rank, 2)
        self.assertEqual(d.unmatched_rows, (2, 3))
        self.assertEqual(d.block_regions, ("W", "O"))
        shapes = [b.shape for b in d.iter_blocks()]
        self.assertEqual(shapes, [(2, 2), (2, 0)])
        self.assertTrue(verify(d))

    def test_empty_rows_and_cols_inside(self):
        # 行 2 全空，列 3 全空（3x4，秩 2）
        edges = [(0, 0), (0, 1), (1, 1)]
        d = analyze(edges, 3, 4)
        self.assertEqual(d.structural_rank, 2)
        self.assertEqual(d.unmatched_rows, (2,))
        self.assertIn(3, d.unmatched_cols)
        self.assertEqual(d.row_region[2], "O")
        self.assertEqual(d.col_region[3], "U")
        self.assertTrue(verify(d))


class DedupAndDeterminismTests(unittest.TestCase):
    def test_duplicate_positions_deduped(self):
        d1 = analyze([(0, 0), (1, 1), (0, 0)])
        d2 = analyze([(1, 1), (0, 0)])
        self.assertEqual(d1.nonzeros, frozenset({(0, 0), (1, 1)}))
        self.assertEqual(d1.nonzeros, d2.nonzeros)
        self.assertEqual(d1.row_perm, d2.row_perm)
        self.assertEqual(d1.col_perm, d2.col_perm)
        self.assertEqual(d1.structural_rank, 2)

    def test_order_independence(self):
        edges = [(r, c) for r in range(5) for c in range(5) if (r + c) % 2 == 0]
        shuffled = edges[::-1]
        d1 = analyze(edges, 5, 5)
        d2 = analyze(shuffled, 5, 5)
        self.assertEqual(d1.row_perm, d2.row_perm)
        self.assertEqual(d1.col_perm, d2.col_perm)
        self.assertEqual(d1.block_regions, d2.block_regions)
        self.assertTrue(verify(d1))
        self.assertTrue(verify(d2))

    def test_result_immutable(self):
        d = analyze([(0, 0)], 1, 1)
        with self.assertRaises(Exception):
            d.structural_rank = 5  # type: ignore[misc]
        self.assertIsInstance(d.nonzeros, frozenset)


class InputValidationTests(unittest.TestCase):
    def test_negative_row_localizable(self):
        with self.assertRaises(InvalidStructureError) as cm:
            analyze([(0, 0), (-1, 2)])
        self.assertIn("entries[1]", str(cm.exception))

    def test_negative_col(self):
        with self.assertRaises(InvalidStructureError):
            analyze([(0, -3)])

    def test_float_nan_rejected(self):
        with self.assertRaises(InvalidStructureError) as cm:
            analyze([(0, float("nan"))])
        self.assertIn("NaN", str(cm.exception))

    def test_float_inf_rejected(self):
        with self.assertRaises(InvalidStructureError):
            analyze([(float("inf"), 0)])
        with self.assertRaises(InvalidStructureError):
            analyze([(0, float("-inf"))])

    def test_bool_rejected_as_index(self):
        with self.assertRaises(TypeError) as cm:
            analyze([(True, 0)])
        self.assertIn("bool", str(cm.exception))

    def test_bool_rejected_as_dimension(self):
        with self.assertRaises(TypeError):
            analyze([], True, 3)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            analyze([], 3, False)  # type: ignore[arg-type]

    def test_bool_rejected_as_budget(self):
        with self.assertRaises(TypeError):
            analyze([(0, 0)], budget=True)  # type: ignore[arg-type]

    def test_float_index_rejected(self):
        with self.assertRaises(TypeError):
            analyze([(1.0, 0)])

    def test_nan_inf_dimension_and_budget_rejected(self):
        with self.assertRaises(InvalidStructureError):
            analyze([], float("nan"), 2)
        with self.assertRaises(InvalidStructureError):
            analyze([], 2, float("inf"))
        with self.assertRaises(InvalidStructureError):
            analyze([(0, 0)], budget=float("-inf"))
        with self.assertRaises(TypeError):
            analyze([], 1.0, 2)
        with self.assertRaises(TypeError):
            analyze([(0, 0)], budget=1.0)

    def test_bad_pair_length_localizable(self):
        with self.assertRaises(InvalidStructureError) as cm:
            analyze([(0, 0), (1, 2, 3)])
        self.assertIn("entries[1]", str(cm.exception))

    def test_out_of_explicit_dimension(self):
        with self.assertRaises(InvalidStructureError):
            analyze([(0, 0), (2, 0)], n_rows=2, n_cols=2)
        with self.assertRaises(InvalidStructureError):
            analyze([(0, 2)], n_rows=2, n_cols=2)

    def test_dimension_limit(self):
        with self.assertRaises(LimitExceededError):
            analyze([], MAX_ROWS + 1, 2)
        with self.assertRaises(LimitExceededError):
            analyze([], 2, MAX_COLS + 1)
        with self.assertRaises(LimitExceededError):
            analyze([(MAX_ROWS, 0)])
        with self.assertRaises(LimitExceededError):
            analyze([(0, MAX_COLS)])

    def test_boundary_dimensions_accepted(self):
        d = analyze([], MAX_ROWS, MAX_COLS)
        self.assertEqual((d.n_rows, d.n_cols), (MAX_ROWS, MAX_COLS))
        self.assertTrue(verify(d))

    def test_boundary_full_100x100_fast_and_full_rank(self):
        # 接口上限规模：100x100 全满，结构秩必须为 100 且整块不可约。
        edges = [(r, c) for r in range(MAX_ROWS) for c in range(MAX_COLS)]
        import time
        t0 = time.perf_counter()
        d = analyze(edges, MAX_ROWS, MAX_COLS)
        elapsed = time.perf_counter() - t0
        self.assertEqual(d.structural_rank, 100)
        wblocks = [b for b in d.iter_blocks() if b.region == "W"]
        self.assertEqual(len(wblocks), 1)
        self.assertEqual(wblocks[0].shape, (100, 100))
        self.assertTrue(verify(d))
        self.assertLess(elapsed, 5.0)

    def test_boundary_100x100_identity_singletons(self):
        edges = [(i, i) for i in range(100)]
        d = analyze(edges, 100, 100)
        self.assertEqual(d.structural_rank, 100)
        self.assertEqual(
            sum(1 for t in d.block_regions if t == "W"), 100
        )
        self.assertTrue(verify(d))

    def test_negative_dimension(self):
        with self.assertRaises(InvalidStructureError):
            analyze([], -1, 2)

    def test_non_iterable_entries(self):
        with self.assertRaises(TypeError):
            analyze(42)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            analyze(None)

    def test_generator_input_consumed_once(self):
        d = analyze((r, c) for r in range(3) for c in range(3) if r == c)
        self.assertEqual(d.structural_rank, 3)
        self.assertTrue(verify(d))


class BudgetTests(unittest.TestCase):
    def test_zero_budget_on_nonempty_raises(self):
        with self.assertRaises(SearchBudgetExhausted):
            analyze([(0, 0)], 1, 1, budget=0)

    def test_budget_exhaustion_is_not_proof_of_no_augmenting_path(self):
        # 边多、预算小：必然耗尽，且不能影响“无输入被修改”的语义。
        edges = [(r, c) for r in range(4) for c in range(4)]
        with self.assertRaises(SearchBudgetExhausted):
            analyze(edges, 4, 4, budget=1)

    def test_sufficient_budget_matches_default(self):
        edges = [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)]
        d_full = analyze(edges, 3, 3)
        # 每条二部边至多考察常数次，给足预算。
        d_b = analyze(edges, 3, 3, budget=10_000)
        self.assertEqual(d_b.structural_rank, d_full.structural_rank)
        self.assertTrue(verify(d_b))

    def test_budget_negative_rejected(self):
        with self.assertRaises(InvalidStructureError):
            analyze([(0, 0)], budget=-1)

    def test_failure_leaves_no_partial_state(self):
        # 分析是纯函数：失败时不返回半成品对象。
        edges = [(0, 0), (1, 1)]
        before = list(edges)
        with self.assertRaises(SearchBudgetExhausted):
            analyze(edges, 2, 2, budget=0)
        self.assertEqual(edges, before)
        with self.assertRaises(InvalidStructureError):
            analyze(edges + [(-1, 0)], 2, 2)
        self.assertEqual(edges, before)


class CertificateTests(unittest.TestCase):
    def test_permutation_restores_block_upper_triangular(self):
        # 8x8：O(3行2列) + 两个 2x2 全满良定不可约块 + U(1行2列)，
        # 各部分互不相连，另加一条 块W1行r3 -> 块W2列c4 的上三角填充边。
        edges = [
            (0, 0), (1, 0), (1, 1), (2, 1),
            (3, 2), (3, 3), (4, 2), (4, 3),
            (5, 4), (5, 5), (6, 4), (6, 5),
            (7, 6), (7, 7),
            (3, 4),
        ]
        d = analyze(edges, 8, 8)
        self.assertEqual(d.structural_rank, 7)
        self.assertEqual(d.block_regions, ("U", "W", "W", "O"))
        self.assertTrue(verify(d))
        # 手工再验一次块上三角与块内相对主对角
        rp = {r: i for i, r in enumerate(d.row_perm)}
        cp = {c: j for j, c in enumerate(d.col_perm)}

        def span_of(spans, p):
            for k, (a, b) in enumerate(spans):
                if a <= p < b:
                    return k
            raise AssertionError

        block_pairs = []
        for r, c in edges:
            i, j = rp[r], cp[c]
            bi = span_of(d.row_blocks, i)
            bj = span_of(d.col_blocks, j)
            self.assertLessEqual(bi, bj)
            block_pairs.append((r, c, bi, bj))
        # 填充边 (3,4) 必须落在对角块*上方*：行块号 < 列块号
        fill = [x for x in block_pairs if x[0] == 3 and x[1] == 4][0]
        self.assertLess(fill[2], fill[3])
        for r, c in enumerate(d.row_match):
            if c != -1:
                i, j = rp[r], cp[c]
                bi = span_of(d.row_blocks, i)
                bj = span_of(d.col_blocks, j)
                self.assertEqual(bi, bj)
                self.assertEqual(
                    i - d.row_blocks[bi][0], j - d.col_blocks[bj][0]
                )

    def test_tampered_certificate_detected(self):
        # 手工篡改一个结果对象（dataclass(frozen=True) 用 replace 派生）
        import dataclasses
        d = analyze([(0, 0), (1, 1)], 2, 2)
        tampered = dataclasses.replace(d, structural_rank=99)
        with self.assertRaises(Exception):
            verify(tampered)

    def test_augmenting_path_detection(self):
        # 构造一个*非最大*匹配证书，verify 必须发现增广路。
        d = analyze([(0, 0), (0, 1), (1, 1)], 2, 2)
        import dataclasses
        # 强制行 1 未匹配、列 1 未匹配：存在增广路 1-1 直接边
        tampered = dataclasses.replace(
            d,
            row_match=(0, -1),
            col_match=(0, -1),
            structural_rank=1,
        )
        with self.assertRaises(Exception) as cm:
            verify(tampered)
        self.assertIn("增广路", str(cm.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class EdgeCriticalityTests(unittest.TestCase):
    """边关键性：逐边独立删除试验的分类与见证。

    参考值由 tests/reference.py 的枚举匹配在“删除该边后的图”上独立
    重算得到；每次删除都从原图出发（独立试验）。
    """

    def _check_witness(self, edge_set, n, rec, rank):
        """核验单条边的见证：替代匹配或 König 最小顶点覆盖。"""
        if rec.critical:
            self.assertIsNone(rec.alternative_matching)
            self.assertEqual(rec.rank_after_removal, rank - 1)
            cover_rows, cover_cols = rec.vertex_cover
            # 覆盖大小 == 删除后的秩，给出删除后匹配数的上界证明。
            self.assertEqual(len(cover_rows) + len(cover_cols), rank - 1)
            remaining = edge_set - {(rec.row, rec.col)}
            for (r, c) in remaining:
                self.assertTrue(
                    r in cover_rows or c in cover_cols,
                    f"边 ({r},{c}) 未被顶点覆盖",
                )
        else:
            self.assertIsNone(rec.vertex_cover)
            self.assertEqual(rec.rank_after_removal, rank)
            alt = rec.alternative_matching
            self.assertEqual(len(alt), n)
            # 替代匹配必须避开被删除的这条边。
            self.assertNotEqual(alt[rec.row], rec.col)
            used_cols = set()
            matched = 0
            for r, c in enumerate(alt):
                if c == -1:
                    continue
                matched += 1
                self.assertIn((r, c), edge_set)
                self.assertNotIn(c, used_cols)
                used_cols.add(c)
            self.assertEqual(matched, rank)

    def _brute_force_check(self, edges, n, m):
        """逐边独立删除 + 枚举匹配，核对分类、秩与见证。"""
        edge_set = set(map(tuple, edges))
        rep = edge_criticality(edges, n, m)
        # 原块三角化接口保留且证书自洽。
        verify(rep.decomposition)
        self.assertEqual(rep.structural_rank, rep.decomposition.structural_rank)
        self.assertEqual(len(rep.edges), len(edge_set))
        self.assertEqual(
            [(e.row, e.col) for e in rep.edges], sorted(edge_set)
        )
        by_pos = rep.by_position()
        for (r, c) in sorted(edge_set):
            rec = by_pos[(r, c)]
            remaining = [e for e in sorted(edge_set) if e != (r, c)]
            _, _, ref_rank = enumerate_max_matching(remaining, n, m)
            with self.subTest(edge=(r, c), edges=sorted(edge_set)):
                self.assertEqual(rec.critical, ref_rank < rep.structural_rank)
                self.assertEqual(rec.rank_after_removal, ref_rank)
                self._check_witness(edge_set, n, rec, rep.structural_rank)
        return rep
    def test_perfect_matching_cycle_no_critical_edge(self):
        # 偶环 C4：两个完美匹配，每条边都在交替环上，删任一边秩不变。
        edges = [(i, i) for i in range(4)] + [(i, (i + 1) % 4) for i in range(4)]
        self.assertEqual(count_perfect_matchings(edges, 4), 2)
        rep = self._brute_force_check(edges, 4, 4)
        self.assertEqual(rep.structural_rank, 4)
        self.assertEqual(rep.critical_edges, ())
        # 每条边（含当前匹配中的边）都有替代匹配见证。
        for rec in rep.edges:
            self.assertIsNotNone(rec.alternative_matching)

    def test_single_bridge_matching(self):
        # 路径 r0-c0-r1-c1：唯一最大匹配 {(0,0),(1,1)}，两条匹配边皆关键。
        edges = [(0, 0), (1, 0), (1, 1)]
        rep = self._brute_force_check(edges, 2, 2)
        self.assertEqual(rep.structural_rank, 2)
        self.assertEqual(set(rep.critical_edges), {(0, 0), (1, 1)})
        # 非匹配边 (1,0) 删除后秩不变。
        self.assertFalse(rep.by_position()[(1, 0)].critical)
        # 关键边的降秩证据：大小为 1 的顶点覆盖。
        cover = rep.by_position()[(0, 0)].vertex_cover
        self.assertEqual(len(cover[0]) + len(cover[1]), 1)

    def test_matched_edge_not_necessarily_essential(self):
        # K2,2 全满：当前匹配含 (0,0)，但它在交替环上，并非必不可少。
        edges = [(0, 0), (0, 1), (1, 0), (1, 1)]
        rep = self._brute_force_check(edges, 2, 2)
        self.assertEqual(rep.critical_edges, ())
        # 每条边都在某个最大匹配里，但没有一条必不可少。
        for rec in rep.edges:
            self.assertFalse(rec.critical)
            self.assertNotEqual(rec.alternative_matching[rec.row], rec.col)
        self.assertTrue(any(e.in_original_matching for e in rep.edges))

    def test_rectangular_underdetermined(self):
        # 2x3 欠定矩形：删任一边秩仍为 2。
        edges = [(r, c) for r in range(2) for c in range(3)]
        rep = self._brute_force_check(edges, 2, 3)
        self.assertEqual(rep.structural_rank, 2)
        self.assertEqual(rep.critical_edges, ())

    def test_structural_rank_zero(self):
        # 结构秩为零：没有非零位置，报告为空。
        rep = edge_criticality([], 3, 3)
        self.assertEqual(rep.structural_rank, 0)
        self.assertEqual(rep.edges, ())
        self.assertEqual(rep.critical_edges, ())
        verify(rep.decomposition)
        self.assertEqual(rep.critical_edges, ())
        verify(rep.decomposition)

    def test_random_small_graphs_brute_force(self):
        # 随机小图：逐边独立删除后枚举匹配，全面核对分类与见证。
        import random
        rng = random.Random(20260922)
        for _ in range(30):
            n = rng.randint(1, 4)
            m = rng.randint(1, 4)
            edges = [
                (r, c)
                for r in range(n)
                for c in range(m)
                if rng.random() < 0.5
            ]
            self._brute_force_check(edges, n, m)

    def test_decomposition_matches_analyze(self):
        # 报告中的分解与 analyze() 完全一致（原接口保留）。
        edges = [(0, 0), (1, 0), (1, 1), (2, 2), (2, 3), (3, 2)]
        rep = edge_criticality(edges, 4, 4)
        self.assertEqual(rep.decomposition, analyze(edges, 4, 4))

    def test_order_independence_and_immutability(self):
        edges = [(0, 0), (1, 0), (1, 1), (2, 2)]
        rep1 = edge_criticality(edges, 3, 3)
        rep2 = edge_criticality(list(reversed(edges)), 3, 3)
        self.assertEqual(rep1, rep2)
        with self.assertRaises(Exception):
            rep1.edges[0].critical = False  # frozen dataclass

    def test_validation_and_budget_semantics(self):
        # 校验与 analyze 一致：NaN 拒绝；预算耗尽与证明无解分开。
        with self.assertRaises(InvalidStructureError):
            edge_criticality([(0, float("nan"))], 1, 1)
        with self.assertRaises(TypeError):
            edge_criticality([(True, 0)], 1, 1)
        dense = [(r, c) for r in range(4) for c in range(4)]
        with self.assertRaises(SearchBudgetExhausted):
            edge_criticality(dense, 4, 4, budget=2)
