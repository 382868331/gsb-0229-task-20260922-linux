"""核心功能测试：结构秩、DM 三区、SCC 块三角化、证书、置换检查。

期望值来自 tests/reference.py 的独立枚举/独立 SCC，不直接调用被测核心。
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests import reference as ref

import structrank as sr
from structrank.errors import InvalidStructureError, SearchBudgetExhaustedError


def render(nz, nr, nc):
    return [
        "".join("#" if (r, c) in nz else "." for c in range(nc))
        for r in range(nr)
    ]


def assert_well_ordered(test, result, nz):
    """对每个测试结果跑独立结构检查（补充性质，不作为期望值来源）。"""
    ok, problems = ref.permuted_structure_check(result, set(nz))
    test.assertTrue(ok, msg="; ".join(problems))


class StructuralRankTests(unittest.TestCase):
    def test_enumerated_small_matrices(self):
        # 枚举所有 0..4 阶（含矩形）的小 0/1 模式，结构秩与独立枚举逐一比对。
        rng = random.Random(20260923)
        for nr in range(0, 5):
            for nc in range(0, 5):
                for trial in range(60):
                    nz = {
                        (r, c)
                        for r in range(nr)
                        for c in range(nc)
                        if rng.random() < 0.4
                    }
                    result = sr.analyze(nz, nr, nc)
                    expected = ref.enumerate_max_matching_size(nz, nr, nc)
                    self.assertEqual(
                        result.structural_rank,
                        expected,
                        msg=(
                            f"rank 不符 nr={nr} nc={nc}\n"
                            + "\n".join(render(nz, nr, nc))
                        ),
                    )
                    report = sr.verify_certificate(
                        nz, result.match_row, result.match_col, nr, nc
                    )
                    self.assertTrue(report.valid)
                    self.assertEqual(
                        len(report.vertex_cover_rows) + len(report.vertex_cover_cols),
                        expected,
                    )
                    assert_well_ordered(self, result, nz)

    def test_full_rank_examples(self):
        nz = {(0, 0), (1, 2), (2, 1)}
        r = sr.analyze(nz, 3, 3)
        self.assertEqual(r.structural_rank, 3)
        self.assertEqual(set(r.regions.well_rows), {0, 1, 2})
        self.assertEqual(set(r.regions.well_cols), {0, 1, 2})

    def test_rank_deficient_square(self):
        # 三行的非零都只落在前两列 -> 结构秩 2。
        nz = {(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)}
        r = sr.analyze(nz, 3, 3)
        self.assertEqual(r.structural_rank, 2)
        # 独立枚举核对。
        self.assertEqual(r.structural_rank, ref.enumerate_max_matching_size(nz, 3, 3))

    def test_empty_matrix(self):
        r = sr.analyze([], 0, 0)
        self.assertEqual(r.structural_rank, 0)
        self.assertEqual(r.row_perm, ())
        self.assertEqual(r.col_perm, ())
        self.assertEqual(r.btf.num_blocks, 0)

    def test_empty_rows_and_cols(self):
        # 4x4：第 3 行全空（孤立行 -> 过定区），第 3 列全空（孤立列 -> 欠定区），
        # 其余是 3x3 对角良定块。
        nz = {(0, 0), (1, 1), (2, 2)}
        r = sr.analyze(nz, 4, 4)
        self.assertEqual(r.structural_rank, 3)
        self.assertIn(3, r.regions.unmatched_rows)
        self.assertIn(3, r.regions.unmatched_cols)
        self.assertIn(3, r.regions.over_rows)
        self.assertIn(3, r.regions.under_cols)
        self.assertEqual(set(r.regions.well_rows), {0, 1, 2})
        self.assertEqual(set(r.regions.well_cols), {0, 1, 2})
        assert_well_ordered(self, r, nz)

    def test_rectangular_underdetermined(self):
        # 2 行 4 列，连通且满行秩：整块欠定（含孤立空列）。
        nz = {(0, 0), (0, 1), (0, 2), (1, 2), (1, 3)}
        r = sr.analyze(nz, 2, 4)
        self.assertEqual(r.structural_rank, 2)
        self.assertEqual(set(r.regions.under_rows), {0, 1})
        self.assertEqual(set(r.regions.under_cols), {0, 1, 2, 3})
        self.assertEqual(len(r.regions.well_rows), 0)
        self.assertEqual(r.btf.num_blocks, 0)
        assert_well_ordered(self, r, nz)

    def test_rectangular_disconnected_splits_regions(self):
        # 对照：2x4 但结构互不连通时，可匹配的方形成分落入良定区，
        # 孤立空列才是欠定区——分区由交替路可达性决定而非仅看形状。
        nz = {(0, 0), (1, 1)}
        r = sr.analyze(nz, 2, 4)
        self.assertEqual(r.structural_rank, 2)
        self.assertEqual(set(r.regions.well_rows), {0, 1})
        self.assertEqual(set(r.regions.well_cols), {0, 1})
        self.assertEqual(set(r.regions.under_cols), {2, 3})
        self.assertEqual(len(r.regions.under_rows), 0)

    def test_rectangular_overdetermined(self):
        # 4 行 2 列：整图过定。
        nz = {(0, 0), (1, 0), (2, 1), (3, 1)}
        r = sr.analyze(nz, 4, 2)
        self.assertEqual(r.structural_rank, 2)
        self.assertEqual(set(r.regions.over_rows), {0, 1, 2, 3})
        self.assertEqual(set(r.regions.over_cols), {0, 1})
        self.assertEqual(sorted(r.regions.unmatched_rows), [1, 3])
        self.assertEqual(len(r.regions.well_cols), 0)

    def test_mixed_three_regions(self):
        # 4x4：一个过定 2x1 块 + 一个良定 1x1 + 一个欠定 1x2 块（互不相连）。
        nz = {(0, 0), (1, 0), (2, 1), (3, 2), (3, 3)}
        r = sr.analyze(nz, 4, 4)
        self.assertEqual(r.structural_rank, 3)
        self.assertEqual(set(r.regions.over_rows), {0, 1})
        self.assertEqual(set(r.regions.over_cols), {0})
        self.assertEqual(set(r.regions.well_rows), {2})
        self.assertEqual(set(r.regions.well_cols), {1})
        self.assertEqual(set(r.regions.under_rows), {3})
        self.assertEqual(set(r.regions.under_cols), {2, 3})
        assert_well_ordered(self, r, nz)

    def test_regions_independent_of_maximum_matching_chosen(self):
        # 有多个最大匹配的图：DM 三区与良定块划分必须不随匹配选择改变。
        # 图：一个不可约 2x2（两个完美匹配）+ 一个固定 1x1。
        nz = {(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)}
        max_matchings = ref.enumerate_all_max_matchings(nz, 3, 3)
        self.assertGreaterEqual(len(max_matchings), 2)
        ref_partitions = set()
        for m in max_matchings:
            (ur, uc, wr, wc, orow, ocol) = ref.independent_dm_regions(nz, 3, 3, m)
            ref_partitions.add((ur, uc, wr, wc, orow, ocol))
        self.assertEqual(len(ref_partitions), 1, "枚举得到的最大匹配 DM 分区不一致")
        r = sr.analyze(nz, 3, 3)
        got = (
            r.regions.under_rows, r.regions.under_cols,
            r.regions.well_rows, r.regions.well_cols,
            r.regions.over_rows, r.regions.over_cols,
        )
        self.assertEqual(got, next(iter(ref_partitions)))

    def test_duplicate_nonzeros_deduped(self):
        with_dup = [(0, 0), (0, 0), (1, 1), (1, 1), (1, 1)]
        r = sr.analyze(iter(with_dup), 2, 2)
        self.assertEqual(r.structural_rank, 2)
        self.assertEqual(len(r.nonzeros), 2)


class BlockTriangularTests(unittest.TestCase):
    def test_irreducible_2x2_block(self):
        # 全满 2x2 才是真正不可约：任何行列置换都无法消成三角，
        # 匹配行有向图上两行互达，是单个 SCC。
        nz = {(0, 0), (0, 1), (1, 0), (1, 1)}
        r = sr.analyze(nz, 2, 2)
        self.assertEqual(r.btf.num_blocks, 1)
        blk = r.btf.blocks[0]
        self.assertEqual(blk.size, 2)
        self.assertEqual(set(blk.rows), {0, 1})
        self.assertEqual(set(blk.cols), {0, 1})

    def test_triangular_pattern_is_reducible(self):
        # 对照：模式 [1 1; 1 0] 在独立行列置换下可约（列交换即三角），
        # 必须给出两个 1x1 块，不能误报成不可约。
        nz = {(0, 0), (0, 1), (1, 0)}
        r = sr.analyze(nz, 2, 2)
        self.assertEqual([b.size for b in r.btf.blocks], [1, 1])
        assert_well_ordered(self, r, nz)

    def test_reducible_to_three_singletons(self):
        # 上三角 3x3（对角 + 严格上三角非零）-> 三个 1x1 块，且本身已按
        # 块上三角排好，行置换为恒等。
        nz = {(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)}
        r = sr.analyze(nz, 3, 3)
        self.assertEqual([b.size for b in r.btf.blocks], [1, 1, 1])
        assert_well_ordered(self, r, nz)
        self.assertEqual(r.row_perm, (0, 1, 2))
        self.assertEqual(r.col_perm, (0, 1, 2))

    def test_lower_triangular_gets_flipped(self):
        # 对照：下三角模式（边只在块对角线下方）必须被翻成上三角。
        nz = {(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)}
        r = sr.analyze(nz, 3, 3)
        self.assertEqual([b.size for b in r.btf.blocks], [1, 1, 1])
        self.assertEqual(r.row_perm, (2, 1, 0))
        self.assertEqual(r.col_perm, (2, 1, 0))
        assert_well_ordered(self, r, nz)

    def test_two_blocks_cycle_and_singleton(self):
        # 块A: 行{0,1} 列{0,1} 全满不可约 2x2；块B: 行2 列2；边 0->2（A 依赖 B）。
        nz = {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 2)}
        r = sr.analyze(nz, 3, 3)
        self.assertEqual(r.btf.num_blocks, 2)
        sizes = [b.size for b in r.btf.blocks]
        self.assertEqual(sizes, [2, 1])
        # 2x2 块在前，单例块（行2）在后，边 (0,2) 落在块对角线上方。
        self.assertEqual(set(r.btf.blocks[0].rows), {0, 1})
        self.assertEqual(set(r.btf.blocks[1].rows), {2})
        self.assertEqual(r.block_bounds, ((0, 2, 0, 2), (2, 3, 2, 3)))
        assert_well_ordered(self, r, nz)

    def test_block_structure_random_enumerated(self):
        # 随机小图：块划分与独立 Tarjan SCC 完全一致，且块间严格上三角。
        rng = random.Random(424242)
        for _ in range(200):
            n = rng.randint(1, 5)
            nz = {
                (r, c)
                for r in range(n) for c in range(n)
                if rng.random() < 0.45
            }
            r = sr.analyze(nz, n, n)
            # 只对满结构秩情形做块数比对（欠秩时 SCC 只在良定区上）。
            if r.structural_rank == n:
                succ = {i: set() for i in range(n)}
                for rr, cc in nz:
                    r2 = r.match_col[cc]
                    if r2 != rr:
                        succ[rr].add(r2)
                sccs = ref.tarjan_scc(list(range(n)), lambda v: succ[v])
                self.assertEqual(
                    sorted(sorted(s) for s in sccs),
                    sorted(sorted(b.rows) for b in r.btf.blocks),
                )
            assert_well_ordered(self, r, nz)


class CertificateTests(unittest.TestCase):
    def test_verify_accepts_maximum(self):
        nz = {(0, 0), (0, 1), (1, 0)}
        r = sr.analyze(nz, 2, 2)
        rep = sr.verify_certificate(nz, r.match_row, r.match_col, 2, 2)
        self.assertTrue(rep.valid)
        self.assertEqual(rep.structural_rank, 2)

    def test_verify_rejects_nonmaximum(self):
        # 满秩 2x2 图上，故意只交一条匹配边（双向映射保持一致）。
        nz = {(0, 0), (0, 1), (1, 0), (1, 1)}
        rep = sr.verify_certificate(nz, [0, -1], [0, -1], 2, 2)
        self.assertFalse(rep.valid)
        self.assertEqual(rep.structural_rank, 1)

    def test_verify_rejects_edge_not_nonzero(self):
        nz = {(0, 0)}
        with self.assertRaises(InvalidStructureError):
            sr.verify_certificate(nz, [1, -1], [-1, 0], 2, 2)

    def test_verify_rejects_inconsistent_pairing(self):
        nz = {(0, 0), (0, 1), (1, 0), (1, 1)}
        # mr 声称 0-0、1-1；mc 却声称 0-1、1-0，双向映射矛盾。
        with self.assertRaises(InvalidStructureError):
            sr.verify_certificate(nz, [0, 1], [1, 0], 2, 2)


class InputValidationTests(unittest.TestCase):
    def test_bool_dimensions_rejected(self):
        with self.assertRaises(InvalidStructureError):
            sr.analyze([], True, 2)
        with self.assertRaises(InvalidStructureError):
            sr.analyze([], 3, False)

    def test_bool_index_rejected(self):
        with self.assertRaises(InvalidStructureError):
            sr.analyze([(True, 0)], 2, 2)
        with self.assertRaises(InvalidStructureError):
            sr.analyze([(0, 1)], 2, 2, max_steps=True)

    def test_nan_infinity_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(InvalidStructureError):
                sr.analyze([(bad, 0)], 2, 2)
            with self.assertRaises(InvalidStructureError):
                sr.analyze([(0, bad)], 2, 2)

    def test_noninteger_float_index_rejected(self):
        with self.assertRaises(InvalidStructureError):
            sr.analyze([(0.5, 0)], 2, 2)

    def test_integer_float_index_accepted(self):
        r = sr.analyze([(0.0, 1.0)], 2, 2)
        self.assertEqual(r.structural_rank, 1)

    def test_out_of_range(self):
        with self.assertRaises(InvalidStructureError):
            sr.analyze([(0, 3)], 2, 2)
        with self.assertRaises(InvalidStructureError):
            sr.analyze([(2, 0)], 2, 2)
        with self.assertRaises(InvalidStructureError):
            sr.analyze([], 101, 2)
        with self.assertRaises(InvalidStructureError):
            sr.analyze([], 2, -1)
        with self.assertRaises(InvalidStructureError):
            sr.analyze([(100, 0)])

    def test_error_is_locatable(self):
        try:
            sr.analyze([(0, 0), (1, 9)], 2, 2)
        except InvalidStructureError as exc:
            self.assertIn("(1, 9)", str(exc))
        else:
            self.fail("应当拒绝越界位置")

    def test_rows_form_input(self):
        r = sr.analyze([[0, 2], [1], []])
        self.assertEqual((r.n_rows, r.n_cols), (3, 3))
        self.assertEqual(r.structural_rank, 2)

    def test_pair_must_be_tuple_not_list(self):
        # 二元 list 永远按“一行”解释：[[0, 1]] 是 1 行 2 列，不是坐标 (0,1)。
        r = sr.analyze([[0, 1]])
        self.assertEqual((r.n_rows, r.n_cols), (1, 2))
        self.assertEqual(r.structural_rank, 1)
        # 混合形式（首项决定坐标对，后续出现 list）必须拒绝，不静默曲解。
        with self.assertRaises(InvalidStructureError):
            sr.analyze([(0, 1), [2, 3]], 3, 3)

    def test_rows_form_rejects_n_rows(self):
        with self.assertRaises(InvalidStructureError):
            sr.analyze([[0], [1]], n_rows=2)

    def test_inference(self):
        r = sr.analyze({(0, 0), (2, 1)})
        self.assertEqual((r.n_rows, r.n_cols), (3, 2))
        r0 = sr.analyze(set())
        self.assertEqual((r0.n_rows, r0.n_cols), (0, 0))

    def test_bad_pair_shape(self):
        with self.assertRaises(InvalidStructureError):
            sr.analyze([(0, 1, 2)], 3, 3)
        with self.assertRaises(InvalidStructureError):
            sr.analyze([5], 3, 3)
        with self.assertRaises(InvalidStructureError):
            sr.analyze(None, 3, 3)


class InvariantAndBoundaryTests(unittest.TestCase):
    def test_partition_invariants_random(self):
        rng = random.Random(7)
        for _ in range(300):
            nr, nc = rng.randint(0, 6), rng.randint(0, 6)
            nz = {
                (r, c)
                for r in range(nr)
                for c in range(nc)
                if rng.random() < 0.4
            }
            r = sr.analyze(nz, nr, nc)
            reg = r.regions
            # 三区行/列各自两两不交且覆盖全部顶点。
            rows = reg.under_rows | reg.well_rows | reg.over_rows
            cols = reg.under_cols | reg.well_cols | reg.over_cols
            self.assertEqual(rows, set(range(nr)))
            self.assertEqual(cols, set(range(nc)))
            self.assertEqual(
                len(reg.under_rows) + len(reg.well_rows) + len(reg.over_rows), nr
            )
            self.assertEqual(
                len(reg.under_cols) + len(reg.well_cols) + len(reg.over_cols), nc
            )
            # Hall 计数：欠定区列多于行，过定区行多于列，良定区等大。
            self.assertGreaterEqual(len(reg.under_cols), len(reg.under_rows))
            self.assertGreaterEqual(len(reg.over_rows), len(reg.over_cols))
            self.assertEqual(len(reg.well_rows), len(reg.well_cols))
            # 未匹配顶点数量与结构秩一致。
            self.assertEqual(
                nr - r.structural_rank, len(reg.unmatched_rows)
            )
            self.assertEqual(
                nc - r.structural_rank, len(reg.unmatched_cols)
            )
            # 匹配边与独立枚举一致。
            self.assertEqual(
                r.structural_rank,
                ref.enumerate_max_matching_size(nz, nr, nc),
            )
            # 独立朴素 BFS 复核分区。
            got = (
                reg.under_rows, reg.under_cols,
                reg.well_rows, reg.well_cols,
                reg.over_rows, reg.over_cols,
            )
            self.assertEqual(
                got, ref.independent_dm_regions(nz, nr, nc, r.match_row)
            )
            assert_well_ordered(self, r, nz)

    def test_deterministic_repeat(self):
        nz = {(0, 1), (1, 0), (1, 1), (2, 2)}
        r1 = sr.analyze(nz, 3, 3)
        r2 = sr.analyze(list(nz), 3, 3)
        self.assertEqual(r1.row_perm, r2.row_perm)
        self.assertEqual(r1.col_perm, r2.col_perm)
        self.assertEqual(r1.match_row, r2.match_row)
        self.assertEqual(
            [tuple(sorted(b.rows)) for b in r1.btf.blocks],
            [tuple(sorted(b.rows)) for b in r2.btf.blocks],
        )

    def test_dimension_boundary_100(self):
        # 100x100 合法：对角匹配 + 覆盖全部顶点的有向环（i -> i+1 -> ... -> 0），
        # 匹配行有向图是一个强连通分量，故整块不可约。
        nz = {(i, i) for i in range(100)} | {
            (i, (i + 1) % 100) for i in range(100)
        }
        r = sr.analyze(nz, 100, 100)
        self.assertEqual(r.structural_rank, 100)
        self.assertEqual(r.btf.num_blocks, 1)
        self.assertEqual(r.btf.blocks[0].size, 100)
        assert_well_ordered(self, r, nz)

    def test_dimension_boundary_dense_100(self):
        # 稠密 100x100：全满图不可约（单块），增广路能即时完成。
        nz = {(r, c) for r in range(100) for c in range(100)}
        r = sr.analyze(nz, 100, 100)
        self.assertEqual(r.structural_rank, 100)
        self.assertEqual(r.btf.num_blocks, 1)

    def test_dimension_above_100_rejected(self):
        with self.assertRaises(InvalidStructureError):
            sr.analyze([], 101, 10)
        with self.assertRaises(InvalidStructureError):
            sr.analyze([(0, 100)], 1, 101)
        with self.assertRaises(InvalidStructureError):
            sr.analyze([(100, 0)])  # 推断维下下标 100 也越界

    def test_all_max_matchings_share_regions_random(self):
        # 随机小图：枚举全部最大匹配，DM 分区必须唯一（DM 分解的匹配无关性）。
        rng = random.Random(99)
        for _ in range(100):
            n = rng.randint(1, 4)
            nz = {
                (r, c)
                for r in range(n) for c in range(n)
                if rng.random() < 0.5
            }
            max_matchings = ref.enumerate_all_max_matchings(nz, n, n)
            partitions = {
                ref.independent_dm_regions(nz, n, n, m) for m in max_matchings
            }
            self.assertEqual(len(partitions), 1, msg=render(nz, n, n))


class BudgetTests(unittest.TestCase):
    def test_budget_exhaustion_is_distinct_from_rejection(self):
        # 稠密大图给极小预算，必然耗尽；异常类型与非法输入拒绝严格不同。
        nz = {(r, c) for r in range(20) for c in range(20)}
        with self.assertRaises(SearchBudgetExhaustedError) as ctx:
            sr.analyze(nz, 20, 20, max_steps=3)
        self.assertIsInstance(ctx.exception, RuntimeError)
        self.assertNotIsInstance(ctx.exception, ValueError)
        self.assertEqual(ctx.exception.budget, 3)

    def test_budget_sufficient_completes(self):
        nz = {(0, 0), (0, 1), (1, 0)}
        r = sr.analyze(nz, 2, 2, max_steps=10_000)
        self.assertEqual(r.structural_rank, 2)

    def test_budget_must_be_positive(self):
        with self.assertRaises(InvalidStructureError):
            sr.analyze({(0, 0)}, 1, 1, max_steps=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
