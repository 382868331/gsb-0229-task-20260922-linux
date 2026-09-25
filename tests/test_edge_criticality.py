"""边关键性（edge criticality）测试：独立删除试验与见证核验。

运行：python -m unittest discover -s tests -v
期望值来自 tests/reference.py 的逐边删除 + 回溯枚举（不导入被测核心）。
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sparsestruct import (  # noqa: E402
    StructuralError,
    analyze,
    edge_criticality,
    verify,
    verify_criticality,
)

from reference import reference_edge_criticality  # noqa: E402


def _check_against_reference(case, edges, n, m):
    d = analyze(edges, n, m)
    report = edge_criticality(d)
    case.assertTrue(verify(d))
    case.assertTrue(verify_criticality(d, report))
    ref = reference_edge_criticality(list(d.nonzeros), n, m)
    case.assertEqual(set(ref), set(d.nonzeros))
    for pos, (critical, rank_after) in ref.items():
        e = report.entry(*pos)
        case.assertEqual(e.critical, critical, f"位置 {pos} 关键性分类不符")
        case.assertEqual(
            e.rank_after_removal, rank_after, f"位置 {pos} 删后秩不符"
        )
        if critical:
            case.assertIsNotNone(e.reduced_matching)
            case.assertIsNotNone(e.vertex_cover_rows)
            case.assertIsNone(e.alternative_matching)
        else:
            case.assertIsNotNone(e.alternative_matching)
            case.assertIsNone(e.reduced_matching)
    return d, report


class PerfectMatchingCycleTests(unittest.TestCase):
    """完全匹配环：交错环上的匹配边全部非关键；纯对角匹配全部关键。"""

    def test_cycle_edges_none_critical(self):
        # 3x3 循环：完美匹配 + 交错环，删任一匹配边都有替代匹配。
        edges = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 0)]
        d, report = _check_against_reference(self, edges, 3, 3)
        self.assertEqual(d.structural_rank, 3)
        self.assertEqual(report.critical_positions, ())
        for e in report.entries:
            self.assertFalse(e.critical)
            self.assertNotIn(e.position, e.alternative_matching)

    def test_pure_diagonal_all_critical(self):
        # 纯置换矩阵：每个匹配边都是唯一支撑，删除即降秩。
        edges = [(i, i) for i in range(4)]
        d, report = _check_against_reference(self, edges, 4, 4)
        self.assertEqual(d.structural_rank, 4)
        self.assertEqual(report.critical_positions, tuple(edges))
        for e in report.entries:
            self.assertEqual(e.rank_after_removal, 3)
            self.assertEqual(
                len(e.vertex_cover_rows) + len(e.vertex_cover_cols), 3
            )


class SingleBridgeMatchingTests(unittest.TestCase):
    """只有一条桥接匹配：恰好一条匹配边是降秩桥，其余有替代。"""

    def test_exactly_one_critical_edge(self):
        # 左部 K2,2（行0,1 × 列0,1）经桥边 (1,2) 接行2？改为：
        # 行0,1 与列0,1 全满（内部可互相替代），行1 另接列2，
        # 列3 只被行2 接，行2 只接列3 -> (2,3) 是唯一桥接匹配。
        edges = [
            (0, 0), (0, 1), (1, 0), (1, 1), (1, 2),
            (2, 3),
        ]
        d, report = _check_against_reference(self, edges, 3, 4)
        self.assertEqual(d.structural_rank, 3)
        self.assertEqual(report.critical_positions, ((2, 3),))
        e = report.entry(2, 3)
        self.assertEqual(e.rank_after_removal, 2)
        # 顶点覆盖证据确实覆盖删除后图的所有边
        cover_r = set(e.vertex_cover_rows)
        cover_c = set(e.vertex_cover_cols)
        for r, c in d.nonzeros - {(2, 3)}:
            self.assertTrue(r in cover_r or c in cover_c)

    def test_matched_but_not_essential(self):
        # 当前匹配使用的边未必必不可少：K2,2 中每条匹配边都有替代。
        edges = [(0, 0), (0, 1), (1, 0), (1, 1)]
        d, report = _check_against_reference(self, edges, 2, 2)
        self.assertEqual(report.critical_positions, ())
        for e in report.entries:
            if e.in_current_matching:
                self.assertFalse(e.critical)
                self.assertNotIn(e.position, e.alternative_matching)


class RectangularUnderdeterminedTests(unittest.TestCase):
    """矩形欠定图：列多于行，未匹配列存在时的边关键性。"""

    def test_underdetermined_2x3(self):
        edges = [(0, 0), (0, 2), (1, 1), (1, 2)]
        d, report = _check_against_reference(self, edges, 2, 3)
        self.assertEqual(d.structural_rank, 2)
        self.assertEqual(len(d.unmatched_cols), 1)

    def test_underdetermined_with_bridge(self):
        # 行0 只接列0 -> (0,0) 是桥；行1 接列1,2 欠定。
        edges = [(0, 0), (1, 1), (1, 2)]
        d, report = _check_against_reference(self, edges, 2, 3)
        self.assertEqual(report.critical_positions, ((0, 0),))


class ZeroRankTests(unittest.TestCase):
    """结构秩零：无非零位置时报告为空。"""

    def test_empty_structure(self):
        for n, m in ((0, 0), (0, 3), (3, 0), (3, 3)):
            with self.subTest(shape=(n, m)):
                d = analyze([], n, m)
                self.assertEqual(d.structural_rank, 0)
                report = edge_criticality(d)
                self.assertEqual(report.entries, ())
                self.assertEqual(report.critical_positions, ())
                self.assertTrue(verify_criticality(d, report))


class IndependentTrialTests(unittest.TestCase):
    """多个删除是独立试验：报告内各条目的证据都基于原图删一条边。"""

    def test_witnesses_based_on_original_graph(self):
        edges = [(i, i) for i in range(3)] + [(0, 1)]
        d = analyze(edges, 3, 3)
        report = edge_criticality(d)
        self.assertTrue(verify_criticality(d, report))
        for e in report.entries:
            witness = (
                e.alternative_matching
                if not e.critical
                else e.reduced_matching
            )
            # 每条见证匹配只用原图删除该位置后的边，不叠加其他删除。
            for r, c in witness:
                self.assertIn((r, c), d.nonzeros)
                self.assertNotEqual((r, c), e.position)


class RandomSmallCrossCheckTests(unittest.TestCase):
    """确定性伪随机小图：逐边删除枚举核对分类与见证。"""

    def test_random_small_against_enumeration(self):
        seed = 987654321

        def lcg():
            nonlocal seed
            seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
            return seed

        for trial in range(80):
            n = lcg() % 5
            m = lcg() % 5
            p = (lcg() % 60) / 100.0 + 0.05
            edges = []
            for r in range(n):
                for c in range(m):
                    if (lcg() % 1000) / 1000.0 < p:
                        edges.append((r, c))
            with self.subTest(trial=trial, shape=(n, m), nedges=len(edges)):
                _check_against_reference(self, edges, n, m)


class ReportInterfaceTests(unittest.TestCase):
    """报告接口与错误语义。"""

    def test_entry_lookup_and_keyerror(self):
        edges = [(0, 0), (1, 1)]
        report = edge_criticality(analyze(edges, 2, 2))
        self.assertEqual(report.entry(0, 0).position, (0, 0))
        with self.assertRaises(KeyError):
            report.entry(0, 1)

    def test_tampered_report_rejected(self):
        edges = [(0, 0), (0, 1), (1, 0), (1, 1)]
        d = analyze(edges, 2, 2)
        report = edge_criticality(d)
        self.assertTrue(verify_criticality(d, report))
        # 篡改：把非关键改成关键（缺证据）必须被检出。
        bad_entry = type(report.entries[0])(
            position=report.entries[0].position,
            critical=True,
            rank_after_removal=1,
            in_current_matching=True,
        )
        bad = type(report)(
            n_rows=2, n_cols=2, structural_rank=2,
            entries=(bad_entry,) + report.entries[1:],
        )
        with self.assertRaises(StructuralError):
            verify_criticality(d, bad)


if __name__ == "__main__":
    unittest.main()

