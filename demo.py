"""固定输入演示：稀疏矩阵的结构秩与块三角化。

运行：python demo.py（无需任何第三方依赖，秒级完成）

演示内容：
  1. 正常结果：一个 10x10 确定性稀疏矩阵，含
       过定区 O(3行2列) | 良定不可约 3x3 块 | 良定不可约 2x2 块 | 欠定区 U(2行3列)
     打印结构秩、匹配证书、行列置换、块边界、置换后块上三角散点图，
     并用 verify() 独立核验证书；
  2. 实际触发的拒绝边界：NaN 坐标（InvalidStructureError）；
  3. 搜索预算耗尽（SearchBudgetExhausted）——明确区别于“证明无解”。

所有数据由本文件内确定性代码生成，无网络、无 sleep、无预录结果。
"""

from __future__ import annotations

import time

from sparsestruct import (
    InvalidStructureError,
    SearchBudgetExhausted,
    analyze,
    verify,
)

REGION_CN = {"U": "欠定(U)", "W": "良定(W)", "O": "过定(O)"}


def build_demo_matrix():
    """10x10 确定性边集：四个互不连通（除一条上三角填充边）的部分。"""
    edges = []
    # 过定部分：3 行争 2 列（秩 2，1 个未匹配行）
    edges += [(0, 0), (1, 0), (1, 1), (2, 1)]
    # 良定不可约块 A：行 3,4,5 × 列 2,3,4，全满 K3,3（一个 SCC）
    edges += [(r, c) for r in (3, 4, 5) for c in (2, 3, 4)]
    # 良定不可约块 B：行 6,7 × 列 5,6，全满 K2,2（一个 SCC）
    edges += [(r, c) for r in (6, 7) for c in (5, 6)]
    # 欠定部分：2 行接 3 列（秩 2，1 个未匹配列）
    edges += [(8, 7), (8, 9), (9, 8), (9, 9)]
    # 一条块对角线上方的填充边：块 A 的行 3 -> 块 B 的列 5，
    # 置换后落在对角块上方，不改变块划分（边方向 A -> B）。
    edges += [(3, 5)]
    return edges


def render_spy_with_blocks(d):
    """渲染带块分隔线的置换后散点图。"""
    nz = d.permuted_nonzeros()
    row_cuts = sorted({b for (_, b) in d.row_blocks} - {d.n_rows})
    col_cuts = sorted({b for (_, b) in d.col_blocks} - {d.n_cols})

    def cell(i, j):
        return "#" if (i, j) in nz else "."

    lines = []
    for i in range(d.n_rows):
        row = "".join(cell(i, j) for j in range(d.n_cols))
        # 在列块边界插入竖线
        for k in sorted(col_cuts, reverse=True):
            row = row[:k] + "|" + row[k:]
        lines.append(row)
        if i + 1 in row_cuts:
            lines.append("-" * (d.n_cols + len(col_cuts)))
    return "\n".join(lines)


def show_normal_result(elapsed_limit=8.0):
    print("=" * 68)
    print("正常结果：10x10 稀疏矩阵结构分析")
    print("=" * 68)
    edges = build_demo_matrix()
    print(f"矩阵规模: 10 x 10；去重后非零位置数: {len(set(edges))}")

    t0 = time.perf_counter()
    d = analyze(edges, 10, 10)
    elapsed = time.perf_counter() - t0

    print(f"结构秩（最大匹配基数）: {d.structural_rank} / 10")
    print(f"未匹配行: {d.unmatched_rows}  未匹配列: {d.unmatched_cols}")
    print(f"行匹配证书 row_match: {d.row_match}")
    print(f"列匹配证书 col_match: {d.col_match}")
    print(f"行区域: {d.row_region}")
    print(f"列区域: {d.col_region}")
    print(f"行置换 row_perm: {d.row_perm}")
    print(f"列置换 col_perm: {d.col_perm}")

    print("\n对角块（置换后坐标，半开区间）:")
    for b in d.iter_blocks():
        kind = (
            "不可约块" if b.region == "W" else
            "欠定矩形块" if b.region == "U" else "过定矩形块"
        )
        print(
            f"  块{b.index}: {REGION_CN[b.region]} {kind} "
            f"行区间{b.row_start}:{b.row_end} 列区间{b.col_start}:{b.col_end} "
            f"形状{b.shape[0]}x{b.shape[1]} "
            f"原始行{b.rows} 原始列{b.cols}"
        )

    print("\n置换后块上三角形式（'#'=非零，'|'/'---'=块边界）:")
    print(render_spy_with_blocks(d))
    print("说明：两个良定对角块分别为 3x3 与 2x2 的不可约块；")
    print("      填充边 (原 3,5) 落在对角块上方，没有任何非零元落在下方。")

    t1 = time.perf_counter()
    verify(d)  # 独立重算核验：匹配最大性、块上三角、强连通不可约性
    verify_elapsed = time.perf_counter() - t1
    print(f"\nverify() 独立证书核验: 通过（分析耗时 {elapsed*1000:.2f} ms，"
          f"核验耗时 {verify_elapsed*1000:.2f} ms）")
    assert elapsed + verify_elapsed < elapsed_limit


def show_rejection_boundary():
    print("\n" + "=" * 68)
    print("异常边界 1：NaN 坐标必须被拒绝（不静默纠正）")
    print("=" * 68)
    bad_entries = [(0, 0), (1, float("nan"))]
    try:
        analyze(bad_entries, 2, 2)
    except InvalidStructureError as exc:
        print(f"实际捕获 {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("NaN 坐标未被拒绝")


def show_budget_boundary():
    print("\n" + "=" * 68)
    print("异常边界 2：搜索预算耗尽 != 证明无增广路")
    print("=" * 68)
    dense = [(r, c) for r in range(4) for c in range(4)]  # 4x4 全满
    try:
        analyze(dense, 4, 4, budget=2)
    except SearchBudgetExhausted as exc:
        print(f"budget=2 时实际捕获 {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("预算耗尽未触发")
    full = analyze(dense, 4, 4)
    print(f"同一输入完整搜索（budget=None）得到证明后的结构秩: "
          f"{full.structural_rank}（可见预算耗尽并不等于证明无解）")


def main():
    t0 = time.perf_counter()
    show_normal_result()
    show_rejection_boundary()
    show_budget_boundary()
    print("\n演示完成，总耗时 %.2f 秒。" % (time.perf_counter() - t0))


if __name__ == "__main__":
    main()
