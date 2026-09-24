"""固定输入演示：稀疏矩阵的结构秩与块三角化（静态 + 动态批量更新）。

运行：python demo.py（无需任何第三方依赖，秒级完成）

演示内容：
  1. 正常结果（静态）：一个 10x10 确定性稀疏矩阵，含
       过定区 O(3行2列) | 良定不可约 3x3 块 | 良定不可约 2x2 块 | 欠定区 U(2行3列)
     打印结构秩、匹配证书、行列置换、块边界、置换后块上三角散点图，
     并用 verify() 独立核验证书；
  2. 正常结果（动态 StructuralModel）：空矩阵起步，批量加边成对角阵、
     再补成不可约块；删除匹配边触发秩下降与欠定/过定重分区；重新加边
     并自动扩展维度（空行空列保留）；全量重算结果与静态 analyze 逐字段一致；
  3. 实际触发的拒绝边界：NaN 坐标（InvalidStructureError）；
  4. 动态非法更新：批次内同一位置既增又删，报错且状态不半更新；
  5. 搜索预算耗尽（SearchBudgetExhausted）——明确区别于“证明无解”，
     动态更新中途预算耗尽同样不留下半更新状态。

所有数据由本文件内确定性代码生成，无网络、无 sleep、无预录结果。
"""

from __future__ import annotations

import time

from sparsestruct import (
    InvalidStructureError,
    SearchBudgetExhausted,
    StructuralModel,
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


def show_dynamic_result():
    print("\n" + "=" * 68)
    print("动态：StructuralModel 批量增删 + 全量重算（3x3 -> 5x5）")
    print("=" * 68)
    # 空的 3x3（显式维度）：U 0x3 + O 3x0
    model = StructuralModel([], 3, 3)
    print("初始空 3x3：", _block_summary(model.decomposition))

    # 1) 批量加边成对角阵：0 -> 3
    rep = model.add_edges([(0, 0), (1, 1), (2, 2)])
    print(f"批量加对角线 3 条：秩 {rep.old_rank} -> {rep.new_rank}；",
          _block_summary(model.decomposition))

    # 2) 批量添三条循环闭合边使整体成为一个强连通（不可约）块
    rep = model.update(add=[(0, 1), (1, 2), (2, 0)])
    print("再补三条循环闭合边 (0,1),(1,2),(2,0)：",
          f"秩 {rep.new_rank}；", _block_summary(model.decomposition))
    print("  （对角 1x1 块合并为一个 3x3 良定不可约块）")

    # 3) 一个批次同时删掉三条闭合边与匹配边 (2,2)：秩 3 -> 2，
    #    行 2 变孤立入过定区、列 2 变孤立入欠定区
    rep = model.update(remove=[(0, 1), (1, 2), (2, 0), (2, 2)])
    d = model.decomposition
    print("批量删除三条闭合边与匹配边 (2,2)："
          f"秩 {rep.old_rank} -> {rep.new_rank}")
    print(f"  实际删除 {len(rep.removed)} 条；"
          f"未匹配行 {d.unmatched_rows}（必在 O 区），"
          f"未匹配列 {d.unmatched_cols}（必在 U 区）")
    print(f"  行区域 {d.row_region}  列区域 {d.col_region}")
    print("  " + _block_summary(d))

    # 4) 重新加回 (2,2)，并顺带加一条远处的 (4,4)：维度自动扩到 5x5
    rep = model.update(add=[(2, 2), (4, 4)])
    print(f"加回 (2,2) 并新增远处 (4,4)：秩 {rep.old_rank} -> {rep.new_rank}；"
          f"维度自动扩展为 {model.n_rows}x{model.n_cols}")
    print("  " + _block_summary(model.decomposition))
    assert model.n_rows == 5 and model.n_cols == 5
    assert (4, 4) in model

    # 5) 全量重算与静态入口逐字段一致
    static = analyze(sorted(model.edges()), model.n_rows, model.n_cols)
    assert model.decomposition == static, "动态重算与静态 analyze 不一致"
    assert verify(model.decomposition)
    # 删边不缩维：把所有边删光，5x5 的空行空列仍保留
    all_edges = sorted(model.edges())
    rep = model.remove_edges(all_edges)
    assert (model.n_rows, model.n_cols) == (5, 5)
    print(f"批量删光全部 {len(all_edges)} 条边：秩 {rep.old_rank} -> 0；"
          f"维度不收缩，仍为 {model.n_rows}x{model.n_cols}")
    print("  " + _block_summary(model.decomposition))
    print("动态结果已通过 verify() 核验，并与静态 analyze 逐字段一致。")


def _block_summary(d):
    parts = []
    for b in d.iter_blocks():
        tag = {"U": "U欠定", "W": "W良定", "O": "O过定"}[b.region]
        parts.append(f"{tag}{b.shape[0]}x{b.shape[1]}")
    return "块序列: " + " | ".join(parts)


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


def show_dynamic_illegal_update():
    print("\n" + "=" * 68)
    print("异常边界 2：非法批量更新报错且不留半更新状态")
    print("=" * 68)
    model = StructuralModel([(0, 0), (1, 1)], 2, 2)
    edges_before = model.edges()
    rank_before = model.decomposition.structural_rank
    # 同一位置在同一批次里既增又删：歧义
    try:
        model.update(add=[(0, 1), (1, 1)], remove=[(1, 1)])
    except InvalidStructureError as exc:
        print(f"实际捕获 {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("歧义更新未被拒绝")
    assert model.edges() == edges_before
    assert model.decomposition.structural_rank == rank_before
    print(f"报错后状态未变：边集仍为 {sorted(model.edges())}，"
          f"秩仍为 {model.decomposition.structural_rank}")

    # 固定维度下越界添加同样原子失败
    fixed = StructuralModel([(0, 0)], 2, 2, auto_dim=False)
    try:
        fixed.add_edges([(2, 0)])
    except InvalidStructureError as exc:
        print(f"固定维度越界实际捕获 {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("越界添加未被拒绝")
    assert fixed.edges() == frozenset({(0, 0)})


def show_budget_boundary():
    print("\n" + "=" * 68)
    print("异常边界 3：搜索预算耗尽 != 证明无增广路")
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

    # 动态路径上预算耗尽：旧状态完整保留
    model = StructuralModel(dense, 4, 4)
    before = model.edges()
    try:
        model.remove_edges([(0, 0)], budget=2)
    except SearchBudgetExhausted as exc:
        print(f"动态更新 budget=2 实际捕获 {type(exc).__name__}，"
              f"边集条数仍为 {len(model.edges())}（应为 {len(before)}），"
              f"秩仍为 {model.decomposition.structural_rank}")
    else:
        raise AssertionError("动态预算耗尽未触发")
    assert model.edges() == before
    rep = model.remove_edges([(0, 0)], budget=None)
    print(f"预算给足后删除成功：秩 {rep.old_rank} -> {rep.new_rank}（未下降，"
          "K4,4 去掉一条边仍存在完美匹配）")


def main():
    t0 = time.perf_counter()
    show_normal_result()
    show_dynamic_result()
    show_rejection_boundary()
    show_dynamic_illegal_update()
    show_budget_boundary()
    print("\n演示完成，总耗时 %.2f 秒。" % (time.perf_counter() - t0))


if __name__ == "__main__":
    main()
