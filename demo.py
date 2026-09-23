"""固定输入演示：结构秩 + DM 三区 + 良定区不可约块三角化。

运行：python demo.py

所有数据在本文件内确定性给出，无网络、无随机种子依赖、无预录结果；
全部输出由 structrank 实时计算。约在 1 秒内完成。
"""

from __future__ import annotations

import time

import structrank as sr
from structrank.errors import InvalidStructureError, SearchBudgetExhaustedError


# 8x8 固定非零模式，刻意把四个结构成分的行列下标打散，
# 以展示置换确实在重排（不是直接读原图顺序）：
#   过定成分：行 {0,4,7} × 列 {1,6}（3 行 2 列，纵向）
#   欠定成分：行 {1,5} × 列 {0,3,4}（2 行 3 列，横向）
#   良定块 A：行 {2,6} × 列 {2,5}，全满 2x2（不可约）
#   良定块 B：行 {3} × 列 {7}，1x1
#   块间依赖边 (2,7)：A -> B，置换后落在块对角线上方
# 列表中故意含两个重复位置（(4,1) 与 (6,5)），analyze 必须按集合去重。
NONZERO_LIST = [
    (0, 1), (0, 6),
    (1, 0), (1, 3),
    (2, 2), (2, 5), (2, 7),
    (3, 7),
    (4, 1), (4, 1),          # 重复
    (5, 3), (5, 4),
    (6, 2), (6, 5), (6, 5),  # 重复
    (7, 6),
]
NONZEROS = set(NONZERO_LIST)
N_ROWS = N_COLS = 8


def render_pattern(nz, nr, nc):
    return "\n".join(
        f"  行{r}: " + "".join("#" if (r, c) in nz else "." for c in range(nc))
        for r in range(nr)
    )


def render_permuted(result):
    """按全局置换重排后打印，标注三区与块边界。"""
    nz = result.nonzeros
    rp, cp = result.row_perm, result.col_perm
    reg = result.regions

    def row_tag(r):
        if r in reg.under_rows:
            return "U"
        if r in reg.over_rows:
            return "O"
        return "W"

    lines = []
    for i, r in enumerate(rp):
        chars = []
        for j, c in enumerate(cp):
            chars.append("#" if (r, c) in nz else ".")
        lines.append(f"  [{row_tag(r)}] 新行{i:>2} (原行{r}): " + "".join(chars))
    head = "             新列: " + "".join(str(j % 10) for j in range(len(cp)))
    return head + "\n" + "\n".join(lines)


def main():
    t0 = time.perf_counter()

    print("=" * 68)
    print("演示一（正常结果）：8x8 稀疏结构秩与块三角化")
    print("=" * 68)
    print("输入非零模式（# 为结构非零，不读任何数值）：")
    print(render_pattern(NONZEROS, N_ROWS, N_COLS))
    print(f"非零元个数（输入 {len(NONZERO_LIST)} 项，重复位置去重后 {len(NONZEROS)} 个）")

    result = sr.analyze(NONZERO_LIST, N_ROWS, N_COLS)
    reg = result.regions

    print()
    print(f"结构秩（最大匹配基数）：{result.structural_rank}")
    print("匹配证书 match_row：", tuple(result.match_row))
    print("匹配证书 match_col：", tuple(result.match_col))
    report = sr.verify_certificate(
        NONZEROS, result.match_row, result.match_col, N_ROWS, N_COLS
    )
    print(
        "König 证书独立复核：valid="
        f"{report.valid}，最小顶点覆盖大小={len(report.vertex_cover_rows) + len(report.vertex_cover_cols)}"
        f"（覆盖行 {sorted(report.vertex_cover_rows)}，覆盖列 {sorted(report.vertex_cover_cols)}）"
    )

    print()
    print("Dulmage–Mendelsohn 三区（原下标）：")
    print(f"  欠定区：行 {sorted(reg.under_rows)}，列 {sorted(reg.under_cols)}")
    print(f"  良定区：行 {sorted(reg.well_rows)}，列 {sorted(reg.well_cols)}")
    print(f"  过定区：行 {sorted(reg.over_rows)}，列 {sorted(reg.over_cols)}")
    print(f"  未匹配行 {sorted(reg.unmatched_rows)}，未匹配列 {sorted(reg.unmatched_cols)}")

    print()
    print(f"良定区不可约块数：{result.btf.num_blocks}")
    for b in result.btf.blocks:
        print(
            f"  块 {b.index}：{b.size}x{b.size}，行 {sorted(b.rows)}，"
            f"列 {sorted(b.cols)}（1x1 平凡不可约；2x2 全满为强连通分量）"
        )
    print("块边界（行起,行止,列起,列止，半开）：", result.block_bounds)

    print()
    print("全局置换（新位置 -> 原下标）：")
    print("  row_perm：", result.row_perm)
    print("  col_perm：", result.col_perm)
    print()
    print("置换后结构（[U]欠定 [W]良定 [O]过定；块间非零元只在对角线上方）：")
    print(render_permuted(result))

    print()
    print("=" * 68)
    print("演示二（实际触发的拒绝边界）：非法数值下标必须被确定性拒绝")
    print("=" * 68)
    bad_input = {(0, 0), (1, float("nan")), (2, 2)}
    try:
        sr.analyze(bad_input, 3, 3)
    except InvalidStructureError as exc:
        print(f"已捕获 InvalidStructureError（ValueError 子类，拒绝而非搜索失败）：")
        print(f"  -> {exc}")
    else:  # pragma: no cover - 演示用，不应发生
        raise AssertionError("NaN 下标未被拒绝")

    print()
    print("另演示搜索预算语义（与拒绝严格分开）：稠密 20x20，max_steps=5")
    dense = {(r, c) for r in range(20) for c in range(20)}
    try:
        sr.analyze(dense, 20, 20, max_steps=5)
    except SearchBudgetExhaustedError as exc:
        print("已捕获 SearchBudgetExhaustedError（RuntimeError 子类）：")
        print(f"  -> {exc}")
        print("  含义：预算内未证明最优，不是“无解”，也不是非法输入；")
        print("  放宽预算重跑即可得到结构秩 20。")
    else:  # pragma: no cover
        raise AssertionError("预算耗尽未按契约抛出")
    rank20 = sr.structural_rank(dense, 20, 20)
    print(f"  不限预算重算结构秩 = {rank20}")

    print()
    print(f"演示完成，实际耗时 {time.perf_counter() - t0:.3f} 秒。")


if __name__ == "__main__":
    main()
