"""固定输入演示：稀疏系统的最少专用输入选址（结构可控性）。

运行：python demo_iteration.py（纯标准库，现场计算，远低于 90 秒）

演示内容：
  1. 正常结果：9 状态混合系统（两个孤立环 + 一条链 + 一个孤立状态），
     打印所选状态、匹配证据（A 自身最大匹配 + 增广匹配行分配）以及
     每个源 SCC 由哪个输入覆盖；并用 analyze() 展示同一非零模式的
     原矩阵分块输出（保持不变的旧功能）；
  2. 退化情形：4x4 零矩阵（每个状态都必须各配一个输入）；
  3. 实际触发的异常边界：状态数超过 20（LimitExceededError）、
     NaN 坐标（InvalidStructureError）。

所有数据由本文件内确定性代码生成，现场枚举求解，无网络、无 sleep、
无预录结果。结论是*结构*可控性意义下的选址，不代替具体数值矩阵的
可控性判断。
"""

from __future__ import annotations

import time

from sparsestruct import (
    InvalidStructureError,
    LimitExceededError,
    analyze,
    minimum_dedicated_inputs,
    verify,
)


def build_demo_system():
    """9 状态确定性系统：环 {0,1}、环 {2,3,4}、链 5->6->7、孤立状态 8。"""
    edges = [
        (0, 1), (1, 0),              # 环 A：0 <-> 1
        (2, 3), (3, 4), (4, 2),      # 环 B：2 -> 3 -> 4 -> 2
        (6, 5), (7, 6),              # 链：5 -> 6 -> 7
        # 状态 8 无任何非零元（零行零列，孤立状态）
    ]
    return edges, 9


def show_placement(edges, n):
    print("=" * 68)
    print(f"正常结果：{n} 状态混合系统的最少专用输入选址")
    print("=" * 68)
    print(f"非零位置（(i,j) 表示状态边 j -> i）: {sorted(set(edges))}")

    t0 = time.perf_counter()
    p = minimum_dedicated_inputs(edges, n)
    elapsed = time.perf_counter() - t0

    print(f"状态数: {p.n_states}；A 的结构秩: {p.structural_rank} / {n} "
          f"（秩缺口 {p.rank_deficiency}）")
    print(f"A 自身最大匹配 col_match: {p.col_match}")
    print(f"A 自身未匹配状态列: {p.unmatched_states}")
    print(f"所选状态（每个状态配一个专用输入）: {p.selected_states}")
    print(f"输入个数: {len(p.selected_states)}")
    print("增广匹配证据 row_assign（行 -> 状态列；-1 = 该行由专用输入覆盖）:")
    print(f"  {p.row_assign}")
    print("源强连通分量及其覆盖输入:")
    for comp, driver in zip(p.source_sccs, p.scc_inputs):
        print(f"  源 SCC {comp} <- 输入作用于状态 {driver}")
    print(f"（现场枚举求解耗时 {elapsed*1000:.2f} ms）")
    return p


def show_original_blocks(edges, n):
    print("\n" + "-" * 68)
    print("同一非零模式的原矩阵分块输出（analyze，旧功能保持不变）")
    print("-" * 68)
    d = analyze(edges, n, n)
    verify(d)
    print(f"结构秩: {d.structural_rank} / {n}；块区域序列: {d.block_regions}")
    for b in d.iter_blocks():
        print(f"  块{b.index}: 区域 {b.region} 形状 "
              f"{b.shape[0]}x{b.shape[1]} 原始行{b.rows} 原始列{b.cols}")
    print("verify() 独立证书核验: 通过")


def show_zero_matrix():
    print("\n" + "=" * 68)
    print("退化情形：4x4 零矩阵（无任何边，每个状态都是孤立源 SCC）")
    print("=" * 68)
    p = minimum_dedicated_inputs([], 4)
    print(f"结构秩: {p.structural_rank}；所选状态: {p.selected_states}")
    print(f"源 SCC: {p.source_sccs}；覆盖输入: {p.scc_inputs}")


def show_limit_boundary():
    print("\n" + "=" * 68)
    print("异常边界 1：状态数超过 20 的接口上限")
    print("=" * 68)
    try:
        minimum_dedicated_inputs([], 21)
    except LimitExceededError as exc:
        print(f"实际捕获 {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("超上限状态数未被拒绝")


def show_nan_boundary():
    print("\n" + "=" * 68)
    print("异常边界 2：NaN 坐标必须被拒绝（不静默纠正）")
    print("=" * 68)
    try:
        minimum_dedicated_inputs([(0, 0), (1, float("nan"))], 2)
    except InvalidStructureError as exc:
        print(f"实际捕获 {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("NaN 坐标未被拒绝")


def main():
    t0 = time.perf_counter()
    edges, n = build_demo_system()
    show_placement(edges, n)
    show_original_blocks(edges, n)
    show_zero_matrix()
    show_limit_boundary()
    show_nan_boundary()
    total = time.perf_counter() - t0
    print(f"\n演示完成，总耗时 {total:.2f} 秒（现场计算，限制 90 秒）。")
    assert total < 90.0


if __name__ == "__main__":
    main()
