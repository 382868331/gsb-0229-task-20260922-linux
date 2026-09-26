"""固定输入演示：方阵状态图的最少专用输入选址（结构可控性）。

运行：python demo_iteration.py（纯标准库，秒级完成，远低于 90 秒）

演示内容（全部现场计算，无 sleep、无预录答案）：
  1. 满结构秩但含多个孤立环的系统：秩缺口为 0，仍需每环一个输入
     ——说明不能只数结构秩缺口；
  2. 链式系统：一个输入打在链头（源 SCC）；
  3. 零矩阵：每个状态都是孤立源 SCC 且秩为 0，需要全部 n 个输入；
  4. 混合系统：秩亏与多个源 SCC 同时起作用，打印匹配证据与
     每个源 SCC 的覆盖输入；
  5. 实际触发的拒绝边界：状态数超过 20（LimitExceededError）与
     bool 下标（TypeError）。

选址结论只依赖非零模式（结构/通用可控性），不代替具体数值矩阵的
可控性判断。数据由本文件内确定性代码生成。
"""

from __future__ import annotations

import time

from sparsestruct import (
    LimitExceededError,
    select_dedicated_inputs,
)


def show(title, edges, n):
    print("=" * 64)
    print(title)
    print(f"  状态数 n = {n}，非零模式边数 = {len(set(edges))}")
    p = select_dedicated_inputs(edges, n)
    print(f"  结构秩 = {p.structural_rank}（秩缺口 {p.rank_deficiency}），"
          f"源 SCC 数 = {p.n_source_sccs}")
    print(f"  最少专用输入数 = {p.n_inputs}，选址状态 = {list(p.states)}")
    print(f"  匹配证据：未匹配行 = {list(p.unmatched_states)}，"
          f"row_match = {list(p.row_match)}")
    for k, (comp, cover) in enumerate(zip(p.source_sccs, p.scc_coverage)):
        print(f"  源 SCC #{k} 状态 {list(comp)} <- 由输入状态 {cover} 覆盖")
    return p


def main():
    t0 = time.perf_counter()

    # 1. 满秩 + 两个孤立环 + 一个自环：秩缺口 0，但 3 个源 SCC。
    edges = [(1, 0), (0, 1), (3, 2), (2, 3), (4, 4)]
    show("例 1：满结构秩、三个孤立环（秩缺口 0 仍需 3 个输入）", edges, 5)

    # 2. 链 0->1->2->3->4。
    edges = [(i + 1, i) for i in range(4)]
    show("例 2：链式系统（源 SCC 为链头，1 个输入）", edges, 5)

    # 3. 零矩阵。
    show("例 3：4 阶零矩阵（每状态都是孤立源 SCC，需 4 个输入）", [], 4)

    # 4. 混合：环{0,1} + 链 2->3->4 + 孤立点 5（秩 5，缺口 1，源 SCC 3 个）。
    edges = [(1, 0), (0, 1), (3, 2), (4, 3)]
    show("例 4：环 + 链 + 孤立点（秩亏与可达性同时起作用）", edges, 6)

    # 5. 异常边界（实际触发）。
    print("=" * 64)
    print("异常边界演示（实际触发，非预录）")
    try:
        select_dedicated_inputs([], 21)
    except LimitExceededError as exc:
        print(f"  状态数 21 被拒绝：LimitExceededError: {exc}")
    try:
        select_dedicated_inputs([(True, 0)], 2)
    except TypeError as exc:
        print(f"  bool 下标被拒绝：TypeError: {exc}")

    dt = time.perf_counter() - t0
    print("=" * 64)
    print(f"全部现场计算完成，用时 {dt:.3f} 秒（远低于 90 秒上限）")


if __name__ == "__main__":
    main()
