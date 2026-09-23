"""核心组合算法：二部图最大匹配、Dulmage–Mendelsohn 分区、SCC 块三角化。

全部基于非零位置（0/1 模式），不做任何数值消元。
"""

from __future__ import annotations

from collections import deque

from .errors import SearchBudgetExhaustedError


def maximum_matching(
    row_adj: list[list[int]],
    n_rows: int,
    n_cols: int,
    max_steps: int | None = None,
) -> tuple[list[int], list[int], int]:
    """Kuhn 增广路算法求二部图最大基数匹配。

    行按升序处理，邻接列升序尝试，结果唯一确定。

    ``max_steps`` 限制搜索中检查的行→列边数：预算耗尽且尚未证明最优时
    抛出 :class:`SearchBudgetExhaustedError`（与“无解/拒绝”严格区分——
    二部图恒存在匹配，这里只是预算内未跑完最优性证明）。
    返回 (match_row, match_col, steps_used)。
    """
    match_row = [-1] * n_rows
    match_col = [-1] * n_cols
    steps = 0

    def augment(start: int) -> bool:
        # 迭代式 DFS 增广路；栈中保存 (row, 下一个待试探邻接下标)。
        seen_col = [False] * n_cols
        seen_row = [False] * n_rows
        seen_row[start] = True
        stack: list[tuple[int, int]] = [(start, 0)]
        parent_col: list[int] = [-1] * n_cols  # 发现列 c 的行
        while stack:
            r, idx = stack[-1]
            if idx >= len(row_adj[r]):
                stack.pop()
                continue
            c = row_adj[r][idx]
            stack[-1] = (r, idx + 1)
            steps_inc()
            if seen_col[c]:
                continue
            seen_col[c] = True
            parent_col[c] = r
            if match_col[c] == -1:
                # 找到增广路，沿路翻转：每个发现行改配发现列，
                # 其原匹配列交给路径上一行。
                cur_c = c
                while cur_c != -1:
                    pr = parent_col[cur_c]
                    prev_c = match_row[pr]
                    match_row[pr] = cur_c
                    match_col[cur_c] = pr
                    cur_c = prev_c
                return True
            nr = match_col[c]
            if not seen_row[nr]:
                seen_row[nr] = True
                stack.append((nr, 0))
        return False

    def steps_inc() -> None:
        nonlocal steps
        steps += 1
        if max_steps is not None and steps > max_steps:
            raise SearchBudgetExhaustedError(
                max_steps, steps, sum(c != -1 for c in match_col), match_row, match_col
            )

    for r in range(n_rows):
        augment(r)
    return match_row, match_col, steps


def alternating_reach_from_rows(
    start_rows: list[int],
    row_adj: list[list[int]],
    match_col: list[int],
) -> tuple[set[int], set[int]]:
    """从给定行出发沿交替路可达性（行→非匹配边→列→匹配边→行）。

    返回 (可达行集合, 可达列集合)，起点行包含在可达行中。
    """
    seen_rows = set(start_rows)
    seen_cols: set[int] = set()
    q: deque[int] = deque(start_rows)
    while q:
        r = q.popleft()
        for c in row_adj[r]:
            if c in seen_cols:
                continue
            seen_cols.add(c)
            nr = match_col[c]
            if nr != -1 and nr not in seen_rows:
                seen_rows.add(nr)
                q.append(nr)
    return seen_rows, seen_cols


def alternating_reach_from_cols(
    start_cols: list[int],
    col_adj: list[list[int]],
    match_row: list[int],
) -> tuple[set[int], set[int]]:
    """从给定列出发沿交替路（列→非匹配边→行→匹配边→列）。"""
    seen_cols = set(start_cols)
    seen_rows: set[int] = set()
    q: deque[int] = deque(start_cols)
    while q:
        c = q.popleft()
        for r in col_adj[c]:
            if r in seen_rows:
                continue
            seen_rows.add(r)
            nc = match_row[r]
            if nc != -1 and nc not in seen_cols:
                seen_cols.add(nc)
                q.append(nc)
    return seen_rows, seen_cols


def kosaraju_scc(vertices: list[int], edges: dict[int, list[int]]) -> list[list[int]]:
    """Kosaraju 强连通分量（两次 DFS，迭代式，确定性顺序）。

    ``edges`` 给出每个顶点的出边表（只需覆盖 ``vertices`` 内部边）。
    返回 SCC 列表，每个 SCC 为其顶点按升序排列的列表；
    SCC 列表本身的顺序由后续拓扑排序决定。
    """
    vset = set(vertices)
    rev: dict[int, list[int]] = {v: [] for v in vertices}
    adj: dict[int, list[int]] = {v: [] for v in vertices}
    for u in vertices:
        for w in edges.get(u, ()):  # w 是另一行（由匹配列映射得到）
            if w in vset:
                adj[u].append(w)
                rev[w].append(u)
    for v in adj.values():
        v.sort()
    for v in rev.values():
        v.sort()

    visited: set[int] = set()
    finish: list[int] = []
    for root in vertices:
        if root in visited:
            continue
        visited.add(root)
        stack: list[tuple[int, int]] = [(root, 0)]
        while stack:
            u, idx = stack[-1]
            if idx < len(adj[u]):
                w = adj[u][idx]
                stack[-1] = (u, idx + 1)
                if w not in visited:
                    visited.add(w)
                    stack.append((w, 0))
            else:
                finish.append(u)
                stack.pop()

    comp: dict[int, int] = {}
    sccs: list[list[int]] = []
    for root in reversed(finish):
        if root in comp:
            continue
        cid = len(sccs)
        members: list[int] = []
        comp[root] = cid
        stack2 = [root]
        while stack2:
            u = stack2.pop()
            members.append(u)
            for w in rev[u]:
                if w not in comp:
                    comp[w] = cid
                    stack2.append(w)
        sccs.append(sorted(members))
    return sccs


def topological_order(
    sccs: list[list[int]], edges: dict[int, list[int]], vertices: list[int]
) -> list[int]:
    """对 SCC 收缩图做 Kahn 拓扑排序，平局按 SCC 最小顶点下标打破。

    边 u→w（u,w 均为良定行）意味着 u 所在块排在 w 所在块之前，
    使块间非零元只出现在块对角线上方（严格块上三角）。
    """
    vset = set(vertices)
    comp_of = {v: i for i, scc in enumerate(sccs) for v in scc}
    k = len(sccs)
    indeg = [0] * k
    cadj: list[set[int]] = [set() for _ in range(k)]
    for u in vertices:
        for w in edges.get(u, ()):
            if w in vset:
                i, j = comp_of[u], comp_of[w]
                if i != j and j not in cadj[i]:
                    cadj[i].add(j)
                    indeg[j] += 1
    min_vertex = [min(s) for s in sccs]
    ready = sorted((i for i in range(k) if indeg[i] == 0), key=lambda i: min_vertex[i])
    order: list[int] = []
    while ready:
        i = ready.pop(0)
        order.append(i)
        for j in sorted(cadj[i], key=lambda x: min_vertex[x]):
            indeg[j] -= 1
            if indeg[j] == 0:
                # 插入并保持按最小顶点排序。
                lo, hi = 0, len(ready)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if min_vertex[ready[mid]] < min_vertex[j]:
                        lo = mid + 1
                    else:
                        hi = mid
                ready.insert(lo, j)
    if len(order) != k:
        raise AssertionError("SCC 收缩图存在环（内部错误）")
    return order
