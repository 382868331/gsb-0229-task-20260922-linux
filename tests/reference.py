"""测试用独立参考实现（禁止导入 sparsestruct）。

与被测核心完全独立：
  * 最大匹配：回溯枚举*所有*匹配，取基数最大者（小矩阵专用）；
  * 区域：按 Dulmage–Mendelsohn 可达性定义独立重写；
  * 不可约块：在良定行有向图上用可达闭包（传递闭包）求互相可达等价类；
  * 块顺序：独立校验核心给出的顺序是合法拓扑序。

枚举法天然给出“证明”：所有候选都数过了，最大值即结构秩。
"""

from __future__ import annotations

from collections import deque


def enumerate_max_matching(edges, n_rows, n_cols):
    """回溯枚举全部匹配，返回 (row_match, col_match, size)。"""
    adj = [[] for _ in range(n_rows)]
    for r, c in edges:
        adj[r].append(c)
    for r in range(n_rows):
        adj[r] = sorted(set(adj[r]))

    best = {"size": -1, "rm": None, "cm": None}
    rm = [-1] * n_rows
    cm = [-1] * n_cols

    def rec(r, used_cols, size):
        if r == n_rows:
            if size > best["size"]:
                best["size"] = size
                best["rm"] = rm.copy()
                best["cm"] = cm.copy()
            return
        # 选项 1：行 r 不参与匹配。
        rec(r + 1, used_cols, size)
        # 选项 2：行 r 匹配某个未占用的邻接列。
        for c in adj[r]:
            if c not in used_cols:
                rm[r] = c
                cm[c] = r
                used_cols.add(c)
                rec(r + 1, used_cols, size + 1)
                used_cols.discard(c)
                cm[c] = -1
                rm[r] = -1

    rec(0, set(), 0)
    return best["rm"], best["cm"], best["size"]


def count_perfect_matchings(edges, n):
    """枚举 n×n 完美匹配个数（用于“多个匹配但同结构区”检查）。"""
    adj = [set() for _ in range(n)]
    for r, c in edges:
        adj[r].add(c)

    def rec(r, used):
        if r == n:
            return 1
        total = 0
        for c in adj[r]:
            if c not in used:
                used.add(c)
                total += rec(r + 1, used)
                used.discard(c)
        return total

    return rec(0, set())


def reference_regions(edges, rm, cm, n_rows, n_cols):
    """独立实现 DM 三区：返回 (under_rows, under_cols, well_rows,
    well_cols, over_rows, over_cols)，均为集合。"""
    adj = [set() for _ in range(n_rows)]
    radj = [set() for _ in range(n_cols)]
    for r, c in edges:
        adj[r].add(c)
        radj[c].add(r)

    over_r, over_c = set(), set()
    q = deque()
    for r in range(n_rows):
        if rm[r] == -1:
            over_r.add(r)  # 源点（未匹配行）本身属于过定区
            q.append(("r", r))
    while q:
        kind, v = q.popleft()
        if kind == "r":
            for c in adj[v] - {rm[v]}:
                if c not in over_c:
                    over_c.add(c)
                    q.append(("c", c))
        else:
            r2 = cm[v]
            if r2 != -1 and r2 not in over_r:
                over_r.add(r2)
                q.append(("r", r2))

    under_r, under_c = set(), set()
    q = deque()
    for c in range(n_cols):
        if cm[c] == -1:
            under_c.add(c)  # 源点（未匹配列）本身属于欠定区
            q.append(("c", c))
    while q:
        kind, v = q.popleft()
        if kind == "c":
            for r in radj[v] - {cm[v]}:
                if r not in under_r:
                    under_r.add(r)
                    q.append(("r", r))
        else:
            c2 = rm[v]
            if c2 != -1 and c2 not in under_c:
                under_c.add(c2)
                q.append(("c", c2))

    well_r = set(range(n_rows)) - over_r - under_r
    well_c = set(range(n_cols)) - over_c - under_c
    return under_r, under_c, well_r, well_c, over_r, over_c


def reference_well_blocks(edges, rm, cm, well_rows):
    """传递闭包求良定行有向图的 SCC，返回 {frozenset(块行)} 与可达闭包。

    边 r -> r' 当且仅当存在非零元 (r, c) 且 c 匹配到 r'。
    """
    well_rows = set(well_rows)
    succ = {r: set() for r in well_rows}
    for r, c in edges:
        if r in well_rows:
            rp = cm[c]
            if rp in well_rows:
                succ[r].add(rp)

    reach = {r: {r} | succ[r] for r in well_rows}
    changed = True
    while changed:
        changed = False
        for r in well_rows:
            new = set(reach[r])
            for w in list(reach[r]):
                new |= reach[w]
            if len(new) != len(reach[r]):
                reach[r] = new
                changed = True

    blocks: dict[frozenset[int], None] = {}
    for r in sorted(well_rows):
        comp = frozenset(w for w in well_rows if w in reach[r] and r in reach[w])
        blocks[comp] = None
    return set(blocks.keys()), reach
