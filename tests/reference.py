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


# --------------------------------------------------------------------------- #
# 输入选址独立参考（小图枚举，禁止导入 sparsestruct）
# --------------------------------------------------------------------------- #


def reference_source_sccs(edges, n):
    """独立求状态有向图的源 SCC。

    非零元 (i, j) 表示边 j -> i。用逐点 BFS 可达闭包判定互相可达
    （传递闭包法，与核心的 Kosaraju 完全独立），源 SCC = 无外部入边
    的分量。返回按最小顶点升序的分量元组（块内升序）。
    """
    succ = [set() for _ in range(n)]
    for i, j in edges:
        succ[j].add(i)
    reach = []
    for s in range(n):
        seen = {s}
        dq = deque([s])
        while dq:
            v = dq.popleft()
            for w in succ[v]:
                if w not in seen:
                    seen.add(w)
                    dq.append(w)
        reach.append(seen)
    comp_of = [None] * n
    comps = []
    for v in range(n):
        if comp_of[v] is not None:
            continue
        comp = tuple(sorted(w for w in range(n) if v in reach[w] and w in reach[v]))
        idx = len(comps)
        comps.append(comp)
        for w in comp:
            comp_of[w] = idx
    has_in = [False] * len(comps)
    for j in range(n):
        for i in succ[j]:
            if comp_of[i] != comp_of[j]:
                has_in[comp_of[i]] = True
    sources = [comps[k] for k in range(len(comps)) if not has_in[k]]
    return tuple(sorted(sources, key=lambda c: c[0]))


def reference_feasible_input_set(edges, n, states):
    """独立判定专用输入集合 states 是否使系统结构可控。

    两条独立条件：
      1. 可达性：每个源 SCC 至少含一个被选状态；
      2. 无膨胀：删去被选行后，剩余行可被完全匹配（用本文件独立的
         回溯枚举匹配判定，不依赖核心算法）。
    """
    states = frozenset(states)
    for comp in reference_source_sccs(edges, n):
        if states.isdisjoint(comp):
            return False
    kept = [(i, j) for (i, j) in edges if i not in states]
    _, _, size = enumerate_max_matching(kept, n, n)
    return size == n - len(states)


def reference_min_dedicated_inputs(edges, n):
    """枚举全部输入集合，返回 (最少个数, 字典序最小状态元组)。

    按规模从小到大、同规模按状态 ID 升序元组字典序枚举，第一个可行
    集合即答案——枚举本身就是“最少”的证明。
    """
    from itertools import combinations

    edges = sorted(set(edges))
    for k in range(n + 1):
        for cand in combinations(range(n), k):
            if reference_feasible_input_set(edges, n, cand):
                return k, cand
    raise AssertionError("全状态集合必然可行，不应到达这里")
