"""测试专用独立参考：与被测核心完全独立的组合算法。

本模块不 import structrank 的任何求解代码，期望值全部用朴素但显然正确
的方法独立得到：

* :func:`enumerate_max_matching_size` —— 回溯枚举全部匹配取最大基数；
* :func:`tarjan_scc` —— 独立的 Tarjan SCC 实现；
* :func:`permuted_structure_check` —— 对置换后的非零模式做区域/块形检查；
* :func:`independent_dm_regions` —— 朴素交替路 BFS 重算 DM 三区。
"""

from __future__ import annotations


def enumerate_max_matching_size(nz, n_rows, n_cols):
    """枚举所有匹配，返回最大基数（O(n!) 朴素，仅用于小矩阵）。"""
    adj = [[] for _ in range(n_rows)]
    for r, c in nz:
        adj[r].append(c)
    best = 0

    def dfs(r, used_cols, size):
        nonlocal best
        if size + (n_rows - r) <= best:
            return
        if r == n_rows:
            best = max(best, size)
            return
        dfs(r + 1, used_cols, size)  # 行 r 不配对
        for c in adj[r]:
            if not used_cols[c]:
                used_cols[c] = True
                dfs(r + 1, used_cols, size + 1)
                used_cols[c] = False

    dfs(0, [False] * n_cols, 0)
    return best


def enumerate_all_max_matchings(nz, n_rows, n_cols):
    """枚举所有最大匹配，返回 match_row 列表（-1 表示未匹配）。"""
    target = enumerate_max_matching_size(nz, n_rows, n_cols)
    adj = [[] for _ in range(n_rows)]
    for r, c in nz:
        adj[r].append(c)
    results = []

    def dfs(r, used_cols, match):
        if r == n_rows:
            if sum(c != -1 for c in match) == target:
                results.append(tuple(match))
            return
        dfs(r + 1, used_cols, match)  # 先放不配对分支
        for c in adj[r]:
            if not used_cols[c]:
                used_cols[c] = True
                match[r] = c
                dfs(r + 1, used_cols, match)
                match[r] = -1
                used_cols[c] = False

    dfs(0, [False] * n_cols, [-1] * n_rows)
    return results


def tarjan_scc(vertices, succ):
    """独立 Tarjan 实现。succ(v) 返回 v 的出边可迭代对象。"""
    index = {}
    low = {}
    onstack = set()
    stack = []
    counter = [0]
    sccs = []

    def strong(v):
        index[v] = counter[0]
        low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        onstack.add(v)
        for w in succ(v):
            if w not in index:
                strong(w)
                low[v] = min(low[v], low[w])
            elif w in onstack:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                onstack.discard(w)
                comp.append(w)
                if w == v:
                    break
            sccs.append(frozenset(comp))

    for v in vertices:
        if v not in index:
            strong(v)
    return sccs


def independent_dm_regions(nz, n_rows, n_cols, match_row):
    """用最朴素的交替路 BFS 从 match_row 出发重算 DM 区（独立实现）。"""
    match_col = [-1] * n_cols
    for r, c in enumerate(match_row):
        if c != -1:
            match_col[c] = r
    radj = [[] for _ in range(n_rows)]
    cadj = [[] for _ in range(n_cols)]
    for r, c in nz:
        radj[r].append(c)
        cadj[c].append(r)

    # 未匹配行 ->(任意边) 列 ->(匹配边) 行
    seen_r, seen_c = set(), set()
    q = [r for r in range(n_rows) if match_row[r] == -1]
    seen_r.update(q)
    head = 0
    while head < len(q):
        r = q[head]
        head += 1
        for c in radj[r]:
            if c not in seen_c:
                seen_c.add(c)
                r2 = match_col[c]
                if r2 != -1 and r2 not in seen_r:
                    seen_r.add(r2)
                    q.append(r2)
    over_rows, over_cols = set(seen_r), set(seen_c)

    # 未匹配列 ->(任意边) 行 ->(匹配边) 列
    seen_r, seen_c = set(), set()
    q = [c for c in range(n_cols) if match_col[c] == -1]
    seen_c.update(q)
    head = 0
    while head < len(q):
        c = q[head]
        head += 1
        for r in cadj[c]:
            if r not in seen_r:
                seen_r.add(r)
                c2 = match_row[r]
                if c2 != -1 and c2 not in seen_c:
                    seen_c.add(c2)
                    q.append(c2)
    under_rows, under_cols = set(seen_r), set(seen_c)

    return (
        frozenset(under_rows), frozenset(under_cols),
        frozenset(set(range(n_rows)) - over_rows - under_rows),
        frozenset(set(range(n_cols)) - over_cols - under_cols),
        frozenset(over_rows), frozenset(over_cols),
    )


def permuted_structure_check(result, nz):
    """对 analyze 结果做独立的置换后结构检查，返回 (ok, problems)。

    检查：
    1. 置换是双射；
    2. 匹配证书的配对边均为非零元，对角线位置一一对应；
    3. 每条非零边在置换后的区域/块位置合法（DM 块形 + 严格块上三角）；
    4. 每个良定块在独立 Tarjan SCC 下确为单个 SCC（不可约）；
    5. block_bounds 与块实际占据位置一致。
    """
    problems = []
    nr, nc = result.n_rows, result.n_cols
    rp, cp = result.row_perm, result.col_perm
    if sorted(rp) != list(range(nr)):
        problems.append("row_perm 不是行双射")
    if sorted(cp) != list(range(nc)):
        problems.append("col_perm 不是列双射")

    rpos = {r: i for i, r in enumerate(rp)}
    cpos = {c: i for i, c in enumerate(cp)}
    reg = result.regions

    # 匹配边检查。矩形矩阵的欠定/过定条带行、列数不等：欠定条带内匹配对
    # 置换后位置相同（差值 0）；良定/过定条带共享固定偏移 delta。
    delta = len(reg.under_rows) - len(reg.under_cols)
    for r, c in enumerate(result.match_row):
        if c == -1:
            continue
        if (r, c) not in nz:
            problems.append(f"匹配边 ({r},{c}) 不是非零元")
        expected = 0 if r in reg.under_rows else delta
        if rpos[r] - cpos[c] != expected:
            problems.append(
                f"匹配对 ({r},{c}) 置换后未对齐（位置 {rpos[r]} vs {cpos[c]}，"
                f"期望差值 {expected}）"
            )

    def region_of_row(r):
        if r in reg.under_rows:
            return "U"
        if r in reg.well_rows:
            return "W"
        return "O"

    def region_of_col(c):
        if c in reg.under_cols:
            return "U"
        if c in reg.well_cols:
            return "W"
        return "O"

    # DM 粗分解块上三角（行序 U,W,O；列序 U,W,O）。由有向图可达性可证：
    # 过定区行的所有非零列必在过定区；欠定区列的所有非零行必在欠定区。
    # 故严格下三角三处 (W,U)/(O,U)/(O,W) 必为 0；右上角 (U,O) 允许非零。
    allowed = {
        ("U", "U"), ("U", "W"), ("U", "O"),
        ("W", "W"), ("W", "O"),
        ("O", "O"),
    }
    for r, c in nz:
        pair = (region_of_row(r), region_of_col(c))
        if pair not in allowed:
            problems.append(f"非零边 ({r},{c}) 落到非法区域组合 {pair}")

    # 块结构检查。
    row_block = {}
    col_block = {}
    for b in result.btf.blocks:
        for r in b.rows:
            row_block[r] = b.index
        for c in b.cols:
            col_block[c] = b.index
    bounds = result.block_bounds
    if [tuple(sorted(b.row_positions)) for b in result.btf.blocks] != [
        tuple(range(lo, hi)) for lo, hi, _, _ in bounds
    ]:
        problems.append("block_bounds 与块行位置不一致")
    if [tuple(sorted(b.col_positions)) for b in result.btf.blocks] != [
        tuple(range(clo, chi)) for _, _, clo, chi in bounds
    ]:
        problems.append("block_bounds 与块列位置不一致")

    for r, c in nz:
        if r in row_block and c in col_block:
            bi, bj = row_block[r], col_block[c]
            if bi != bj:
                # 严格块上三角：列块 bj 必须排在行块 bi 之后（位置区间不重叠
                # 且更靠后），块索引也必须严格更大。
                rlo, rhi, _, _ = bounds[bi]
                _, _, clo, chi = bounds[bj]
                if bj < bi or clo < rhi:
                    problems.append(
                        f"边 ({r},{c}) 违反块三角顺序：块 {bi}->块 {bj}"
                    )
        # 仅一端在良定区的边（欠定行->良定列、良定行->过定列）由上面的
        # 区域合法性检查覆盖，这里不误报。

    # 独立 Tarjan：每个良定块必须恰为一个 SCC（用良定区全部内部非零边建图）。
    well_rows = set(reg.well_rows)
    well_cols = set(reg.well_cols)
    succ_map = {r: [] for r in well_rows}
    for r, c in nz:
        if r in well_rows and c in well_cols:
            r2 = result.match_col[c]
            if r2 != r:
                succ_map[r].append(r2)
    sccs = tarjan_scc(sorted(well_rows), lambda v: succ_map[v])
    for b in result.btf.blocks:
        if b.rows not in sccs:
            problems.append(f"块 {b.index} 不是单个强连通分量（可约）")

    return (len(problems) == 0, problems)
