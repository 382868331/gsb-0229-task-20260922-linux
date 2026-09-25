"""稀疏矩阵的结构秩与块三角化（纯组合结构分析，不做任何数值消元）。

公开接口见 :func:`analyze` 与 :class:`StructuralDecomposition`。
本模块只读取非零元的*位置*，从不读取或推断数值，因此结构结论是精确的
离散结论，不涉及浮点误差。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Optional

__all__ = [
    "MAX_ROWS",
    "MAX_COLS",
    "StructuralError",
    "InvalidStructureError",
    "LimitExceededError",
    "SearchBudgetExhausted",
    "Block",
    "StructuralDecomposition",
    "EdgeCriticality",
    "EdgeCriticalityReport",
    "analyze",
    "verify",
    "edge_criticality",
    "verify_criticality",
]

#: 接口允许的最大行数（题面给定的硬边界）。
MAX_ROWS = 100
#: 接口允许的最大列数（题面给定的硬边界）。
MAX_COLS = 100

_U = "U"  # underdetermined 欠定区（列多于行）
_W = "W"  # well-determined 良定区（方阵块）
_O = "O"  # overdetermined 过定区（行多于列）


class StructuralError(ValueError):
    """结构分析相关错误的基类。"""


class InvalidStructureError(StructuralError):
    """输入结构非法（越界、负数、NaN/Infinity、错误坐标个数等）。"""


class LimitExceededError(StructuralError):
    """触发接口边界（维度上限等）。"""


class SearchBudgetExhausted(LimitExceededError):
    """交替路径搜索预算耗尽。

    注意：这*不*表示证明了不存在增广路（即不表示当前匹配已最大），
    只是未能在预算内完成搜索。它与“已证明无增广路”必须区分。
    """


@dataclass(frozen=True)
class Block:
    """块三角形式中的一个对角块（置换后坐标）。

    Attributes:
        index: 块编号，按置换后的先后顺序（0 起）。
        region: ``'U'`` 欠定 / ``'W'`` 良定 / ``'O'`` 过定。
        row_start, row_end: 该块在置换后行序中的半开区间。
        col_start, col_end: 该块在置换后列序中的半开区间。
        rows: 该块包含的*原始*行号（按置换顺序）。
        cols: 该块包含的*原始*列号（按置换顺序）。
    """

    index: int
    region: str
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    rows: tuple[int, ...]
    cols: tuple[int, ...]

    @property
    def shape(self) -> tuple[int, int]:
        """块形状 ``(行数, 列数)``。"""
        return (self.row_end - self.row_start, self.col_end - self.col_start)


@dataclass(frozen=True)
class StructuralDecomposition:
    """结构分析结果证书。

    所有字段均为不可变的确定性结果；给定相同输入（位置集合，重复位置
    自动去重）输出唯一，与传入顺序无关。

    Attributes:
        n_rows / n_cols: 矩阵维度。
        nonzeros: 去重后的非零位置集合 ``{(row, col), ...}``。
        structural_rank: 结构秩（最大匹配基数）。
        row_match: ``row_match[r]`` 为行 r 匹配的列，未匹配为 -1。
        col_match: ``col_match[c]`` 为列 c 匹配的行，未匹配为 -1。
        unmatched_rows / unmatched_cols: 未匹配行、列（升序元组）。
        row_perm: 行置换，``row_perm[i]`` 是置换后第 i 行的原始行号。
        col_perm: 列置换，``col_perm[j]`` 是置换后第 j 列的原始列号。
        row_blocks / col_blocks: 对角块在置换后坐标上的半开区间。
        block_regions: 每个对角块的区域标签（与块编号对齐）。
        row_region / col_region: 每个*原始*行/列所属区域标签。
    """

    n_rows: int
    n_cols: int
    nonzeros: frozenset[tuple[int, int]]
    structural_rank: int
    row_match: tuple[int, ...]
    col_match: tuple[int, ...]
    unmatched_rows: tuple[int, ...]
    unmatched_cols: tuple[int, ...]
    row_perm: tuple[int, ...]
    col_perm: tuple[int, ...]
    row_blocks: tuple[tuple[int, int], ...]
    col_blocks: tuple[tuple[int, int], ...]
    block_regions: tuple[str, ...]
    row_region: tuple[str, ...]
    col_region: tuple[str, ...]

    def iter_blocks(self) -> tuple[Block, ...]:
        """按块编号返回全部对角块（含区域标签与原始行列号）。"""
        out = []
        for k, (region, (rs, re_), (cs, ce)) in enumerate(
            zip(self.block_regions, self.row_blocks, self.col_blocks)
        ):
            out.append(
                Block(
                    index=k,
                    region=region,
                    row_start=rs,
                    row_end=re_,
                    col_start=cs,
                    col_end=ce,
                    rows=tuple(self.row_perm[rs:re_]),
                    cols=tuple(self.col_perm[cs:ce]),
                )
            )
        return tuple(out)

    def permuted_nonzeros(self) -> frozenset[tuple[int, int]]:
        """把非零位置换算到置换后坐标，返回 ``{(i, j), ...}``。"""
        rp = {r: i for i, r in enumerate(self.row_perm)}
        cp = {c: j for j, c in enumerate(self.col_perm)}
        return frozenset((rp[r], cp[c]) for (r, c) in self.nonzeros)

    def spy(self, filled: str = "#", empty: str = ".") -> str:
        """返回置换后矩阵的 ASCII 散点图（不含块分隔线）。"""
        nz = self.permuted_nonzeros()
        lines = []
        for i in range(self.n_rows):
            lines.append(
                "".join(filled if (i, j) in nz else empty for j in range(self.n_cols))
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 输入校验
# --------------------------------------------------------------------------- #


def _as_index(value, where: str) -> int:
    """校验单个行列下标：必须是 int（明确拒绝 bool），有限、非负。"""
    if isinstance(value, bool):
        raise TypeError(f"{where}: bool 不能用作整数下标")
    if isinstance(value, float):
        # 契约：数值输入拒绝 NaN/Infinity；其余浮点也不接受为下标。
        if not math.isfinite(value):
            raise InvalidStructureError(f"{where}: 拒绝 NaN/Infinity")
        raise TypeError(f"{where}: 下标必须为 int，收到 float {value!r}")
    if not isinstance(value, int):
        raise TypeError(f"{where}: 下标必须为 int，收到 {type(value).__name__}")
    if value < 0:
        raise InvalidStructureError(f"{where}: 下标不能为负，收到 {value}")
    return value


def _as_dim(value, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name}: bool 不能用作整数维度")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidStructureError(f"{name}: 拒绝 NaN/Infinity")
        raise TypeError(f"{name}: 维度必须为 int，收到 float {value!r}")
    if not isinstance(value, int):
        raise TypeError(f"{name}: 维度必须为 int，收到 {type(value).__name__}")
    if value < 0:
        raise InvalidStructureError(f"{name}: 维度不能为负，收到 {value}")
    return value


def _as_budget(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("budget: bool 不能用作整数")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidStructureError("budget: 拒绝 NaN/Infinity")
        raise TypeError(f"budget: 必须为 int 或 None，收到 float {value!r}")
    if not isinstance(value, int):
        raise TypeError("budget: 必须为 int 或 None")
    if value < 0:
        raise InvalidStructureError(f"budget: 不能为负，收到 {value}")
    return value


def _normalize(entries, n_rows, n_cols):
    """规范化输入：校验类型/范围，重复位置去重。

    返回 ``(n_rows, n_cols, sorted adjacency, reverse adjacency, edge_set)``。
    纯函数式：任何失败都抛出异常，不产生部分结果。
    """
    if n_rows is not None:
        n_rows = _as_dim(n_rows, "n_rows")
        if n_rows > MAX_ROWS:
            raise LimitExceededError(
                f"n_rows={n_rows} 超过上限 {MAX_ROWS}"
            )
    if n_cols is not None:
        n_cols = _as_dim(n_cols, "n_cols")
        if n_cols > MAX_COLS:
            raise LimitExceededError(
                f"n_cols={n_cols} 超过上限 {MAX_COLS}"
            )

    if entries is None:
        raise TypeError("entries: 不能为 None，需要可迭代的 (row, col) 对")
    try:
        iterator = iter(entries)
    except TypeError:
        raise TypeError("entries: 必须是可迭代的 (row, col) 对") from None

    edge_set: set[tuple[int, int]] = set()
    max_r = -1
    max_c = -1
    for i, pair in enumerate(iterator):
        if isinstance(pair, (str, bytes)):
            raise TypeError(f"entries[{i}]: 坐标对不能是 str/bytes")
        try:
            length = len(pair)
        except TypeError:
            raise TypeError(
                f"entries[{i}]: 每个元素必须是含 2 个下标的对"
            ) from None
        if length != 2:
            raise InvalidStructureError(
                f"entries[{i}]: 坐标对长度必须为 2，收到 {length}"
            )
        r = _as_index(pair[0], f"entries[{i}] 的行号")
        c = _as_index(pair[1], f"entries[{i}] 的列号")
        if r >= MAX_ROWS:
            raise LimitExceededError(
                f"entries[{i}] 行号 {r} 超过上限 {MAX_ROWS - 1}"
            )
        if c >= MAX_COLS:
            raise LimitExceededError(
                f"entries[{i}] 列号 {c} 超过上限 {MAX_COLS - 1}"
            )
        edge_set.add((r, c))  # 重复非零位置在此去重
        if r > max_r:
            max_r = r
        if c > max_c:
            max_c = c

    if n_rows is None:
        n_rows = max_r + 1
    if n_cols is None:
        n_cols = max_c + 1

    for (r, c) in edge_set:
        if r >= n_rows:
            raise InvalidStructureError(
                f"非零位置 ({r}, {c}) 越界：行号 >= n_rows={n_rows}"
            )
        if c >= n_cols:
            raise InvalidStructureError(
                f"非零位置 ({r}, {c}) 越界：列号 >= n_cols={n_cols}"
            )

    adj = [[] for _ in range(n_rows)]
    radj = [[] for _ in range(n_cols)]
    for r, c in edge_set:
        adj[r].append(c)
        radj[c].append(r)
    for r in range(n_rows):
        adj[r].sort()
    for c in range(n_cols):
        radj[c].sort()
    return n_rows, n_cols, adj, radj, frozenset(edge_set)


# --------------------------------------------------------------------------- #
# 核心算法
# --------------------------------------------------------------------------- #


def _maximum_matching(adj, n_rows, n_cols, budget):
    """Kuhn 增广路算法求二部图最大匹配。

    每尝试沿一条二部边搜索消耗 1 个预算单位；预算耗尽抛
    :class:`SearchBudgetExhausted`（绝不把它与“无增广路”混淆）。
    返回 ``(row_match, col_match, matched_count)``。
    """
    rmatch = [-1] * n_rows
    cmatch = [-1] * n_cols
    remaining = budget

    def augment(r, seen):
        nonlocal remaining
        for c in adj[r]:
            if c in seen:
                continue
            if remaining is not None:
                if remaining <= 0:
                    raise SearchBudgetExhausted(
                        "交替路径搜索预算耗尽：未能证明当前匹配最大"
                    )
                remaining -= 1
            seen.add(c)
            if cmatch[c] == -1 or augment(cmatch[c], seen):
                cmatch[c] = r
                rmatch[r] = c
                return True
        return False

    for r in range(n_rows):
        # 每次增广尝试独立的“已访问列”集合。
        augment(r, set())

    return rmatch, cmatch, sum(1 for c in rmatch if c != -1)


def _dm_regions(adj, radj, rmatch, cmatch):
    """Dulmage–Mendelsohn 区域划分（沿交替路可达性）。

    交替有向图 G_M：
      * 非匹配边 row -> col；
      * 匹配边     col -> row。
    从*未匹配行*前向可达的行列 = 过定区 O（行多于列）；
    沿反向边从*未匹配列*可达（即能到达未匹配列）的行列 = 欠定区 U
    （列多于行）；其余 = 良定区 W。
    最大匹配下不存在“未匹配行 -> 未匹配列”的交替路，否则那是增广路，
    所以三个区域必然互不相交。
    """
    n_rows = len(adj)
    n_cols = len(radj)

    # 过定区：从未匹配行出发沿 G_M 前向搜索。
    over_r: set[int] = set()
    over_c: set[int] = set()
    queue = deque()
    for r in range(n_rows):
        if rmatch[r] == -1:
            over_r.add(r)
            queue.append(("r", r))
    while queue:
        kind, v = queue.popleft()
        if kind == "r":
            r = v
            for c in adj[r]:
                if c == rmatch[r]:
                    continue  # 匹配边方向是 col -> row
                if c not in over_c:
                    over_c.add(c)
                    queue.append(("c", c))
        else:
            c = v
            r2 = cmatch[c]
            if r2 != -1 and r2 not in over_r:
                over_r.add(r2)
                queue.append(("r", r2))

    # 欠定区：从未匹配列出发沿 G_M 的反向边搜索。
    under_r: set[int] = set()
    under_c: set[int] = set()
    queue = deque()
    for c in range(n_cols):
        if cmatch[c] == -1:
            under_c.add(c)
            queue.append(("c", c))
    while queue:
        kind, v = queue.popleft()
        if kind == "c":
            c = v
            for r in radj[c]:
                if rmatch[r] == c:
                    continue  # 匹配边反向：row <- col，这里只走非匹配边
                if r not in under_r:
                    under_r.add(r)
                    queue.append(("r", r))
        else:
            r = v
            c2 = rmatch[r]
            if c2 != -1 and c2 not in under_c:
                under_c.add(c2)
                queue.append(("c", c2))

    if over_r & under_r or over_c & under_c:
        ## 理论上不可达：相交意味着存在增广路，与最大匹配矛盾。
        raise StructuralError("内部错误：区域相交，匹配证书可能非最大")

    well_r = [r for r in range(n_rows) if r not in over_r and r not in under_r]
    well_c = [c for c in range(n_cols) if c not in over_c and c not in under_c]
    return under_r, under_c, set(well_r), set(well_c), over_r, over_c


def _kosaraju(nodes, adjf):
    """迭代 Kosaraju 强连通分量。

    返回按拓扑序排列的 SCC 列表（若有边 u->v，则 u 所在块不晚于 v
    所在块），块内节点升序，保证结果确定。
    """
    visited: set[int] = set()
    finish_order: list[int] = []
    for start in nodes:  # nodes 已升序
        if start in visited:
            continue
        visited.add(start)
        stack: list[list[int]] = [[start, 0]]
        while stack:
            v, i = stack[-1]
            neigh = adjf.get(v, ())
            if i < len(neigh):
                w = neigh[i]
                stack[-1][1] = i + 1
                if w not in visited:
                    visited.add(w)
                    stack.append([w, 0])
            else:
                finish_order.append(v)
                stack.pop()

    radj: dict[int, list[int]] = {v: [] for v in nodes}
    for v in nodes:
        for w in adjf.get(v, ()):
            radj[w].append(v)

    seen: set[int] = set()
    components: list[list[int]] = []
    for start in reversed(finish_order):
        if start in seen:
            continue
        comp = []
        stack = [start]
        seen.add(start)
        while stack:
            v = stack.pop()
            comp.append(v)
            for w in radj[v]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        components.append(sorted(comp))
    return components


def analyze(
    entries: Iterable[tuple[int, int]],
    n_rows: Optional[int] = None,
    n_cols: Optional[int] = None,
    *,
    budget: Optional[int] = None,
) -> StructuralDecomposition:
    """对非零位置集合做结构秩与块三角化分析。

    Args:
        entries: ``(row, col)`` 非零位置的可迭代集合；重复位置自动去重，
            结果与传入顺序无关。下标必须为 ``int``（拒绝 bool）、非负且
            不超过 99；拒绝 NaN/Infinity。
        n_rows / n_cols: 可选显式维度（0..100）。省略时由最大下标 +1
            推断（空集合推断为 0）。显式给定时，位置越界报
            :class:`InvalidStructureError`。
        budget: 可选交替路搜索预算（每考察一条二部边计 1）。``None``
            （默认）表示完整搜索，必定给出证明后的最大匹配；非负整数在
            预算耗尽时抛 :class:`SearchBudgetExhausted`，它*不*表示证明
            无解。

    Returns:
        :class:`StructuralDecomposition` 证书。

    Raises:
        TypeError: 类型错误（含 bool 冒充 int、浮点下标）。
        InvalidStructureError: 非法结构（NaN/Infinity、负数、越界、
            坐标对长度不为 2 等），信息带可定位下标。
        LimitExceededError: 超过 100×100 的接口边界。
        SearchBudgetExhausted: 搜索预算耗尽（区别于证明无增广路）。
    """
    budget = _as_budget(budget)
    n_rows, n_cols, adj, radj, edge_set = _normalize(entries, n_rows, n_cols)

    rmatch, cmatch, rank = _maximum_matching(adj, n_rows, n_cols, budget)

    under_r, under_c, well_r, well_c, over_r, over_c = _dm_regions(
        adj, radj, rmatch, cmatch
    )

    # 良定区中行、列一一匹配。以良定行为顶点构造有向图 H：
    # 非零元 (r, c)（c 匹配到 r'）给出边 r -> r'。匹配边成为自环。
    # H 的 SCC 即不可约块；按拓扑序排列后，非零元只落在对角块及其上方，
    # 得到块上三角形式。
    well_rows_sorted = sorted(well_r)
    h_adj: dict[int, list[int]] = {r: [] for r in well_rows_sorted}
    for r in well_rows_sorted:
        targets = sorted({cmatch[c] for c in adj[r] if cmatch[c] in well_r})
        h_adj[r] = targets
    sccs = _kosaraju(well_rows_sorted, h_adj)

    # 区域顺序 U | W | O：跨块非零元只会出现在对角块右上方。
    # U 块（列多于行）：匹配行按序排列，其匹配列同行号排在块内主对角，
    # 未匹配列附在右侧；O 块对偶，未匹配行附在底部。
    under_rows = sorted(under_r)
    u_matched_cols = [rmatch[r] for r in under_rows]
    under_cols = u_matched_cols + sorted(under_c - set(u_matched_cols))
    well_row_perm: list[int] = []
    well_col_perm: list[int] = []
    for comp in sccs:
        well_row_perm.extend(comp)
        well_col_perm.extend(rmatch[r] for r in comp)  # 配对列同行置换，匹配元落主对角
    over_cols = sorted(over_c)
    o_matched_rows = [cmatch[c] for c in over_cols]
    over_rows = o_matched_rows + sorted(over_r - set(o_matched_rows))

    row_perm = tuple(under_rows + well_row_perm + over_rows)
    col_perm = tuple(under_cols + well_col_perm + over_cols)

    row_blocks: list[tuple[int, int]] = []
    col_blocks: list[tuple[int, int]] = []
    block_regions: list[str] = []
    if under_rows or under_cols:
        row_blocks.append((0, len(under_rows)))
        col_blocks.append((0, len(under_cols)))
        block_regions.append(_U)
    w_r0 = len(under_rows)
    w_c0 = len(under_cols)
    for comp in sccs:
        k = len(comp)
        row_blocks.append((w_r0, w_r0 + k))
        col_blocks.append((w_c0, w_c0 + k))
        w_r0 += k
        w_c0 += k
        block_regions.append(_W)
    if over_rows or over_cols:
        row_blocks.append((w_r0, w_r0 + len(over_rows)))
        col_blocks.append((w_c0, w_c0 + len(over_cols)))
        block_regions.append(_O)

    row_region = [""] * n_rows
    for r in under_rows:
        row_region[r] = _U
    for r in well_row_perm:
        row_region[r] = _W
    for r in over_rows:
        row_region[r] = _O
    col_region = [""] * n_cols
    for c in under_cols:
        col_region[c] = _U
    for c in well_col_perm:
        col_region[c] = _W
    for c in over_cols:
        col_region[c] = _O

    return StructuralDecomposition(
        n_rows=n_rows,
        n_cols=n_cols,
        nonzeros=edge_set,
        structural_rank=rank,
        row_match=tuple(rmatch),
        col_match=tuple(cmatch),
        unmatched_rows=tuple(r for r in range(n_rows) if rmatch[r] == -1),
        unmatched_cols=tuple(c for c in range(n_cols) if cmatch[c] == -1),
        row_perm=row_perm,
        col_perm=col_perm,
        row_blocks=tuple(row_blocks),
        col_blocks=tuple(col_blocks),
        block_regions=tuple(block_regions),
        row_region=tuple(row_region),
        col_region=tuple(col_region),
    )


# --------------------------------------------------------------------------- #
# 证书核验（独立重算，供调用方与 demo 使用）
# --------------------------------------------------------------------------- #


def verify(d: StructuralDecomposition) -> bool:
    """独立核验结果证书，失败抛 :class:`StructuralError`，成功返回 ``True``。

    核验内容：置换合法性、匹配边合法且一一对应、秩等于匹配基数、
    用独立的分层 BFS 确认不存在增广路（匹配确实最大）、区域标签完备
    互斥、置换后非零元只在对角块及其上方（块上三角），且良定区匹配元
    落在主对角线上。
    """
    n, m = d.n_rows, d.n_cols

    if sorted(d.row_perm) != list(range(n)):
        raise StructuralError("row_perm 不是合法行置换")
    if sorted(d.col_perm) != list(range(m)):
        raise StructuralError("col_perm 不是合法列置换")
    if len(d.row_region) != n or len(d.col_region) != m:
        raise StructuralError("区域标签长度与维度不符")
    if not set(d.row_region) <= {_U, _W, _O}:
        raise StructuralError("存在未知行区域标签")
    if not set(d.col_region) <= {_U, _W, _O}:
        raise StructuralError("存在未知列区域标签")

    if len(d.row_match) != n or len(d.col_match) != m:
        raise StructuralError("匹配数组长度与维度不符")
    matched = 0
    for r, c in enumerate(d.row_match):
        if c == -1:
            continue
        matched += 1
        if not (0 <= c < m) or (r, c) not in d.nonzeros:
            raise StructuralError(f"行匹配 ({r},{c}) 不是非零边")
        if d.col_match[c] != r:
            raise StructuralError(f"匹配在 ({r},{c}) 处不互逆")
    for c, r in enumerate(d.col_match):
        if r != -1 and not (0 <= r < n):
            raise StructuralError(f"列 {c} 的匹配指向越界行 {r}")
    if matched != d.structural_rank:
        raise StructuralError("结构秩与匹配基数不一致")

    # 独立分层 BFS：未匹配行 -> 非匹配边 -> 列 -> 匹配边 -> 行 ...
    # 若能到达任一未匹配列，则存在增广路，匹配非最大。
    adj_rows: dict[int, list[int]] = {r: [] for r in range(n)}
    for r, c in d.nonzeros:
        adj_rows[r].append(c)
    seen: set[tuple[str, int]] = set()
    queue = deque()
    for r in range(n):
        if d.row_match[r] == -1:
            seen.add(("r", r))
            queue.append(("r", r))
    while queue:
        kind, v = queue.popleft()
        if kind == "r":
            r = v
            for c in adj_rows[r]:
                if c == d.row_match[r]:
                    continue  # 只走非匹配边 row -> col
                if ("c", c) not in seen:
                    seen.add(("c", c))
                    queue.append(("c", c))
        else:
            c = v
            if d.col_match[c] == -1:
                raise StructuralError("存在增广路：匹配不是最大匹配")
            r2 = d.col_match[c]  # 沿匹配边 col -> row
            if ("r", r2) not in seen:
                seen.add(("r", r2))
                queue.append(("r", r2))

    if len(d.row_blocks) != len(d.col_blocks) or len(d.row_blocks) != len(
        d.block_regions
    ):
        raise StructuralError("块边界与区域标签数量不一致")

    def _tiles(spans, total, axis):
        # U 块可以 0 行（0×k）、O 块可以 0 列（k×0），故允许 a <= b；
        # 但必须首尾相接覆盖整个维度。
        prev = 0
        for a, b in spans:
            if a != prev or b < a:
                raise StructuralError(f"{axis} 块边界不连续")
            prev = b
        if prev != total:
            raise StructuralError(f"{axis} 块未覆盖整个维度")

    _tiles(d.row_blocks, n, "行")
    _tiles(d.col_blocks, m, "列")

    def _block_index(spans, pos):
        for k, (a, b) in enumerate(spans):
            if a <= pos < b:
                return k
        raise StructuralError(f"位置 {pos} 不属于任何块")

    rp = {r: i for i, r in enumerate(d.row_perm)}
    cp = {c: j for j, c in enumerate(d.col_perm)}

    # 良定行有向图 H（由非零位置与匹配独立重建）：r -> cmatch[c]。
    w_succ: dict[int, set[int]] = {
        r: set() for r in range(n) if d.row_region[r] == _W
    }
    for r, c in d.nonzeros:
        if d.row_region[r] == _W:
            r2 = d.col_match[c]
            if 0 <= r2 < n and d.row_region[r2] == _W:
                w_succ[r].add(r2)

    def _reaches(src, within):
        seen = {src}
        dq = deque([src])
        while dq:
            v = dq.popleft()
            for w in w_succ[v]:
                if w in within and w not in seen:
                    seen.add(w)
                    dq.append(w)
        return seen

    for r, c in d.nonzeros:
        i, j = rp[r], cp[c]
        bi = _block_index(d.row_blocks, i)
        bj = _block_index(d.col_blocks, j)
        if bi > bj:
            raise StructuralError(
                f"非零元 ({r},{c}) 置换后位于 ({i},{j})，落在对角块下方，"
                "不满足块上三角"
            )
        # 匹配元必须落在所在块的*相对*主对角上（U/O 块单侧可能零宽，
        # 故不能用全局 i == j）。
        if bi == bj and d.row_match[r] == c:
            rs = d.row_blocks[bi][0]
            cs = d.col_blocks[bi][0]
            if i - rs != j - cs:
                raise StructuralError(
                    f"块 {bi} 内匹配元 ({r},{c}) 未落在块内相对主对角"
                    f"（块内坐标 {i - rs},{j - cs}）"
                )

    # 每个良定块必须是 H 上的强连通分量（不可约），且不可再分。
    for k, region in enumerate(d.block_regions):
        if region != _W:
            continue
        members = set(d.row_perm[d.row_blocks[k][0]:d.row_blocks[k][1]])
        if not members:
            raise StructuralError(f"良定块 {k} 为空")
        for src in members:
            if _reaches(src, members) != members:
                raise StructuralError(
                    f"良定块 {k} 不是强连通分量（不可约块划分有误）"
                )
    return True


# --------------------------------------------------------------------------- #
# 边关键性（edge criticality）：单独删除每个非零位置是否降低结构秩
# --------------------------------------------------------------------------- #
#
# 理论依据（纯组合，不读数值）：
#   在交替有向图 G_M 中（非匹配边 row -> col，匹配边 col -> row）：
#   * 当前匹配未使用的边：删除后原匹配仍然合法，秩必不下降，绝不关键；
#   * 匹配边 e=(r,c)：另一个最大匹配 M' 与 M 的对称差由交错环和偶长
#     交错路径组成，因此存在不含 e 的最大匹配，当且仅当满足三者之一：
#       1. e 在交错环上 <=> G_M 中有 r -> c 路径 <=> r 与 c 同 SCC
#          （c -> r 弧恒存在；且 r 的唯一入弧就是这条匹配弧，从 r 出发
#          的简单路径不会用到它，无需逐边删弧重算）；
#       2. G_M 中 r 能到达某条未匹配列（沿交错路翻转后 c 变为未匹配，
#          未匹配列被匹配，秩不变）——即 r 位于欠定区 U；
#       3. G_M 中某条未匹配行能到达 c（对偶翻转）——即 c 位于过定区 O。
#   因此一次交替图 SCC 分解加 DM 区域标签即可分类全部匹配边；只有被
#   判定为关键的边才在独立删除后的图上重跑最大匹配，并用 König 最小
#   顶点覆盖给出降秩上界证据。每个位置的删除都是基于原图的独立试验，
#   互不累积。


@dataclass(frozen=True)
class EdgeCriticality:
    """单个非零位置的独立删除试验结果。

    Attributes:
        position: 非零位置 (row, col)（原始坐标）。
        critical: 单独删除该位置后结构秩是否下降。
        rank_after_removal: 删除后的结构秩（不降 = 原秩，降 = 原秩 - 1）。
        in_current_matching: 该位置是否被当前匹配证书使用。
        alternative_matching: 非关键时的证据：删除后图上大小等于原秩的
            匹配（不使用被删位置），边元组按行列升序。当前匹配未使用该
            位置时即原匹配本身。
        reduced_matching: 关键时的证据之一：删除后图上大小为原秩 - 1
            的最大匹配。
        vertex_cover_rows / vertex_cover_cols: 关键时的证据之二：删除后
            图上的最小顶点覆盖（König），其大小等于原秩 - 1，证明删除后
            任何匹配都不超过原秩 - 1（降秩上界）。
    """

    position: tuple[int, int]
    critical: bool
    rank_after_removal: int
    in_current_matching: bool
    alternative_matching: Optional[tuple[tuple[int, int], ...]] = None
    reduced_matching: Optional[tuple[tuple[int, int], ...]] = None
    vertex_cover_rows: Optional[tuple[int, ...]] = None
    vertex_cover_cols: Optional[tuple[int, ...]] = None


@dataclass(frozen=True)
class EdgeCriticalityReport:
    """全部非零位置的边关键性报告（各删除试验相互独立）。

    Attributes:
        n_rows / n_cols: 矩阵维度。
        structural_rank: 原图结构秩。
        entries: 每个非零位置一条 EdgeCriticality，按位置升序。
    """

    n_rows: int
    n_cols: int
    structural_rank: int
    entries: tuple[EdgeCriticality, ...]

    @property
    def critical_positions(self) -> tuple[tuple[int, int], ...]:
        """所有关键非零位置（删除后结构秩下降），按位置升序。"""
        return tuple(e.position for e in self.entries if e.critical)

    def entry(self, row: int, col: int) -> EdgeCriticality:
        """按位置查询单个结果；位置不存在抛 KeyError。"""
        for e in self.entries:
            if e.position == (row, col):
                return e
        raise KeyError(f"({row}, {col}) 不是非零位置")


def _alternating_graph(adj, rmatch, cmatch, n_rows, n_cols):
    """构造交替有向图 G_M 的邻接表（复用内部分析）。

    节点编号：行 0..n_rows-1，列 n_rows..n_rows+n_cols-1。
    弧：非匹配边 row -> col；匹配边 col -> row。
    """
    graph: dict[int, list[int]] = {}
    for r in range(n_rows):
        graph[r] = [n_rows + c for c in adj[r] if c != rmatch[r]]
    for c in range(n_cols):
        graph[n_rows + c] = [cmatch[c]] if cmatch[c] != -1 else []
    return graph


def _find_path(graph, src, dst, targets=None):
    """BFS 求 src -> dst（或 targets 中任一节点）的一条节点路径。"""
    if targets is None:
        targets = {dst}
    if src in targets:
        return [src]
    parent = {src: None}
    queue = deque([src])
    while queue:
        v = queue.popleft()
        for w in graph.get(v, ()):
            if w in parent:
                continue
            parent[w] = v
            if w in targets:
                path = [w]
                while path[-1] is not src:
                    path.append(parent[path[-1]])
                path.reverse()
                return path
            queue.append(w)
    return None


def _reverse_graph(graph, n_total):
    rev: dict[int, list[int]] = {v: [] for v in range(n_total)}
    for v, outs in graph.items():
        for w in outs:
            rev[w].append(v)
    return rev


def _flip_along_path(rmatch, cmatch, path, n_rows):
    """沿交错路径翻转匹配：path 是 G_M 中从行节点出发的节点路径
    [r0, c1, r1, ..., rk, c_{k+1}]。把每条非匹配弧 r_i -> c_{i+1}
    变为匹配边，返回 (row_match, col_match) 列表副本（调用方负责
    处理路径另一端释放出的悬空端点）。"""
    new_rm = list(rmatch)
    new_cm = list(cmatch)
    for i in range(0, len(path) - 1, 2):
        r_i = path[i]
        c_next = path[i + 1] - n_rows
        new_rm[r_i] = c_next
        new_cm[c_next] = r_i
    return new_rm, new_cm


def _matching_edges(rmatch):
    return tuple(sorted((r, c) for r, c in enumerate(rmatch) if c != -1))


def _konig_cover(adj, rmatch, cmatch, n_rows, n_cols):
    """由最大匹配构造 König 最小顶点覆盖。

    Z = 从未匹配行出发沿交替路（非匹配边 row->col，匹配边 col->row）
    可达的顶点；覆盖 = （不在 Z 中的行） ∪ （在 Z 中的列），
    其大小等于最大匹配基数。
    """
    z_rows: set[int] = set()
    z_cols: set[int] = set()
    queue = deque()
    for r in range(n_rows):
        if rmatch[r] == -1:
            z_rows.add(r)
            queue.append(("r", r))
    while queue:
        kind, v = queue.popleft()
        if kind == "r":
            for c in adj[v]:
                if c == rmatch[v]:
                    continue
                if c not in z_cols:
                    z_cols.add(c)
                    queue.append(("c", c))
        else:
            r2 = cmatch[v]
            if r2 != -1 and r2 not in z_rows:
                z_rows.add(r2)
                queue.append(("r", r2))
    cover_rows = tuple(r for r in range(n_rows) if r not in z_rows)
    cover_cols = tuple(c for c in range(n_cols) if c in z_cols)
    return cover_rows, cover_cols


def edge_criticality(d: StructuralDecomposition) -> EdgeCriticalityReport:
    """对每个已存在非零位置做*独立*的单边删除试验。

    判断单独删除该位置是否降低最大匹配大小（结构秩），并给出证据：
    非关键位置给出删除后仍达到原秩的替代匹配；关键位置给出删除后的
    最大匹配与最小顶点覆盖（大小 = 原秩 - 1，证明降秩上界）。

    所有删除都基于原图，互不在前一次删除后的图上继续删。复用一次
    交替图强连通分解完成全部分类，仅对关键边在删除后的图上重跑
    最大匹配（独立试验）。

    Args:
        d: analyze() 返回的分解证书。

    Returns:
        EdgeCriticalityReport，可用 verify_criticality() 独立核验。
    """
    n_rows, n_cols = d.n_rows, d.n_cols
    rank = d.structural_rank
    rmatch = list(d.row_match)
    cmatch = list(d.col_match)

    adj = [[] for _ in range(n_rows)]
    for r, c in d.nonzeros:
        adj[r].append(c)
    for r in range(n_rows):
        adj[r].sort()

    # 一次交替图 SCC 分解 + 反向图，分类所有匹配边。
    graph = _alternating_graph(adj, rmatch, cmatch, n_rows, n_cols)
    nodes = list(range(n_rows + n_cols))
    comp_id: dict[int, int] = {}
    for k, comp in enumerate(_kosaraju(nodes, graph)):
        for v in comp:
            comp_id[v] = k
    rgraph = _reverse_graph(graph, n_rows + n_cols)
    unmatched_col_nodes = {
        n_rows + c for c in range(n_cols) if cmatch[c] == -1
    }
    unmatched_row_nodes = {r for r in range(n_rows) if rmatch[r] == -1}

    def _alternative_for_matched(r, c):
        """为匹配边 (r, c) 构造不含它的同基数匹配；不存在返回 None。"""
        # 情形 1：交错环（r 与 c 同 SCC，等价于 r -> c 可达）。
        if comp_id[r] == comp_id[n_rows + c]:
            path = _find_path(graph, r, n_rows + c)
            if path is None:  # 理论不可达：同 SCC 必有路径
                raise StructuralError("内部错误：同一 SCC 内无交错路径")
            return _flip_along_path(rmatch, cmatch, path, n_rows)
        # 情形 2：r 可达某未匹配列，翻转后 c 释放为未匹配。
        path = _find_path(graph, r, None, targets=unmatched_col_nodes)
        if path is not None:
            new_rm, new_cm = _flip_along_path(rmatch, cmatch, path, n_rows)
            new_cm[c] = -1  # 原匹配列被释放
            return new_rm, new_cm
        # 情形 3：某未匹配行可达 c（反向 BFS），翻转后 r 释放为未匹配。
        path = _find_path(rgraph, n_rows + c, None, targets=unmatched_row_nodes)
        if path is not None:
            path.reverse()  # 还原为 未匹配行 -> ... -> c 的正向路径
            new_rm, new_cm = _flip_along_path(rmatch, cmatch, path, n_rows)
            new_rm[r] = -1  # 原匹配行被释放
            return new_rm, new_cm
        return None

    entries: list[EdgeCriticality] = []
    for r, c in sorted(d.nonzeros):
        if rmatch[r] != c:
            # 当前匹配未使用该边：删除后原匹配仍合法，秩不下降。
            entries.append(
                EdgeCriticality(
                    position=(r, c),
                    critical=False,
                    rank_after_removal=rank,
                    in_current_matching=False,
                    alternative_matching=_matching_edges(rmatch),
                )
            )
            continue

        alternative = _alternative_for_matched(r, c)
        if alternative is not None:
            entries.append(
                EdgeCriticality(
                    position=(r, c),
                    critical=False,
                    rank_after_removal=rank,
                    in_current_matching=True,
                    alternative_matching=_matching_edges(alternative[0]),
                )
            )
            continue

        # 关键边：在独立删除后的图上重跑最大匹配（不累积删除），
        # 并用 König 最小顶点覆盖给出降秩上界证据。
        del_adj = [list(cols) for cols in adj]
        del_adj[r] = [c2 for c2 in del_adj[r] if c2 != c]
        new_rm, new_cm, new_rank = _maximum_matching(
            del_adj, n_rows, n_cols, None
        )
        if new_rank != rank - 1:  # 理论保证恰降 1
            raise StructuralError("内部错误：关键边删除后秩变化异常")
        cover_rows, cover_cols = _konig_cover(
            del_adj, new_rm, new_cm, n_rows, n_cols
        )
        entries.append(
            EdgeCriticality(
                position=(r, c),
                critical=True,
                rank_after_removal=new_rank,
                in_current_matching=True,
                reduced_matching=_matching_edges(new_rm),
                vertex_cover_rows=cover_rows,
                vertex_cover_cols=cover_cols,
            )
        )

    return EdgeCriticalityReport(
        n_rows=n_rows,
        n_cols=n_cols,
        structural_rank=rank,
        entries=tuple(entries),
    )


def _check_matching(edges, matching, n_rows, n_cols, expect_size, what):
    """核验 matching 是 edges 上的合法匹配且大小为 expect_size。"""
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    for r, c in matching:
        if not (0 <= r < n_rows and 0 <= c < n_cols):
            raise StructuralError(f"{what}: 匹配边 ({r},{c}) 越界")
        if (r, c) not in edges:
            raise StructuralError(f"{what}: 匹配边 ({r},{c}) 不是非零边")
        if r in used_rows or c in used_cols:
            raise StructuralError(f"{what}: 匹配在 ({r},{c}) 处冲突")
        used_rows.add(r)
        used_cols.add(c)
    if len(matching) != expect_size:
        raise StructuralError(
            f"{what}: 匹配大小 {len(matching)} != 期望 {expect_size}"
        )


def verify_criticality(
    d: StructuralDecomposition, report: EdgeCriticalityReport
) -> bool:
    """独立核验边关键性报告的全部证据，失败抛 StructuralError。

    核验内容：报告覆盖且仅覆盖全部非零位置；秩字段与原分解一致；
    每个非关键位置的替代匹配是删除后图上大小等于原秩的合法匹配
    （且不使用被删位置）；每个关键位置的降秩匹配合法，且顶点覆盖
    覆盖删除后图的所有边、大小等于原秩 - 1（从而证明删除后任何
    匹配都不超过原秩 - 1，降秩结论成立）。
    """
    n_rows, n_cols = d.n_rows, d.n_cols
    rank = d.structural_rank
    if report.structural_rank != rank:
        raise StructuralError("报告结构秩与分解证书不一致")
    if report.n_rows != n_rows or report.n_cols != n_cols:
        raise StructuralError("报告维度与分解证书不一致")

    positions = [e.position for e in report.entries]
    if sorted(positions) != sorted(d.nonzeros) or len(set(positions)) != len(
        positions
    ):
        raise StructuralError("报告未恰好覆盖每个非零位置一次")

    for e in report.entries:
        r, c = e.position
        remaining = d.nonzeros - {(r, c)}
        if e.critical:
            if e.rank_after_removal != rank - 1:
                raise StructuralError(
                    f"位置 {e.position}: 关键边降秩后秩应为原秩 - 1"
                )
            if e.reduced_matching is None:
                raise StructuralError(f"位置 {e.position}: 缺少降秩匹配证据")
            _check_matching(
                remaining, e.reduced_matching, n_rows, n_cols, rank - 1,
                f"位置 {e.position} 的降秩匹配",
            )
            if e.vertex_cover_rows is None or e.vertex_cover_cols is None:
                raise StructuralError(
                    f"位置 {e.position}: 缺少最小顶点覆盖证据"
                )
            cover_r = set(e.vertex_cover_rows)
            cover_c = set(e.vertex_cover_cols)
            if len(cover_r) + len(cover_c) != rank - 1:
                raise StructuralError(
                    f"位置 {e.position}: 顶点覆盖大小不等于原秩 - 1"
                )
            for r2, c2 in remaining:
                if r2 not in cover_r and c2 not in cover_c:
                    raise StructuralError(
                        f"位置 {e.position}: 顶点覆盖未覆盖边 ({r2},{c2})"
                    )
        else:
            if e.rank_after_removal != rank:
                raise StructuralError(
                    f"位置 {e.position}: 非关键边删除后秩不应变化"
                )
            if e.alternative_matching is None:
                raise StructuralError(f"位置 {e.position}: 缺少替代匹配证据")
            _check_matching(
                remaining, e.alternative_matching, n_rows, n_cols, rank,
                f"位置 {e.position} 的替代匹配",
            )
            if (r, c) in e.alternative_matching:
                raise StructuralError(
                    f"位置 {e.position}: 替代匹配仍使用被删位置"
                )
    return True
