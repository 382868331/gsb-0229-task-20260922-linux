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
    "UpdateReport",
    "StructuralModel",
    "analyze",
    "verify",
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


def _check_pair(pair, where):
    """校验单个坐标对：长度必须为 2，下标必须为有限非负 int。

    ``where`` 是可定位前缀，如 ``"entries[7]"`` / ``"add[0]"``。
    """
    if isinstance(pair, (str, bytes)):
        raise TypeError(f"{where}: 坐标对不能是 str/bytes")
    try:
        length = len(pair)
    except TypeError:
        raise TypeError(
            f"{where}: 每个元素必须是含 2 个下标的对"
        ) from None
    if length != 2:
        raise InvalidStructureError(
            f"{where}: 坐标对长度必须为 2，收到 {length}"
        )
    r = _as_index(pair[0], f"{where} 的行号")
    c = _as_index(pair[1], f"{where} 的列号")
    if r >= MAX_ROWS:
        raise LimitExceededError(
            f"{where} 行号 {r} 超过上限 {MAX_ROWS - 1}"
        )
    if c >= MAX_COLS:
        raise LimitExceededError(
            f"{where} 列号 {c} 超过上限 {MAX_COLS - 1}"
        )
    return r, c


def _check_edges_within(edge_set, n_rows, n_cols):
    """所有非零位置必须在显式维度内，否则报可定位错误。"""
    for (r, c) in edge_set:
        if r >= n_rows:
            raise InvalidStructureError(
                f"非零位置 ({r}, {c}) 越界：行号 >= n_rows={n_rows}"
            )
        if c >= n_cols:
            raise InvalidStructureError(
                f"非零位置 ({r}, {c}) 越界：列号 >= n_cols={n_cols}"
            )


def _normalize(entries, n_rows, n_cols):
    """规范化输入：校验类型/范围，重复位置去重。

    返回 ``(n_rows, n_cols, adjacency, reverse adjacency, edge_set)``。
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
        r, c = _check_pair(pair, f"entries[{i}]")
        edge_set.add((r, c))  # 重复非零位置在此去重
        if r > max_r:
            max_r = r
        if c > max_c:
            max_c = c

    if n_rows is None:
        n_rows = max_r + 1
    if n_cols is None:
        n_cols = max_c + 1

    _check_edges_within(edge_set, n_rows, n_cols)

    adj, radj = _build_adjacency(edge_set, n_rows, n_cols)
    return n_rows, n_cols, adj, radj, frozenset(edge_set)


def _build_adjacency(edge_set, n_rows, n_cols):
    """由去重边集构造排序后的行/列邻接表。"""
    adj = [[] for _ in range(n_rows)]
    radj = [[] for _ in range(n_cols)]
    for r, c in edge_set:
        adj[r].append(c)
        radj[c].append(r)
    for r in range(n_rows):
        adj[r].sort()
    for c in range(n_cols):
        radj[c].sort()
    return adj, radj


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


def _decompose(n_rows, n_cols, edge_set, budget):
    """由已校验的去重边集计算完整结构分解。

    静态入口 :func:`analyze` 与动态入口 :class:`StructuralModel` 的全量
    重算共用本函数，因此两条路径对同一边集给出*逐字段相同*的证书。
    """
    adj, radj = _build_adjacency(edge_set, n_rows, n_cols)

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
        nonzeros=frozenset(edge_set),
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
    n_rows, n_cols, _, _, edge_set = _normalize(entries, n_rows, n_cols)
    return _decompose(n_rows, n_cols, edge_set, budget)


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
# 有状态批量更新（非零模式）
# --------------------------------------------------------------------------- #

_BUDGET_UNSET = object()


@dataclass(frozen=True)
class UpdateReport:
    """一次批量更新（及随后全量重算）的变更账目与新证书。

    Attributes:
        added: 本次实际*新增*的位置（之前不存在），升序元组。
        removed: 本次实际*删除*的位置（之前存在），升序元组。
        added_duplicates: 出现在 add 中但模型里已存在、因此无变化的位置。
        removed_missing: 出现在 remove 中但模型里本不存在、因此无变化的位置。
        old_rank / new_rank: 更新前后的结构秩。
        decomposition: 更新后的 :class:`StructuralDecomposition` 证书。
    """

    added: tuple[tuple[int, int], ...]
    removed: tuple[tuple[int, int], ...]
    added_duplicates: tuple[tuple[int, int], ...]
    removed_missing: tuple[tuple[int, int], ...]
    old_rank: int
    new_rank: int
    decomposition: StructuralDecomposition

    @property
    def changed(self) -> bool:
        """边集是否真的发生了变化（忽略重复增 / 缺失删）。"""
        return bool(self.added or self.removed)


class StructuralModel:
    """可批量增删非零边、随后全量重算的有状态结构模型。

    语义：

    * 每次 :meth:`update`（以及 :meth:`add_edges` / :meth:`remove_edges`）
      先在*副本*上完成全部校验与边集修改，再做一次全量重算，成功后才
      原子替换旧状态——任何非法输入或搜索预算耗尽都不会留下半更新状态，
      :attr:`decomposition` 仍是上一次成功的结果；
    * 重算走与静态入口 :func:`analyze` 完全相同的代码路径
      （:func:`_decompose`），因此 ``model.decomposition`` 与
      ``analyze(model.edges(), model.n_rows, model.n_cols)`` 逐字段一致；
    * 同一位置在一次批次里同时增、删是歧义输入，直接拒绝
      （:class:`InvalidStructureError`），不依赖批次内出现顺序；
    * 删除不存在的边不是错误，记入报告的 ``removed_missing``；添加已存在
      的边同理，记入 ``added_duplicates``。

    维度语义：

    * ``auto_dim``（默认）时维度为宽松边界：显式给出的 ``n_rows`` /
      ``n_cols`` 只是初始下界，之后加边可自动扩大维度，删边不会缩小
      （空行 / 空列作为显式维度的一部分保留）；
    * ``auto_dim=False`` 时维度在构造时固定，加边越界报
      :class:`InvalidStructureError`。
    """

    def __init__(
        self,
        entries: Iterable[tuple[int, int]] = (),
        n_rows: Optional[int] = None,
        n_cols: Optional[int] = None,
        *,
        auto_dim: bool = True,
        budget: Optional[int] = None,
    ):
        if not isinstance(auto_dim, bool):
            raise TypeError("auto_dim: 必须为 bool")
        self._auto_dim = auto_dim
        self._budget = _as_budget(budget)

        # 复用静态入口的完整校验（含显式维度边界、NaN/bool 拒绝等），
        # 得到首份证书；维度与边集从证书取得，保证初始状态即合法。
        d = analyze(entries, n_rows, n_cols, budget=self._budget)
        self._n_rows = d.n_rows
        self._n_cols = d.n_cols
        self._edges: set[tuple[int, int]] = set(d.nonzeros)
        self._decomposition = d

    # ----- 只读视图 ----------------------------------------------------- #

    @property
    def n_rows(self) -> int:
        """当前行数（空行也计入，维度只扩不缩）。"""
        return self._n_rows

    @property
    def n_cols(self) -> int:
        """当前列数（空列也计入，维度只扩不缩）。"""
        return self._n_cols

    @property
    def decomposition(self) -> StructuralDecomposition:
        """最近一次成功重算的结构分解证书。"""
        return self._decomposition

    @property
    def budget(self) -> Optional[int]:
        """当前生效的搜索预算（``None`` 表示完整搜索）。"""
        return self._budget

    def edges(self) -> frozenset[tuple[int, int]]:
        """当前去重非零位置集合的不可变快照。"""
        return frozenset(self._edges)

    def __len__(self) -> int:
        return len(self._edges)

    def __contains__(self, pair) -> bool:
        try:
            return pair in self._edges
        except TypeError:
            return False

    # ----- 配置 --------------------------------------------------------- #

    def set_budget(self, budget: Optional[int]) -> None:
        """设置后续重算使用的搜索预算（立即校验，不触发重算）。"""
        self._budget = _as_budget(budget)

    # ----- 批量更新 ----------------------------------------------------- #

    def update(
        self,
        add: Iterable[tuple[int, int]] = (),
        remove: Iterable[tuple[int, int]] = (),
        *,
        budget=_BUDGET_UNSET,
    ) -> UpdateReport:
        """批量增、删非零边，然后全量重算，原子提交。

        Args:
            add: 要新增的 ``(row, col)`` 位置（重复自动去重）。
            remove: 要删除的位置；删除不存在的边记为 ``removed_missing``，
                不算错误。
            budget: 本次重算的搜索预算；省略时沿用模型当前预算，
                显式传 ``None`` 表示本次完整搜索。

        Returns:
            :class:`UpdateReport`：实际增/删账目与新证书。

        Raises:
            TypeError / InvalidStructureError / LimitExceededError:
                非法坐标、批次内同一位置既增又删、固定维度下越界等。
            SearchBudgetExhausted: 本次搜索预算耗尽；此时旧状态完整保留。
        """
        eff_budget = self._budget if budget is _BUDGET_UNSET else _as_budget(budget)

        if add is None or remove is None:
            raise TypeError(
                "add/remove: 必须为可迭代的 (row, col) 对，不能为 None"
            )
        try:
            add_list = list(add)
        except TypeError:
            raise TypeError("add: 必须为可迭代的 (row, col) 对") from None
        try:
            remove_list = list(remove)
        except TypeError:
            raise TypeError("remove: 必须为可迭代的 (row, col) 对") from None

        # 第一阶段：校验全部坐标（错误信息可定位到批次内序号）。
        add_pairs = [_check_pair(pair, f"add[{i}]") for i, pair in enumerate(add_list)]
        rem_pairs = [
            _check_pair(pair, f"remove[{i}]") for i, pair in enumerate(remove_list)
        ]

        add_set = set(add_pairs)
        rem_set = set(rem_pairs)
        conflict = add_set & rem_set
        if conflict:
            pos = sorted(conflict)[0]
            raise InvalidStructureError(
                f"同一位置 {pos} 在一个批次中既被添加又被删除（歧义更新）"
            )

        # 第二阶段：在副本上计算新边集与新维度，校验固定维度边界。
        actually_added = add_set - self._edges
        actually_removed = rem_set & self._edges
        new_edges = (self._edges | add_set) - rem_set

        new_n_rows = self._n_rows
        new_n_cols = self._n_cols
        if self._auto_dim:
            for r, _ in add_set:
                if r + 1 > new_n_rows:
                    new_n_rows = r + 1
            for _, c in add_set:
                if c + 1 > new_n_cols:
                    new_n_cols = c + 1
        else:
            for r, c in add_set:
                if r >= new_n_rows or c >= new_n_cols:
                    raise InvalidStructureError(
                        f"添加位置 ({r}, {c}) 越界：固定维度 "
                        f"{new_n_rows}x{new_n_cols}（auto_dim=False）"
                    )

        # 第三阶段：全量重算（可能抛 SearchBudgetExhausted）。此时尚未触碰
        # 任何实例状态；成功后一次性提交，杜绝半更新。
        new_d = _decompose(new_n_rows, new_n_cols, new_edges, eff_budget)

        old_rank = self._decomposition.structural_rank
        report = UpdateReport(
            added=tuple(sorted(actually_added)),
            removed=tuple(sorted(actually_removed)),
            added_duplicates=tuple(sorted(add_set & self._edges)),
            removed_missing=tuple(sorted(rem_set - self._edges)),
            old_rank=old_rank,
            new_rank=new_d.structural_rank,
            decomposition=new_d,
        )
        self._edges = new_edges
        self._n_rows = new_n_rows
        self._n_cols = new_n_cols
        self._decomposition = new_d
        return report

    def add_edges(self, entries: Iterable[tuple[int, int]], **kwargs) -> UpdateReport:
        """只增批量入口，等价于 :meth:`update` 的 ``add=entries``。"""
        return self.update(add=entries, **kwargs)

    def remove_edges(self, entries: Iterable[tuple[int, int]], **kwargs) -> UpdateReport:
        """只删批量入口，等价于 :meth:`update` 的 ``remove=entries``。"""
        return self.update(remove=entries, **kwargs)

    def recompute(self, *, budget=_BUDGET_UNSET) -> StructuralDecomposition:
        """不改变边集，做一次全量重算（空批次）并返回新证书。

        可用于切换预算后重新搜索，或核对结果与静态入口一致。
        """
        self.update((), (), budget=budget)
        return self._decomposition
