"""公开门面：:func:`analyze` 串起最大匹配、DM 分区与 SCC 块三角化。"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ._algorithms import (
    alternating_reach_from_cols,
    alternating_reach_from_rows,
    kosaraju_scc,
    maximum_matching,
    topological_order,
)
from ._input import iter_adjacency, normalize_nonzeros
from .errors import InvalidStructureError
from .types import (
    Block,
    BlockTriangularForm,
    Regions,
    StructuralAnalysis,
    VerifyReport,
)

__all__ = [
    "analyze",
    "structural_rank",
    "dm_regions",
    "verify_certificate",
]


def _coerce(nonzeros: Iterable, n_rows, n_cols):
    nr, nc, nz = normalize_nonzeros(nonzeros, n_rows, n_cols)
    row_adj, col_adj = iter_adjacency(nz, nr, nc)
    return nr, nc, nz, row_adj, col_adj


def dm_regions(
    row_adj: list[list[int]],
    col_adj: list[list[int]],
    match_row: list[int],
    match_col: list[int],
) -> Regions:
    """Dulmage–Mendelsohn 三区划分（要求输入匹配是最大匹配）。

    从所有未匹配行出发沿交替路（非匹配边先行）可达的行/列为过定区；
    从所有未匹配列出发可达的行/列为欠定区；其余为良定区。
    最大匹配保证两类起点的可达集不交（否则存在增广路）。
    """
    n_rows, n_cols = len(row_adj), len(col_adj)
    unmatched_rows = [r for r in range(n_rows) if match_row[r] == -1]
    unmatched_cols = [c for c in range(n_cols) if match_col[c] == -1]

    over_rows, over_cols = alternating_reach_from_rows(unmatched_rows, row_adj, match_col)
    under_rows, under_cols = alternating_reach_from_cols(unmatched_cols, col_adj, match_row)

    well_rows = set(range(n_rows)) - over_rows - under_rows
    well_cols = set(range(n_cols)) - over_cols - under_cols
    return Regions(
        under_rows=frozenset(under_rows),
        under_cols=frozenset(under_cols),
        well_rows=frozenset(well_rows),
        well_cols=frozenset(well_cols),
        over_rows=frozenset(over_rows),
        over_cols=frozenset(over_cols),
        unmatched_rows=frozenset(unmatched_rows),
        unmatched_cols=frozenset(unmatched_cols),
    )


def analyze(
    nonzeros: Iterable,
    n_rows: int | None = None,
    n_cols: int | None = None,
    *,
    max_steps: int | None = None,
) -> StructuralAnalysis:
    """对非零位置集合做结构秩与块三角化分析。

    参数
    ----
    nonzeros:
        ``(row, col)`` 对的可迭代对象（重复位置自动去重），或“行序列”形式：
        每行是列下标的可迭代对象（此时不可再传 ``n_rows``）。
    n_rows, n_cols:
        维度，0..100 的整数；为 None 时由非零位置推断（空集合推断为 0）。
    max_steps:
        增广路搜索检查边数的硬预算。None（默认）表示不限；耗尽时抛
        :class:`~structrank.errors.SearchBudgetExhaustedError`，这是“尚无
        结论”，不是“证明无解”。bool 一律拒绝。

    返回
    ----
    :class:`~structrank.types.StructuralAnalysis`，含结构秩、匹配证书、
    DM 三区、良定区不可约块、全局行列置换与块边界。
    """
    if max_steps is not None:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int):
            raise InvalidStructureError(
                f"max_steps 必须为 int 或 None（收到 {type(max_steps).__name__}）"
            )
        if max_steps <= 0:
            raise InvalidStructureError(f"max_steps 必须为正整数（收到 {max_steps}）")

    nr, nc, nz, row_adj, col_adj = _coerce(nonzeros, n_rows, n_cols)
    match_row, match_col, _ = maximum_matching(row_adj, nr, nc, max_steps)
    rank = sum(c != -1 for c in match_col)

    regions = dm_regions(row_adj, col_adj, match_row, match_col)

    # ---- 良定区：在“匹配行有向图”上求 SCC ----
    well_rows = regions.well_rows
    well_cols = regions.well_cols
    well_row_list = sorted(well_rows)
    edges: dict[int, list[int]] = {r: [] for r in well_row_list}
    for r in well_row_list:
        for c in row_adj[r]:
            if c in well_cols:
                r2 = match_col[c]
                if r2 != r:  # 匹配边是自环，不参与块间结构
                    edges[r].append(r2)

    sccs = kosaraju_scc(well_row_list, edges) if well_row_list else []
    order = topological_order(sccs, edges, well_row_list) if sccs else []

    blocks: list[Block] = []
    well_row_perm: list[int] = []
    well_col_perm: list[int] = []
    for new_idx, scc_id in enumerate(order):
        rows = sorted(sccs[scc_id])
        cols = [match_row[r] for r in rows]  # 与行对齐：匹配对落在对角线上
        well_row_perm.extend(rows)
        well_col_perm.extend(cols)
        blocks.append(
            Block(
                index=new_idx,
                rows=frozenset(rows),
                cols=frozenset(cols),
                row_positions=frozenset(),  # 全局位置待装配后回填
                col_positions=frozenset(),
            )
        )

    # ---- 欠定/过定条带内部的确定性排列（匹配对靠左上，未匹配靠后） ----
    under_rows = sorted(regions.under_rows)
    under_cols = [match_row[r] for r in under_rows] + sorted(regions.unmatched_cols)
    over_cols = sorted(regions.over_cols)
    over_rows = [match_col[c] for c in over_cols] + sorted(regions.unmatched_rows)

    row_perm = tuple(under_rows + well_row_perm + over_rows)
    col_perm = tuple(under_cols + well_col_perm + over_cols)

    row_pos = {r: i for i, r in enumerate(row_perm)}
    col_pos = {c: i for i, c in enumerate(col_perm)}

    bounds: list[tuple[int, int, int, int]] = []
    pos = len(under_rows)
    cpos = len(under_cols)
    filled_blocks: list[Block] = []
    for blk in blocks:
        s = blk.size
        rpos_set = frozenset(row_pos[r] for r in blk.rows)
        cpos_set = frozenset(col_pos[c] for c in blk.cols)
        filled_blocks.append(
            Block(
                index=blk.index,
                rows=blk.rows,
                cols=blk.cols,
                row_positions=rpos_set,
                col_positions=cpos_set,
            )
        )
        bounds.append((pos, pos + s, cpos, cpos + s))
        pos += s
        cpos += s

    btf = BlockTriangularForm(
        blocks=tuple(filled_blocks),
        well_row_perm=tuple(well_row_perm),
        well_col_perm=tuple(well_col_perm),
    )

    return StructuralAnalysis(
        n_rows=nr,
        n_cols=nc,
        structural_rank=rank,
        nonzeros=frozenset(nz),
        match_row=tuple(match_row),
        match_col=tuple(match_col),
        regions=regions,
        btf=btf,
        row_perm=row_perm,
        col_perm=col_perm,
        block_bounds=tuple(bounds),
    )


def structural_rank(
    nonzeros: Iterable,
    n_rows: int | None = None,
    n_cols: int | None = None,
    *,
    max_steps: int | None = None,
) -> int:
    """仅求结构秩（最大匹配基数）的便捷函数。"""
    return analyze(nonzeros, n_rows, n_cols, max_steps=max_steps).structural_rank


def verify_certificate(
    nonzeros: Iterable,
    match_row: Sequence[int],
    match_col: Sequence[int],
    n_rows: int | None = None,
    n_cols: int | None = None,
) -> VerifyReport:
    """独立复核匹配证书：合法性 + König 最小顶点覆盖证明。

    复核逻辑（König 定理）与求解用的增广路算法相互独立：从所有未匹配行
    沿交替路搜索，取“不可达行 ∪ 可达列”为顶点覆盖。当且仅当该覆盖大小
    等于匹配基数时匹配为最大匹配；同时逐边检查覆盖确实覆盖全部非零元。
    """
    nr, nc, nz, row_adj, _ = _coerce(nonzeros, n_rows, n_cols)

    mr = list(match_row)
    mc = list(match_col)
    if len(mr) != nr or len(mc) != nc:
        raise InvalidStructureError(
            f"证书长度不符：match_row 需 {nr} 项（收到 {len(mr)}），"
            f"match_col 需 {nc} 项（收到 {len(mc)}）"
        )
    for r, c in enumerate(mr):
        if not isinstance(c, int) or isinstance(c, bool) or not (-1 <= c < nc):
            raise InvalidStructureError(f"match_row[{r}]={c!r} 非法")
    for c, r in enumerate(mc):
        if not isinstance(r, int) or isinstance(r, bool) or not (-1 <= r < nr):
            raise InvalidStructureError(f"match_col[{c}]={r!r} 非法")

    # 1) 匹配合法性：每条配对边是非零元，且双向映射一致。
    for r, c in enumerate(mr):
        if c != -1:
            if (r, c) not in nz:
                raise InvalidStructureError(f"证书非法：({r}, {c}) 不是非零位置")
            if mc[c] != r:
                raise InvalidStructureError(f"证书非法：match_row[{r}]={c} 但 match_col[{c}]={mc[c]}")
    for c, r in enumerate(mc):
        if r != -1 and mr[r] != c:
            raise InvalidStructureError(f"证书非法：match_col[{c}]={r} 与 match_row 不一致")
    rank = sum(c != -1 for c in mr)

    # 2) König 覆盖：未匹配行起交替可达。
    reach_rows, reach_cols = alternating_reach_from_rows(
        [r for r in range(nr) if mr[r] == -1], row_adj, mc
    )
    cover_rows = frozenset(set(range(nr)) - reach_rows)
    cover_cols = frozenset(reach_cols)

    valid = True
    if len(cover_rows) + len(cover_cols) != rank:
        valid = False
    for r, c in nz:  # 覆盖必须覆盖每一条边，否则证书不是最大匹配。
        if r not in cover_rows and c not in cover_cols:
            valid = False
            break

    return VerifyReport(
        valid=valid,
        structural_rank=rank,
        vertex_cover_rows=cover_rows,
        vertex_cover_cols=cover_cols,
    )
