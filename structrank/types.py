"""公开数据结构（冻结、可比较、可复现）。

约定：矩阵为 ``n_rows`` 行 × ``n_cols`` 列；行列下标均从 0 开始。
所有置换都是“新位置 -> 原下标”的映射，即 ``perm[i] = j`` 表示新顺序的第
``i`` 个位置放原来的第 ``j`` 行/列。
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "Regions",
    "Block",
    "BlockTriangularForm",
    "StructuralAnalysis",
    "VerifyReport",
]


@dataclass(frozen=True)
class Regions:
    """Dulmage–Mendelsohn 三区划分（按原矩阵下标给出，两两不交）。

    * ``under_rows`` / ``under_cols``：欠定区（列多于行的横向条带）；
    * ``well_rows`` / ``well_cols``：良定区（可完美匹配的方形成分）；
    * ``over_rows`` / ``over_cols``：过定区（行多于列的纵向条带）；
    * ``unmatched_rows``：最大匹配未覆盖的行（``over_rows`` 的子集）；
    * ``unmatched_cols``：最大匹配未覆盖的列（``under_cols`` 的子集）。

    六个分区集合（under/well/over 各行列）两两不交且覆盖全部顶点；
    孤立行落入过定区，孤立列落入欠定区。
    """

    under_rows: frozenset[int]
    under_cols: frozenset[int]
    well_rows: frozenset[int]
    well_cols: frozenset[int]
    over_rows: frozenset[int]
    over_cols: frozenset[int]
    unmatched_rows: frozenset[int]
    unmatched_cols: frozenset[int]


@dataclass(frozen=True)
class Block:
    """良定区粗粒度块三角化中的一个对角块（不可约块）。

    块内行集合与列集合等大；``rows``/``cols`` 为**原矩阵下标**，
    ``row_positions``/``col_positions`` 为这些顶点在全局置换中的新位置
    （两者占据同一段位置区间）。每个块不可约：其匹配行有向图是单个强
    连通分量（1×1 块平凡不可约）。
    """

    index: int
    rows: frozenset[int]
    cols: frozenset[int]
    row_positions: frozenset[int]
    col_positions: frozenset[int]

    @property
    def size(self) -> int:
        return len(self.rows)


@dataclass(frozen=True)
class BlockTriangularForm:
    """良定区的块三角（Dulmage–Mendelsohn 粗分解）结果。

    ``blocks`` 已按拓扑序排列：块间非零元只出现在块对角线上方
    （严格块上三角，块序小的行 -> 块序大的列）。
    """

    blocks: tuple[Block, ...]
    well_row_perm: tuple[int, ...]
    well_col_perm: tuple[int, ...]

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)


@dataclass(frozen=True)
class VerifyReport:
    """:func:`structrank.verify_certificate` 的独立复核报告。

    ``valid`` 为真时，``vertex_cover_rows`` ∪ ``vertex_cover_cols`` 是一个
    大小恰好等于 ``structural_rank`` 的顶点覆盖（König 证书），从组合上
    证明匹配基数达到最大，与求解用的增广路算法相互独立。
    """

    valid: bool
    structural_rank: int
    vertex_cover_rows: frozenset[int]
    vertex_cover_cols: frozenset[int]


@dataclass(frozen=True)
class StructuralAnalysis:
    """``analyze`` 的完整结果。

    匹配证书
    --------
    ``match_row``：长度 ``n_rows``，``match_row[r]`` 为与行 r 配对的列，
    未匹配行为 -1；``match_col``：长度 ``n_cols``，反向映射，未匹配列为 -1。
    ``structural_rank`` = 配对数 = 最大匹配基数。证书可用
    :func:`structrank.verify_certificate` 独立复核。

    全局置换
    --------
    ``row_perm`` / ``col_perm``（新位置 -> 原下标）把非零模式排成
    Dulmage–Mendelsohn 标准块形::

              欠定列   良定列(B1|B2|..)   过定列
        欠定行 [  E          *              *  ]
        良定行 [  0       B1  *   *         *  ]
               [  0       0  B2   *         *  ]
               [  0       0   0  B_k        *  ]
        过定行 [  0       0   0   0         O  ]

    E 为欠定横向块（行少列多），O 为过定纵向块（行多列少），B_i 为
    良定区不可约对角块（块对角线全为匹配非零元，块间非零元只在对角线上方
    的严格块上三角位置；右上角欠定行×过定列也允许非零，其余下三角三处
    必为 0）。``block_bounds`` 与块一一对应，每项为
    ``(行起, 行止, 列起, 列止)`` 两个半开位置区间。
    纯欠定/纯过定（良定区为空）时不产生对角块。
    """

    n_rows: int
    n_cols: int
    structural_rank: int
    nonzeros: frozenset[tuple[int, int]]
    match_row: tuple[int, ...]
    match_col: tuple[int, ...]
    regions: Regions
    btf: BlockTriangularForm
    row_perm: tuple[int, ...]
    col_perm: tuple[int, ...]
    block_bounds: tuple[tuple[int, int], ...]
