"""输入解析与校验：只接受符合契约的结构信息，错误必须可定位。"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from typing import Any

from .errors import InvalidStructureError

MAX_DIM = 100


def _reject_bool(obj: Any, what: str) -> None:
    # bool 是 int 的子类，按契约整数参数必须显式拒绝 bool。
    if isinstance(obj, bool):
        raise InvalidStructureError(f"{what} 不得为 bool 类型（收到 {obj!r}）")


def _is_real_number(obj: Any) -> bool:
    # 只接受真正的 int/float（排除 bool、complex、字符串等）。
    if isinstance(obj, bool):
        return False
    return isinstance(obj, (int, float))


def _check_finite(value: float, where: str) -> None:
    if not math.isfinite(value):
        # NaN 与 Infinity 一律拒绝，不参与任何结构判断。
        raise InvalidStructureError(f"{where} 必须有限，拒绝 NaN/Infinity（收到 {value!r}）")


def normalize_nonzeros(
    nonzeros: Iterable[Any],
    n_rows: Any,
    n_cols: Any,
) -> tuple[int, int, set[tuple[int, int]]]:
    """把非零位置集合规范化为 ``set[(row, col)]``。

    接受两种形式：

    1. ``nonzeros`` 为 ``(row, col)`` 对的可迭代对象，必须同时给出
       ``n_rows`` / ``n_cols``（正整数或 None 自动推断）；
    2. 稀疏矩阵的类序列输入：行可迭代对象，每行是列下标的可迭代对象
       （该形式下 ``n_rows`` 必须为 None，行数由外层长度决定）。

    位置下标必须为非负整数（``bool``、浮点一律拒绝）；重复位置静默
    **去重**是本接口明确允许的唯一“规范化”，其余非法输入一律抛出带
    定位信息的 :class:`InvalidStructureError`，不静默纠正。
    """
    _reject_bool(n_rows, "n_rows")
    _reject_bool(n_cols, "n_cols")

    if n_rows is not None:
        if not isinstance(n_rows, int) or isinstance(n_rows, bool):
            raise InvalidStructureError(f"n_rows 必须为 int 或 None（收到 {type(n_rows).__name__}）")
        if not 0 <= n_rows <= MAX_DIM:
            raise InvalidStructureError(f"n_rows 必须在 [0, {MAX_DIM}]（收到 {n_rows}）")
    if n_cols is not None:
        if not isinstance(n_cols, int) or isinstance(n_cols, bool):
            raise InvalidStructureError(f"n_cols 必须为 int 或 None（收到 {type(n_cols).__name__}）")
        if not 0 <= n_cols <= MAX_DIM:
            raise InvalidStructureError(f"n_cols 必须在 [0, {MAX_DIM}]（收到 {n_cols}）")

    if nonzeros is None:
        raise InvalidStructureError("nonzeros 不得为 None")

    # 形式判定（公开契约，按首元素决定整体形式）：
    #   * 坐标对形式：首项是恰好两个数值的 tuple (r, c)；
    #   * 行序列形式：首项可迭代但不是二元数值 tuple（如 [0, 2]、[]）。
    # 二元 list 一律按“一行列下标”解释，不会静默当成坐标对。
    if isinstance(nonzeros, (str, bytes)):
        raise InvalidStructureError("nonzeros 必须是 (row, col) 对的可迭代对象或行的序列")

    try:
        materialized = list(nonzeros)
    except TypeError as exc:
        raise InvalidStructureError("nonzeros 必须可迭代") from exc

    rows_form = False
    if materialized:
        first = materialized[0]
        if (
            not isinstance(first, (str, bytes))
            and isinstance(first, Iterable)
            and not (isinstance(first, tuple) and len(first) == 2
                     and _is_real_number(first[0]) and _is_real_number(first[1]))
        ):
            rows_form = True

    result: set[tuple[int, int]] = set()

    if rows_form:
        if n_rows is not None:
            raise InvalidStructureError(
                "行序列形式（每行是列下标可迭代对象）不允许同时显式传入 n_rows"
            )
        inferred_rows = len(materialized)
        if inferred_rows > MAX_DIM:
            raise InvalidStructureError(f"行数必须 <= {MAX_DIM}（收到 {inferred_rows}）")
        inferred_cols = 0 if n_cols is None else n_cols
        for r, row in enumerate(materialized):
            if isinstance(row, (str, bytes)) or not isinstance(row, Iterable):
                raise InvalidStructureError(f"第 {r} 行必须是列下标的可迭代对象（收到 {row!r}）")
            for item in row:
                c = _valid_index(item, f"第 {r} 行的列下标")
                if c >= MAX_DIM:
                    raise InvalidStructureError(
                        f"列下标必须 < {MAX_DIM}（第 {r} 行收到 c={c}）"
                    )
                if n_cols is not None and c >= n_cols:
                    raise InvalidStructureError(
                        f"位置 ({r}, {c}) 越界：n_cols={n_cols}"
                    )
                inferred_cols = max(inferred_cols, c + 1)
                result.add((r, c))
        if inferred_cols > MAX_DIM:
            raise InvalidStructureError(f"列数必须 <= {MAX_DIM}（推断为 {inferred_cols}）")
        if n_cols is not None:
            inferred_cols = n_cols
        return inferred_rows, inferred_cols, result

    # 坐标对形式。
    inferred_rows = 0 if n_rows is None else n_rows
    inferred_cols = 0 if n_cols is None else n_cols
    for k, item in enumerate(materialized):
        where = f"nonzeros 第 {k} 项"
        if isinstance(item, (str, bytes)) or not isinstance(item, Iterable):
            raise InvalidStructureError(f"{where} 必须是 (row, col) 对（收到 {item!r}）")
        if not isinstance(item, tuple):
            raise InvalidStructureError(
                f"{where} 坐标对必须是 tuple (row, col)；list 按行形式解释"
                f"（收到 {item!r}）"
            )
        pair = list(item)
        if len(pair) != 2:
            raise InvalidStructureError(f"{where} 必须恰好含两个下标（收到长度 {len(pair)}）")
        r = _valid_index(pair[0], f"{where} 的行下标")
        c = _valid_index(pair[1], f"{where} 的列下标")
        if r >= MAX_DIM:
            raise InvalidStructureError(f"{where} 行下标必须 < {MAX_DIM}（收到 r={r}）")
        if c >= MAX_DIM:
            raise InvalidStructureError(f"{where} 列下标必须 < {MAX_DIM}（收到 c={c}）")
        if n_rows is not None and r >= n_rows:
            raise InvalidStructureError(f"{where} 越界：({r}, {c})，n_rows={n_rows}")
        if n_cols is not None and c >= n_cols:
            raise InvalidStructureError(f"{where} 越界：({r}, {c})，n_cols={n_cols}")
        inferred_rows = max(inferred_rows, r + 1)
        inferred_cols = max(inferred_cols, c + 1)
        result.add((r, c))  # set 去重：重复非零位置视为同一个。
    if n_rows is not None:
        inferred_rows = n_rows
    if n_cols is not None:
        inferred_cols = n_cols
    return inferred_rows, inferred_cols, result


def _valid_index(obj: Any, where: str) -> int:
    _reject_bool(obj, where)
    if isinstance(obj, float):
        _check_finite(obj, where)
        if not obj.is_integer():
            raise InvalidStructureError(f"{where} 必须为整数下标（收到 {obj!r}）")
        value = int(obj)
    elif isinstance(obj, int):
        value = obj
    else:
        raise InvalidStructureError(f"{where} 必须为 int（收到 {type(obj).__name__}: {obj!r}）")
    if value < 0:
        raise InvalidStructureError(f"{where} 必须非负（收到 {value}）")
    return value


def iter_adjacency(
    nonzeros: set[tuple[int, int]], n_rows: int, n_cols: int
) -> tuple[list[list[int]], list[list[int]]]:
    """构造行列二部图的行/列邻接表（下标升序，保证结果确定性）。"""
    row_adj: list[list[int]] = [[] for _ in range(n_rows)]
    col_adj: list[list[int]] = [[] for _ in range(n_cols)]
    for r, c in nonzeros:
        row_adj[r].append(c)
        col_adj[c].append(r)
    for adj in row_adj:
        adj.sort()
    for adj in col_adj:
        adj.sort()
    return row_adj, col_adj
