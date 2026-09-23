"""structrank —— 稀疏矩阵结构秩与块三角化（仅基于非零位置）。

公开接口：

* :func:`analyze` —— 一次返回结构秩、匹配证书、DM 三区、不可约块与置换；
* :func:`structural_rank` —— 仅取结构秩；
* :func:`dm_regions` —— 给定最大匹配求 Dulmage–Mendelsohn 三区；
* :func:`verify_certificate` —— 用 König 顶点覆盖独立复核匹配证书。

纯 Python 3.14 标准库实现，全程不做数值消元。
"""

from .errors import (
    InvalidStructureError,
    SearchBudgetExhaustedError,
    StructuralRankError,
)
from .facade import analyze, dm_regions, structural_rank, verify_certificate
from .types import (
    Block,
    BlockTriangularForm,
    Regions,
    StructuralAnalysis,
    VerifyReport,
)

MAX_DIM = 100

__all__ = [
    "MAX_DIM",
    "analyze",
    "structural_rank",
    "dm_regions",
    "verify_certificate",
    "StructuralAnalysis",
    "Regions",
    "Block",
    "BlockTriangularForm",
    "VerifyReport",
    "StructuralRankError",
    "InvalidStructureError",
    "SearchBudgetExhaustedError",
]
