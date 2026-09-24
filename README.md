# 稀疏矩阵的结构秩与块三角化（sparsestruct）

离线稀疏数值预处理用的**纯组合结构分析**库：只读取非零元的位置，不读取、
推断或消元任何数值。基于行列二部图的最大匹配求**结构秩**，按
Dulmage–Mendelsohn 可达性划分欠定 / 良定 / 过定三区，并对良定匹配图做
强连通分量（SCC）块三角排序，输出行列置换、块边界与可独立核验的匹配证书。

- **Python 版本**：Python 3.14.7（仅使用标准库，`dataclasses` / `collections` / `math`）
- **依赖**：无第三方依赖，安装 Python 后即可运行；核心、测试、演示全部离线
- **输入上限**：矩阵维度与下标为接口硬边界：行数、列数均 ≤ **100**，
  下标范围 0..99；非零位置数量无单独上限（至多 100×100=10000 个去重位置）
- **浮点口径**：本库的输入是整数位置、输出是离散的置换/分区/块边界，
  **不产生任何浮点结果，因此不存在浮点误差或容差**。唯一接触浮点数的地方
  是输入校验：`NaN` / `Infinity` 一律拒绝（见“错误语义”）

## 运行命令

```bash
# 必要测试（静态入口 + 批量增删；独立枚举参考 + 结构检查），约 0.15 秒
python -m unittest discover -s tests -v

# 固定输入演示：静态分析 + 批量增删（正常/边界/原子失败/预算耗尽），0.1 秒内
python demo.py
```

两条命令都不需要网络、账号、密钥、数据库或任何额外安装。

## 公开接口

```python
from sparsestruct import analyze, verify, StructuralDecomposition, Block
from sparsestruct import SparsePattern, UpdateReport
from sparsestruct import (
    StructuralError,            # 本库错误基类（ValueError 子类）
    InvalidStructureError,      # 非法结构
    LimitExceededError,          # 超过 100×100 接口边界
    SearchBudgetExhausted,       # 搜索预算耗尽（不等于证明无解）
)

d = analyze(entries, n_rows=None, n_cols=None, *, budget=None)

p = SparsePattern(entries=(), n_rows=None, n_cols=None)
report = p.add(entries)          # 批量增加非零位置
report = p.remove(entries)       # 批量删除非零位置
report = p.update(add=..., remove=...)  # 一次提交批量增删
p.resize(n_rows=None, n_cols=None)     # 显式调整维度
d = p.result                     # 当前 StructuralDecomposition（每次更新后自动重算）
```

### `analyze(entries, n_rows=None, n_cols=None, *, budget=None)`

- `entries`：`(row, col)` 非零位置的可迭代对象（列表、生成器、集合均可）。
  **重复位置自动去重**，结果与传入顺序无关（排序规范化，输出确定）。
- `n_rows` / `n_cols`：可选显式维度。省略时由“最大下标 + 1”推断
  （空集合推断为 0×0）。显式给定时，位置超出维度报
  `InvalidStructureError`。
- `budget`：可选非负整数，交替路径搜索中每考察一条二部边计 1。
  - `None`（默认）：完整搜索，结束时给出**经证明**的最大匹配与结构秩；
  - 整数：预算耗尽时抛 `SearchBudgetExhausted`。该异常**只表示搜索没做完**，
    绝不表示“证明无增广路 / 证明无解”——两者在类型与文档上严格分开。

返回不可变（`frozen dataclass`）的 `StructuralDecomposition`：

| 字段 | 含义 |
|---|---|
| `n_rows`, `n_cols` | 维度 |
| `nonzeros` | 去重后的非零位置 `frozenset` |
| `structural_rank` | 结构秩 = 最大匹配基数 |
| `row_match`, `col_match` | 匹配证书；未匹配为 `-1`，两侧互逆 |
| `unmatched_rows`, `unmatched_cols` | 未匹配行 / 列（升序） |
| `row_perm`, `col_perm` | 行列置换，`row_perm[i]` 是置换后第 i 行的原始行号 |
| `row_blocks`, `col_blocks` | 各对角块在置换后坐标上的半开区间 |
| `block_regions` | 每块区域：`'U'` 欠定 / `'W'` 良定 / `'O'` 过定 |
| `row_region`, `col_region` | 每个原始行 / 列的区域标签 |

方法：`iter_blocks()` 返回 `Block`（含块形状与块内原始行列号）；
`permuted_nonzeros()` 返回置换后坐标的非零位置；`spy()` 返回 ASCII 散点图。

### `verify(decomposition) -> True`

独立重算核验证书（与主算法共享图模型但搜索代码独立）：

1. 置换是合法排列；匹配边都是非零边且两侧互逆；秩等于匹配基数；
2. 用独立分层 BFS 确认**不存在增广路**（匹配确实最大），篡改的非最大
   匹配证书会被检出并报“存在增广路”；
3. 块边界连续覆盖整个维度；置换后所有非零元只在对角块及其右上方
   （块上三角），匹配元落在各块的块内相对主对角；
4. 每个良定块独立 BFS 复验为强连通分量（**不可约**）。

失败抛 `StructuralError`，成功返回 `True`。

### `SparsePattern`：有状态模式与批量增删

需要在同一张图上反复增删非零位置时使用。它保存去重边集、维度和最近
一次分解证书；批量入口**先完整校验整批输入，通过后才一次性提交并重算**，
任何非法输入都抛异常且状态不变（不存在半更新）。重算走与 `analyze`
相同的共享管线（本迭代为全量重算，非增量算法），因此更新后的证书与对
同一位置集合调用 `analyze(..., n_rows, n_cols)` **逐字段一致**。

```python
p = SparsePattern([(0, 0), (0, 1), (1, 0)], n_rows=3, n_cols=3)
p.result                 # 当前 StructuralDecomposition（构造时即算好）
rep = p.add([(1, 1), (2, 2)])     # 批量加；重复位置计入 unchanged
rep = p.remove([(0, 0)])          # 批量删（可删匹配边，匹配/分区整体重算）
rep = p.update(add=[(0, 2)], remove=[(2, 2)])  # 先删后增，一次提交
p.nonzeros                # frozenset 快照；len(p)、(r,c) in p、iter(p) 可用
```

- 构造参数与 `analyze` 相同：`entries` 可迭代、可选显式 `n_rows/n_cols`
  （0..100）。省略维度时按最大下标 +1 推断；**推断维度只增不减**
  （删除不收缩，可用 `resize` 显式调整）；显式维度在构造后固定，向其外
  新增位置报 `InvalidStructureError`。
- `add` / `remove` 接受任意可迭代坐标对（含生成器），自动去重；空批次
  是合法空操作，返回秩不变的台账。
- `update(add=..., remove=...)` 先删后增；同一位置同时出现在两批时
  **以增加为准**（最终存在）。
- 返回不可变台账 `UpdateReport`：

| 字段 | 含义 |
|---|---|
| `added` / `removed` | 实际新增 / 删除的去重位置（升序） |
| `unchanged` | 批次中本就存在（add）/本就不存在（remove）的位置 |
| `rank_before` / `rank_after` | 更新前后结构秩 |
| `n_nonzeros` | 更新后去重非零位置总数 |
| `rebuilt` | 是否全量重算（本版本恒为 `True`） |

  批内重复位置只按*更新前*状态唯一归类一次（例如对已存在位置重复 add，
  只计一次 `unchanged`）。
- 其他方法：`p.result` / `p.decomposition()` 取当前证书；`p.contains((r,c))`
  查询（严格校验下标；`in` 运算符按容器惯例对非法下标返回 `False`）；
  `p.resize(n_rows=, n_cols=)` 显式扩缩维度（缩到放不下现有边时拒绝且
  状态不变）；`p.recompute()` 显式触发一次全量重算。
- 删除匹配边后，最大匹配、结构秩、欠定/良定/过定分区与良定块全部重算：
  秩可能下降（该行失去唯一可匹配列），也可能因交替路改配而保持
  （匹配证书随之变化）；一切以重算结果与 `verify()` 为准。

## 错误语义（可定位，不静默纠正）

- 下标必须是 `int`：**`bool` 被明确拒绝**（`True`/`False` 不当作 1/0），
  普通 `float` 下标拒绝（`TypeError`），`float('nan')`/`float('inf')`
  拒绝（`InvalidStructureError`）；维度、`budget` 参数同规则。
- 负下标、坐标对长度 ≠ 2、位置超出显式维度：`InvalidStructureError`，
  错误信息带可定位前缀，如 `entries[7] 的列号: ...`；批量入口的定位
  前缀分别为 `add 批次[i]` / `remove 批次[i]` /
  `update 的 add 批次[i]`（或 `remove 批次[i]`）。
- 行号 ≥ 100 或列号 ≥ 100、维度 > 100：`LimitExceededError`。
- 分析是纯函数：任何校验失败都在构造结果前抛出，**不存在部分变更**；
  输入可迭代对象不会被就地修改。
- `SparsePattern` 的批量增删同样原子：整批校验先于任何状态修改，
  失败后边集、维度与缓存证书全部保持更新前状态；调用方传入的容器
  不被修改。

## 设计取舍

1. **结构判断不使用数值消元**。最大匹配用 Kuhn 增广路算法（上界
   O(V·E)，100×100 内为微秒~毫秒级）；“无增广路”由完整 BFS/DFS 终止
   性给出，是组合意义上的证明。`analyze` 与 `SparsePattern` 共用同一
   条“规范化图模型 → 最大匹配 → DM 分区 → SCC → 证书”管线（提取为
   内部 `_decompose`），因此全量重算结果逐字段一致，不存在两套实现
   漂移的可能。
2. **区域划分（Dulmage–Mendelsohn）**：以匹配边（列→行）与非匹配边
   （行→列）构成交替有向图。从未匹配行前向可达 = 过定区 O；
   沿反向边从未匹配列可达 = 欠定区 U；其余 = 良定区 W。三区与*选择哪一个*
   最大匹配无关（测试用枚举得到的另一最大匹配交叉验证）。
   未匹配行必在 O、未匹配列必在 U；最大匹配下 U∩O 必为空（否则存在增广路）。
3. **良定区块三角化**：以良定行为顶点，非零元 `(r,c)` 且列 c 匹配到 r′
   给出有向边 r→r′（匹配边为自环）；其 SCC 即不可约块，Kosaraju
   （迭代实现）输出拓扑序。列置换与行置换按匹配配对，使匹配元落在主对角，
   全部非零元只在对角块及其上方，得到块上三角形式。
4. **块排列固定为 U | W(各 SCC 拓扑序) | O**；U 块允许 0 行（如 0×3）、
   O 块允许 0 列（如 3×0），仍各自作为一个对角块输出。
5. **确定性**：所有邻接表排序、SCC 内顶点升序；相同位置集合无论输入顺序
   都产生同一结果。
6. **证书可核验**：匹配、置换、区域、块边界全部显式返回，`verify()`
   独立重算；测试中的期望值来自 `tests/reference.py` 的**回溯枚举所有匹配**
   与传递闭包 SCC（小矩阵专用），该参考不导入被测库；往返一致仅作补充。

## 已知限制 / 非目标

- 仅实现本题契约的数学子集：输入为二值的非零位置集合，不读数值、不计算
  数值秩、不做符号/容差判断，也不宣称兼容 Harwell-Boeing、Matrix Market
  等行业格式。
- 维度硬上限 100×100；超出即拒绝，不做大规模稀疏的伸缩优化。
- `SparsePattern` 的增删当前一律全量重算（与 `analyze` 共享管线），
  不做动态局部增量匹配/增量 SCC；100×100 内为微秒~毫秒级，更大规模
  的增量复杂度是明确的非目标。台账 `rebuilt` 字段当前恒为 `True`，
  预留给将来的增量实现。
- 块三角化的块顺序是拓扑序之一（确定性的一种），不保证对所有可能的
  拓扑序做枚举；不可约块（SCC）本身的划分是唯一的。
- `budget` 是搜索中断机制而非性能契约；预算耗尽时无结果返回（纯函数，
  无部分状态）。`SparsePattern` 的重算不接受 `budget`（始终完整搜索，
  保证更新后证书是证明后的最大匹配）。
- 不提供网站 / CLI 产品 / 报表；终端命令仅用于测试与演示。

## 仓库结构

```
sparsestruct/__init__.py    # 本题库：校验、最大匹配、DM 分区、SCC、
                            #   verify、SparsePattern 批量增删
tests/reference.py          # 独立参考：枚举所有匹配 + 传递闭包 SCC（不导入核心）
tests/test_sparsestruct.py  # 静态入口 analyze 的必要测试（首轮留档，全部保留）
tests/test_batch_updates.py # 批量增删：正常/边界/失败原子性/一致性测试
demo.py                     # 固定输入演示（确定性数据，真实计算）
README.md
```
