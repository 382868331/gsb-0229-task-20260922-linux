# 稀疏矩阵的结构秩、块三角化与最少专用输入选址（sparsestruct）

离线稀疏数值预处理用的**纯组合结构分析**库：只读取非零元的位置，不读取、
推断或消元任何数值。基于行列二部图的最大匹配求**结构秩**，按
Dulmage–Mendelsohn 可达性划分欠定 / 良定 / 过定三区，并对良定匹配图做
强连通分量（SCC）块三角排序，输出行列置换、块边界与可独立核验的匹配证书。
在此之上，本库还把方阵非零模式视作状态有向图，给出**最少专用输入选址**
（结构可控性意义下，每个输入只直接作用于一个状态）。

- **Python 版本**：Python 3.14.7（仅使用标准库，`dataclasses` / `collections` / `math`）
- **依赖**：无第三方依赖，安装 Python 后即可运行；核心、测试、演示全部离线
- **输入上限**：矩阵维度与下标为接口硬边界：行数、列数均 ≤ **100**，
  下标范围 0..99；非零位置数量无单独上限（至多 100×100=10000 个去重位置）。
  输入选址的状态数另有更严上限：状态数 ≤ **20**
- **浮点口径**：本库的输入是整数位置、输出是离散的置换/分区/块边界，
  **不产生任何浮点结果，因此不存在浮点误差或容差**。唯一接触浮点数的地方
  是输入校验：`NaN` / `Infinity` 一律拒绝（见“错误语义”）

## 运行命令

```bash
# 准备虚拟环境（可选但推荐；本库零第三方依赖，venv 仅为隔离）
python3 -m venv .venv
. .venv/bin/activate        # 或者直接使用 .venv/bin/python 代替 python

# 必要测试（独立枚举参考 + 结构检查），约 0.1 秒
python -m unittest discover -s tests -v

# 固定输入演示 1：结构秩与块三角化（含不可约块）+ 异常边界，约 0.1 秒内
python demo.py

# 固定输入演示 2：最少专用输入选址（现场枚举计算）+ 异常边界，远小于 90 秒
python demo_iteration.py
```

以上命令都不需要网络、账号、密钥、数据库或任何额外安装。

## 公开接口

```python
from sparsestruct import analyze, verify, StructuralDecomposition, Block
from sparsestruct import minimum_dedicated_inputs, InputPlacement
from sparsestruct import (
    StructuralError,            # 本库错误基类（ValueError 子类）
    InvalidStructureError,      # 非法结构
    LimitExceededError,          # 超过 100×100 接口边界
    SearchBudgetExhausted,       # 搜索预算耗尽（不等于证明无解）
)

d = analyze(entries, n_rows=None, n_cols=None, *, budget=None)
p = minimum_dedicated_inputs(entries, n_states=None)
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

### `minimum_dedicated_inputs(entries, n_states=None)`

把**方阵**非零模式视作状态有向图：非零 `(i, j)` 即边 `j -> i`
（列状态 j 驱动行状态 i，对应 `ẋ = Ax` 的稀疏模式）。求**最少专用
输入选址**：每个输入只直接作用于一个状态，所选状态集合同时满足两条
结构条件——

1. **增广结构秩条件（无膨胀）**：在 `A` 的二部图上为每个被选状态
   `s` 增加一个只连接行 `s` 的输入列后，存在覆盖全部 `n` 行的
   匹配（即 `[A B]` 模式满结构秩）——只数结构秩缺口是不够的；
2. **源 SCC 可达条件**：状态图的每个源强连通分量（缩点后无入边的
   分量）都含有至少一个被选状态，等价于所有状态都可由输入到达。

在满足两条件的集合中取大小最小者；**并列按状态 ID 升序元组取字典序
最小者**。下界 `max(源 SCC 数, 结构秩缺口)` 起按子集规模枚举，
`n ≤ 20` 内为毫秒级。`n_states` 省略时由最大下标 +1 推断，推断
结果必须是方阵，否则报 `InvalidStructureError`；状态数超过 20 报
`LimitExceededError`。

返回不可变的 `InputPlacement`：

| 字段 | 含义 |
|---|---|
| `n_states`, `nonzeros` | 状态数与去重后的非零位置 |
| `structural_rank`, `rank_deficiency` | `A` 自身结构秩与秩缺口 |
| `col_match`, `unmatched_states` | `A` 自身最大匹配证书与未匹配状态列 |
| `selected_states` | 所选状态（升序，字典序最小解） |
| `row_assign` | 增广匹配证据：行 → 状态列，`-1` 表示该行由专用输入覆盖 |
| `source_sccs`, `scc_inputs` | 各源 SCC 及覆盖它的输入状态（一一对齐） |

> 注意：这是**结构**可控性结论（只依赖非零模式），不代替任何具体
> 数值矩阵的可控性判断。

## 错误语义（可定位，不静默纠正）

- 下标必须是 `int`：**`bool` 被明确拒绝**（`True`/`False` 不当作 1/0），
  普通 `float` 下标拒绝（`TypeError`），`float('nan')`/`float('inf')`
  拒绝（`InvalidStructureError`）；维度、`budget` 参数同规则。
- 负下标、坐标对长度 ≠ 2、位置超出显式维度：`InvalidStructureError`，
  错误信息带可定位前缀，如 `entries[7] 的列号: ...`。
- 行号 ≥ 100 或列号 ≥ 100、维度 > 100：`LimitExceededError`。
- 分析是纯函数：任何校验失败都在构造结果前抛出，**不存在部分变更**；
  输入可迭代对象不会被就地修改。

## 设计取舍

1. **结构判断不使用数值消元**。最大匹配用 Kuhn 增广路算法（上界
   O(V·E)，100×100 内为微秒~毫秒级）；“无增广路”由完整 BFS/DFS 终止
   性给出，是组合意义上的证明。
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
7. **输入选址的精确枚举**：选址条件（增广结构秩满秩 + 源 SCC 全覆盖）
   直接按定义判定，从下界 `max(源 SCC 数, 秩缺口)` 起枚举子集，
   第一个可行者即最小且字典序最小解。测试用
   `tests/test_input_placement.py` 内**独立**的 BFS 可达性与回溯
   增广匹配参考逐字段对照（不调用被测库），覆盖满结构秩多孤立环、
   链、零矩阵与并列字典序情形。

## 已知限制 / 非目标

- 仅实现本题契约的数学子集：输入为二值的非零位置集合，不读数值、不计算
  数值秩、不做符号/容差判断，也不宣称兼容 Harwell-Boeing、Matrix Market
  等行业格式。
- 维度硬上限 100×100；超出即拒绝，不做大规模稀疏的伸缩优化。
- 输入选址状态数硬上限 20；选址是**结构**可控性结论，不代替具体
  数值矩阵的可控性判断；每个输入只直接作用于一个状态（专用输入），
  不讨论一个输入驱动多个状态的最少输入数问题。
- 块三角化的块顺序是拓扑序之一（确定性的一种），不保证对所有可能的
  拓扑序做枚举；不可约块（SCC）本身的划分是唯一的。
- `budget` 是搜索中断机制而非性能契约；预算耗尽时无结果返回（纯函数，
  无部分状态）。
- 不提供网站 / CLI 产品 / 报表；终端命令仅用于测试与演示。

## 仓库结构

```
sparsestruct/__init__.py        # 本题库：校验、最大匹配、DM 分区、SCC、verify、输入选址
tests/reference.py              # 独立参考：枚举所有匹配 + 传递闭包 SCC（不导入核心）
tests/test_sparsestruct.py      # 结构秩与块三角化测试
tests/test_input_placement.py   # 输入选址测试（独立枚举参考对照 + 异常路径）
demo.py                         # 结构秩与块三角化演示（确定性数据，真实计算）
demo_iteration.py               # 最少专用输入选址演示（现场枚举计算）
README.md
```
