# 稀疏矩阵的结构秩、块三角化与最少专用输入选址（sparsestruct）

离线稀疏数值预处理用的**纯组合结构分析**库：只读取非零元的位置，不读取、
推断或消元任何数值。基于行列二部图的最大匹配求**结构秩**，按
Dulmage–Mendelsohn 可达性划分欠定 / 良定 / 过定三区，并对良定匹配图做
强连通分量（SCC）块三角排序，输出行列置换、块边界与可独立核验的匹配证书。
此外把方阵非零模式视作状态有向图，给出**结构可控**所需的**最少专用输入
选址**：每个输入只直接作用于一个状态，同时满足最大匹配未匹配点覆盖与
源强连通分量可达两条要求（不只是数结构秩缺口）。

- **Python 版本**：Python 3.14.7（仅使用标准库，`dataclasses` / `collections` / `math`）
- **依赖**：无第三方依赖，安装 Python 后即可运行；核心、测试、演示全部离线
- **输入上限**：矩阵维度与下标为接口硬边界：行数、列数均 ≤ **100**，
  输入选址接口的状态数（方阵阶数）≤ **20**
- **浮点口径**：本库的输入是整数位置、输出是离散的置换/分区/块边界，
  **不产生任何浮点结果，因此不存在浮点误差或容差**。唯一接触浮点数的地方
  是输入校验：`NaN` / `Infinity` 一律拒绝（见“错误语义”）

## 环境准备与运行命令

```bash
# 准备 venv（可选；库只用标准库，系统 Python 3.14.7 亦可直接运行）
python3.14 -m venv .venv
source .venv/bin/activate

# 必要测试（独立枚举参考 + 结构检查），约 0.3 秒
python -m unittest discover -s tests -v

# 固定输入演示一：结构秩与块三角化（不可约块 + 异常边界），约 0.1 秒内
python demo.py

# 固定输入演示二：最少专用输入选址（满秩多孤立环 / 链 / 零矩阵 / 混合 +
# 实际触发的异常边界），全部现场计算，远低于 90 秒
python demo_iteration.py
```

三条命令都不需要网络、账号、密钥、数据库、Docker 或任何额外安装。

## 公开接口

```python
from sparsestruct import analyze, verify, StructuralDecomposition, Block
from sparsestruct import (
    select_dedicated_inputs, InputPlacement,
    MAX_STATES,                  # 输入选址状态数上限 20
    StructuralError,            # 本库错误基类（ValueError 子类）
    InvalidStructureError,      # 非法结构
    LimitExceededError,          # 超过接口边界（100×100 / 20 状态）
    SearchBudgetExhausted,       # 搜索预算耗尽（不等于证明无解）
)

d = analyze(entries, n_rows=None, n_cols=None, *, budget=None)
p = select_dedicated_inputs(entries, n_states=None)
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

### `select_dedicated_inputs(entries, n_states=None)`

把 n 阶方阵的非零模式视作状态有向图：非零元 `(i, j)`（A[i][j] ≠ 0）表示
有向边 `j -> i`（状态 j 直接影响状态 i）。求**结构可控**所需的最少专用
输入选址——每个输入只直接作用于一个状态（B 的列为单位向量）。

- `entries`：`(i, j)` 非零位置可迭代集合，校验与去重规则同 `analyze`。
- `n_states`：可选显式方阵阶数 0..20；省略时取行列两个方向推断值的
  较大者。超过 20 报 `LimitExceededError`。
- 返回不可变的 `InputPlacement`：

| 字段 | 含义 |
|---|---|
| `states` | 选中的输入状态（升序）；并列时取状态 ID 集合字典序最小者 |
| `n_inputs` | 最少输入数 = `max(源 SCC 数, n - 结构秩)` |
| `structural_rank` / `rank_deficiency` | 结构秩与秩缺口 |
| `row_match` / `col_match` / `unmatched_states` | 匹配证据：删去所选行后其余行全部被匹配，未匹配行恰好是所选状态 |
| `source_sccs` | 状态有向图的全部源 SCC（无外部入边的强连通分量） |
| `scc_coverage` | 与 `source_sccs` 对齐：每个源 SCC 被哪个输入状态覆盖 |

可行性判据（Lin 1974 专用输入情形）同时要求两条，**不能只数结构秩缺口**：

1. 可达性：每个源 SCC 至少含一个被选状态；
2. 无膨胀：删去被选状态对应的行后，剩余行在行列二部图中全部可匹配。

实现按规模从小到大、同规模按状态 ID 升序枚举候选集合，首个可行者即
答案，因此并列时自动给出字典序最小的状态集合。该结论是*结构*（通用）
可控性，**不代替**对具体数值矩阵的可控性判断（如 PBH 判据）。

### `verify(decomposition) -> True`

独立重算核验证书（与主算法共享图模型但搜索代码独立）：

1. 置换是合法排列；匹配边都是非零边且两侧互逆；秩等于匹配基数；
2. 用独立分层 BFS 确认**不存在增广路**（匹配确实最大），篡改的非最大
   匹配证书会被检出并报“存在增广路”；
3. 块边界连续覆盖整个维度；置换后所有非零元只在对角块及其右上方
   （块上三角），匹配元落在各块的块内相对主对角；
4. 每个良定块独立 BFS 复验为强连通分量（**不可约**）。

失败抛 `StructuralError`，成功返回 `True`。

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

## 已知限制 / 非目标

- 仅实现本题契约的数学子集：输入为二值的非零位置集合，不读数值、不计算
  数值秩、不做符号/容差判断，也不宣称兼容 Harwell-Boeing、Matrix Market
  等行业格式。
- 输入选址是**结构可控性**结论：只回答“非零模式允许可控”，不代替对
  具体数值矩阵的可控性判断；某些数值矩阵即使模式可行也可能数值不可控。
- 输入选址状态数硬上限 20；候选集合按字典序枚举，病态图（下界接近
  n/2 且大量候选不可行）下枚举可能明显变慢，演示与测试规模均在毫秒级。
- 只支持专用输入（每个输入作用于单个状态），不处理多状态共享输入的
  一般选址问题。
- 维度硬上限 100×100；超出即拒绝，不做大规模稀疏的伸缩优化。
- 块三角化的块顺序是拓扑序之一（确定性的一种），不保证对所有可能的
  拓扑序做枚举；不可约块（SCC）本身的划分是唯一的。
- `budget` 是搜索中断机制而非性能契约；预算耗尽时无结果返回（纯函数，
  无部分状态）。
- 不提供网站 / CLI 产品 / 报表；终端命令仅用于测试与演示。

## 仓库结构

```
sparsestruct/__init__.py      # 本题库：校验、最大匹配、DM 分区、SCC、verify、输入选址
tests/reference.py            # 独立参考：枚举匹配 + 传递闭包 SCC + 枚举输入集合（不导入核心）
tests/test_sparsestruct.py    # 结构秩与块三角化测试
tests/test_input_placement.py # 输入选址测试（独立参考对照 + 异常路径）
demo.py                       # 块三角化演示（确定性数据，真实计算）
demo_iteration.py             # 最少专用输入选址演示（现场计算，< 90 秒）
README.md
```
