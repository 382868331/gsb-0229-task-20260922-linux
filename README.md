# 稀疏矩阵的结构秩与块三角化（sparsestruct）

离线稀疏数值预处理用的**纯组合结构分析**库：只读取非零元的位置，不读取、
推断或消元任何数值。基于行列二部图的最大匹配求**结构秩**，按
Dulmage–Mendelsohn 可达性划分欠定 / 良定 / 过定三区，并对良定匹配图做
强连通分量（SCC）块三角排序，输出行列置换、块边界与可独立核验的匹配证书。

除一次性的**静态入口** `analyze()` 外，还提供**有状态模型**
`StructuralModel`：对非零模式做批量增、删，随后全量重算最大匹配与三区，
更新过程原子提交（失败不留半更新状态），结果与静态入口逐字段一致。

- **Python 版本**：Python 3.14.7（仅使用标准库，`dataclasses` / `collections` / `math`）
- **依赖**：无第三方依赖，安装 Python 后即可运行；核心、测试、演示全部离线
- **输入上限**：矩阵维度与下标为接口硬边界：行数、列数均 ≤ **100**，
  下标范围 0..99；非零位置数量无单独上限（至多 100×100=10000 个去重位置）
- **浮点口径**：本库的输入是整数位置、输出是离散的置换/分区/块边界，
  **不产生任何浮点结果，因此不存在浮点误差或容差**。唯一接触浮点数的地方
  是输入校验：`NaN` / `Infinity` 一律拒绝（见“错误语义”）

## 运行命令

```bash
# 必要测试（独立枚举参考 + 结构检查，含动态批量更新），约 0.1 秒
python -m unittest discover -s tests -v

# 固定输入演示：静态正常结果（含不可约块）+ 动态批量增删 + 三个实际触发的
# 异常边界，约 0.1 秒内
python demo.py
```

两条命令都不需要网络、账号、密钥、数据库或任何额外安装。

## 公开接口

```python
from sparsestruct import analyze, verify, StructuralDecomposition, Block
from sparsestruct import StructuralModel, UpdateReport  # 有状态批量更新
from sparsestruct import (
    StructuralError,            # 本库错误基类（ValueError 子类）
    InvalidStructureError,      # 非法结构
    LimitExceededError,          # 超过 100×100 接口边界
    SearchBudgetExhausted,       # 搜索预算耗尽（不等于证明无解）
)

d = analyze(entries, n_rows=None, n_cols=None, *, budget=None)
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

### `StructuralModel`：有状态的批量增删 + 全量重算

```python
model = StructuralModel(entries=(), n_rows=None, n_cols=None,
                        *, auto_dim=True, budget=None)
report = model.update(add=(), remove=(), *, budget=省略则沿用模型预算)
report = model.add_edges(entries)      # 只增的便捷入口
report = model.remove_edges(entries)   # 只删的便捷入口
d = model.recompute()                  # 空批次全量重算，返回新证书
```

- **构造**：参数与 `analyze()` 相同（同样的下标/维度/budget 校验），另加
  `auto_dim`：
  - `True`（默认）：`n_rows` / `n_cols` 是初始**下界**，之后加边可自动
    扩大维度；删边**不收缩**维度（空行 / 空列作为显式维度保留）；
  - `False`：维度构造时固定，加边越界报 `InvalidStructureError`。
- **`update(add, remove)`**：一次批次内完成批量新增与删除，然后做一次
  **全量重算**。批次内：
  - 位置自动去重；添加已存在的边记为 `added_duplicates`，删除不存在的边
    记为 `removed_missing`，二者都**不是错误**；
  - 同一位置在同一批次里既增又删属于歧义，报 `InvalidStructureError`；
  - 坐标校验规则与静态入口一致（拒绝 bool / float / NaN / 负数 / 超限），
    错误信息带批次内定位，如 `remove[1] 的行号: ...`。
- **原子性**：先在副本上完成“校验全部输入 → 改边集 → 全量重算”，成功后
  才一次性替换旧状态。任一步失败（非法输入、固定维度越界、
  `SearchBudgetExhausted`）都**不留下半更新状态**：`edges()`、维度与
  `decomposition` 仍是上一次成功的结果。传入的生成器若中途抛错，已产出
  的边也不会入库。
- **结果一致性**：全量重算与 `analyze()` 走同一实现，因此恒有
  `model.decomposition == analyze(model.edges(), model.n_rows, model.n_cols)`
  （测试逐字段断言）。
- 只读视图：`model.n_rows` / `n_cols` / `decomposition` / `budget`、
  `model.edges()`（`frozenset` 快照）、`len(model)`、`(r,c) in model`；
  `model.set_budget(b)` 改后续重算预算。

返回不可变的 `UpdateReport`：

| 字段 | 含义 |
|---|---|
| `added` / `removed` | 本次**实际**新增 / 删除的位置（升序） |
| `added_duplicates` / `removed_missing` | 重复增 / 缺失删（无变化）的位置 |
| `old_rank` / `new_rank` | 更新前后结构秩（加边不减、删边不增） |
| `changed` | 边集是否真的变化 |
| `decomposition` | 更新后的完整 `StructuralDecomposition` 证书 |

## 错误语义（可定位，不静默纠正）

- 下标必须是 `int`：**`bool` 被明确拒绝**（`True`/`False` 不当作 1/0），
  普通 `float` 下标拒绝（`TypeError`），`float('nan')`/`float('inf')`
  拒绝（`InvalidStructureError`）；维度、`budget` 参数同规则。
- 负下标、坐标对长度 ≠ 2、位置超出显式维度：`InvalidStructureError`，
  错误信息带可定位前缀，如 `entries[7] 的列号: ...`；批量更新中为
  `add[3] ...` / `remove[1] ...`。
- 行号 ≥ 100 或列号 ≥ 100、维度 > 100：`LimitExceededError`。
- 分析与更新都是“先校验、后提交”：任何校验失败都在构造结果前抛出，
  **不存在部分变更**；输入可迭代对象不会被就地修改。

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
7. **动态批量更新是全量重算**：`StructuralModel.update()` 复用静态路径
   （同一 `_decompose`），因此“动态结果 == 静态结果”是构造性保证，不做
   增广路的动态局部维护（本期非目标）；维度一旦因加边扩大便不再收缩，
   空行 / 空列始终显式保留，与 `analyze(entries, n_rows, n_cols)` 的显式
   维度语义一致。

## 已知限制 / 非目标

- 仅实现本题契约的数学子集：输入为二值的非零位置集合，不读数值、不计算
  数值秩、不做符号/容差判断，也不宣称兼容 Harwell-Boeing、Matrix Market
  等行业格式。
- 维度硬上限 100×100；超出即拒绝，不做大规模稀疏的伸缩优化。
- 块三角化的块顺序是拓扑序之一（确定性的一种），不保证对所有可能的
  拓扑序做枚举；不可约块（SCC）本身的划分是唯一的。
- `budget` 是搜索中断机制而非性能契约；预算耗尽时无结果返回（静态为纯
  函数、动态为原子不提交，均无部分状态）。
- 动态更新每次整批重算，不维护跨更新的增量匹配/增量分区；无操作历史、
  版本号或撤销接口；维度只扩不缩（需要“收缩”时用 `analyze(model.edges())`
  重新推断维度构造新模型）。
- 不提供网站 / CLI 产品 / 报表；终端命令仅用于测试与演示。

## 仓库结构

```
sparsestruct/__init__.py   # 本题库：校验、最大匹配、DM 分区、SCC、verify、
                           # StructuralModel 批量增删与全量重算
tests/reference.py         # 独立参考：枚举所有匹配 + 传递闭包 SCC（不导入核心）
tests/test_sparsestruct.py # 静态入口必要测试（原样保留）
tests/test_dynamic.py      # 动态批量更新测试：匹配边删除、空行、矩形、
                           # 非法更新原子性、枚举参考随机游走
demo.py                    # 固定输入演示（静态 + 动态，确定性数据，真实计算）
README.md
```
