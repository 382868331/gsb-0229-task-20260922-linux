"""异常类型与错误语义。

所有本库抛出的异常都继承 :class:`StructuralRankError`，便于调用方统一捕获。

错误分两类，严格区分：

* :class:`InvalidStructureError` —— 非法输入被**确定性拒绝**（这是一个证明过的
  结论，不是搜索失败）；
* :class:`SearchBudgetExhaustedError` —— 给定搜索预算内**无法得出结论**
  （既没有证明最优，也没有证明无解），与“证明无解/拒绝”完全分开。
"""


class StructuralRankError(Exception):
    """本库所有异常的公共基类。"""


class InvalidStructureError(StructuralRankError, ValueError):
    """输入违反公开契约（维度、下标、取值等），错误信息带可定位位置。"""


class SearchBudgetExhaustedError(StructuralRankError, RuntimeError):
    """增广路搜索步数预算耗尽，匹配是否最大尚无结论。

    这不是“证明无解”：二部匹配对任何合法图都存在一个（至少是空匹配）。
    该异常仅表示在 ``max_steps`` 限制内算法未能跑完最优性证明。
    异常对象携带部分匹配状态，仅供诊断，不代表最大匹配。
    """

    def __init__(self, budget, steps_used, matched_count, match_row, match_col):
        self.budget = budget
        self.steps_used = steps_used
        self.matched_count = matched_count
        self.match_row = tuple(match_row)
        self.match_col = tuple(match_col)
        super().__init__(
            f"搜索预算耗尽：budget={budget}，已检查 {steps_used} 条边，"
            f"当前部分匹配大小={matched_count}（尚未证明为最大匹配）"
        )
