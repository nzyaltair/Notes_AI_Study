# 搜索与 MCTS

A* 按 \(f(n)=g(n)+h(n)\) 扩展节点；启发式可容许时可得到最优路径。MCTS 通过选择、扩展、评估和回传，在大分支空间中分配搜索预算。

LLM 可生成候选行动、提供启发式或估值，但错误估值会被搜索放大。应限制分支与成本、去重，并将事实和副作用判断交给工具或模拟器。

References: Hart et al., *A Formal Basis for A\**; Kocsis & Szepesvári, *UCT*.