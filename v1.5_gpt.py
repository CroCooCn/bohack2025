def solve_with_kaiwu(n, m, strs, pos):
    """
    方案A：只用 x+y，不用 z
    - x[i,j,c]：棋盘格子(i,j)是否取字符c（one-hot）
    - y[k,t]：第k个字符串是否选择第t个候选放置位置
    目标：最大化匹配字符串条数 => 最小化 -w * sum(y[k,t])
    关键：如果 y[k,t]=1，则该放置对应路径上每个字符必须匹配，否则重罚
    """

    # ====== 建议的系数（先跑通逻辑，再调参）======
    w_hit = 120            # 选中一个“完全匹配”的放置带来的收益（等价于命中一条字符串）
    lam_cell = 5000        # one-hot 约束（必须很大，保证每格恰好一个字符）
    lam_one  = 800         # 同一字符串“最多选一个放置”的惩罚（要 > w_hit）
    lam_mis  = 300         # 选了但不匹配的惩罚（必须 > w_hit，保证宁可不选也不乱选）

    # ====== 变量定义 ======
    # x: n*n*|SIGMA| 个二进制变量，表示棋盘每格取哪个字符
    x = kw.core.ndarray((n, n, len(SIGMA)), "x", kw.core.Binary)

    # y: 每条字符串k，对它的候选位置t，建一个二进制变量
    y = []
    for k in range(m):
        y.append([kw.core.Binary(f"y[{k},{t}]") for t in range(len(pos[k]))])

    model = kw.qubo.QuboModel()
    obj = 0

    # ====== (1) 每个格子必须恰好选择一个字符：sum_c x[i,j,c] == 1 ======
    # 注意：lam_cell 不能为0，否则 x 全0 也“合法”，输出会崩
    for i in range(n):
        for j in range(n):
            model.add_constraint(x[i, j, :].sum() == 1, f"cell[{i},{j}]", penalty=lam_cell)

    # ====== (2) 同一字符串最多选一个放置：对任意 t<u，罚 y[k,t]*y[k,u] ======
    # 这是 “at most one” 的经典 QUBO 写法：只要同时选两个，就产生正惩罚
    one_terms = []
    for k in range(m):
        pk = len(pos[k])
        for t in range(pk):
            for u in range(t + 1, pk):
                one_terms.append(y[k][t] * y[k][u])
    if one_terms:
        obj += lam_one * kw.core.quicksum(one_terms)

    # ====== (3) 匹配惩罚：如果选了某放置 y[k,t]=1，则路径上每格必须匹配 ======
    # 对每个字符位置 l：罚 y[k,t] * (1 - x[cell, required_char])
    # - 如果该格选对字符 x=1 => (1-x)=0 => 该项为0（不罚）
    # - 如果该格没选对字符 x=0 => (1-x)=1 => 该项= y[k,t] => 重罚
    mis_terms = []
    for k, s in enumerate(strs):
        L = len(s)
        for t, p in enumerate(pos[k]):
            sx, sy, direc = int2pos(p, n)
            path = get_path(s, sx, sy, direc, n)
            for l in range(L):
                ii, jj = path[l]
                ci = CH2INT[s[l]]
                mis_terms.append(y[k][t] * (1 - x[ii, jj, ci]))
    if mis_terms:
        obj += lam_mis * kw.core.quicksum(mis_terms)

    # ====== (4) 目标：最大化匹配条数（命中越多越好）======
    # 因为同一字符串最多选一个放置，所以 sum_t y[k,t] 就是“第k条是否命中”的0/1指示
    hit_terms = []
    for k in range(m):
        hit_terms += y[k]
    if hit_terms:
        obj += -w_hit * kw.core.quicksum(hit_terms)

    # （可选）你现在还没到 c=M 的阶段时，不建议奖励 '.'，会干扰命中
    # 真要做二阶段（c=M 后最大化 '.'），那应该另开一次优化或做后处理，不要混在这里

    model.set_objective(obj)

    # ====== 求解器参数（可先用你原来的）======
    optimizer = kw.classical.SimulatedAnnealingOptimizer(
        initial_temperature=1e6,
        alpha=0.995,
        cutoff_temperature=0.01,
        iterations_per_t=200,
        size_limit=50
    )
    solver = kw.solver.SimpleSolver(optimizer)
    sol_dict, qubo_val = solver.solve_qubo(model)

    # ====== 解码 x 得到棋盘 ======
    x_val = kw.core.get_array_val(x, sol_dict)
    grid = [['.'] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            best_c = int(max(range(len(SIGMA)), key=lambda c: x_val[i, j, c]))
            grid[i][j] = SIGMA[best_c]

    return grid
