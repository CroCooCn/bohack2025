def read_input():
    # TODO: 按你的输入格式改
    # 例：第一行 M，然后接 M 行字符串
    import sys
    data = sys.stdin.read().strip().split()
    M = int(data[0])
    strings = data[1:1+M]
    return strings

N = 20
SIGMA = list("ABCDEFGH.")  # 9
C2I = {ch: i for i, ch in enumerate(SIGMA)}

def torus_path(i0, j0, direction, L):
    # direction: 0 -> right, 1 -> down
    path = []
    for t in range(L):
        if direction == 0:
            path.append((i0 % N, (j0 + t) % N))
        else:
            path.append(((i0 + t) % N, j0 % N))
    return path

def all_placements_for_length(L):
    # 20*20*2 = 800 个 placements
    res = []
    for i0 in range(N):
        for j0 in range(N):
            res.append(torus_path(i0, j0, 0, L))
            res.append(torus_path(i0, j0, 1, L))
    return res

from collections import Counter

def build_initial_grid(strings):
    cnt = Counter()
    for s in strings:
        cnt.update(s)
    base = max("ABCDEFGH", key=lambda ch: cnt[ch])
    grid = [[base for _ in range(N)] for _ in range(N)]
    return grid

def placement_score(s, path, grid):
    score = 0
    for ell, (i, j) in enumerate(path):
        if grid[i][j] == s[ell]:
            score += 1
    return score

def prune_placements(strings, K=20, Lmin=1):
    grid = build_initial_grid(strings)
    placements = []
    cache = {}  # length -> all placements (800) cache

    for s in strings:
        L = len(s)
        if L < Lmin:
            placements.append([])
            continue

        if L not in cache:
            cache[L] = all_placements_for_length(L)
        allp = cache[L]

        scored = [(placement_score(s, path, grid), path) for path in allp]
        scored.sort(key=lambda x: x[0], reverse=True)
        placements.append([path for _, path in scored[:K]])
    return placements

import kaiwu as kw

def solve_with_kaiwu(strings, placements,
                     wK=10.0, wD=2.0,
                     lam_cell=200.0, lam_match=50.0, lam_link=50.0):
    """
    目标能量（越小越好）：
      H = lam_cell * Σ (Σ_c x[p,c] - 1)^2
        + lam_match * Σ y[k,t] * (1 - x[pos, ch])
        + lam_link * Σ (Σ_t y[k,t] - z[k])^2
        + (-wK * Σ z[k] + wD * Σ x[p,'.'])
    """

    M = len(strings)

    # --- 变量：注意在 1.3.0 用 kw.core.Binary / kw.core.ndarray ---
    # x[i,j,c]：格子(i,j)是否选字符 c
    x = kw.core.ndarray((N, N, len(SIGMA)), "x", kw.core.Binary)  # 官方 core 示例就是这种风格:contentReference[oaicite:2]{index=2}

    # z[k]：字符串 k 是否命中
    z = [kw.core.Binary(f"z[{k}]") for k in range(M)]

    # y[k][t]：字符串 k 是否选第 t 个 placement
    y = []
    for k in range(M):
        y.append([kw.core.Binary(f"y[{k},{t}]") for t in range(len(placements[k]))])

    # --- QuboModel：用 kw.qubo.QuboModel ---
    qubo_model = kw.qubo.QuboModel()

    # 1) one-hot：每格恰好一个字符
    # 注意：1.3.0 的 changelog 提到 add_constraint 接收逻辑表达式（关系式）:contentReference[oaicite:3]{index=3}
    for i in range(N):
        for j in range(N):
            qubo_model.add_constraint(x[i, j, :].sum() == 1, f"cell[{i},{j}]", penalty=lam_cell)

    # 2) score：-wK * sum(z) + wD * sum(dot)
    obj = -wK * kw.core.quicksum(z) + wD * x[:, :, DOT].sum()

    # 3) match：lam_match * sum y*(1-x)
    match_terms = []
    for k, s in enumerate(strings):
        if not placements[k]:
            continue
        L = len(s)
        for t, path in enumerate(placements[k]):
            for ell in range(L):
                ii, jj = path[ell]
                ci = C2I[s[ell]]
                match_terms.append(y[k][t] * (1 - x[ii, jj, ci]))
    obj += lam_match * kw.core.quicksum(match_terms)

    # 4) link：lam_link * sum (sum_t y - z)^2
    link_terms = []
    for k in range(M):
        if not y[k]:
            # 没建这个字符串的 placements，就强制 z[k]=0 更省（可选）
            # 这里用一个轻约束：z[k] == 0
            qubo_model.add_constraint(z[k] == 0, f"skip_z[{k}]", penalty=lam_link)
            continue
        link_terms.append((kw.core.quicksum(y[k]) - z[k]) ** 2)
    obj += lam_link * kw.core.quicksum(link_terms)

    qubo_model.set_objective(obj)

    # --- 求解：用 SA + Solver ---
    # SA 类在 kaiwu.classical 文档中有定义:contentReference[oaicite:4]{index=4}
    optimizer = kw.classical.SimulatedAnnealingOptimizer(
        initial_temperature=1e6,
        alpha=0.995,
        cutoff_temperature=0.01,
        iterations_per_t=100,
        size_limit=50
    )
    solver = kw.solver.PenaltyMethodSolver(optimizer=optimizer, controller=None)
    # 有的版本也可以用 SimpleSolver；但 1.3.0 文档里 solver 体系是 PenaltyMethodSolver 为主:contentReference[oaicite:5]{index=5}

    # solve_qubo 返回解（sol_dict 等），接口在 solver 文档里有说明:contentReference[oaicite:6]{index=6}
    result = solver.solve_qubo(qubo_model)
    # 不同 solver 可能返回结构略有差异；常见是 dict 或 SolutionResult
    # 我这里做一个兼容提取：
    if isinstance(result, dict) and "sol_dict" in result:
        sol_dict = result["sol_dict"]
    elif hasattr(result, "sol_dict"):
        sol_dict = result.sol_dict
    else:
        # 有些实现直接返回 (sol_dict, obj_val)
        sol_dict = result[0]

    # --- 解码 x -> 20x20 ---
    grid = [["."] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            picked = "."
            for c, ch in enumerate(SIGMA):
                # core 的 get_array_val 可把 ndarray 变量批量代入 sol_dict（更稳）
                # 但这里用最朴素的取值方式：变量名直接索引
                # 如果你发现拿不到，就用 kw.core.get_array_val（见下方注释）
                name = f"x[{i}][{j}][{c}]"
                if sol_dict.get(name, 0.0) > 0.5:
                    picked = ch
                    break
            grid[i][j] = picked

    return grid