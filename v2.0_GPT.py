def solve_y_only(n, m, strs, pos): #y
    w_hit = 1
    lam_one = 3          # 必须 > w_hit
    lam_conf = 5         # 冲突惩罚，通常要明显大于 w_hit

    # 给每个 y[k,t] 一个全局编号，方便放到同一个列表里处理
    y = []
    vid = 0
    y_id = [[-1]*len(pos[k]) for k in range(m)]
    for k in range(m):
        for t in range(len(pos[k])):
            y.append(kw.core.Binary(f"y[{k},{t}]"))
            y_id[k][t] = vid
            vid += 1

    model = kw.qubo.QuboModel()
    obj = 0

    # (1) 命中奖励：-w_hit * sum y
    obj += -w_hit * kw.core.quicksum(y)

    # (2) 同一串最多选一个：lam_one * sum_{u<t} y_u y_t
    one_terms = []
    for k in range(m):
        L = len(pos[k])
        for u in range(L):
            for t in range(u):
                one_terms.append(y[y_id[k][u]] * y[y_id[k][t]])
    obj += lam_one * kw.core.quicksum(one_terms)

    # (3) 格子冲突项
    #    cell_map[cell] = dict(letter -> [var_ids...])
    p_up = 2 * n * n
    cell_map = [dict() for _ in range(n*n)]

    for k, s in enumerate(strs):
        for t, p in enumerate(pos[k]):
            x0, y0, dir = int2pos(p, n)
            path = get_path(s, x0, y0, dir, n)
            v = y_id[k][t]
            for ch, (i, j) in zip(s, path):
                cell = i*n + j
                need = CH2INT[ch]
                cell_map[cell].setdefault(need, []).append(v)

    conf_terms = []
    for cell in range(n*n):
        groups = list(cell_map[cell].items())  # [(need_letter, [vids...]), ...]
        # 不同字母组之间两两冲突
        for a in range(len(groups)):
            va = groups[a][1]
            for b in range(a):
                vb = groups[b][1]
                for ida in va:
                    for idb in vb:
                        conf_terms.append(y[ida] * y[idb])

    obj += lam_conf * kw.core.quicksum(conf_terms)

    model.set_objective(obj)

    optimizer = kw.classical.SimulatedAnnealingOptimizer(
        initial_temperature=1000,
        alpha=0.995,
        cutoff_temperature=0.01,
        iterations_per_t=200,
        size_limit=100
    )
    solver = kw.solver.SimpleSolver(optimizer)
    sol_dict, qubo_val = solver.solve_qubo(model)

    # 还原 grid：对每个 cell，看被选中的 placements 要求什么字母（应当一致）
    chosen = [0]*len(y)
    for i in range(len(y)):
        chosen[i] = int(sol_dict.get(str(y[i]), 0))

    grid = [['.']*n for _ in range(n)]
    # 写入字母：如果一个 cell 被多个选中 placement 覆盖，它们应当都要求同一字母
    for cell in range(n*n):
        best_need = None
        for need, vids in cell_map[cell].items():
            if any(chosen[v] for v in vids):
                best_need = need
                break
        if best_need is not None:
            i, j = divmod(cell, n)
            grid[i][j] = SIGMA[best_need]
    return grid