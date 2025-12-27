def build_pos_bootstrap(n, strs, POS_CNT=20, ROUNDS=120,  # 轮数越多越稳
                        SAMPLE=120, beta=4.0, top_pick=3, seed=0):
    """
    自举式贪心拼盘投票：输出每个字符串的候选位置列表 pos[i]（长度<=POS_CNT）
    - beta：冲突惩罚系数，越大越“怕冲突”
    - SAMPLE：每次给一个字符串评估多少个候选位置（从800里抽样）
    - top_pick：从前 top_pick 名里随机挑一个，避免太贪心
    """
    random.seed(seed)
    m = len(strs)
    P = 2 * n * n

    # 预计算：cand[i][p] = [(cell, need_letter_index), ...]
    cand = []
    for s in strs:
        one = []
        for p in range(P):
            x, y, dir = int2pos(p, n)
            path = get_path(s, x, y, dir, n)
            cons = []
            for ch, (ii, jj) in zip(s, path):
                cons.append((ii * n + jj, CH2INT[ch]))  # 只会是0..7
            one.append(cons)
        cand.append(one)

    # 按长度降序放（长串优先）
    order = list(range(m))
    order.sort(key=lambda i: -len(strs[i]))

    # 统计每个串每个位置被选中的“票数/分数”
    vote = [ [0.0] * P for _ in range(m) ]

    for _ in range(ROUNDS):
        grid = [-1] * (n * n)

        for i in order:
            # 从 0..P-1 抽样一些候选位置
            if SAMPLE >= P:
                sampled_ps = range(P)
            else:
                sampled_ps = random.sample(range(P), SAMPLE)

            best = []  # 存 (score, p)
            for p in sampled_ps:
                match = 0
                conflict = 0
                for cell, need in cand[i][p]:
                    cur = grid[cell]
                    if cur == -1:
                        continue
                    if cur == need:
                        match += 1
                    else:
                        conflict += 1
                sc = match - beta * conflict
                best.append((sc, p))

            best.sort(reverse=True)
            pick_k = min(top_pick, len(best))
            sc, p_star = random.choice(best[:pick_k])

            # 记录投票（也可以 vote[i][p_star] += 1，只按频次）
            vote[i][p_star] += sc + 10.0  # +常数避免全负时没区分度（你也可去掉）

            # 把该位置写进临时网格：只填未定格子，减少“覆盖引入矛盾”
            for cell, need in cand[i][p_star]:
                if grid[cell] == -1:
                    grid[cell] = need

    # 对每个串取票数最高的 POS_CNT 个位置
    pos = []
    for i in range(m):
        ps = list(range(P))
        ps.sort(key=lambda p: vote[i][p], reverse=True)
        pos.append(ps[:min(POS_CNT, P)])
    return pos
