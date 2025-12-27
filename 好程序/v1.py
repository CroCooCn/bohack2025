'''
奖惩公式及系数
放置位置选择
'''

import random
import sys
import kaiwu as kw
import math

SIGMA = list("ABCDEFGH.")
CH2INT = {ch: i for i, ch in enumerate(SIGMA)}


RAND_MAP_CNT=20 #上限100
POS_CNT=20

def read():
    n,m=map(int,sys.stdin.readline().split())
    strs=[]
    for i in range(m):
        strs.append(sys.stdin.readline().strip())
    return n,m,strs

def test_read(n,m,strs):
    print(f"n={n}");
    print(f"m={m}");
    for i in range(m):
        print(strs[i])

def get_path(s:str,i:int,j:int,dir:int,n):
    path=[]
    l=len(s)
    if(dir==0): #右
        for d in range(l):
            path.append((i,(j+d)%n))
    elif(dir==1): #下
        for d in range(l):
            path.append(((i+d)%n,j))
    return path

def all_placement(s:str,n):
    pm=[]
    for i in range(n):
        for j in range(n):
            pm.append(get_path(s,i,j,0,n))
            pm.append(get_path(s,i,j,1,n))
    return pm

def test_all_placement(s:str,n):
    pm=all_placement(s,n)
    for path in pm:
        print(path)
    
def gen_randmap_list(strs):
    res=[]
    for s in strs:
        for ch in s:
            res.append(ch)
    res.sort()
    #print(res)
    return res

def gen_randmap(rdlist,n):
    l=len(rdlist)
    res=[]
    for i in range(n):
        row=""
        for j in range(n):
            idx=random.randint(0,l-1)
            row+=rdlist[idx]
        res.append(row)
    return res 

def pos2int(n,i,j,dir) :
    res=i*n+j
    if(dir==1):
        res+=n*n
    return res

def int2pos(p,n):
    dir=0
    if(p>=n*n):
        dir=1
        p-=n*n
    i=p//n
    j=p%n
    return i,j,dir

    

def calc_score(n,m,strs,rdmap,sco):
    for i,s in enumerate(strs):
        for x in range(n):
            for y in range(n):
                for dir in range(2):
                    cursco=0
                    path=get_path(s,x,y,dir,n)
                    for j,(x1,y1) in enumerate(path):
                        if(s[j]==rdmap[x1][y1]):
                            cursco+=1
                    t1,t2=sco[i][pos2int(n,x,y,dir)]
                    sco[i][pos2int(n,x,y,dir)]=(t1+cursco,t2)


def solve_with_kaiwu(n,m,strs,pos):
    
    wk=800
    wd=2
    lam_cell=1
    lam_match=10
    lam_link=5
    

    
    #创建变量
    x = kw.core.ndarray((n, n, len(SIGMA)), "x", kw.core.Binary)
    #print(x)
    z = [kw.core.Binary(f"z[{k}]") for k in range(m)]
    #print(z)
    y=[]
    for i in range(m):
        t=[]
        for j in range(len(pos[i])):
            t.append(kw.core.Binary(f"y[{i},{j}]"))
        y.append(t)
    
    model=kw.qubo.QuboModel()
    obj=0

    
    #cell
    for i in range(n):
        for j in range(n):
            model.add_constraint(x[i, j, :].sum() == 1, f"cell[{i},{j}]", penalty=lam_cell)    
    

    #match
    match_terms = []
    for k, s in enumerate(strs):
        L = len(s)
        for t, p in enumerate(pos[k]):
            xx,yy,dir=int2pos(p,n)
            path=get_path(s,xx,yy,dir,n)
            for l in range(L):
                ii, jj = path[l]
                ci = CH2INT[s[l]]
                match_terms.append(y[k][t] * (1 - x[ii, jj, ci]))
    obj += lam_match * kw.core.quicksum(match_terms)

    #link
    link_terms = []
    for k in range(m):
        link_terms.append((kw.core.quicksum(y[k]) - z[k]) ** 2)
    obj += lam_link * kw.core.quicksum(link_terms)

    #score
    obj += -wk * kw.core.quicksum(z) + wd * x[:, :, CH2INT['.']].sum()

    model.set_objective(obj)

    optimizer = kw.classical.SimulatedAnnealingOptimizer(
        initial_temperature=1e6,
        alpha=0.995,
        cutoff_temperature=0.01,
        iterations_per_t=100,
        size_limit=50
    )
    solver = kw.solver.SimpleSolver(optimizer)
    sol_dict, qubo_val = solver.solve_qubo(model)
    
    x_val = kw.core.get_array_val(x, sol_dict)
    grid=[['.']*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            best_c = int(max(range(len(SIGMA)), key=lambda c: x_val[i, j, c]))
            grid[i][j]=SIGMA[best_c]
    
    return grid


def print_sol(sol):
    for row in sol:
        for ch in row:
            print(ch,end="")
        print("")


if __name__=="__main__":
    #读入
    n,m,strs=read()
    #test_read(n,m,strs)
    #test_all_placement(strs[0],n)
    rdlist=gen_randmap_list(strs)
    #初始化sco
    sco=[]
    for _ in range(m):
        row=[]
        for i in range(2*n*n):
            row.append((0,i))
        sco.append(row)

    for i in range(RAND_MAP_CNT):
        rdmap=gen_randmap(rdlist,n)
        calc_score(n,m,strs,rdmap,sco)

    pos=[]
    for i,each_sco in enumerate(sco):
        each_sco.sort(reverse=True)
        top=each_sco[:min(POS_CNT,len(each_sco))]
        toppos=[]
        for sco,p in top:
            toppos.append(p)
        pos.append(toppos)

    '''
    #输出pos
    for i in range(m):
        print(f"{strs[i]}的候选位置")
        for p in pos[i]:
            print(int2pos(p,n))
    '''

    print("开始求解!")
    
    sol=solve_with_kaiwu(n,m,strs,pos)
    print_sol(sol)

        