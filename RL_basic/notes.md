强化学习包括env、agent两个部分；
    - env：包含奖励函数P(r|s,a)和状态转移函数P(s'|s,a)
        - env包括agent当前状态、step函数（根据action跳转到下一个状态，并产生当前step的奖励值）
    
    - agent：包含策略函数pi(a|s)
        - agent包含动作空间、策略pi、和choosr_action函数（根据当前状态选择下一step需要采取的action）
