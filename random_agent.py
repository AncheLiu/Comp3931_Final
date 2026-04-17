import gymnasium as gym

# 1. 创建环境 (CartPole-v1 倒立摆)
# render_mode="human" 表示我们要以图形化界面观看它的运行
env = gym.make("CartPole-v1", render_mode="human")

# 2. 初始化环境，获得初始状态 (state)
# reset() 返回两个值：初始状态和一些附加信息字典
state, info = env.reset()
print(f"初始状态: {state}")

total_reward = 0  # 记录得分

# 3. 开始交互循环 (运行 1000 步看看效果)
for step in range(1000):
    # env.render() # 渲染画面（在Gymnasium中，由于make时指定了human，这一步会自动进行）
    
    # 随机采样一个动作。
    # 对于 CartPole，动作空间是离散的：0 代表向左推车，1 代表向右推车
    action = env.action_space.sample()
    
    # 将动作输入给环境，环境会返回 5 个值 (这是新版 Gymnasium 的标准 API)
    next_state, reward, terminated, truncated, info = env.step(action)
    
    total_reward += reward

    # terminated: 游戏是否达到了失败条件（例如杆子倾斜角度过大，或小车跑出边界）
    # truncated: 游戏是否达到了最大步数（例如坚持了 500 步，系统强制结束）
    if terminated or truncated:
        print(f"回合结束！坚持了 {step+1} 步，总得分为 {total_reward}")
        # 重置环境，开始下一回合
        state, info = env.reset()
        total_reward = 0
    else:
        # 更新状态，继续下一步
        state = next_state

# 4. 运行结束，关闭环境窗口
env.close()