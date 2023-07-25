import math
import re
from tokenize import Double
import torch
import numpy as np
import random


torch.random.initial_seed()

seed = 66
torch.manual_seed(seed) # 为CPU设置随机种子
torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.

# #每个agent类 
# class user:
#     #初始化参数: 
#     def __init__(self,):
#         self.location_x     #当前位置（x，y）
#         self.location_y
#         self.task           #任务量
#         self.f              #本地cpu频率
#         self.p              #设备功率
#         self.action_space=spaces.Box(low=np.array([0,0,0]), high=np.array([1,1,1]), dtype=np.float32)
#         self.observation_space=spaces.Box(low=np.array([0,0,0]), high=np.array([100, 1, 10]), dtype=np.float32)


# class server:
#     def init(self):
#         self.location_x
#         self.location_y
#         self.f

# #每个agent的设置：task, p , 全部设置相同k 或者全部相同
# user_data = np.array([2,1,2],[],[],[],[],[])

class mec_env():
    def __init__(self,n_agents, n_obs, n_action, task_rate,):
        #self.user = torch.Tensor(user_data)  #传入二维数组表示用户参数
        #设置用户的参数:生成随机任务量task，移动速度，cpu芯片参数k，用户功率
        self.task_rate = task_rate 
        self.walk_rate = 2 #每秒往任意方向移动最多为1m
        self.k = 10E-27 #cpu芯片
        self.power = 1.5 #功率
        self.L = 500 
        # self.user_param = np.empty(shape=[0,4],dytpe = np.float32)
        # for i in range(n_agents):
        #     np.append(self.user_param,[task_rate, self.walk_rate, self.k, self.p], axis = 0)
        self.weight = [3, 2, 2, 2] #表示奖励函数的权重:隐私，能耗， 延迟，任务丢失惩罚
        
        #设置server的位置参数（x,y），服务器计算能力 一秒处理多少MB任务
        self.n_server = np.array([[333,333], [333,666], [666,333], [666,666]])
        self.serverCompute = 25 #mbps
        
        self.counts = 0
        self.n_agents = n_agents # 终端设备数量
        self.n_action = n_action # 动作维度: (服务器 + 1)*2
        self.n_obs = n_obs  #三维：task，x，y

        self.state_batch = torch.zeros(self.n_agents,self.n_obs) #系统所有agent总状态
        #self.action_batch =  torch.zeros(self.n_agents,self.n_action) #生成总状态

        #环境参数
        self.g0 = 1 #距离服务器1m的信道增益
        self.sigma = 10E-7 #计算信道增益的信道内部的噪声功率
        self.B = 30E+9  # 带宽 3Ghz

    def reset(self):
        for i in range(self.n_agents):
            #初始每个agent的状态
            self.state_batch[i,0] = np.clip(np.random.normal(self.task_rate),0.1,10) #生成任务量
            self.state_batch[i,1] = np.random.randint(0,1000)  #x坐标
            self.state_batch[i,2] = np.random.randint(0,1000) #y坐标
        return self.state_batch

    
    def step(self, action_batch):
        #action_batch = np.asarrays(action_batch)
        reward_n = []
        done_n = []
        new_action = []
        n_gain = []
        n_p = []
        m_computeTime = [] # m一维  放着每个服务器m的计算延迟
        n_transTime = [] # 设备l的传输时间 n * m
        n_offDelay = [] # 比较l卸载到服务器花费的总共时间  计算延迟 + 传输延迟 的最大值
        n_localDelay = [] # 每个l本地计算时间
        n_sumDelay = [] # 卸载总共用时, 比较本地时间和传输时间的最大值
        n_data = []
        n_offEnergy = [] # 通过传输时间*传输功率计算 
        n_localEnergy = []
        n_sumEnergy = []
        n_punish = []
        n_privacy = [] #每个设备的privacy


        for i in range(self.n_agents): # 循环每个agent
            action = (action_batch[i].reshape(2, self.n_action//2) + 1)/2 # 取出每个agent的动作转化为二维
            action = np.clip(action,0, 1)
            # 动作全是边界值为0时，任务给本地做
            if sum(action[0])==0 and action[0][1] == 0:
                action[0][1] = 1
            if sum(action[1]) == 0 and action[1][1] == 0:
                action[1][1] = 1
            action[0] = action[0] / sum(action[0])
            action[1] = action[1] / sum(action[1])
            #action[0] = np.clip(action[0],0.1,self.power)
            action_batch[i] = action.reshape(self.n_action)

            action[0] = self.power * action[0]  # 功率分配
            action[1] = self.state_batch[i][0] * action[1] # 计算数据量
            
            done_n.append(False)

            # 本地处理
            localtime = self.get_localtime(action[0][0], action[1][0]) # 计算设备l本地计算时间
            localtime = min(1,localtime)
            n_localDelay.append(localtime)
            n_localEnergy.append(localtime * action[0][0])
            n_punish.append(max(0, action[1][0] - action[0][0]*2)) # 丢失的任务： 计算任务 - 1 * 功率
            # 用于计算传输速度
            m_p = action[0][1:] 
            m_gain = self.get_h(self.state_batch[i][1], self.state_batch[i][2])
            n_gain.append(m_gain) 
            n_p.append(m_p) 
            # 用于传输时间
            n_data.append(action[1][-4:]) # l传输到每个服务器的数据量
            action[0] = action[0]/self.power   # 功率分配
            action[1] =  action[1]/self.state_batch[i][0] # 计算数据量


        n_localDelay = np.clip(n_localDelay, 0 ,1)
        m_ph = self.get_ph(n_gain, n_p)
        n_rate = self.get_rate(m_ph, n_p, n_gain) # rate(i,j) 表示 设备i到服务器j的传输速度
        m_computeTime = self.get_computeTime(n_data)

        n_transTime, n_taskDrop = self.get_tranTime(n_data,n_rate)
        n_transTime = np.clip(n_transTime, 0 ,1)

        n_offDelay = self.get_offDelay(n_transTime, m_computeTime)
        n_offDelay = np.clip(n_offDelay, 0, 1)

        n_sumDelay = self.get_sumdelay(n_offDelay, n_localDelay) # 总延迟
        n_offEnergy = self.get_offEnergy(n_transTime, n_p)

        for i in range(self.n_agents):
            n_sumEnergy.append(n_localEnergy[i] + n_offEnergy[i])  #总能耗
            n_punish[i] = n_punish[i] + n_taskDrop[i] #本地丢失任务+卸载丢失任务
        n_privacy = self.get_privacy(n_data)

        for i in range(self.n_agents):
            reward_n.append(self.weight[0]*n_privacy[i] - self.weight[1] * n_sumEnergy[i] - self.weight[2] * n_sumDelay[i] -  self.weight[3]*n_punish[i])
            self.state_batch[i][0] = np.clip(np.random.normal(self.task_rate),0.1,10)  #生成任务量
            # self.state_batch[i][1] = np.clip(self.state_batch[i][1] + random.uniform(-self.walk_rate, self.walk_rate),0, 1000)
            # self.state_batch[i][2] = np.clip(self.state_batch[i][2] + random.uniform(-self.walk_rate, self.walk_rate),0, 1000)
            self.state_batch[i][1] = np.clip(self.state_batch[i][1] + random.choice((-1, 1)) * self.walk_rate,0, 1000)
            self.state_batch[i][2] = np.clip(self.state_batch[i][2] + random.choice((-1, 1)) * self.walk_rate,0, 1000)
        #print(reward_n)
        return self.state_batch, reward_n, done_n, (n_privacy, n_sumEnergy, n_sumDelay, n_punish), action_batch

    #input: 本地计算功率, 本地任务大小  
    def get_localtime(self, p_local, a_local):
        # f_l = pow(p_lcoal/self.k, 1/3)
        if(a_local == 0):
            return 0
        if(p_local == 0):
            return 1
        f_l = p_local*2
        # localtime = a_local * self.L / f_l / 10E+6  # bit/s -》mbps
        if(f_l == 0):
            print("f_l is zero")
        localtime = a_local / f_l
        return localtime
            
    # 根据位置 求设备l到不同服务器的信道增益  
    def get_h(self, x, y):
        m_gain = []
        for i in range(4):
            dis = pow((pow(x-self.n_server[i][0],2)+pow(y-self.n_server[i][1],2)), 0.5)
            if(dis <= 1): 
                dis = 1
            m_gain.append(self.g0/dis)
        return m_gain
    
    # 返回每个服务器的信道干扰
    def get_ph(self, n_gain, n_p):
        m_ph = []
        for i in range(4):
            temp  = 0
            for j in range(self.n_agents):
                temp += n_gain[j][i] * n_p[j][i]
            m_ph.append(temp)
        return m_ph

    def get_rate(self, m_ph, n_p , n_gain):
        n_rate = []
        for i in range(self.n_agents):
            m_rate = []
            for j in range(4):
                if m_ph[j] == 0:
                    sinr = 0
                else: 
                    sinr = n_p[i][j]*n_gain[i][j]/m_ph[j]
                # print(sinr)
                # rate = self.B * math.log(1 + sinr, 2) / 10E+6 # bit -》mbps
                rate = self.B * sinr / 10E+8
                m_rate.append(rate)
            n_rate.append(m_rate)
        return n_rate

    def get_tranTime(self, n_data, n_rate):
        n_transTime = []
        n_taskDrop = []
        for i in range(self.n_agents):
            transTime = []
            task_drop = 0
            for j in range(4):
                if(n_data[i][j] == 0):
                    transTime.append(0)
                elif(n_rate[i][j]== 0):
                    transTime.append(1)
                    task_drop = task_drop + n_data[i][j]
                else: 
                    transTime.append(n_data[i][j]/n_rate[i][j])
                    task_drop = task_drop + max(n_data[i][j] - n_rate[i][j], 0) #丢失的任务 = data - 1s * rate
                # if(n_rate[i][j]== 0):
                #     print("n_rate is zero")
            n_transTime.append(transTime)
            n_taskDrop.append(task_drop)
        return n_transTime, n_taskDrop
    
    def get_computeTime(self, n_data):
        m_computeData = [0,0,0,0]
        for i in range(self.n_agents):
            for j in range(4):
                m_computeData[j] += n_data[i][j]
        for j in range(4):
            m_computeData[j] = m_computeData[j]/self.serverCompute
        return m_computeData

    def get_offDelay(self, n_transTime, m_computeTime):
        n_offDelay = []
        for i in range(self.n_agents):
            temp = 0
            for j in range(4):
                temp = max(temp, n_transTime[i][j] + m_computeTime[j])
            n_offDelay.append(temp)
        return n_offDelay
    
    def get_sumdelay(self, n_offDelay, n_localDelay):
        n_sumdelay = []
        for i in range(self.n_agents):
            temp = max(n_offDelay[i], n_localDelay[i])
            n_sumdelay.append(temp)
        return n_sumdelay

    def get_offEnergy(self, n_transTime, n_p):
        n_offEnergy = []
        for i in range(self.n_agents):
            offEnergy = 0
            for j in range(4):
                offEnergy += n_transTime[i][j] * n_p[i][j]
            n_offEnergy.append(offEnergy)
        return n_offEnergy

    def get_privacy(self, n_data):
        n_privacy = []
        for i in range(self.n_agents):
            privacy = 0
            sum_data = sum(n_data[i])
            for j in range(4):
                if(sum_data==0) : privacy = 2 #全部本地计算为2
                else :
                    perfer = n_data[i][j]/sum_data
                    if(perfer == 0):
                        perfer = 1
                    privacy += - perfer * math.log(perfer,2)
            n_privacy.append(privacy)
        return n_privacy