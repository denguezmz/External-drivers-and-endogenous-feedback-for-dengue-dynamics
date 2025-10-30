import matplotlib.pyplot as plt
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torchvision
from torch import nn
from torch.nn import Sequential, init
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.manual_seed(10)
color1 = (179 / 255, 70 / 255, 138 / 255)
color2 = (228 / 255, 192 / 255, 214 / 255)
color3 = (164 / 255, 136 / 255, 186 / 255)


def normalize_0_1(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


class BORN(nn.Module):
    def __init__(self):
        super(BORN, self).__init__()
        self.layer1 = nn.Linear(2, 10)
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        x = torch.sigmoid(self.layer3(x))
        return x


# 创建 PyTorch 模型实例


born = BORN()
model_path = f'E:\\登革热广州\\新旧蚊子\\\yuan_state_dict_born.pth'
model_weight = torch.load(model_path)
born.load_state_dict(model_weight)

num_steps = 1826
TT = torch.linspace(1, num_steps, num_steps).view(-1, 1)  # [100, 1]
TT = TT / num_steps  # 归一化时间序列 (可选)

import statsmodels.api as sm

file_path = 'E:\\登革热广州\\新旧蚊子\\Local.xlsx'
df = pd.read_excel(file_path)
Local1 = df['All2'].values


Local = torch.tensor(Local1, dtype=torch.float32)  # 原始数据
# Local_sum = torch.tensor(rolling_sum_values, dtype=torch.float32)
from scipy.ndimage import gaussian_filter1d

sigma = 2 # 高斯核的标准差
Local_smooth = gaussian_filter1d(Local.numpy(), sigma=sigma)

plt.plot(Local, label='Original')
plt.plot(Local_smooth, label='Smoothed', color='b')
plt.legend()
plt.show()

Local_smooth = torch.tensor(Local)
window_size = 7
Local_7_smooth = Local_smooth[365:].unfold(0, window_size, window_size).sum(dim=1)
plt.plot(Local_7_smooth)
plt.show()


Local_7_smooth = torch.tensor(Local_7_smooth)
# Local_365_smooth = torch.tensor(Local_365)

file_path = 'E:\\登革热广州\\新旧蚊子\\Input.xlsx'
df = pd.read_excel(file_path)
Import1 = df['All2'].values

# Import1_series = pd.Series(Import1)
# # 计算前7天（包括当天）的累计值
# rolling_sum = Import1_series.rolling(window=7, min_periods=1).sum()
# # 将结果转换为数组
# rolling_sum_values = rolling_sum.values
# # 将 Import1 和计算后的值转换为 PyTorch 张量
Import = torch.tensor(Import1, dtype=torch.float32)  # 原始数据
# Import_sum = torch.tensor(rolling_sum_values, dtype=torch.float32)


file_path = 'E:\\登革热广州\\新旧蚊子\\Temp.xlsx'
df = pd.read_excel(file_path)
T = df['All']
file_path = 'E:\\登革热广州\\新旧蚊子\\Rain.xlsx'
df = pd.read_excel(file_path)
R = df['All']
file_path = 'E:\\登革热广州\\新旧蚊子\\Expanded_Half_Month_Averages.xlsx'
df = pd.read_excel(file_path)
AT = df['All'].values
file_path = 'E:\\登革热广州\\新旧蚊子\\Seven_Day_Smoothed_Rain.xlsx'
df = pd.read_excel(file_path)
AR = df['All'].values

import scipy.io
import pandas as pd


class SineActivation(nn.Module):
    def __init__(self, amplitude=1.0, frequency=1.0, offset=0.0):
        super(SineActivation, self).__init__()
        self.amplitude = amplitude
        self.frequency = frequency
        self.offset = offset

    def forward(self, x):
        # 正弦激活函数，支持调整幅度、频率、偏移
        return self.amplitude * torch.sin(self.frequency * x + self.offset)


import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


ax = 0.0135
hax = 28116.4141
hhx = 35378.2344
thx = 301.6750
# mup 和 mua 不再使用 nn.Parameter
mup = 0.9991  # 初始化值
mua = 0.6308  # 初始化值
tp = 22
vp = 20
ta = 30
va = 12.1170
r = 0.2845
ZT = 21.7535

inputs = torch.tensor(list(zip(T.values, R.values)), dtype=torch.float32)
myborn = born(inputs) * 18 + 4.0091
mu = ax * ((T + 273.15) / 298.15) * np.exp(hax / 1.987 * (1 / 298.15 - 1 / (T + 273.15))) / (
        1 + np.exp(hhx / 1.987 * (1 / thx - 1 / (T + 273.15))))
mymu = np.minimum(mu * (AT > ZT) + mu * r * (AT <= ZT), 1)
dp1 = np.abs(1 - mup * np.exp(-(T - tp) ** 2 / (vp ** 2)))
dp2 = np.abs(1 - mup * np.exp(-(ZT - tp) ** 2 / (vp ** 2)))
mydp = np.minimum(dp1 * (AT > ZT) + dp2 * (AT <= ZT), 1)
myda = np.minimum(np.abs(1 - mua * np.exp(-(T - ta) ** 2 / (va ** 2))), 1)
AR = torch.tensor(AR, dtype=torch.float32)
myda = torch.tensor(myda.to_numpy(), dtype=torch.float32).view(-1, 1)
mydp = torch.tensor(mydp.to_numpy(), dtype=torch.float32).view(-1, 1)
mymu = torch.tensor(mymu.to_numpy(), dtype=torch.float32).view(-1, 1)


# wen = torch.tensor(wen1[0:12*5], dtype=torch.float32)
Import = torch.tensor(Import1, dtype=torch.float32)
Local = torch.tensor(Local1, dtype=torch.float32)
AR = torch.tensor(AR, dtype=torch.float32)
# AR = normalize_0_1(AR)

# b = T.apply(lambda t: 0.000202 * t * (t - 13.35) * np.sqrt(40.08 - t) if 13.35 <= t <= 40.08 else 0)
b = T.apply(lambda t: 0.0043 * t + 0.0943 if 10 <= t <= 40 else 0)
bh = T.apply(lambda t: 0.001044 * t * (t - 12.286) * np.sqrt(32.461 - t) if 12.286 <= t <= 32.461 else 0)
# bv = T.apply(lambda t: -0.9037 + 0.0729 * t if 12.4 <= t < 26.1 else (1 if 26.1 <= t <= 32.5 else 0))
bv = T.apply(lambda t: -0.9037 + 0.0729 * t if 12.4 <= t < 26.1 else 1)

bh = torch.tensor((bh * b).to_numpy(), dtype=torch.float32).view(-1, 1)
bv = torch.tensor((bv * b).to_numpy(), dtype=torch.float32).view(-1, 1)
b = torch.tensor((b).to_numpy(), dtype=torch.float32).view(-1, 1)

T = torch.tensor(T.to_numpy(), dtype=torch.float32).view(-1, 1)

class ODENN(nn.Module):
    def __init__(self):
        super(ODENN, self).__init__()
        self._all_layers = []
        self.init_state = []
        self.init_res = []
        self.init_sym = []
        # self.Myk = Myk

        self.Nh = torch.tensor([113460000])
        self.S0 = torch.tensor([113460000])
        self.E0 = torch.tensor([0])
        self.I0 = torch.tensor([0])
        self.Iin0 = torch.tensor([0])
        self.R0 = torch.tensor([0])

        # self.Sp0 = torch.tensor([10000000])
        # self.Sp0 = torch.tensor([7.0])
        self.Sp0 = torch.nn.Parameter(torch.tensor([7.17], dtype=torch.float32), requires_grad=True)
        self.Ip0 = torch.tensor([0])
        self.Sa0 = torch.tensor([0])
        self.Ea0 = torch.tensor([0])
        self.Ia0 = torch.tensor([0])

        self.res0 = torch.tensor([0])
        self.M0 = torch.tensor([0])
        self.N0 = torch.tensor([0])
        self.gamma = torch.nn.Parameter(torch.tensor([0.1256], dtype=torch.float32), requires_grad=True)
        self.U = torch.nn.Parameter(torch.tensor([0.9072], dtype=torch.float32),requires_grad=True)  # 0.9944  0.014-0.174
        self.detaa = torch.nn.Parameter(torch.tensor([0.1007], dtype=torch.float32), requires_grad=True)
        self.detah = torch.nn.Parameter(torch.tensor([0.1506], dtype=torch.float32), requires_grad=True)
        self.out = torch.nn.Parameter(torch.tensor([0.1309], dtype=torch.float32), requires_grad=True)
        # self.SpK = torch.tensor([50000000])

    def clamp_parameters(self):

        self.gamma.data = F.hardtanh(self.gamma, min_val=0.125, max_val=0.3333)
        self.U.data = F.hardtanh(self.U, min_val=0.826, max_val=0.986)
        self.detaa.data = F.hardtanh(self.detaa, min_val=0.0833, max_val=0.1250)
        # self.detah.data = F.hardtanh(self.detah, min_val=0.125, max_val=0.2500)
        self.detah.data = F.hardtanh(self.detah, min_val=0.10, max_val=0.25)
        self.out.data = F.hardtanh(self.out, min_val=0.1, max_val=0.5)
        # self.Myk.data = F.hardtanh(self.Myk, min_val=0.00000001, max_val=1.0)


    def forward(self, num_steps, mymu, mydp, myda, myborn, bh, bv, Import, inpu,b):
        Sp0 = torch.pow(torch.tensor([10]), self.Sp0)
        SpK = Sp0 * 5
        self.init_state = torch.cat(
            (self.S0, self.E0, self.I0, self.Iin0, self.R0, Sp0, self.Ip0, self.Sa0, self.Ea0, self.Ia0), dim=0)
        self.init_res = torch.cat((self.res0,  self.M0, self.M0, self.M0), dim=0)

        # A = torch.tensor([0.000022])
        # deathh = torch.tensor([0.00002])
        A = torch.tensor([0])
        deathh = torch.tensor([0])
        outputs = [self.init_state]
        outres = [self.init_res]

        # print(myk)
        betah = bh
        betav = bv


        for step in range(num_steps):
            if step == 0:
                h = self.init_state
                internal_state = h
            h = internal_state
            S = h[0]
            E = h[1]
            I = h[2]
            Iin = h[3]
            R = h[4]

            Sp = h[5]
            Ip = h[6]

            Sa = h[7]
            Ea = h[8]
            Ia = h[9]

            N = S + E + I + R + Iin


            dSp = myborn[step] * Sa + self.U * myborn[step] * (Ia + Ea) - mydp[step] * Sp - mymu[step] * Sp
            dIp = (1 - self.U) * myborn[step] * (Ia + Ea) - mydp[step] * Ip - mymu[step] * Ip
            ef = torch.abs(
                torch.exp(-0.1 * (1 + (mymu[step] * Sp + mymu[step] * Ip) / (SpK * (1 + 0.0147 * AR[step])))))

            dSa = ef * mymu[step] * Sp - betav[step] * Sa * (I + Iin) / N  - myda[step] * Sa
            dEa = betav[step] * Sa * (I + Iin) / N - myda[step] * Ea - self.detaa * Ea
            dIa = ef * mymu[step] * Ip + self.detaa * Ea - myda[step] * Ia

            dS = A * (S + E + I + R) - deathh * S - betah[step] * Ia * S / N
            dE = betah[step] * Ia * S / N - self.detah * E - deathh * E
            dI = self.detah * E - self.gamma * I - deathh * I
            dIin = Import[step] - self.out * Iin
            dR = self.gamma * I - deathh * R

            S_next = torch.clamp(S + dS, min=0.0)
            E_next = torch.clamp(E + dE, min=0.0)
            I_next = torch.clamp(I + dI, min=0.0)
            Iin_next = torch.clamp(Iin + dIin, min=0.0)
            R_next = torch.clamp(R + dR, min=0.0)
            Sp_next = torch.clamp(Sp + dSp, min=0.0)
            Ea_next = torch.clamp(Ea + dEa, min=0.0)
            Ip_next = torch.clamp(Ip + dIp, min=0.0)
            Sa_next = torch.clamp(Sa + dSa, min=0.0)
            Ia_next = torch.clamp(Ia + dIa, min=0.0)
            res = 0.8 * self.detah * E
            sol = torch.cat((S_next, E_next, I_next, Iin_next, R_next, Sp_next, Ip_next, Sa_next, Ea_next, Ia_next),
                            dim=0)
            resap = torch.cat((res, betah[step], betav[step], b[step]), dim=0)
            internal_state = sol
            outputs.append(sol)
            outres.append(resap)
        return outputs, outres


def train_model(model, optimizer, epochs):
    model.train()
    loss = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    # Add_loss = [(212, 333), (579, 699), (943, 1064), (1308, 1429), (1673, 1794)]
    # indices = torch.cat([torch.arange(start, end) for start, end in Add_loss])
    Add_loss = [(186, 204), (239, 256)]
    indices4 = torch.cat([torch.arange(start, end) for start, end in Add_loss])
    Add_loss2 = [(0, 58), (334, 424), (700, 789), (1065, 1154), (1430, 1519), (1795, 1825)]
    # indices2 = torch.cat([torch.arange(start, end) for start, end in Add_loss2])
    # Add_loss3 = [(943, 1064), (1308, 1429)]
    Add_loss3 = [(134, 152), (186, 204), (239, 256)]
    indices3 = torch.cat([torch.arange(start, end) for start, end in Add_loss3])
    Add_loss4 = [(80, 100), (130, 150)]
    indices4 = torch.cat([torch.arange(start, end) for start, end in Add_loss4])
    # Add_loss4 = [(273, 303), (639, 669), (1004, 1034), (1358, 1380), (1715, 1755)]
    # indices4 = torch.cat([torch.arange(start, end) for start, end in Add_loss4])
    for epoch in range(epochs):
        optimizer.zero_grad()
        sol_pred, sol_res = model(num_steps, mymu, mydp, myda, myborn, bh, bv, Import, Tinput,b)
        sol_res = torch.stack(sol_res)
        H_pred = sol_res[1:, 0]
        # b_pred = sol_res[1:, 3]
        window_size = 7

        # 计算每7天的总和（假设H_pred和Local的长度是7的倍数）
        H_pred_sum = H_pred[365:].unfold(0, window_size, window_size).sum(dim=1)
        Local_sum = Local[365:].unfold(0, window_size, window_size).sum(dim=1)
        # H_pred_sum = H_pred
        # Local_sum = Local
        # mask = Local_sum > 15

        myloss = 0.8 * loss(H_pred_sum, Local_sum) + 2 * loss(H_pred_sum[indices4], Local_sum[indices4])
        # + 100*loss(b_pred,torch.ones_like(b_pred))
        # myloss = loss(H_pred, Local)

        myloss.backward(retain_graph=True)
        optimizer.step()
        model.clamp_parameters()

        if epoch % 10 == 0:
            print(
                f"Epoch [{epoch}/{epochs}], Loss: {myloss.item()}, gamma: {model.gamma.item()}, U: {model.U.item()}, detaa: {model.detaa.item()}. detah: {model.detah.item()}, out: {model.out.item()}")
            # , radio1: {model.radio1.item()}, radio2: {model.radio2.item()},radio1: {model.radio1},detaa: {model.detaa.item()}
        if epoch % 20 == 0:
            sol_pred = torch.stack(sol_pred)
            plt.style.use('ggplot')  # 更美观的背景样式

            t = range(0, len(H_pred_sum))
            E2 = sol_pred[1:, 1]
            I2 = sol_pred[1:, 2]

            Sv2 = sol_pred[1:, 7]
            Ev2 = sol_pred[1:, 8]
            Iv2 = sol_pred[1:, 9]

            fig = plt.figure(figsize=(24, 12))
            gs = fig.add_gridspec(4, 3)  # 创建一个 3 行 3 列的网格
            ax1 = fig.add_subplot(gs[0, 0:3])
            ax1.scatter(t, Local_sum.detach().numpy(), color=color3, label='Exact Data Points', s=10, edgecolor='black')
            ax1.plot(t, H_pred_sum.detach().numpy(), color=color1, label='Predicted Curve', linewidth=4,
                     linestyle='-')
            ax1.set_title('True vs Predicted (res)')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Data Values')
            ax1.legend()

            ax2 = fig.add_subplot(gs[1, 0])
            ax2.plot(E2.detach().numpy(), label='Ih')
            ax2.set_title('Eh')
            ax2.set_ylabel('Eh')
            ax2.legend()

            ax3 = fig.add_subplot(gs[1, 1])
            ax3.plot(I2.detach().numpy(), label='Ih')
            ax3.set_title('Ih')
            ax3.set_ylabel('Ih')
            ax3.legend()

            ax4 = fig.add_subplot(gs[1, 2])
            ax4.plot(Sv2.detach().numpy(), label='Sv', color='r')
            ax4.set_title('Sv')
            ax4.set_ylabel('Sv')
            ax4.legend()

            ax5 = fig.add_subplot(gs[2, 0])
            ax5.plot(Ev2.detach().numpy(), label='Ev', color='r')
            ax5.set_title('Ev')
            ax5.set_ylabel('Ev')
            ax5.legend()

            ax6 = fig.add_subplot(gs[2, 1])
            ax6.plot(Iv2.detach(), label='Iv', color='r')
            ax6.set_title('Iv')
            ax6.set_ylabel('Iv')
            ax6.legend()

            M2 = sol_res[1:, 1]

            # bh2 = bh * M2.unsqueeze(1)
            # bv2 = bv * M2.unsqueeze(1)
            bh2 = sol_res[1:, 1]
            bv2 = sol_res[1:, 2]
            b2 = sol_res[1:, 1]

            ax7 = fig.add_subplot(gs[2, 2])
            ax7.plot(bh2.detach().numpy(), label='betah', color='g')
            ax7.set_title('betah')
            ax7.set_ylabel('betah')
            ax7.legend()

            ax10 = fig.add_subplot(gs[3, 0])
            ax10.plot(bv2.detach().numpy(), label='betav', color='b')
            ax10.set_title('betav')
            ax10.set_ylabel('betav')
            ax10.legend()

            ax8 = fig.add_subplot(gs[3, 1])
            ax8.plot(b2.detach().numpy(), label='b', color='g')
            ax8.set_title('b')
            ax8.set_ylabel('b')
            ax8.legend()

            ax9 = fig.add_subplot(gs[3, 2])

            wenresult = Sv2 + Ev2 + Iv2
            # df = pd.DataFrame(wenresult.detach().numpy(), columns=['AD'])
            # df.to_excel('E:\\登革热广州\\新旧蚊子\\AD.xlsx', index=False, header=False)
            # Sp2 = sol_pred[1:,5]
            # Ip2 = sol_pred[1:,6]
            # wenresult2 = Sp2 + Ip2
            # df = pd.DataFrame(wenresult2.detach().numpy(), columns=['W'])
            # df.to_excel('E:\\登革热广州\\新旧蚊子\\W.xlsx', index=False, header=False)
            # ax9.scatter(normalize_0_1(wen).detach().numpy(), label='Nv_pred', color='orange')
            ax9.plot(normalize_0_1(wenresult).detach().numpy(), label='Nv_real', color='blue')
            ax9.set_title('Nv')
            ax9.set_ylabel('Nv')
            ax9.legend()
            plt.tight_layout()
            # fig_path = f'E:/登革热广州/1231多网络组合/beha0119.png'
            # plt.savefig(fig_path, dpi=600, bbox_inches='tight')  # bbox_inches='tight' 避免裁剪内容
            plt.show()

num_steps = 1826
Tinput = (torch.linspace(0, num_steps, num_steps).view(-1, 1)) / num_steps  # [100, 1]
model = ODENN()
# optimizer = torch.optim.Adam([
#     # {'params': model.Sp0, 'lr': 0.01},
#     # {'params': model.k2, 'lr': 0.001},
#     {'params': [model.gamma, model.out, model.detah, model.U], 'lr': 0.001}
# ], lr=0.000001)
# train_model(model, optimizer, epochs=10000)




#
sol_pred, sol_res = model(num_steps, mymu, mydp, myda, myborn, bh, bv, Import, Tinput, b)
sol_res = torch.stack(sol_res)
sol_pred = torch.stack(sol_pred)


S = sol_pred[1:, 0]
E = sol_pred[1:, 1]
I = sol_pred[1:, 2]
Iin = sol_pred[1:, 3]
R = sol_pred[1:, 4]
Sp=sol_pred[1:, 5]
Ip=sol_pred[1:, 6]
Sa = sol_pred[1:, 7]
Ea = sol_pred[1:, 8]
Ia = sol_pred[1:, 9]

new = sol_res[1:,0]
betah = sol_res[1:,1]
betav = sol_res[1:,2]
bite = sol_res[1:,3]

if isinstance(S, torch.Tensor):
    S = S.detach().numpy()
if isinstance(E, torch.Tensor):
    E = E.detach().numpy()
if isinstance(I, torch.Tensor):
    I = I.detach().numpy()
if isinstance(Iin, torch.Tensor):
    Iin = Iin.detach().numpy()
if isinstance(R, torch.Tensor):
    R = R.detach().numpy()
if isinstance(Sp, torch.Tensor):
    Sp = Sp.detach().numpy()
if isinstance(Ip, torch.Tensor):
    Ip = Ip.detach().numpy()
if isinstance(Sa, torch.Tensor):
    Sa = Sa.detach().numpy()
if isinstance(Ea, torch.Tensor):
    Ea = Ea.detach().numpy()
if isinstance(Ia, torch.Tensor):
    Ia = Ia.detach().numpy()
if isinstance(new, torch.Tensor):
    new = new.detach().numpy()
if isinstance(betah, torch.Tensor):
    betah = betah.detach().numpy()
if isinstance(betav, torch.Tensor):
    betav = betav.detach().numpy()
if isinstance(bite, torch.Tensor):
    bite = bite.detach().numpy()

# 创建DataFrame
data = {
    'S': S,
    'E': E,
    'I': I,
    'Iin': Iin,
    'R': R,
    'Sp': Sp,
    'Ip': Ip,
    'Sa': Sa,
    'Ea': Ea,
    'Ia': Ia,
    'new': new,
    'betah': betah,
    'betav': betav,
    'bite': bite,
    'real':Local
}

df = pd.DataFrame(data)

# 保存为Excel文件
df.to_excel('E:\登革热广州\结果数据\model2_results.xlsx', index=False, engine='openpyxl')

print("数据已成功保存为 model2_results.xlsx 文件")


#
#
#
# from datetime import datetime, timedelta
# color4 = (242 / 255, 146 / 255, 33 / 255)
# color5 = (193 / 255, 40 / 255, 45 / 255)
# H_pred = sol_res[1:, 0]
# H_pred_sum = H_pred
# Local_sum = Local[365:]
# window_size = 7
# H_pred_sum = H_pred[365:].unfold(0, window_size, window_size).sum(dim=1)
# Local_sum = Local[365:].unfold(0, window_size, window_size).sum(dim=1)
# t = range(0, len(H_pred_sum))
# fig = plt.figure(figsize=(8, 4))
# gs = fig.add_gridspec(1, 1)
# ax1 = fig.add_subplot(gs[0, 0])
# ax1.scatter(t, Local_sum.detach().numpy(), color=color5, label='Exact Data Points', s=50, edgecolor="white")
# ax1.plot(t, H_pred_sum.detach().numpy(), color=color1, label='Predicted Curve', linewidth=5,
#          linestyle='-')
# ax1.set_title('GuangDong',
#               fontsize=12, pad=5, fontweight='bold')
# ax1.set_xlabel('Time', fontsize=12, labelpad=5)
# ax1.set_ylabel('Cases', fontsize=12, labelpad=5)
#
# start_date = datetime(2016, 1, 1)
# date_list = [start_date + timedelta(days=int(x)*7) for x in t]
# num_ticks = 8  # 可以根据需要调整
# ax1.set_xticks(np.linspace(0, len(H_pred_sum)-1, num_ticks))  # 均匀分布的刻度
# ax1.set_xticklabels([(start_date + timedelta(days=int(x)*7)).strftime('%y-%m')
#                    for x in np.linspace(0, len(H_pred_sum)-1, num_ticks)])
#
# # Rotate date labels for better readability
# plt.setp(ax1.get_xticklabels(), rotation=0, ha='right')
#
# ax1.grid(False)
# ax1.set_facecolor('white')
# for spine in ax1.spines.values():
#     spine.set_visible(True)
#     spine.set_linewidth(1.5)
#     spine.set_color('black')
# ax1.legend(facecolor='none', frameon=False)
# plt.tight_layout()
# plt.show()
#
#
# from scipy import stats
# x = H_pred[365:].detach().numpy()
# y = Local[365:].detach().numpy()
# residuals = y - x
#
# fig, ax = plt.subplots(figsize=(8, 6))
# sm.ProbPlot(residuals, dist=stats.norm).ppplot(line='45', ax=ax)
# ax.set_title("PP Plot for Normality Check", fontsize=14)
# ax.set_xlabel("Theoretical Cumulative Probability", fontsize=12)
# ax.set_ylabel("Sample Cumulative Probability", fontsize=12)
# plt.grid(True)
# plt.show()
#
#
#
#
#
# M2 = sol_res[365+1:, 2]
# t = range(0, len(M2))
# fig = plt.figure(figsize=(8, 4))
# gs = fig.add_gridspec(1, 1)
# ax1 = fig.add_subplot(gs[0, 0])
# ax1.plot(t,M2.detach().numpy(), color=color5, label='Predicted Curve', linewidth=5,
#          linestyle='-')
# ax1.set_title('Biting rate',
#               fontsize=12, pad=5, fontweight='bold')
# ax1.set_xlabel('Time', fontsize=12, labelpad=5)
# ax1.set_ylabel('Biting rate', fontsize=12, labelpad=5)
# ax1.grid(False)
# start_date = datetime(2016, 1, 1)
# date_list = [start_date + timedelta(days=int(x)) for x in t]
# ax1.set_xticks(np.linspace(0, 1461, 8))  # 6 ticks from start to end
# ax1.set_xticklabels([(start_date + timedelta(days=int(x))).strftime('%y-%m')
#                    for x in np.linspace(0, 1461, 8)])
#
# # Rotate date labels for better readability
# plt.setp(ax1.get_xticklabels(), rotation=0, ha='right')
# ax1.set_facecolor('white')
# for spine in ax1.spines.values():
#     spine.set_visible(True)
#     spine.set_linewidth(1.5)
#     spine.set_color('black')
# # ax1.legend(facecolor='none', frameon=False)
# plt.tight_layout()
# plt.show()
#
#
#
#
#
#
#
#
#
# t = range(0, len(H_pred_sum))
# fig = plt.figure(figsize=(26, 4))
# gs = fig.add_gridspec(1, 3)  # 创建一个 3 行 3 列的网格
# ax1 = fig.add_subplot(gs[0, 0:2])
# ax1.scatter(t, Local_sum.detach().numpy(), color=color3, label='Exact Data Points', s=40, edgecolor='black')
# ax1.plot(t, H_pred_sum.detach().numpy(), color=color1, label='Predicted Curve', linewidth=4,
#          linestyle='-')
# ax1.set_title('Local_I Model1',
#               fontsize=16, pad=5, fontweight='bold')
# ax1.set_xlabel('Time', fontsize=16, labelpad=5)
# ax1.set_ylabel('Cases', fontsize=16, labelpad=5)
# start_date = datetime(2016, 1, 1)
# date_list = [start_date + timedelta(days=int(x)*7) for x in t]
# num_ticks = 24  # 可以根据需要调整
# ax1.set_xticks(np.linspace(0, len(H_pred_sum)-1, num_ticks))  # 均匀分布的刻度
# ax1.set_xticklabels([(start_date + timedelta(days=int(x)*7)).strftime('%y-%m')
#                    for x in np.linspace(0, len(H_pred_sum)-1, num_ticks)])
#
# # Rotate date labels for better readability
# plt.setp(ax1.get_xticklabels(), rotation=0, ha='right')
# ax1.legend()
# Iin2 = sol_pred[1:, 3]
# ax2 = fig.add_subplot(gs[0, 2])
# ax2.plot(Iin2.detach().numpy(), color=color5, linewidth=2,linestyle='-')
# ax2.set_title('Import_I',
#               fontsize=16, pad=5, fontweight='bold')
# ax2.set_xlabel('Time', fontsize=16, labelpad=5)
# ax2.set_ylabel('Cases', fontsize=16, labelpad=5)
# start_date = datetime(2016, 1, 1)
# date_list = [start_date + timedelta(days=int(x)) for x in t]
# num_ticks = 8  # 可以根据需要调整
# ax2.set_xticks(np.linspace(0, len(Iin2)-1, num_ticks))  # 均匀分布的刻度
# ax2.set_xticklabels([(start_date + timedelta(days=int(x))).strftime('%y-%m')
#                    for x in np.linspace(0, len(Iin2)-1, num_ticks)])
# plt.tight_layout()
# plt.show()

