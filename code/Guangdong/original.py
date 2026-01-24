import os
import time
import math
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.ndimage import gaussian_filter1d
import statsmodels.api as sm
from scipy.ndimage import gaussian_filter1d

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
torch.manual_seed(10)

color1 = (179 / 255, 70 / 255, 138 / 255)
color2 = (228 / 255, 192 / 255, 214 / 255)
color3 = (164 / 255, 136 / 255, 186 / 255)

def normalize_0_1(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

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

born = BORN()
model_path = f'./data/yuan_state_dict_born.pth'
model_weight = torch.load(model_path)
born.load_state_dict(model_weight)

num_steps = 1826
TT = torch.linspace(1, num_steps, num_steps).view(-1, 1)  # [100, 1]
TT = TT / num_steps

file_path = './data/Local.xlsx'
df = pd.read_excel(file_path)
Local1 = df['All2'].values
Local = torch.tensor(Local1, dtype=torch.float32)

sigma = 2 # 高斯核的标准差
Local_smooth = gaussian_filter1d(Local.numpy(), sigma=sigma)
Local_smooth = torch.tensor(Local)
window_size = 7
Local_7_smooth = Local_smooth[365:].unfold(0, window_size, window_size).sum(dim=1)
Local_7_smooth = torch.tensor(Local_7_smooth)

file_path = './data/Input.xlsx'
df = pd.read_excel(file_path)
Import1 = df['All2'].values
Import = torch.tensor(Import1, dtype=torch.float32)

file_path = './data/Temp.xlsx'
df = pd.read_excel(file_path)
T = df['All']
file_path = './data/Rain.xlsx'
df = pd.read_excel(file_path)
R = df['All']
file_path = './data/Expanded_Half_Month_Averages.xlsx'
df = pd.read_excel(file_path)
AT = df['All'].values
file_path = './data/Seven_Day_Smoothed_Rain.xlsx'
df = pd.read_excel(file_path)
AR = df['All'].values

ax = 0.0135
hax = 28116.4141
hhx = 35378.2344
thx = 301.6750
mup = 0.9991
mua = 0.6308
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

b = T.apply(lambda t: 0.0043 * t + 0.0943 if 10 <= t <= 40 else 0)
bh = T.apply(lambda t: 0.001044 * t * (t - 12.286) * np.sqrt(32.461 - t) if 12.286 <= t <= 32.461 else 0)
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

        self.Nh = torch.tensor([113460000])
        self.S0 = torch.tensor([113460000])
        self.E0 = torch.tensor([0])
        self.I0 = torch.tensor([0])
        self.Iin0 = torch.tensor([0])
        self.R0 = torch.tensor([0])

        self.Sp0 = torch.nn.Parameter(torch.tensor([7.17], dtype=torch.float32), requires_grad=True)
        self.rateIp0 = torch.nn.Parameter(torch.tensor([2.4933e-08], dtype=torch.float32), requires_grad=True)
        self.Ip0 = torch.tensor([0])
        self.Sa0 = torch.tensor([0])
        self.Ea0 = torch.tensor([0])
        self.Ia0 = torch.tensor([0])

        self.res0 = torch.tensor([0])
        self.M0 = torch.tensor([0])
        self.N0 = torch.tensor([0])
        self.gamma = torch.nn.Parameter(torch.tensor([0.1012], dtype=torch.float32), requires_grad=True)
        self.out = torch.nn.Parameter(torch.tensor([0.01], dtype=torch.float32), requires_grad=True)
        self.U = torch.nn.Parameter(torch.tensor([0.9436], dtype=torch.float32), requires_grad=True)
        self.detah = torch.nn.Parameter(torch.tensor([0.1037], dtype=torch.float32), requires_grad=True)

    def clamp_parameters(self):

        self.gamma.data = F.hardtanh(self.gamma, min_val=0.0714, max_val=0.3333)
        self.U.data = F.hardtanh(self.U, min_val=0.826, max_val=0.9436)
        self.detah.data = F.hardtanh(self.detah, min_val=0.10, max_val=0.25)
        self.out.data = F.hardtanh(self.out, min_val=0.01, max_val=0.03)
        # self.Myk.data = F.hardtanh(self.Myk, min_val=0.00000001, max_val=1.0)

    def forward(self, num_steps, mymu, mydp, myda, myborn, bh, bv, Import, inpu,b):
        Sp0 = torch.pow(torch.tensor([10]), self.Sp0)
        SpK = Sp0 * 5
        self.init_state = torch.cat(
            (self.S0, self.E0, self.I0,self.I0, self.Iin0, self.R0, Sp0*(1-self.rateIp0), Sp0*self.rateIp0, self.Sa0, self.Ea0, self.Ia0), dim=0)
        self.init_res = torch.cat((self.res0,  self.M0, self.M0, self.M0), dim=0)
        deathh = torch.tensor([0])
        outputs = [self.init_state]
        outres = [self.init_res]

        betah = bh
        betav = bv
        detaa = 1 / (4 + np.exp(5.15 - 0.123 * T))

        for step in range(num_steps):
            if step == 0:
                h = self.init_state
                internal_state = h
            h = internal_state
            S, E, I, A, Iin, R, Sp, Ip, Sa, Ea, Ia = h
            N = S + E + I + A + R + Iin
            dSp = myborn[step] * Sa + self.U * myborn[step] * (Ia + Ea) - mydp[step] * Sp - mymu[step] * Sp
            dIp = (1 - self.U) * myborn[step] * (Ia + Ea) - mydp[step] * Ip - mymu[step] * Ip
            ef = torch.abs(
                torch.exp(-0.1 * (1 + (mymu[step] * Sp + mymu[step] * Ip) / (SpK * (1 + 0.0239 * AR[step])))))

            dSa = ef * mymu[step] * Sp - betav[step] * Sa * (I + A + Iin) / N  - myda[step] * Sa
            dEa = betav[step] * Sa * (I + A + Iin) / N - myda[step] * Ea - detaa[step] * Ea
            dIa = ef * mymu[step] * Ip + detaa[step] * Ea - myda[step] * Ia

            dS = - deathh * S - betah[step] * Ia * S / N
            dE = betah[step] * Ia * S / N - self.detah * E - deathh * E
            dI = 0.3125 * self.detah * E - self.gamma * I - deathh * I
            dA = 0.6875 * self.detah * E - self.gamma * A - deathh * A
            dIin = Import[step] - (self.out + self.gamma) * Iin
            dR =  self.gamma * I - deathh * R + (self.out + self.gamma) * Iin + self.gamma * A

            S_next = torch.clamp(S + dS, min=0.0)
            E_next = torch.clamp(E + dE, min=0.0)
            I_next = torch.clamp(I + dI, min=0.0)
            A_next = torch.clamp(A + dA, min=0.0)
            Iin_next = torch.clamp(Iin + dIin, min=0.0)
            R_next = torch.clamp(R + dR, min=0.0)
            Sp_next = torch.clamp(Sp + dSp, min=0.0)
            Ea_next = torch.clamp(Ea + dEa, min=0.0)
            Ip_next = torch.clamp(Ip + dIp, min=0.0)
            Sa_next = torch.clamp(Sa + dSa, min=0.0)
            Ia_next = torch.clamp(Ia + dIa, min=0.0)
            res = 0.3125 * self.detah * E
            sol = torch.cat((S_next, E_next, I_next, A_next, Iin_next, R_next, Sp_next, Ip_next, Sa_next, Ea_next, Ia_next),
                            dim=0)
            resap = torch.cat((res, betah[step], betav[step], b[step]), dim=0)
            internal_state = sol
            outputs.append(sol)
            outres.append(resap)
        return outputs, outres

def train_model(model, optimizer, epochs):
    model.train()
    loss = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    Add_loss = [(80, 100),(130,150)]
    indices = torch.cat([torch.arange(start, end) for start, end in Add_loss])

    for epoch in range(epochs):
        optimizer.zero_grad()
        sol_pred, sol_res = model(num_steps, mymu, mydp, myda, myborn, bh, bv, Import, Tinput,b)
        sol_res = torch.stack(sol_res)
        H_pred = sol_res[1:, 0]
        window_size = 7
        H_pred_sum = H_pred[365:].unfold(0, window_size, window_size).sum(dim=1)
        Local_sum = Local[365:].unfold(0, window_size, window_size).sum(dim=1)
        myloss = 1.0 * loss(H_pred_sum, Local_sum) +  1.5 * loss(H_pred_sum[indices], Local_sum[indices])
        myloss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        model.clamp_parameters()

        if epoch % 10 == 0:
            print(
                f"Epoch [{epoch}/{epochs}], Loss: {myloss.item()}, gamma: {model.gamma.item()}, U: {model.U.item()}, detah: {model.detah.item()}, out: {model.out.item()}")
        if epoch % 20 == 0:
            sol_pred = torch.stack(sol_pred)
            plt.style.use('ggplot')
            t = range(0, len(H_pred_sum))
            E2 = sol_pred[1:, 1]
            I2 = sol_pred[1:, 2]
            Sv2 = sol_pred[1:, 8]
            Ev2 = sol_pred[1:, 9]
            Iv2 = sol_pred[1:, 10]
            fig = plt.figure(figsize=(24, 12))
            gs = fig.add_gridspec(4, 3)
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
            ax9.plot(normalize_0_1(wenresult).detach().numpy(), label='Nv_real', color='blue')
            ax9.set_title('Nv')
            ax9.set_ylabel('Nv')
            ax9.legend()
            plt.tight_layout()
            plt.show()

num_steps = 1826
Tinput = (torch.linspace(0, num_steps, num_steps).view(-1, 1)) / num_steps  # [100, 1]
model = ODENN()
optimizer = torch.optim.Adam([
    {'params': [model.gamma, model.out, model.detah, model.U], 'lr': 0.001}
], lr=0.000001)
# train_model(model, optimizer, epochs=10000)

sol_pred, sol_res = model(num_steps, mymu, mydp, myda, myborn, bh, bv, Import, Tinput, b)
sol_res = torch.stack(sol_res)
sol_pred = torch.stack(sol_pred)

S = sol_pred[1:, 0]
E = sol_pred[1:, 1]
I = sol_pred[1:, 2]
A = sol_pred[1:, 3]
Iin = sol_pred[1:, 4]
R = sol_pred[1:, 5]
Sp=sol_pred[1:, 6]
Ip=sol_pred[1:, 7]
Sa = sol_pred[1:, 8]
Ea = sol_pred[1:, 9]
Ia = sol_pred[1:, 10]
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
if isinstance(A, torch.Tensor):
    A = A.detach().numpy()
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

data = {
    'S': S,
    'E': E,
    'I': I,
    'A':A,
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
df.to_excel('./result/Model_2_results.xlsx', index=False, engine='openpyxl')

