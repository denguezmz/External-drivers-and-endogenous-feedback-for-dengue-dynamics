import matplotlib.pyplot as plt
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import os
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
import math
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
torch.manual_seed(25000)

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


born = BORN()
model_path = './data/yuan_state_dict_born.pth'
model_weight = torch.load(model_path)
born.load_state_dict(model_weight)

file_path = './data/Local.xlsx'
df = pd.read_excel(file_path)
Local1 = df['Guangxi'].values
Local = torch.tensor(Local1, dtype=torch.float32)

file_path = './data/Input.xlsx'
df = pd.read_excel(file_path)
Import1 = df['Guangxi'].values
Import = torch.tensor(Import1, dtype=torch.float32)

file_path = './data/Temp.xlsx'
df = pd.read_excel(file_path)
T = df['Temp']
file_path = './data/Rain.xlsx'
df = pd.read_excel(file_path)
R = df['Rain']
file_path = './data/Temp.xlsx'
df = pd.read_excel(file_path)
AT = df['Ave_Temp'].values
file_path = './data/Rain.xlsx'
df = pd.read_excel(file_path)
AR = df['Avg_Rain'].values

def sigmoid(x):
    return 1 / (1 + math.exp(-x))
ax = 0.0135
hax = 28116.4141
hhx = 35378.2344
thx = 301.6750
# mup and mua are fixed values rather than nn.Parameter objects.
mup = 0.9991  # Initial value.
mua = 0.6308  # Initial value.
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
T = torch.tensor(T.to_numpy(), dtype=torch.float32).view(-1, 1)

class MosquitoModel:
    def __init__(self, init_state, myborn, mydp, mymu, myda, AR):
        self.init_state = init_state
        self.myborn = myborn
        self.mydp = mydp
        self.mymu = mymu
        self.myda = myda
        self.AR = AR
        self.SpK = torch.tensor([5.0e+07])

    def simulate(self, num_steps=365):
        outputs = []
        outputs2 = []
        internal_state = self.init_state
        for step in range(num_steps):
            if step == 0:
                internal_state = self.init_state
            h = internal_state
            Sp = h[0]
            Sa = h[1]
            dSp = self.myborn[step] * Sa - self.mydp[step] * Sp - self.mymu[step] * Sp
            ef = torch.abs(torch.exp(-0.1 * (1 + (self.mymu[step] * Sp) / (self.SpK * (1 + 0.0239 * self.AR[step])))))
            dSa = ef * self.mymu[step] * Sp - self.myda[step] * Sa
            Sp_next = Sp + dSp
            Sa_next = Sa + dSa
            sol = torch.cat((Sp_next, Sa_next), dim=0)
            internal_state = sol
            outputs.append(sol)
        return outputs

init_state = torch.tensor([1.0e+07, 0.0])
model = MosquitoModel(init_state, myborn, mydp, mymu, myda, AR)
outputs = model.simulate()
sol_res = torch.stack(outputs)
W_pred = sol_res[-1, 0]
A_pred = sol_res[-1, 1]

class SineActivation(nn.Module):
    def __init__(self, amplitude=1.0, frequency=1.0, offset=0.0):
        super(SineActivation, self).__init__()
        self.amplitude = amplitude
        self.frequency = frequency
        self.offset = offset
    def forward(self, x):
        return self.amplitude * torch.sin(self.frequency * x + self.offset)


class ODENN(nn.Module):
    def __init__(self):
        super(ODENN, self).__init__()
        self._Yunnan_layers = []
        self.init_state = []
        self.init_res = []
        self.init_sym = []
        self.Nh = torch.tensor([56590000])
        self.S0 = torch.tensor([56590000])
        self.E0 = torch.tensor([0])
        self.I0 = torch.tensor([0])
        self.Iin0 = torch.tensor([0])
        self.R0 = torch.tensor([0])
        self.rateS0 = torch.nn.Parameter(torch.tensor([7.2446], dtype=torch.float32), requires_grad=True)
        self.res0 = torch.tensor([0])
        self.M0 = torch.tensor([0])
        self.N0 = torch.tensor([0])
        self.gamma = torch.nn.Parameter(torch.tensor([0.0834], dtype=torch.float32), requires_grad=True)
        self.gammaimp = torch.nn.Parameter(torch.tensor([0.0505], dtype=torch.float32), requires_grad=True)
        self.U = torch.nn.Parameter(torch.tensor([0.1324], dtype=torch.float32), requires_grad=True)
        self.detah = torch.nn.Parameter(torch.tensor([0.1038], dtype=torch.float32), requires_grad=True)
        self.beta0 = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=False)
        self.beta1 = torch.nn.Parameter(torch.tensor([500], dtype=torch.float32), requires_grad=True)
        self.rate1 = torch.nn.Parameter(torch.tensor([6.9141], dtype=torch.float32), requires_grad=True)
        self.rate2 = torch.nn.Parameter(torch.tensor([1.4215], dtype=torch.float32), requires_grad=True)
        self.rate3 = torch.nn.Parameter(torch.tensor([9.1214], dtype=torch.float32), requires_grad=True)

    def clamp_parameters(self):
        self.gamma.data = F.hardtanh(self.gamma, min_val=0.0714, max_val=0.3)
        self.gammaimp.data = F.hardtanh(self.gammaimp, min_val=0.01, max_val=0.3)
        self.U.data = F.hardtanh(self.U, min_val=0.0140, max_val=0.1340)
        self.detah.data = F.hardtanh(self.detah, min_val=0.100, max_val=0.25)

    def forward(self, num_steps, mymu, mydp, myda, myborn, AR, Import, T):
        rate1  = self.rate1 * (1e-11)
        rate2 = self.rate2 * (1e-5)
        rate3 = self.rate3 * (1e-8)
        powSP = torch.pow(torch.tensor([10]), self.rateS0)
        Sp0 = powSP
        Ip0 = torch.tensor([0])
        Sa0 = torch.tensor([0])
        Ea0 = torch.tensor([0])
        Ia0 = torch.tensor([0])
        init_res = torch.cat((self.res0, self.M0, self.M0, self.M0), dim=0)
        internal_state = torch.cat((self.S0, self.E0, self.I0, self.I0, self.Iin0, self.R0, Sp0, Ip0, Sa0, Ea0, Ia0), dim=0)
        self.SpK = Sp0 * 5
        deathh = torch.tensor([0])
        outputs = [internal_state]
        outres = [init_res]
        detaa = 1 / (4 + np.exp(5.15 - 0.123 * T))
        b = 0.0043 * T + 0.0943
        bh = torch.where((T >= 12.286) & (T <= 32.461), 0.001044 * T * (T - 12.286) * torch.sqrt(32.461 - T),
                         torch.tensor(0.0))
        bv = torch.where((T >= 12.4) & (T < 26.1), -0.9037 + 0.0729 * T, torch.tensor(0.0))
        bv = torch.where((T >= 26.1) & (T <= 32.5), 1.0, bv)
        betah = torch.abs(bh * b * self.beta0)
        betav = torch.abs(bv * b * self.beta0)
        for step in range(num_steps):
            h = internal_state
            S, E, I, A, Iin, R, Sp, Ip, Sa, Ea, Ia = h
            N = S + E + I + R + Iin + A
            input = I + Iin
            M = rate1 * input * input / (1 + rate2 * input * input) + rate3
            dSp = myborn[step] * Sa + (1-self.U) * myborn[step] * (Ia + Ea) - mydp[step] * Sp - mymu[step] * Sp
            dIp = self.U * myborn[step] * (Ia + Ea) - mydp[step] * Ip - mymu[step] * Ip
            ef = torch.abs(torch.exp(-0.1 * (1 + (mymu[step] * Sp + mymu[step] * Ip) / (self.SpK * (1 + 0.0239 * AR[step])))))
            dSa = ef * mymu[step] * Sp -  M * torch.log(1 + betav[step]  * (I + A + Iin) / (M * N)) * Sa - myda[step] * Sa
            dEa = M * torch.log(1 + betav[step]  * (I + A + Iin) / (M * N)) * Sa - myda[step] * Ea - detaa[step] * Ea
            dIa = ef * mymu[step] * Ip + detaa[step] * Ea - myda[step] * Ia
            dS = - M * torch.log(1 + betah[step]  * Ia / (M * N)) * S
            dE =  M * torch.log(1 + betah[step]  * Ia / (M * N)) * S - self.detah * E - deathh * E
            dI = 0.3125 * self.detah * E - self.gamma * I - deathh * I
            dA = 0.6875 * self.detah * E - self.gamma * A - deathh * A
            dIin = Import[step] -  (self.gamma + self.gammaimp) * Iin - deathh * Iin
            dR =  self.gamma * I - deathh * R +  (self.gamma + self.gammaimp) * Iin + self.gamma * A
            S_next = torch.clamp(S + dS, min=0.0)
            E_next = torch.clamp(E + dE, min=0.0)
            I_next = torch.clamp(I + dI, min=0.0)
            A_next = torch.clamp(A + dA, min=0.0)
            Iin_next = torch.clamp(Iin + dIin, min=0.0)
            R_next = torch.clamp(R + dR, min=0.0)
            Sp_next = torch.clamp(Sp + dSp, min=0.0)
            Ip_next = torch.clamp(Ip + dIp, min=0.0)
            Sa_next = torch.clamp(Sa + dSa, min=0.0)
            Ea_next = torch.clamp(Ea + dEa, min=0.0)
            Ia_next = torch.clamp(Ia + dIa, min=0.0)
            # M_next = torch.clamp(M + dM, min=0.0, max = 1.0)
            true_cases = 0.3125 * self.detah * E
            # observed_cases = torch.binomial(true_cases, torch.tensor([0.8]))
            internal_state = torch.cat((S_next, E_next, I_next, A_next, Iin_next, R_next, Sp_next, Ip_next, Sa_next, Ea_next, Ia_next),
                            dim=0)
            outputs.append(internal_state)
            outres.append(torch.cat((true_cases, betah[step], betav[step], M), dim=0))
        return outputs, outres


def custom_loss(H_pred_sum, Local_sum, weight_above_100=1.5, weight_below_100=1.0):
    base_loss = (H_pred_sum - Local_sum) ** 2
    weights = torch.where(Local_sum > 100, weight_above_100, weight_below_100)
    weighted_loss = base_loss * weights
    return weighted_loss.mean()


def train_model(model, optimizer, epochs):
    model.train()
    loss = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    Add_loss4 = [(37,42)]
    indices4 = torch.cat([torch.arange(start, end) for start, end in Add_loss4])
    window_size = 7
    for epoch in range(epochs):
        optimizer.zero_grad()
        # sol_pred, sol_res = model(num_steps, mymu[365:], mydp[365:], myda[365:], myborn[365:], AR[365:], Import[365:],
        #                           T[365:])
        sol_pred, sol_res = model(num_steps, mymu[-730:], mydp[-730:], myda[-730:], myborn[-730:], AR[-730:], Import[-730:],
                                  T[-730:])
        sol_res = torch.stack(sol_res)
        sol_pred = torch.stack(sol_pred)
        H_pred = sol_res[1:, 0]
        H_pred_sum = H_pred[-365:].unfold(0, window_size, window_size).sum(dim=1)
        Local_sum = Local[-365:].unfold(0, window_size, window_size).sum(dim=1)
        myloss = custom_loss(H_pred_sum, Local_sum, weight_above_100=1.5, weight_below_100=1.0)
        # myloss = loss(H_pred_sum, Local_sum)+ 1.5*loss(H_pred_sum[indices4], Local_sum[indices4])
        myloss.backward(retain_graph=True)
        optimizer.step()
        model.clamp_parameters()
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {myloss.item()}")
        if epoch % 20 == 0:
            plt.style.use('ggplot')
            t = range(0, len(H_pred_sum))
            S2 = sol_pred[1:, 0] / 10000000
            I2 = sol_pred[1:, 2]
            E2 = sol_pred[1:, 1]
            Iin2 = sol_pred[1:, 3]
            Sv2 = sol_pred[1:, 7]
            Ev2 = sol_pred[1:, 8]
            Iv2 = sol_pred[1:, 9]
            fig = plt.figure(figsize=(24, 12))
            gs = fig.add_gridspec(4, 3)
            ax1 = fig.add_subplot(gs[0, 0:2])
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
            ax4.plot(Iin2.detach().numpy(), label='Sv', color='r')
            ax4.set_title('Iin')
            ax4.set_ylabel('Iin')
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

            M2 = sol_res[1:, 3]
            bh2 = sol_res[1:, 1]
            bv2 = sol_res[1:, 2]

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
            ax8.plot(M2.detach().numpy(), label='M', color='g')
            ax8.set_title('M')
            ax8.set_ylabel('M')
            ax8.legend()

            ax9 = fig.add_subplot(gs[3, 2])

            wenresult = Sv2 + Ev2 + Iv2
            ax9.plot(normalize_0_1(wenresult).detach().numpy(), label='Nv_real', color='blue')
            ax9.set_title('Nv')
            ax9.set_ylabel('Nv')
            ax9.legend()
            plt.tight_layout()
            plt.show()
num_steps = 365+ 365
model = ODENN()
optimizer = torch.optim.Adam([
    # {'params': model.bate.parameters(), 'lr': 0.00001},
    # {'params': [model.beta0,model.beta1], 'lr': 0.00001},
    # {'params': [model.myrate], 'lr': 0.00001},
    # {'params': [model.rateIp0], 'lr': 0.000000001},
    {'params': model.rateS0, 'lr': 0.001},
    {'params': [model.detah,model.U], 'lr': 0.001},
    {'params': [model.gamma,model.gammaimp,], 'lr': 0.001},
    # {'params': [model.rate3], 'lr': 0.001},
    {'params': [model.rate1,], 'lr': 0.001},
    {'params': [model.rate2, ], 'lr': 0.001},
])
# train_model(model, optimizer, epochs=5000)

sol_pred, sol_res = model(num_steps, mymu[-730:], mydp[-730:], myda[-730:], myborn[-730:], AR[-730:], Import[-730:],
                          T[-730:])
sol_res = torch.stack(sol_res)
sol_pred = torch.stack(sol_pred)


S = sol_pred[-365:, 0]
E = sol_pred[-365:, 1]
I = sol_pred[-365:, 2]
A = sol_pred[-365:, 3]
Iin = sol_pred[-365:, 4]
R = sol_pred[-365:, 5]
Sp=sol_pred[-365:, 6]
Ip=sol_pred[-365:, 7]
Sa = sol_pred[-365:, 8]
Ea = sol_pred[-365:, 9]
Ia = sol_pred[-365:, 10]

H_pred_sum = sol_res[-365:, 0]
Local_sum = Local[-365:]


M = sol_res[-365:, 3]
#
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
if isinstance(H_pred_sum, torch.Tensor):
    H_pred_sum = H_pred_sum.detach().numpy()
if isinstance(Local_sum, torch.Tensor):
    Local_sum = Local_sum.detach().numpy()
if isinstance(M, torch.Tensor):
    M = M.detach().numpy()
data = {
    'H_pred':H_pred_sum,
    'Local':Local_sum,
    'M':M,
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
}
df = pd.DataFrame(data)
df.to_excel('./result/Guangxi_heterogeneity.xlsx', index=False, engine='openpyxl')
