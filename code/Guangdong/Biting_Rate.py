import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os
from torch import nn
from torch.nn import Sequential, init
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

# 创建 PyTorch 模型实例
born = BORN()
model_path = f'./data/yuan_state_dict_born.pth'
model_weight = torch.load(model_path)
born.load_state_dict(model_weight)

file_path = './data/Local.xlsx'
df = pd.read_excel(file_path)
Local1 = df['All2'].values
Local = torch.tensor(Local1, dtype=torch.float32)

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

class SineActivation(nn.Module):
    def __init__(self, amplitude=1.0, frequency=1.0, offset=0.0):
        super(SineActivation, self).__init__()
        self.amplitude = amplitude
        self.frequency = frequency
        self.offset = offset

    def forward(self, x):
        return self.amplitude * torch.sin(self.frequency * x + self.offset)
import math
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

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
T = torch.tensor(T.to_numpy(), dtype=torch.float32).view(-1, 1)

class MosquitoModel:
    def __init__(self, init_state, myborn, mydp, mymu, myda, AR):
        self.init_state = init_state
        self.myborn = myborn
        self.mydp = mydp
        self.mymu = mymu
        self.myda = myda
        self.AR = AR
        self.SpK = torch.pow(torch.tensor([10]), 7.17) * 5

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

Sp0 = torch.pow(torch.tensor([10]),7.17)
init_state = torch.tensor([Sp0, 0.0])
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


class mycell5(nn.Module):
    def __init__(self):
        super(mycell5, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Tanh()
        )
    def forward(self, input):
        output = torch.square(self.model(input))
        return output
bate = mycell5()
model_path = f'./weight/bitting.pth'

class ODENN(nn.Module):
    def __init__(self):
        super(ODENN, self).__init__()
        self._all_layers = []
        self.init_state = []
        self.init_res = []
        self.init_sym = []
        self.bate = bate
        self.Nh = torch.tensor([113460000])
        self.S0 = torch.tensor([113460000])
        self.E0 = torch.tensor([0])
        self.I0 = torch.tensor([0])
        self.Iin0 = torch.tensor([0])
        self.R0 = torch.tensor([0])

        self.rateIp0 = torch.nn.Parameter(torch.tensor([2.4933e-08], dtype=torch.float32), requires_grad=True)
        self.res0 = torch.tensor([0])
        self.M0 = torch.tensor([0])
        self.N0 = torch.tensor([0])
        self.gamma = torch.nn.Parameter(torch.tensor([0.0714], dtype=torch.float32), requires_grad=True)
        self.myrate = torch.nn.Parameter(torch.tensor([0.0374], dtype=torch.float32), requires_grad=True) #adjust the range of the network output
        self.myrate2 = torch.nn.Parameter(torch.tensor([0.5102], dtype=torch.float32), requires_grad=True)#adjust the range of the network output
        self.U = torch.nn.Parameter(torch.tensor([0.1183], dtype=torch.float32),requires_grad=True)
        self.detah = torch.nn.Parameter(torch.tensor([0.1045], dtype=torch.float32), requires_grad=True)
        self.gammaimp = torch.nn.Parameter(torch.tensor([0.0306], dtype=torch.float32), requires_grad=True)

        self.myrate7 = torch.nn.Parameter(torch.tensor([0.0132], dtype=torch.float32), requires_grad=True) #adjust the range of the network input
        self.SpK = torch.pow(torch.tensor([10]), 7.17) * 5

    def clamp_parameters(self):
        self.gamma.data = F.hardtanh(self.gamma, min_val=0.0714, max_val=0.15)
        self.gammaimp.data = F.hardtanh(self.gammaimp, min_val=0.01, max_val=0.05)
        self.U.data = F.hardtanh(self.U, min_val=0.0140, max_val=0.1240)
        self.detah.data = F.hardtanh(self.detah, min_val=0.100, max_val=0.13)
        self.rateIp0.data = F.hardtanh(self.rateIp0, min_val=1.5e-8, max_val=1e-07)

    def forward(self,num_steps,mymu,mydp,myda,myborn,AR,Import,T):
        Sp0 = torch.tensor([W_pred.item()]) * (1- self.rateIp0)
        Ip0 = torch.tensor([W_pred.item()]) * self.rateIp0
        Sa0 = torch.tensor([A_pred.item()]) * (1- self.rateIp0)
        Ea0 = torch.tensor([0])
        Ia0 = torch.tensor([A_pred.item()]) * self.rateIp0
        init_res = torch.cat((self.res0, self.M0, self.M0, self.M0), dim=0)
        internal_state = torch.cat((self.S0, self.E0, self.I0, self.I0, self.Iin0, self.R0, Sp0, Ip0, Sa0, Ea0, Ia0), dim=0)

        deathh = torch.tensor([0])
        outputs = [internal_state]
        outres = [init_res]

        bh = torch.where((T >= 12.286) & (T <= 32.461), 0.001044 * T * (T - 12.286) * torch.sqrt(32.461 - T),
                         torch.tensor(0.0))
        bv = torch.where((T >= 12.4) & (T < 26.1), -0.9037 + 0.0729 * T, torch.tensor(0.0))
        bv = torch.where((T >= 26.1) & (T <= 32.5), 1.0, bv)
        betah =  bh
        betav =  bv
        detaa = 1 / (4 + np.exp(5.15 - 0.123 * T))

        for step in range(num_steps):
            h = internal_state
            S, E, I, A, Iin, R, Sp, Ip, Sa, Ea, Ia = h
            N = S + E + I + R + Iin + A
            x2 = self.myrate7 * (h[2]+h[4]).unsqueeze(0)
            input = torch.cat((T[step]/35,x2), dim=0)
            bt = self.bate(input) * self.myrate2 + self.myrate
            dSp = myborn[step] * Sa + (1-self.U) * myborn[step] * (Ia + Ea) - mydp[step] * Sp - mymu[step] * Sp
            dIp = self.U * myborn[step] * (Ia + Ea) - mydp[step] * Ip - mymu[step] * Ip
            ef = torch.abs(torch.exp(-0.1 * (1 + (mymu[step] * Sp + mymu[step] * Ip) / (self.SpK * (1 + 0.0239 * AR[step])))))
            dSa = ef * mymu[step] * Sp - bt * betav[step] * Sa * (I + A) / N - bt * betav[step] * Sa * Iin / N - myda[step] * Sa
            dEa = bt * betav[step] * Sa * (I + A) / N + bt * betav[step] * Sa * Iin / N - myda[step] * Ea - detaa[step] * Ea
            dIa = ef * mymu[step] * Ip + detaa[step] * Ea - myda[step] * Ia

            dS = - deathh * S - bt * betah[step] * S * Ia / N
            dE = bt * betah[step] * S * Ia / N - self.detah * E - deathh * E
            dI = 0.3125 * self.detah * E - self.gamma * I - deathh * I
            dA = 0.6875 * self.detah * E - self.gamma * A - deathh * A
            dIin = Import[step] - (self.gammaimp + self.gamma) * Iin - deathh * Iin
            dR = self.gamma * I - deathh * R + (self.gammaimp + self.gamma) * Iin + self.gamma * A

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

            true_cases = 0.3125 * self.detah * E
            internal_state = torch.cat((S_next, E_next, I_next, A_next, Iin_next, R_next, Sp_next, Ip_next, Sa_next, Ea_next, Ia_next),dim=0)
            outputs.append(internal_state)

            outres.append(torch.cat((true_cases, T[step], betav[step], bt), dim=0))

        return outputs, outres

def custom_loss(H_pred_sum, Local_sum, weight_above_100=1.5, weight_below_100=1.0):
    base_loss = (H_pred_sum - Local_sum) ** 2
    weights = torch.where(Local_sum > 100, weight_above_100, weight_below_100)
    weighted_loss = base_loss * weights
    return weighted_loss.mean()
def calculate_r_squared(y_pred, y_true):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    return r_squared
def train_model(model, optimizer,epochs):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        sol_pred, sol_res = model(num_steps,mymu[365:],mydp[365:],myda[365:],myborn[365:],AR[365:],Import[365:],T[365:])
        sol_res = torch.stack(sol_res)
        H_pred = sol_res[1:,0]
        window_size = 7
        H_pred_sum = H_pred.unfold(0, window_size, window_size).sum(dim=1)
        Local_sum = Local[365:365+num_steps].unfold(0, window_size, window_size).sum(dim=1)
        myloss = custom_loss(H_pred_sum, Local_sum)
        myloss.backward(retain_graph=True)
        optimizer.step()
        model.clamp_parameters()
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {myloss.item()}")
            r_squared = calculate_r_squared(H_pred_sum, Local_sum)
            print(r_squared)
        if epoch % 20 == 0:
            sol_pred = torch.stack(sol_pred)
            plt.style.use('ggplot')
            t = range(0, len(H_pred_sum))
            S2 = sol_pred[1:, 0]/10000000
            I2 = sol_pred[1:, 2]

            Sv2 = sol_pred[1:,7]
            Ev2 = sol_pred[1:,8]
            Iv2 = sol_pred[1:,9]

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
            ax2.plot(S2.detach().numpy(), label='Ih')
            ax2.set_title('Sh')
            ax2.set_ylabel('Sh')
            ax2.legend()

            ax3 = fig.add_subplot(gs[1,1])
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

            ax6 = fig.add_subplot(gs[2,1])
            ax6.plot(Iv2.detach(), label='Iv', color='r')
            ax6.set_title('Iv')
            ax6.set_ylabel('Iv')
            ax6.legend()

            M2 = sol_res[1:, 3]
            bh2 = sol_res[1:,1]
            bv2 = sol_res[1:,2]
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

            colors = [
                (0.0, '#FFFFCC'),
                (0.2, '#FFCC99'),
                (0.5, '#FF9966'),
                (0.6, '#FF6666'),
                (0.7, '#CC3366'),
                (0.8, '#993366'),
                (1.0, '#660033')
            ]
            from matplotlib.colors import LinearSegmentedColormap
            cmaps = LinearSegmentedColormap.from_list("custom_diverging", colors)
            temp_range = np.linspace(0 / 35, 35 / 35, 50)
            I_range = np.linspace(0 / 100, 1000 / 100, 50)
            X, Y = np.meshgrid(temp_range, I_range)
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    input_data = torch.tensor([[X[i, j], Y[i, j]]], dtype=torch.float32)
                    Z[i, j] = model.bate(input_data).item()
            fig, ax = plt.subplots(figsize=(10, 4))
            contour = ax.contourf(X, Y, Z, 20, cmap=cmaps, alpha=0.8)
            fig.colorbar(contour, ax=ax, label='Network Output')
            ax.set_xlabel('Temperature (Normalized)')
            ax.set_ylabel('Cases (Normalized)')
            ax.set_title('Model Output with Real Data Boundary (Convex Hull)')
            plt.tight_layout()
            plt.show()

num_steps = 1826-365
model = ODENN()
# optimizer = torch.optim.Adam([
    # {'params': model.bate.parameters(), 'lr': 0.000001},
    # {'params': [model.gamma,model.gammaimp, model.U, model.detah], 'lr': 0.0001},
    # {'params': model.b.parameters(), 'lr': 0.000001},
    # {'params': model.bh.parameters(), 'lr': 0.00001},
    # {'params': model.b.parameters(), 'lr': 0.0001},
#     {'params': model.myrate, 'lr': 0.0001},
#     {'params': model.myrate7, 'lr': 0.0001},
#     {'params': model.myrate2, 'lr': 0.0001},
#     # {'params': model.rateIp0, 'lr': 0.00000000000000000000001},
#     {'params': model.detah, 'lr': 0.0001},
#     {'params': model.U, 'lr': 0.0001},
#     {'params': model.gamma, 'lr': 0.0001},
#     {'params': model.gammaimp, 'lr': 0.0001},
# ], lr=0.000001)
# train_model(model, optimizer, epochs=5000)
# model_path = './weight/bitting.pth'
# torch.save(model.bate.state_dict(), model_path)
sol_pred, sol_res = model(num_steps, mymu[365:], mydp[365:], myda[365:], myborn[365:], AR[365:], Import[365:], T[365:])
sol_res = torch.stack(sol_res)
sol_pred = torch.stack(sol_pred)
M = sol_res[1:, 3]
I1 = sol_pred[1:,2]
I2 = sol_pred[1:,4]
I1 = I1.detach().numpy()
I2 = I2.detach().numpy()
T1 = sol_res[1:, 1]
T1 = T1.detach().numpy()
M = M.detach().numpy()
data = np.column_stack((I1,I2,T1,M))

df = pd.DataFrame(data, columns=['I1','I2','T1','M'])
df.to_csv('./result/biting.csv', index=False)

sol_pred, sol_res = model(num_steps, mymu[365:], mydp[365:], myda[365:], myborn[365:], AR[365:], Import[365:], T[365:])
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
local  = Local[365:365+num_steps]
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
if isinstance(local, torch.Tensor):
    local = local.detach().numpy()

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
    'real':local
}

df = pd.DataFrame(data)

df.to_excel('./result/Model_4_results.xlsx', index=False, engine='openpyxl')



temp_range = np.linspace(0 / 35, 35 / 35, 20)
segment1 = np.linspace(0, 300, 30)
segment2 = np.linspace(300, 1000, 20)
I_range = np.concatenate([segment1, segment2[1:]]) * model.myrate7.item()
X, Y = np.meshgrid(temp_range, I_range)
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        input_data = torch.tensor([[X[i, j], Y[i, j]]], dtype=torch.float32)
        Z[i, j] = (model.bate(input_data)* model.myrate2 + model.myrate).item()
df = pd.DataFrame({
    'T': X.flatten(),
    'I': Y.flatten(),
    'M': Z.flatten()
})
df.to_csv('./result/model_bate.csv', index=False)


