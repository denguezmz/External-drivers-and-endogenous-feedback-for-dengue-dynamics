"""Empirical temperature-dependent dengue transmission model."""

import matplotlib.pyplot as plt
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path

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

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(
    os.environ.get('DENGUE_DATA_DIR', str(SCRIPT_DIR / 'data'))
)
OUTPUT_DIR = Path(
    os.environ.get('DENGUE_OUTPUT_DIR', str(SCRIPT_DIR / 'outputs'))
)


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


# Create the PyTorch model.


born = BORN()
model_path = DATA_DIR / 'yuan_state_dict_born.pth'
model_weight = torch.load(model_path)
born.load_state_dict(model_weight)

num_steps = 1826
TT = torch.linspace(1, num_steps, num_steps).view(-1, 1)  # [100, 1]
TT = TT / num_steps  # Optionally normalize the time series.

import statsmodels.api as sm

file_path = DATA_DIR / 'Local.xlsx'
df = pd.read_excel(file_path)
Local1 = df['All2'].values


Local = torch.tensor(Local1, dtype=torch.float32)  # Original data.
# Local_sum = torch.tensor(rolling_sum_values, dtype=torch.float32)
from scipy.ndimage import gaussian_filter1d

sigma = 2  # Gaussian-kernel standard deviation.
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

file_path = DATA_DIR / 'Input.xlsx'
df = pd.read_excel(file_path)
Import1 = df['All2'].values

Import = torch.tensor(Import1, dtype=torch.float32)  # Original data.
# Import_sum = torch.tensor(rolling_sum_values, dtype=torch.float32)


file_path = DATA_DIR / 'Temp.xlsx'
df = pd.read_excel(file_path)
T = df['All']
file_path = DATA_DIR / 'Rain.xlsx'
df = pd.read_excel(file_path)
R = df['All']
file_path = DATA_DIR / 'Expanded_Half_Month_Averages.xlsx'
df = pd.read_excel(file_path)
AT = df['All'].values
file_path = DATA_DIR / 'Seven_Day_Smoothed_Rain.xlsx'
df = pd.read_excel(file_path)
AR = df['All'].values

import scipy.io
import pandas as pd


import math


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
        # self.SpK = torch.tensor([50000000])

        # self.myrate = torch.nn.Parameter(torch.tensor([1.3522], dtype=torch.float32), requires_grad=True)

    def clamp_parameters(self):

        self.gamma.data = F.hardtanh(self.gamma, min_val=0.0714, max_val=0.3333)
        self.U.data = F.hardtanh(self.U, min_val=0.826, max_val=0.9436)
        # self.detaa.data = F.hardtanh(self.detaa, min_val=0.0833, max_val=0.1250)
        # self.detah.data = F.hardtanh(self.detah, min_val=0.125, max_val=0.2500)
        self.detah.data = F.hardtanh(self.detah, min_val=0.10, max_val=0.25)
        self.out.data = F.hardtanh(self.out, min_val=0.01, max_val=0.03)
        # self.Myk.data = F.hardtanh(self.Myk, min_val=0.00000001, max_val=1.0)


    def forward(self, num_steps, mymu, mydp, myda, myborn, bh, bv, Import, inpu,b):
        Sp0 = torch.pow(torch.tensor([10]), self.Sp0)
        SpK = Sp0 * 5
        self.init_state = torch.cat(
            (self.S0, self.E0, self.I0,self.I0, self.Iin0, self.R0, Sp0*(1-self.rateIp0), Sp0*self.rateIp0, self.Sa0, self.Ea0, self.Ia0), dim=0)
        self.init_res = torch.cat((self.res0,  self.M0, self.M0, self.M0), dim=0)

        # A = torch.tensor([0.000022])
        # deathh = torch.tensor([0.00002])
        # A = torch.tensor([0])
        deathh = torch.tensor([0])
        outputs = [self.init_state]
        outres = [self.init_res]

        # print(myk)
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
    # Add_loss = [(212, 333), (579, 699), (943, 1064), (1308, 1429), (1673, 1794)]
    # indices = torch.cat([torch.arange(start, end) for start, end in Add_loss])
    Add_loss = [(186, 204), (239, 256)]
    indices4 = torch.cat([torch.arange(start, end) for start, end in Add_loss])
    Add_loss2 = [(0, 58), (334, 424), (700, 789), (1065, 1154), (1430, 1519), (1795, 1825)]
    # indices2 = torch.cat([torch.arange(start, end) for start, end in Add_loss2])
    # Add_loss3 = [(943, 1064), (1308, 1429)]
    Add_loss3 = [(134, 152), (186, 204), (239, 256)]
    indices3 = torch.cat([torch.arange(start, end) for start, end in Add_loss3])
    Add_loss4 = [(80, 100),(130,150)]
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

        # Sum complete seven-day windows.
        H_pred_sum = H_pred[365:].unfold(0, window_size, window_size).sum(dim=1)
        Local_sum = Local[365:].unfold(0, window_size, window_size).sum(dim=1)
        # H_pred_sum = H_pred
        # Local_sum = Local
        # mask = Local_sum > 15

        myloss = loss(H_pred_sum, Local_sum)

        # Calculate weekly R-squared over the same period as the training loss.
        with torch.no_grad():
            ss_res = torch.sum((Local_sum - H_pred_sum) ** 2)
            ss_tot = torch.sum((Local_sum - torch.mean(Local_sum)) ** 2)
            train_r2 = 1.0 - ss_res / torch.clamp(ss_tot, min=1e-12)

        # Backward pass and optimization
        myloss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients.

        optimizer.step()
        model.clamp_parameters()


        if epoch % 10 == 0:
            print(
                f"Epoch [{epoch}/{epochs}], "
                f"Loss: {myloss.item():.6f}, "
                f"R2: {train_r2.item():.4f}, "
                f"gamma: {model.gamma.item():.6f}, "
                f"U: {model.U.item():.6f}, "
                f"detah: {model.detah.item():.6f}, "
                f"out: {model.out.item():.6f}"
            )
            # , radio1: {model.radio1.item()}, radio2: {model.radio2.item()},radio1: {model.radio1},detaa: {model.detaa.item()}
        if epoch % 20 == 0:
            sol_pred = torch.stack(sol_pred)
            plt.style.use('ggplot')  # Apply a consistent plotting style.

            t = range(0, len(H_pred_sum))
            E2 = sol_pred[1:, 1]
            I2 = sol_pred[1:, 2]

            Sv2 = sol_pred[1:, 8]
            Ev2 = sol_pred[1:, 9]
            Iv2 = sol_pred[1:, 10]

            fig = plt.figure(figsize=(24, 12))
            gs = fig.add_gridspec(4, 3)  # Create a 4-row, 3-column grid.
            ax1 = fig.add_subplot(gs[0, 0:3])
            ax1.scatter(t, Local_sum.detach().numpy(), color=color3, label='Exact Data Points', s=10, edgecolor='black')
            ax1.plot(t, H_pred_sum.detach().numpy(), color=color1, label='Predicted Curve', linewidth=4,
                     linestyle='-')
            ax1.set_title(f'True vs Predicted (R²={train_r2.item():.4f})')
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
            b2 = sol_res[1:, 3]  # bite rate

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
            # df.to_excel(DATA_DIR / 'AD.xlsx', index=False, header=False)
            # Sp2 = sol_pred[1:,5]
            # Ip2 = sol_pred[1:,6]
            # wenresult2 = Sp2 + Ip2
            # df = pd.DataFrame(wenresult2.detach().numpy(), columns=['W'])
            # df.to_excel(DATA_DIR / 'W.xlsx', index=False, header=False)
            # ax9.scatter(normalize_0_1(wen).detach().numpy(), label='Nv_pred', color='orange')
            ax9.plot(normalize_0_1(wenresult).detach().numpy(), label='Nv_real', color='blue')
            ax9.set_title('Nv')
            ax9.set_ylabel('Nv')
            ax9.legend()
            plt.tight_layout()
            # fig_path = OUTPUT_DIR / 'beha0119.png'
            # plt.savefig(fig_path, dpi=600, bbox_inches='tight')  # Avoid clipping.
            plt.show()

num_steps = 1826
Tinput = (torch.linspace(0, num_steps, num_steps).view(-1, 1)) / num_steps  # [100, 1]
model = ODENN()

optimizer = torch.optim.Adam([
    # {'params': model.myrate, 'lr': 0.0001},
    # {'params': model.U, 'lr': 0.001},
    {'params': [model.gamma, model.out, model.detah, model.U], 'lr': 0.0001}
], lr=0.000001)
train_model(model, optimizer, epochs=200)
#



#
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

# ============================================================================
# Goodness-of-fit calculations and outputs
# ============================================================================
def calculate_goodness_of_fit(observed, predicted):
    """Calculate common goodness-of-fit metrics."""
    observed = np.asarray(observed, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)

    if observed.shape != predicted.shape:
        raise ValueError(
            "Observed and predicted series have different lengths: "
            f"{len(observed)} vs {len(predicted)}"
        )
    if len(observed) == 0:
        raise ValueError("No data are available for goodness-of-fit metrics.")
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("Observed or predicted values contain NaN or infinity.")

    residual = observed - predicted
    mse = float(np.mean(residual ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residual)))
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((observed - observed.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    if np.std(observed) > 0 and np.std(predicted) > 0:
        pearson_r = float(np.corrcoef(observed, predicted)[0, 1])
        regression_slope, regression_intercept = np.polyfit(
            observed, predicted, deg=1
        )
        regression_slope = float(regression_slope)
        regression_intercept = float(regression_intercept)
    else:
        pearson_r = np.nan
        regression_slope = np.nan
        regression_intercept = np.nan

    return {
        'n': int(len(observed)),
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson_r': pearson_r,
        'regression_slope': regression_slope,
        'regression_intercept': regression_intercept,
        'observed_total': float(observed.sum()),
        'predicted_total': float(predicted.sum()),
        'observed_mean': float(observed.mean()),
        'predicted_mean': float(predicted.mean()),
        'observed_peak': float(observed.max()),
        'predicted_peak': float(predicted.max()),
        'mean_residual_observed_minus_predicted': float(residual.mean()),
    }


# Convert observed cases to NumPy if needed.
if isinstance(Local, torch.Tensor):
    local_np = Local.detach().cpu().numpy().reshape(-1)
else:
    local_np = np.asarray(Local).reshape(-1)

# Create the daily results table.
n_days = len(new)
dates = pd.date_range('2015-01-01', periods=n_days, freq='D')
data = {
    'date': dates,
    'is_fit_period': np.arange(n_days) >= 365,
    'S': S,
    'E': E,
    'I': I,
    'A': A,
    'Iin': Iin,
    'R': R,
    'Sp': Sp,
    'Ip': Ip,
    'Sa': Sa,
    'Ea': Ea,
    'Ia': Ia,
    'predicted_daily_cases': new,
    'betah': betah,
    'betav': betav,
    'bite': bite,
    'observed_daily_cases': local_np,
}
daily_df = pd.DataFrame(data)
daily_df['residual_observed_minus_predicted'] = (
    daily_df['observed_daily_cases'] - daily_df['predicted_daily_cases']
)

# Match training exactly: exclude the first 365 days and retain complete weeks.
# With 1,461 remaining days, the final five days are excluded from weekly metrics.
fit_observed_daily = local_np[365:]
fit_predicted_daily = np.asarray(new, dtype=np.float64).reshape(-1)[365:]
n_complete_weeks = min(len(fit_observed_daily), len(fit_predicted_daily)) // 7
n_used_days = n_complete_weeks * 7

observed_weekly = fit_observed_daily[:n_used_days].reshape(-1, 7).sum(axis=1)
predicted_weekly = fit_predicted_daily[:n_used_days].reshape(-1, 7).sum(axis=1)
week_start = pd.date_range(
    dates[365], periods=n_complete_weeks, freq='7D'
)

weekly_df = pd.DataFrame({
    'week_index': np.arange(n_complete_weeks),
    'week_start': week_start,
    'observed_weekly_cases': observed_weekly,
    'predicted_weekly_cases': predicted_weekly,
})
weekly_df['residual_observed_minus_predicted'] = (
    weekly_df['observed_weekly_cases'] - weekly_df['predicted_weekly_cases']
)
weekly_df['absolute_error'] = np.abs(
    weekly_df['residual_observed_minus_predicted']
)
weekly_df['squared_error'] = (
    weekly_df['residual_observed_minus_predicted'] ** 2
)

# Weekly metrics are primary because the loss is weekly; daily metrics are secondary.
weekly_metrics = calculate_goodness_of_fit(observed_weekly, predicted_weekly)
daily_metrics = calculate_goodness_of_fit(
    fit_observed_daily[:n_used_days], fit_predicted_daily[:n_used_days]
)
metrics_df = pd.DataFrame([
    {'scale': 'weekly_fit_period_primary', **weekly_metrics},
    {'scale': 'daily_fit_period_supplementary', **daily_metrics},
])

# Export final parameter values.
parameters_df = pd.DataFrame({
    'parameter': [
        'Sp0_log10', 'Sp0', 'rateIp0', 'gamma', 'out', 'U', 'detah'
    ],
    'value': [
        float(model.Sp0.detach().cpu()),
        float(torch.pow(torch.tensor(10.0), model.Sp0.detach().cpu())),
        float(model.rateIp0.detach().cpu()),
        float(model.gamma.detach().cpu()),
        float(model.out.detach().cpu()),
        float(model.U.detach().cpu()),
        float(model.detah.detach().cpu()),
    ],
    'optimized_in_current_code': [
        False, False, False, True, True, True, True
    ],
})

print('\n===== Empirical temperature model: goodness of fit =====', flush=True)
print(f"Complete weeks N : {weekly_metrics['n']}")
print(f"R²              : {weekly_metrics['r2']:.4f}")
print(f"MSE             : {weekly_metrics['mse']:.4f}")
print(f"RMSE            : {weekly_metrics['rmse']:.4f}")
print(f"MAE             : {weekly_metrics['mae']:.4f}")
print(f"Pearson r       : {weekly_metrics['pearson_r']:.4f}")
print(f"Observed total   : {weekly_metrics['observed_total']:.2f}")
print(f"Predicted total  : {weekly_metrics['predicted_total']:.2f}")
print(f"Observed peak    : {weekly_metrics['observed_peak']:.2f}")
print(f"Predicted peak   : {weekly_metrics['predicted_peak']:.2f}")
print("Metrics use complete seven-day weeks after the warm-up period.")

# Save the workbook and figures.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_PATH = os.path.join(
    OUTPUT_DIR, 'temperature_empirical_model_results_with_fit_metrics.xlsx'
)
FIT_FIGURE_PATH = os.path.join(
    OUTPUT_DIR, 'temperature_empirical_model_weekly_fit.png'
)
REGRESSION_FIGURE_PATH = os.path.join(
    OUTPUT_DIR, 'temperature_empirical_model_observed_vs_predicted.png'
)

with pd.ExcelWriter(RESULT_PATH, engine='openpyxl') as writer:
    daily_df.to_excel(writer, sheet_name='DailyResults', index=False)
    weekly_df.to_excel(writer, sheet_name='WeeklyFit', index=False)
    metrics_df.to_excel(writer, sheet_name='FitMetrics', index=False)
    parameters_df.to_excel(writer, sheet_name='Parameters', index=False)

# Weekly case time-series fit
fig, ax = plt.subplots(figsize=(12, 5))
ax.scatter(
    weekly_df['week_start'], weekly_df['observed_weekly_cases'],
    s=18, label='Observed weekly cases'
)
ax.plot(
    weekly_df['week_start'], weekly_df['predicted_weekly_cases'],
    linewidth=2.0, label='Predicted weekly cases'
)
ax.set_title(
    f"Empirical temperature-dependent model | "
    f"R²={weekly_metrics['r2']:.3f}, "
    f"RMSE={weekly_metrics['rmse']:.2f}, "
    f"MAE={weekly_metrics['mae']:.2f}"
)
ax.set_xlabel('Week')
ax.set_ylabel('Local cases per week')
ax.legend(frameon=False)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIT_FIGURE_PATH, dpi=300, bbox_inches='tight')
plt.close(fig)

# Observed-versus-predicted scatter plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(observed_weekly, predicted_weekly, s=24, alpha=0.75)
xy_min = float(min(observed_weekly.min(), predicted_weekly.min()))
xy_max = float(max(observed_weekly.max(), predicted_weekly.max()))
ax.plot([xy_min, xy_max], [xy_min, xy_max], linestyle='--', label='Identity line')
if np.isfinite(weekly_metrics['regression_slope']):
    x_line = np.linspace(xy_min, xy_max, 200)
    y_line = (
        weekly_metrics['regression_slope'] * x_line
        + weekly_metrics['regression_intercept']
    )
    ax.plot(x_line, y_line, linewidth=1.8, label='Regression fit')
ax.set_title(
    f"Observed vs predicted weekly cases\n"
    f"R²={weekly_metrics['r2']:.3f}, r={weekly_metrics['pearson_r']:.3f}"
)
ax.set_xlabel('Observed weekly cases')
ax.set_ylabel('Predicted weekly cases')
ax.legend(frameon=False)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(REGRESSION_FIGURE_PATH, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"\nResults workbook saved to: {RESULT_PATH}")
print(f"Weekly case-fit plot saved to: {FIT_FIGURE_PATH}")
print(f"Observed-versus-predicted plot saved to: {REGRESSION_FIGURE_PATH}")
