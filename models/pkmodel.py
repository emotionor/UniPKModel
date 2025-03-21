import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchdiffeq import odeint

class UniPKModel(nn.Module):
    def __init__(self, num_cmpts=3, curve_weight=1.0, compartment_penalty_wight=0.1, weighting="abs_log_error", route="i.v.", method='linear', **kwargs):
        super(UniPKModel, self).__init__()
        self.num_cmpts = num_cmpts
        self.curve_weight = curve_weight
        self.compartment_penalty_wight = compartment_penalty_wight
        self.weighting = weighting
        self.route = route
        self.method = method # 'linear' or 'mcmodel' or 'mmmodel' or 'induction' or 'inhibition'
        if self.method == 'mcmodel':
            self.cmptmodel = MultiCompartmentModel(self.num_cmpts, route=self.route)
        elif self.method == 'mmmodel':
            self.cmptmodel = MichaelisMentenModel(self.num_cmpts, route=self.route)
        elif self.method == 'induction':
            self.cmptmodel = AutoInductionModel(self.num_cmpts, route=self.route)
        elif self.method == 'induction2':
            self.cmptmodel = AutoInductionModel2(self.num_cmpts, route=self.route)
        elif self.method == 'inhibition':
            self.cmptmodel = AutoInhibitionModel(self.num_cmpts, route=self.route)
        elif self.method == 'mixmodel1':
            self.cmptmodel = MixPKModel1(self.num_cmpts, route=self.route)
        elif self.method == 'mixmodel2':
            self.cmptmodel = MixPKModel2(self.num_cmpts, route=self.route)
        elif self.method == 'mixmodel3':
            self.cmptmodel = MixPKModel3(self.num_cmpts, route=self.route)
        elif self.method == 'NeuralODE':
            node_mid_dim = kwargs.get('node_mid_dim', 64)
            vd_mid_dim = kwargs.get('vd_mid_dim', 32)
            self.cmptmodel = NeuralODE(self.num_cmpts, route=self.route, input_dim=512, middle_dim=node_mid_dim)
            self.volumeD = VolumeD(input_dim=512, middle_dim=vd_mid_dim)
        elif self.method == 'NeuralODE2':
            self.cmptmodel = NeuralODE2(self.num_cmpts, route=self.route, input_dim=512)
            self.volumeD = VolumeD(input_dim=512)
        else:
            raise ValueError(f"method {self.method} not supported")
           
    def forward(self, params, route, doses, meas_times):
        if len(params.shape)==1:
            params = params.unsqueeze(0)  # for single sample
        
        if self.method in ['NeuralODE', 'NeuralODE2']:
            V0 = self.volumeD(params)
        else:
            V0 = params[:,1]
        C1 = doses / V0  # Dose / Vc as initial condition
        
        batch_size = C1.shape[0]

        if self.route == 'i.v.':
            C = torch.zeros((self.num_cmpts, batch_size), dtype=C1.dtype, device=C1.device)
            C[0] = C1
        elif self.route == 'p.o.':
            C = torch.zeros((self.num_cmpts + 1, batch_size), dtype=C1.dtype, device=C1.device)
            #C[-1] = C1
            C[0][route == 1] = C1[route == 1]
            C[-1][route == 0] = C1[route == 0]

        init_conditions = C
        if self.method in ['NeuralODE', 'NeuralODE2']:
            solution  = odeint(lambda t, y: self.cmptmodel(t, y, params, V0), init_conditions, meas_times, options={"min_step": 0.01},rtol=1e-3,atol=1e-4)
        else:
            solution  = odeint(lambda t, y: self.cmptmodel(t, y, params), init_conditions, meas_times, options={"min_step": 0.01},rtol=1e-3,atol=1e-4)
        return solution

def get_model_params(method, route, num_cmpts):
    if method in ['mcmodel']:
        num_classes = 2 * num_cmpts
    elif method in ['mmmodel', 'induction', 'inhibition']:
        num_classes = 2 * num_cmpts + 1
    elif method in ['induction2']:
        num_classes = 2 * num_cmpts + 2
    elif method in ['mixmodel1']:
        num_classes = 2 * num_cmpts + 3
    elif method in ['mixmodel2', 'mixmodel3']:
        num_classes = 2 * num_cmpts + 4
    elif method in ['NeuralODE', 'NeuralODE2']:
        num_classes = 2 * num_cmpts
    else:
        raise ValueError(f"method {method} not supported")
    if route == 'p.o.':
        num_classes += 1
    return num_classes, method in ['NeuralODE', 'NeuralODE2']

class BaseCompartmentModel(nn.Module):
    def __init__(self, num_compartments, route='i.v.'):
        super(BaseCompartmentModel, self).__init__()
        self.num_compartments = num_compartments
        self.route = route

    def forward(self, t, y, params, V0=None):
        if self.route == 'i.v.':
            return self.forward_iv(t, y, params, V0=V0)
        elif self.route == 'p.o.':
            return self.forward_po(t, y, params, V0=V0)
        else:
            raise ValueError(f"Unsupported route: {self.route}")

    def forward_iv(self, t, y, params, V0=None):
        raise NotImplementedError

    def forward_po(self, t, y, params, V0=None):
        raise NotImplementedError

class VolumeD(nn.Module):
    def __init__(self, input_dim, middle_dim=32):
        super(VolumeD, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, middle_dim),
            nn.ReLU(),
            nn.Linear(middle_dim, 1),
            nn.Softplus(),
        )
    
    def forward(self, x):
        return self.net(x).squeeze(1)


class NeuralODE(BaseCompartmentModel):
    def __init__(self, num_compartments, route='i.v.', input_dim=512, middle_dim=64):
        super(NeuralODE, self).__init__(num_compartments=num_compartments, route=route)
        self.input_dim = input_dim
        output_dim = num_compartments * 2 - 1
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, middle_dim),
            nn.ReLU(),
            nn.Linear(middle_dim, output_dim),
            nn.Softplus(),
        )
        if route == 'p.o.':
            self.poka = nn.Sequential(
                nn.Linear(input_dim + 1, middle_dim),
                nn.ReLU(),
                nn.Linear(middle_dim, 1),
                nn.Softplus(),
            )

    def forward_iv(self, t, y, params, V0):
        if len(params.shape) == 1:
            params = params.unsqueeze(0)

        net_output = self.net(torch.cat([params, y[0].unsqueeze(1)], dim=-1))
        Cl = net_output[:, 0]

        C = y[:self.num_compartments]
        dC_dt = torch.zeros_like(C)
        dC_dt[0] = - Cl / V0 * C[0]

        if self.num_compartments > 1:
            rate_constants = net_output[:, 1:].view(-1, 2, self.num_compartments - 1)
            for i in range(1, self.num_compartments):
                dC_dt[0] +=  - rate_constants[:, 0, i-1] * C[0] + rate_constants[:, 1, i-1] * C[i]
                dC_dt[i] = rate_constants[:, 0, i-1] * C[0] - rate_constants[:, 1, i-1] * C[i]

        return dC_dt
    
    def forward_po(self, t, y, params, V0):
        if len(params.shape) == 1:
            params = params.unsqueeze(0)

        net_output = self.net(torch.cat([params, y[0].unsqueeze(1)], dim=-1))
        Cl = net_output[:, 0]

        ka = self.poka(torch.cat([params, y[0].unsqueeze(1)], dim=-1)).squeeze(1)

        C = y[:self.num_compartments + 1]
        dC_dt = torch.zeros_like(C)
        dC_dt[0] = - Cl / V0 * C[0] + ka * C[-1]
        dC_dt[-1] = - ka * C[-1]

        if self.num_compartments > 1:
            rate_constants = net_output[:, 1:].view(-1, 2, self.num_compartments - 1)
            for i in range(1, self.num_compartments):
                dC_dt[0] +=  - rate_constants[:, 0, i-1] * C[0] + rate_constants[:, 1, i-1] * C[i]
                dC_dt[i] = rate_constants[:, 0, i-1] * C[0] - rate_constants[:, 1, i-1] * C[i]

        return dC_dt

    
class NeuralODE2(BaseCompartmentModel):
    def __init__(self, num_compartments, route='i.v.', input_dim=512):
        super(NeuralODE2, self).__init__(num_compartments=num_compartments, route=route)
        self.input_dim = input_dim
        output_dim = num_compartments * 2 - 1
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softplus(),
        )

    def forward_iv(self, t, y, params, V0):
        if len(params.shape) == 1:
            params = params.unsqueeze(0)

        net_output = self.net(torch.cat([params, y[0].unsqueeze(1)], dim=-1))
        Cl = net_output[:, 0]
        
        C = y[:self.num_compartments]
        dC_dt = torch.zeros_like(C)
        dC_dt[0] = - Cl / V0

        if self.num_compartments > 1:
            rate_constants = net_output[:, 1:].view(-1, 2, self.num_compartments - 1)
            for i in range(1, self.num_compartments):
                dC_dt[0] +=  - rate_constants[:, 0, i-1] * C[0] + rate_constants[:, 1, i-1] * C[i]
                dC_dt[i] = rate_constants[:, 0, i-1] * C[0] - rate_constants[:, 1, i-1] * C[i]

        return dC_dt

class MultiCompartmentModel(BaseCompartmentModel):
    def forward_iv(self, t, y, params, **kwargs):
        if len(params.shape) == 1:
            params = params.unsqueeze(0)  # for single sample
        
        Cl, V1 = params[:,0], params[:,1]
        rate_constants = params[:, 2:].view(-1, 2, self.num_compartments - 1)
        
        C = y[:self.num_compartments]
        dC_dt = torch.zeros_like(C)
        dC_dt[0] = - Cl / V1 * C[0]

        if self.num_compartments > 1:
            for i in range(1, self.num_compartments):
                dC_dt[0] +=  - rate_constants[:, 0, i-1] * C[0] + rate_constants[:, 1, i-1] * C[i]
                dC_dt[i] = rate_constants[:, 0, i-1] * C[0] - rate_constants[:, 1, i-1] * C[i]

        return dC_dt

    def forward_po(self, t, y, params, **kwargs):
        if len(params.shape) == 1:
            params = params.unsqueeze(0)  # for single sample

        Cl, V1, ka = params[:,0], params[:,1], params[:,2]
        rate_constants = params[:, 3:].view(-1, 2, self.num_compartments - 1)

        C = y[:self.num_compartments + 1]

        dC_dt = torch.zeros_like(C)
        dC_dt[0] = - Cl / V1 * C[0] + ka * C[-1]
        dC_dt[-1] = - ka * C[-1]

        if self.num_compartments > 1:
            for i in range(1, self.num_compartments):
                dC_dt[0] +=  - rate_constants[:, 0, i-1] * C[0] + rate_constants[:, 1, i-1] * C[i]
                dC_dt[i] = rate_constants[:, 0, i-1] * C[0] - rate_constants[:, 1, i-1] * C[i]

        return dC_dt
    
class MichaelisMentenModel(BaseCompartmentModel): 
    def forward_iv(self, t, y, params, **kwargs):
        if len(params.shape) == 1:
            params = params.unsqueeze(0)  # for single sample
            
        V_m, V, K_m = params[:,0], params[:,1], params[:,2]
        rate_constants = params[:, 3:].view(-1, 2, self.num_compartments - 1)

        C = y[:self.num_compartments]
        dC_dt = torch.zeros_like(C)
        dC_dt[0] = - (V_m / V) * C[0] / (K_m + C[0])

        if self.num_compartments > 1:
            for i in range(1, self.num_compartments):
                dC_dt[0] +=  - rate_constants[:, 0, i-1] * C[0] + rate_constants[:, 1, i-1] * C[i]
                dC_dt[i] = rate_constants[:, 0, i-1] * C[0] - rate_constants[:, 1, i-1] * C[i]

        return dC_dt
    
    def forward_po(self, t, y, params, **kwargs):
        if len(params.shape) == 1:
            params = params.unsqueeze(0)  # for single sample

        V_m, V, K_m, ka = params[:,0], params[:,1], params[:,2], params[:,3]
        rate_constants = params[:, 4:].view(-1, 2, self.num_compartments - 1)
        
        C = y[:self.num_compartments + 1]
        dC_dt = torch.zeros_like(C)
        dC_dt[0] = - (V_m / V) * C[0] / (K_m + C[0]) + ka * C[-1]
        dC_dt[-1] = - ka * C[-1]

        if self.num_compartments > 1:
            for i in range(1, self.num_compartments):
                dC_dt[0] +=  - rate_constants[:, 0, i-1] * C[0] + rate_constants[:, 1, i-1] * C[i]
                dC_dt[i] = rate_constants[:, 0, i-1] * C[0] - rate_constants[:, 1, i-1] * C[i]

        return dC_dt
    
class AutoInductionModel(BaseCompartmentModel):       
    def forward_iv(self, t, y, params, **kwargs):
        if len(params.shape) == 1:
            params = params.unsqueeze(0) # for single sample

        Cl, V1, kint = params[:,0], params[:,1], params[:,2]
        rate_constants = params[:, 3:].view(-1, 2, self.num_compartments - 1)
        
        C = y[:self.num_compartments]
        dC_dt = torch.zeros_like(C)
        dC_dt[0] = - (Cl + kint * C[0]) / V1 * C[0]

        if self.num_compartments > 1:
            for i in range(1, self.num_compartments):
                dC_dt[0] +=  - rate_constants[:, 0, i-1] * C[0] + rate_constants[:, 1, i-1] * C[i]
                dC_dt[i] = rate_constants[:, 0, i-1] * C[0] - rate_constants[:, 1, i-1] * C[i]

        return dC_dt
    
    def forward_po(self, t, y, params, **kwargs):
        if len(params.shape) == 1:
            params = params.unsqueeze(0) # for single sample

        Cl, V1, kint, ka = params[:,0], params[:,1], params[:,2], params[:,3]
        rate_constants = params[:, 4:].view(-1, 2, self.num_compartments - 1)

        C = y[:self.num_compartments + 1]
        dC_dt = torch.zeros_like(C)
        dC_dt[0] = - (Cl + kint * C[0]) / V1 * C[0] + ka * C[-1]
        dC_dt[-1] = - ka * C[-1]

        if self.num_compartments > 1:
            for i in range(1, self.num_compartments):
                dC_dt[0] +=  - rate_constants[:, 0, i-1] * C[0] + rate_constants[:, 1, i-1] * C[i]
                dC_dt[i] = rate_constants[:, 0, i-1] * C[0] - rate_constants[:, 1, i-1] * C[i]

        return dC_dt

class AutoInductionModel2(BaseCompartmentModel):       
    def forward_iv(self, t, y, params, **kwargs):
        if len(params.shape) == 1:
            params = params.unsqueeze(0) # for single sample

        Cl, V1, k1, k2 = params[:,0], params[:,1], params[:,2], params[:,3]
        rate_constants = params[:, 4:].view(-1, 2, self.num_compartments - 1)
        
        C = y[:self.num_compartments]
        dC_dt = torch.zeros_like(C)
        dC_dt[0] = - (Cl + k1 * C[0] / (k2 + C[0])) / V1 * C[0]

        if self.num_compartments > 1:
            for i in range(1, self.num_compartments):
                dC_dt[0] +=  - rate_constants[:, 0, i-1] * C[0] + rate_constants[:, 1, i-1] * C[i]
                dC_dt[i] = rate_constants[:, 0, i-1] * C[0] - rate_constants[:, 1, i-1] * C[i]

        return dC_dt

class AutoInhibitionModel(BaseCompartmentModel):
    def forward_iv(self, t, y, params, **kwargs):
        if len(params.shape) == 1:
            params = params.unsqueeze(0)

        Cl, V1, kinh = params[:,0], params[:,1], params[:,2]
        rate_constants = params[:, 3:].view(-1, 2, self.num_compartments - 1)

        C = y[:self.num_compartments]
        dC_dt = torch.zeros_like(C)
        dC_dt[0] = - (Cl / (1 + kinh * C[0])) / V1 * C[0]

        if self.num_compartments > 1:
            for i in range(1, self.num_compartments):
                dC_dt[0] +=  - rate_constants[:, 0, i-1] * C[0] + rate_constants[:, 1, i-1] * C[i]
                dC_dt[i] = rate_constants[:, 0, i-1] * C[0] - rate_constants[:, 1, i-1] * C[i]

        return dC_dt
    
    def forward_po(self, t, y, params, **kwargs):
        if len(params.shape) == 1:
            params = params.unsqueeze(0)

        Cl, V1, kinh, ka = params[:,0], params[:,1], params[:,2], params[:,3]
        rate_constants = params[:, 4:].view(-1, 2, self.num_compartments - 1)

        C = y[:self.num_compartments + 1]
        dC_dt = torch.zeros_like(C)
        dC_dt[0] = - (Cl / (1 + kinh * C[0])) / V1 * C[0] + ka * C[-1]
        dC_dt[-1] = - ka * C[-1]

        if self.num_compartments > 1:
            for i in range(1, self.num_compartments):
                dC_dt[0] +=  - rate_constants[:, 0, i-1] * C[0] + rate_constants[:, 1, i-1] * C[i]
                dC_dt[i] = rate_constants[:, 0, i-1] * C[0] - rate_constants[:, 1, i-1] * C[i]

        return dC_dt
    
class MixPKModel1(BaseCompartmentModel):
    def forward_iv(self, t, y, params, **kwargs):
        if len(params.shape) == 1:
            params = params.unsqueeze(0)

        Cl, V1, k1, k2, k3 = params[:,0], params[:,1], params[:,2], params[:,3], params[:,4]
        rate_constants = params[:, 5:].view(-1, 2, self.num_compartments - 1)

        C = y[:self.num_compartments]
        dC_dt = torch.zeros_like(C)
        dC_dt[0] = - (Cl + k1 * C[0] + k2 / (k3 + C[0])) / V1 * C[0]

        if self.num_compartments > 1:
            for i in range(1, self.num_compartments):
                dC_dt[0] +=  - rate_constants[:, 0, i-1] * C[0] + rate_constants[:, 1, i-1] * C[i]
                dC_dt[i] = rate_constants[:, 0, i-1] * C[0] - rate_constants[:, 1, i-1] * C[i]

        return dC_dt
    
    def forward_po(self, t, y, params, **kwargs):
        if len(params.shape) == 1:
            params = params.unsqueeze(0)

        Cl, V1, k1, k2, k3, ka = params[:,0], params[:,1], params[:,2], params[:,3], params[:,4], params[:,5]
        rate_constants = params[:, 6:].view(-1, 2, self.num_compartments - 1)

        C = y[:self.num_compartments + 1]
        dC_dt = torch.zeros_like(C)
        dC_dt[0] = - (Cl + k1 * C[0] + k2 / (k3 + C[0])) / V1 * C[0] + ka * C[-1]
        dC_dt[-1] = - ka * C[-1]

        if self.num_compartments > 1:
            for i in range(1, self.num_compartments):
                dC_dt[0] +=  - rate_constants[:, 0, i-1] * C[0] + rate_constants[:, 1, i-1] * C[i]
                dC_dt[i] = rate_constants[:, 0, i-1] * C[0] - rate_constants[:, 1, i-1] * C[i]

        return dC_dt
    
class MixPKModel2(BaseCompartmentModel):
    def forward_iv(self, t, y, params, **kwargs):
        if len(params.shape) == 1:
            params = params.unsqueeze(0)

        Cl, V1, k1, k2, k3, k4 = params[:,0], params[:,1], params[:,2], params[:,3], params[:,4], params[:,5]
        rate_constants = params[:, 6:].view(-1, 2, self.num_compartments - 1)

        C = y[:self.num_compartments]
        dC_dt = torch.zeros_like(C)
        dC_dt[0] = - (Cl + k1 * C[0] / (k2 + C[0]) + k3 / (k4 + C[0])) / V1 * C[0]

        if self.num_compartments > 1:
            for i in range(1, self.num_compartments):
                dC_dt[0] +=  - rate_constants[:, 0, i-1] * C[0] + rate_constants[:, 1, i-1] * C[i]
                dC_dt[i] = rate_constants[:, 0, i-1] * C[0] - rate_constants[:, 1, i-1] * C[i]

        return dC_dt
    
class MixPKModel3(BaseCompartmentModel):
    def forward_iv(self, t, y, params, **kwargs):
        if len(params.shape) == 1:
            params = params.unsqueeze(0)

        Cl, V1, k1, k2, k3, k4 = params[:,0], params[:,1], params[:,2], params[:,3], params[:,4], params[:,5]
        rate_constants = params[:, 6:].view(-1, 2, self.num_compartments - 1)

        C = y[:self.num_compartments]
        dC_dt = torch.zeros_like(C)
        dC_dt[0] = - (Cl + k1 * C[0] / (k2 + C[0]) - k3 / (k4 + C[0])) / V1 * C[0]

        if self.num_compartments > 1:
            for i in range(1, self.num_compartments):
                dC_dt[0] +=  - rate_constants[:, 0, i-1] * C[0] + rate_constants[:, 1, i-1] * C[i]
                dC_dt[i] = rate_constants[:, 0, i-1] * C[0] - rate_constants[:, 1, i-1] * C[i]

        return dC_dt