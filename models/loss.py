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
        self.device = kwargs.get('device', 'cpu')
        self.method = method # 'linear' or 'mcmodel' or 'mmmodel' or 'induction' or 'inhibition'
        if self.method == 'mcmodel':
            self.routel = MultiCompartmentModel(self.num_cmpts, route=self.route)
        elif self.method == 'mmmodel':
            self.routel = MichaelisMentenModel(self.num_cmpts, route=self.route)
        elif self.method == 'induction':
            self.routel = AutoInductionModel(self.num_cmpts, route=self.route)
        elif self.method == 'induction2':
            self.routel = AutoInductionModel2(self.num_cmpts, route=self.route)
        elif self.method == 'inhibition':
            self.routel = AutoInhibitionModel(self.num_cmpts, route=self.route)
        elif self.method == 'mixmodel1':
            self.routel = MixPKModel1(self.num_cmpts, route=self.route)
        elif self.method == 'mixmodel2':
            self.routel = MixPKModel2(self.num_cmpts, route=self.route)
        elif self.method == 'mixmodel3':
            self.routel = MixPKModel3(self.num_cmpts, route=self.route)
        else:
            raise ValueError(f"method {self.method} not supported")
           
    def forward(self, params, route, doses, meas_times):
        if len(params.shape)==1:
            params = params.unsqueeze(0)  # for single sample
        C1 = doses / params[:,1]  # Dose / Vc as initial condition
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
        solution  = odeint(lambda t, y: self.routel(t, y, params), init_conditions, meas_times, options={"min_step": 0.01},rtol=1e-3,atol=1e-4)
        return solution


    # def get_QDE_solution(self, params, dose=None, times=None, route=None, **kwargs):
    #     if len(params.shape)==2:
    #         C1 = dose / params[:,1]
    #         batch_size = C1.shape[0]
    #     else:
    #         C1 = dose / params[1]
    #         batch_size = 1
    #     if len(times.shape) == 1:
    #         time = times
    #     else:
    #         time = times[0] # assume all the same time points
    #     if self.route == 'i.v.':
    #         C = torch.zeros((self.num_cmpts, batch_size), dtype=C1.dtype, device=C1.device)
    #         C[0] = C1
    #     elif self.route == 'p.o.':
    #         C = torch.zeros((self.num_cmpts + 1, batch_size), dtype=C1.dtype, device=C1.device)
    #         #C[-1] = C1
    #         C[0][route == 1] = C1[route == 1]
    #         C[-1][route == 0] = C1[route == 0]

    #     init_conditions = C
    #     with torch.no_grad():
    #         solution  = odeint(lambda t, y: self.routel(t, y, params), init_conditions, time, options={"min_step": 0.01}, rtol=1e-3,atol=1e-4)
    #         if len(solution.shape) == 3:
    #             y_pred = solution[:,0].transpose(0, 1)
    #         else:
    #             y_pred = solution[:,0]
    #         return time.detach().cpu().numpy(), y_pred.detach().cpu().numpy()

class BaseCompartmentModel(nn.Module):
    def __init__(self, num_compartments, route='i.v.'):
        super(BaseCompartmentModel, self).__init__()
        self.num_compartments = num_compartments
        self.route = route

    def forward(self, t, y, params):
        if self.route == 'i.v.':
            return self.forward_iv(t, y, params)
        elif self.route == 'p.o.':
            return self.forward_po(t, y, params)
        else:
            raise ValueError(f"Unsupported route: {self.route}")

    def forward_iv(self, t, y, params):
        raise NotImplementedError

    def forward_po(self, t, y, params):
        raise NotImplementedError
    
class MultiCompartmentModel(BaseCompartmentModel):
    def forward_iv(self, t, y, params):
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

    def forward_po(self, t, y, params):
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
    def forward_iv(self, t, y, params):
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
    
    def forward_po(self, t, y, params):
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
    def forward_iv(self, t, y, params):
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
    
    def forward_po(self, t, y, params):
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
    def forward_iv(self, t, y, params):
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
    def forward_iv(self, t, y, params):
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
    
    def forward_po(self, t, y, params):
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
    def forward_iv(self, t, y, params):
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
    
    def forward_po(self, t, y, params):
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
    def forward_iv(self, t, y, params):
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
    def forward_iv(self, t, y, params):
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