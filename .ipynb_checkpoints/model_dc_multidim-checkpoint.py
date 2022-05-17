# import packages 
import numpy as np
import tools
from types import SimpleNamespace
import egm_dc_multidim as egm
from numba import njit, int64, double, boolean, int32,void
from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import Process
from scipy import optimize

class model_dc_multidim():

    def __init__(self,name=None):
        """ defines default attributes """

        # a. name
        self.par = SimpleNamespace()
        self.sol = SimpleNamespace() 
        self.sim = SimpleNamespace() 

    def setup(self):

        par = self.par
        
        #par.Tmax = 50
        #par.Tmin = 25
        #par.T = par.Tmax -  par.Tmin 
        par.T = 90
        par.Tmin = 20
        
        # Model parameters
        par.beta = 0.96963
        
        par.interest = 1.05
        
        par.credit_constraint = 20
        
        # Grids and numerical integration
        par.m_max = 1000
        par.m_phi = 1.1 # Curvature parameters
        par.a_max = 1000
        par.a_phi = 1.1  # Curvature parameters
        par.h_max = 1.0
        par.p_phi = 1.0 # Curvature parameters

        par.Nxi = 8
        par.Nm = 200
        par.Na = 200
        par.Ne = 100


        par.Nm_b = 50
        par.Rh = 1
        
        # preference
        par.zeta = 0.79488 # CRRA coefficient in consumption
        par.zeta_beq = 0.48834 # CRRA coefficient in utility of bequest
        
        par.gamma = {0: 0, 1000: 1.4139, 2000: 2.0088, 2250: 2.9213, 2500: 2.8639, 3000: 3.8775}
        par.kappa_1 = 0.50321
        par.kappa_2 = 0.00008
        par.kappa_3 =  0.05083
        par.b_scale = 0.68659
        
        par.lambda_scale = 0.29950 # taste shock scale
        par.betas = {"dr": 0.96963, "hs":  0.96732, "cg": 0.96806}
        
        # human capital function parameters
        par.eta_0 = {"low": 0, "high": 0.39311, "dr": 2.456471, "hs":  2.56761, "cg": 2.78766}
        par.eta_1 = {"dr": 0.01974, "hs":  0.02164, "cg": 0.03041}
        par.eta_2 = {"dr": 0, "hs":0.00002 * (-1), "cg": 0.00017*(-1)}
        par.eta_3 = 0.02676
        par.eta_4 = -0.00076
        
        # miscellaneous structural parameters
        # shocks
        par.sigma_1 =  0.24485 # shock distribution: constant
        par.sigma_2 =  0.00421 # shock distribution: age
        par.initial_wealth_sigma = 1.48960
        
        par.tr = 5.51308 # transfer from parents
        #par.tr = 0
        par.rho = {"dr": 6.47838, "hs": 5.43473, "cg": 6.30347} # superannuation
        par.high_proportion = {"dr": 0.69306, "hs": 0.80130, "cg":  0.90089} # proportion of workers being high type
    
        # 6. simulation
        par.sim_mini = 2.5 # initial m in simulation
        par.simN = 100 # number of persons in simulation
        par.simT = 10 # number of periods in simulation
    
    def create_grids(self):
        par = self.par

        # End of period assets
        par.grid_a = np.nan + np.zeros([par.T,par.Na])
        for t in range(par.T):
            # credit
            par.grid_a[t,:] = tools.nonlinspace(0+1e-6,par.a_max,par.Na,par.a_phi)

        # Cash-on-hand
        par.grid_m =  np.concatenate([np.linspace(0+1e-6,1-1e-6,par.Nm_b), tools.nonlinspace(1+1e-6,par.m_max,par.Nm-par.Nm_b,par.m_phi)]) 

        par.grid_e = tools.nonlinspace(0+1e-4,par.h_max,par.Ne,par.p_phi)
        par.H_bunches = np.array([0, 1000, 2000, 2250, 2500, 3000])
        par.NH = 6

        par.Nshocks = par.Nxi * par.Na
        
        # Set seed
        np.random.seed(2020)

    def solve(self, print_iteration_number = True):
        import threading
        # Initialize
        par = self.par
        sol = self.sol

        shape=(par.T, par.NH, par.Nm, par.Ne)
        sol.m = np.nan+np.zeros(shape)
        sol.c = np.nan+np.zeros(shape)
        sol.v = np.nan+np.zeros(shape)
        
        def obj(c, m, h_plus, par):
            b = m - c
            return -(egm.util(c, h_plus, par.T-1, par) + egm.util_bequest(b, par, a_is_grid = False) )
        
        for i_c, m in enumerate(par.grid_m):
            c = optimize.minimize_scalar(obj, args=(m, 0, par), bounds = (0.000001, m), method = "bounded").x
            sol.c[par.T -1,:,i_c,:] = c
            sol.v[par.T -1,:,i_c,:] = egm.util(c,0,par.T - 1,par)
        #return
        """
        # Last period, (= consume all) 
        for i_e in range(par.Ne):
            for i_h, h_plus in enumerate(par.H_bunches):
                for i_c, m in enumerate(par.grid_m):
                    c = optimize.minimize_scalar(obj, args=(h_plus, par), bounds = (0, m)).x
                    
                    sol.c[par.T-1,i_h,i_c,i_e] = c
                    sol.v[par.T-1,i_h,i_c,i_e] = egm.util(c,h_plus,par.T-1,par)
                
                    #sol.m[par.T-1,i_h,:,i_e] = par.grid_m
                    #sol.c[par.T-1,i_h,:,i_e] = par.grid_m
                    #sol.v[par.T-1,i_h,:,i_e] = egm.util(sol.c[par.T-1,i_h,:,i_e],h_plus,par.T-1,par)
        """
        # Before last period
        # T_plus is time choice [T^w, T^H], e.g. [5, 10]  
        for t in range(par.T-2, par.Tmin -1,-1):
            print(t)
            #Choice specific function
            for i_e, e_plus in enumerate(par.grid_e):
                if t > 84:
                    h_plus = 0
                    c,v = egm.EGM(sol,h_plus,e_plus,t,par)
                    sol.c[t,:,:,i_e] = c
                    sol.v[t,:,:,i_e] = v
                    #print(c.shape)
                    continue
                    
                for i_h, h_plus in enumerate(par.H_bunches):
                    # Solve model with EGM
                    c,v = egm.EGM(sol,h_plus,e_plus,t,par)
                    sol.c[t,i_h,:,i_e] = c
                    sol.v[t,i_h,:,i_e] = v
            #return