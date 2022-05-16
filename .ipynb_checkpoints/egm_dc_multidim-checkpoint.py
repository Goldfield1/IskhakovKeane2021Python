import numpy as np
import tools
import warnings

def EGM (sol,h_plus,e_plus, t,par): 

    #if z_plus == 1:     #Retired =  Not working
    #    w_raw, avg_marg_u_plus = retired(sol,z_plus,p,t,par)
    #else:               # Working
    #    w_raw, avg_marg_u_plus = working(sol,z_plus,p,t,par)

    w_raw, avg_marg_u_plus = first_step(sol,h_plus,e_plus,t,par)

    # raw c, m and v
    # + (1-delta) * marg_util_bequest b() # bequest
    
    # inv RHS euler
    delta = death_chance(t)
    marg_u_bequest = marg_util_bequest(par.grid_a[t,:], par)
    #marg_u_pens
    
    c_raw = inv_marg_util((1-delta) * marg_u_bequest + delta * par.beta * par.interest * avg_marg_u_plus, par)
    m_raw = c_raw + par.grid_a[t,:]
   
    # Upper Envelope
    #c,v = c_raw, m_raw
    c,v = upper_envelope(t,h_plus,c_raw,m_raw,w_raw,par)
    #c, v = c_raw, m_raw
    return c,v

# delta -> death chance
def death_chance(age):
    return 1 - 0.0006569 * (np.exp(0.1078507 * (age - 40)) - 1) * (age >= 40)

# tau is type
def human_capital(e, t, par, tau=("cg","high")):
    edu, skill = tau
    
    return np.exp(par.eta_0[edu] + par.eta_0[skill] + par.eta_1[edu] * t*e + par.eta_2[edu] * (t*e)**2 + par.eta_3*t + par.eta_4*t**2)

def soft_max(x, y):
    v = 0.1
    return v * np.log(np.exp(x/v) + np.exp(y/v))

def pens_fun(m, wage):
    warnings.filterwarnings('error')
    try:
        year = 2015
        m_asset_test = m
        pens = soft_max(10.75973 + 1.84692 * (year > 2010) 
                        - soft_max(0, soft_max(0.27749 * wage, 0.00499 * (m_asset_test - 117.08260))), 0)
    except:
        print(pens)
        return "dd"
        
    return pens
    

def tax_fun_(income, par):
    thld_1 = 17.3918
    thld_2 = 73.1766

    if income < thld_1:
        return 0
    if income < thld_2:
        return 0.29907 * (income - thld_1)
    return 0.37930 * (income - thld_2) + 0.29907 * thld_2

def first_step(sol, h_plus, e, t, par, tau=("cg","high")):

    
    edu, skill = tau
    # Prepare
    #print(t)
    
    xi, xi_w = tools.GaussHermite_lognorm(par.sigma_1 + par.sigma_2 * t, par.Nxi)
    xi = np.tile(xi,par.Na)
    
    e = np.tile(e, par.Na*par.Nxi)
    a = np.repeat(par.grid_a[t],par.Nxi) 
    w = np.tile(xi_w,(par.Na,1))

    # Next period states
    tax_fun = np.vectorize(tax_fun_)
          
    e_plus = (1/(1 + t)) * (t * e + h_plus/par.H_bunches[5])
    K_plus = human_capital(e_plus, t, par)
    wage_plus = K_plus * xi # K
    income = h_plus * wage_plus / 1000
    
    #print(income)
    #print(tax_fun(h_plus * wage_plus, par))
    
    super_payment = K_plus * par.rho[edu] * (t+1 == 65)   
    m_plus = par.interest * a + income - tax_fun(income, par) + par.tr * (t+1 <= 3) # wealth before pens

    #pens_fun(m_plus, wage_plus)
    #m_plus += pens_fun(m_plus, wage_plus) * (t+1 > 65) # wealth after pens
    
    # Value, consumption, marg_util
    shape = (par.NH,m_plus.size)
    v_plus = np.nan+np.zeros(shape)
    c_plus = np.nan+np.zeros(shape)
    marg_u_plus = np.nan+np.zeros(shape)

    # range over possible hours worked next period
    for i, h_i in enumerate(par.H_bunches):
        # Choice specific value 
        v_plus[i,:] = tools.interp_2d_vec(par.grid_m, par.grid_e, sol.v[t+1,i], m_plus, e_plus)
    
        # Choice specific consumption    
        c_plus[i,:] = tools.interp_2d_vec(par.grid_m, par.grid_e, sol.c[t+1,i], m_plus, e_plus)
        c_plus[i,:] = np.maximum(c_plus[i,:], 0.001)
                
        # Choice specific Marginal utility
        marg_u_plus[i,:] = marg_util(c_plus[i,:], par) 
       
    # Expected value
    V_plus, prob = logsum(v_plus, par.lambda_scale) 
    w_raw = w*np.reshape(V_plus,(par.Na,par.Nxi))
    w_raw = np.sum(w_raw,1) 
    marg_u_plus = np.sum(prob * marg_u_plus, axis = 0)

    #Expected  average marg. utility
    avg_marg_u_plus = w*np.reshape(marg_u_plus,(par.Na,par.Nxi))
    avg_marg_u_plus = np.sum(avg_marg_u_plus ,1)

    return w_raw, avg_marg_u_plus


def upper_envelope(t,h_plus,c_raw,m_raw,w_raw,par):
    
    # Add a point at the bottom
    c_raw = np.append(1e-6,c_raw)  
    m_raw = np.append(1e-6,m_raw) 
    a_raw = np.append(0,par.grid_a[t,:]) 
    w_raw = np.append(w_raw[0],w_raw)

    # Initialize c and v   
    c = np.nan + np.zeros((par.Nm))
    v = -np.inf + np.zeros((par.Nm))
    
    # Loop through the endogenous grid
    size_m_raw = m_raw.size
    for i in range(size_m_raw-1):    

        c_now = c_raw[i]        
        m_low = m_raw[i]
        m_high = m_raw[i+1]
        c_slope = (c_raw[i+1]-c_now)/(m_high-m_low)
        
        w_now = w_raw[i]
        a_low = a_raw[i]
        a_high = a_raw[i+1]
        w_slope = (w_raw[i+1]-w_now)/(a_high-a_low)

        # Loop through the common grid
        for j, m_now in enumerate(par.grid_m):

            interp = (m_now >= m_low) and (m_now <= m_high) 
            extrap_above = (i == size_m_raw-1) and (m_now > m_high)

            if interp or extrap_above:
                # Consumption
                c_guess = c_now+c_slope*(m_now-m_low)
                
                # post-decision values
                a_guess = m_now - c_guess
                w = w_now+w_slope*(a_guess-a_low)
                
                # Value of choice
                v_guess = util(c_guess,h_plus,t,par)+par.beta*death_chance(t)*w
                
                # Update
                if v_guess >v[j]:
                    v[j]=v_guess
                    c[j]=c_guess
    return c,v

def marg_util_bequest(b, par):
    a = np.ones(b.shape) * par.credit_constraint + 0.00001
    return par.b_scale * (b + a)**(1-par.zeta_beq) 
    
def util_bequest(b, par, a_is_grid = True): 
    if a_is_grid:
        a = np.ones(b.shape) * par.credit_constraint
    else:
        a = par.credit_constraint
    return par.b_scale * ((b + a)**(1-par.zeta_beq) - a**(1-par.zeta_beq)) / (1 - par.zeta_beq)


def v(h, t, par, tau=("cg","high")):
    edu, skill = tau
    
    k_type = 1 + par.kappa_1 * (skill == "low")
    k_age = 1 + par.kappa_2 * (t - 40)**2 * (t > 40) + par.kappa_3 * (t - 25) * (t < 25)
    
    return (h > 0) * k_type * k_age * par.gamma[h]

def util(c, h, t, par):
    return ((c**(1.0-par.zeta))/(1.0-par.zeta)- v(h, t, par))


#def marg_util_bequest():
    


def marg_util(c,par):
    if np.min(c) < 0:
        print(np.min(c))
    return c**(-par.zeta)

def inv_marg_util(u,par):
    return u**(-1/par.zeta)


def logsum(V, sigma):
    dd = np.isnan(V)
    de = np.isneginf(V)
    if (np.any(de)):
        print(dd)
        return 
    # Maximum over the discrete choices (0 = for each column, i.e., for each "competing choice")
    mxm = V.max(0)
    # numerically robust log-sum
    log_sum = mxm + sigma*(np.log(np.sum(np.exp((V - mxm) / sigma),axis=0)))
    # d. numerically robust probability
    prob = np.exp((V- log_sum) / sigma)    
    return log_sum,prob