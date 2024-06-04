import numpy as np
from scipy.stats import gamma, norm, beta, truncnorm, lognorm
import pandas as pd
import batman
import rmfit
from dynesty import DynamicNestedSampler
from dynesty.utils import resample_equal
import contextlib
from multiprocessing import Pool
from astropy import units as u
from astropy import constants as c
import itertools

def transform_uniform(x, hyperparameters):
    a, b = hyperparameters
    return a + (b-a)*x

def transform_loguniform(x, hyperparameters):
    a, b = hyperparameters
    la = np.log(a)
    lb = np.log(b)
    return np.exp(la + x * (lb - la))

def transform_normal(x, hyperparameters):
    mu, sigma = hyperparameters
    return norm.ppf(x, loc=mu, scale=sigma)

def transform_beta(x, hyperparameters):
    a, b = hyperparameters
    return beta.ppf(x, a, b)

def transform_exponential(x, hyperparameters):
    a = hyperparameters
    return gamma.ppf(x, a)

def transform_truncated_normal(x, hyperparameters):
    mu, sigma, a, b = hyperparameters
    ar, br = (a - mu) / sigma, (b - mu) / sigma
    return truncnorm.ppf(x, ar, br, loc=mu, scale=sigma)

def transform_modifiedjeffreys(x, hyperparameters):
    turn, hi = hyperparameters
    return turn * (np.exp((x + 1e-10) * np.log(hi / turn + 1)) - 1)

def u1_u2_from_q1_q2(q1,q2):
    u1, u2 = 2.*np.sqrt(q1)*q2, np.sqrt(q1)*(1.-2*q2)
    return u1, u2

def q1_q2_from_u1_u2(u1,u2):
    q1, q2 = (u1 + u2)**2. , u1/(2.*(u1+u2))
    return q1, q2

def Transit(bjd, t0, P, RpRs, sma, inc, e, w, u1, u2):                 #Tierra
    params = batman.TransitParams()
    params.t0 = t0                  #epoch of mid-transit
    params.per = P                  #orbital period
    params.rp = RpRs                 #planet radius (in units of stellar radii)
    params.a =  sma              #semi-major axis (in units of stellar radii)
    params.inc =  inc            #orbital inclination (in degrees)
    params.ecc = e                     #eccentricity
    params.w = w                       #longitude of periastron (in degrees)
    params.u = [u1,u2]              #limb darkening coefficients [u1, u2]
    params.limb_dark = "quadratic"       #limb darkening model
    m = batman.TransitModel(params, bjd)    #initializes model
    flux_lc = m.light_curve(params)          #calculates light curve
    return flux_lc

def create_RM_data(bjd,dct_params):
        lam = dct_params["lam_p1"]
        vsini = dct_params["vsini_star"]
        P =  dct_params["per_p1"]
        t0 = dct_params["t0_p1"]
        RpRs = dct_params["p_p1"] 
        e = dct_params["e_p1"]
        w = dct_params["omega_p1"]
        beta = dct_params["beta"]
        u1 = dct_params["u"][0]
        u2 = dct_params["u"][1]
        sma = dct_params["aRs_p1"]
        inc = dct_params["inc_p1"]
        exp_time = dct_params["exp_time"]/60./60./24.
        rv_prec = dct_params["rv_prec"]
        errors = np.ones(len(bjd))*rv_prec
        RM = rmfit.RMHirano(lam,vsini,P,t0,sma,inc,RpRs,e,w,[u1,u2],beta,vsini/1.31,limb_dark="quadratic",supersample_factor=10,exp_time=exp_time)
        RM_model = RM.evaluate(bjd,base_error=rv_prec)
        return RM_model, errors

def get_vals(vec):
    fvec   = np.sort(vec)

    fval  = np.median(fvec)
    nn = int(np.around(len(fvec)*0.15865))

    vali,valf = fval - fvec[nn],fvec[-nn] - fval
    return fval,vali,valf

class Data_Org:
    def __init__(self,lc_time=None,lc_flux=None,lc_flux_err=None,rv_time=None,rv=None,rv_err=None,rm_time=None,rm=None,rm_err=None,verbose = True,exp_times=None):
        self.verbose = verbose
        self.exp_times = exp_times
        if self.verbose:
            print("Reading LCs...")
        self.lc_time = lc_time
        self.lc_flux = lc_flux
        self.lc_flux_err = lc_flux_err
        if verbose:
            print("Reading RVs...")
        self.rv_time = rv_time
        self.rv = rv
        self.rv_err = rv_err
        if verbose:
            print("Reading RM data...")
        self.rm_time = rm_time
        self.rm = rm
        self.rm_err = rm_err
        self.lc_instruments = list(self.lc_time.keys())
        self.rv_instruments = list(self.rv_time.keys())
        self.rm_instruments = list(self.rm_time.keys())
        self.x = dict(itertools.chain(self.lc_time.items(),self.rv_time.items(),self.rm_time.items()))
        self.y = dict(itertools.chain(self.lc_flux.items(),self.rv.items(),self.rm.items()))
        self.yerr = dict(itertools.chain(self.lc_flux_err.items(),self.rv_err.items(),self.rm_err.items()))

class Priors:
    def __init__(self,file_,data,verbose = True):
        self.data = data
        self.dct = {}
        self.verbose = verbose
        with open(file_) as priors_file:
            for line in priors_file:
                line = line.split()
                hyp = ()
                for element in line[2].split(","):
                    hyp = hyp + (float(element),)
                self.dct[line[0]] = [line[1], hyp]
        if self.verbose:
            print("Priors dictionary ready...")
        self.parameters = list(self.dct.keys())
        if self.verbose:
            print("Detecting fixed parameters...")
        fixed = []
        varying = []
        for parameter in self.parameters:
            if self.dct[parameter][0] == "FIXED":
                if self.verbose:
                    print("Fixed {0} detected!".format(parameter))
                fixed.append(parameter)
            else:
                varying.append(parameter)
        self.fixed_parameters = fixed
        self.varyin_parameters = varying
        
        if "r_star" in self.parameters and "m_star" in self.parameters:
            self.m_star_param = True
            self.rho_param = True
        elif "rho_star" in self.parameters:
            self.m_star_param = False
            self.rho_param = True
        else:
            self.m_star_param = False
            self.rho_param = False
    
        if "b_p1" in self.parameters:
            self.b_param = True
        else:
            self.b_param = False

        if "vsini_star" in self.parameters:
            self.true_obliquity_param = False
        elif "cosi_star" in self.parameters and "Prot_star" in self.parameters and "r_star" in self.parameters:
            self.true_obliquity_param = True
        else:
            print("Something went wrong with the obliquity parametrization. Please check the parameters. For deriving the true obliquity include cosi_star, Prot_star, and r_star. For the sky-projected obliquity include vsini_star.")

class Fit:
    def __init__(self,data,priors):
        self.data = data
        self.priors = priors
        self.ndim = len(self.priors.varyin_parameters)

    def priors_transform(self,params):
            tranformed_values = []
            for index,parameter in enumerate(self.priors.varyin_parameters):
                dist = self.priors.dct[parameter][0]
                if dist != "FIXED":
                    value = params[index]
                    if dist == "uniform":
                        transformed = transform_uniform(value, self.priors.dct[parameter][1])
                    if dist == "normal":
                        transformed = transform_normal(value, self.priors.dct[parameter][1])
                    if dist == "loguniform":
                        transformed = transform_loguniform(value, self.priors.dct[parameter][1])
                    if dist == "truncatednormal":
                        transformed = transform_truncated_normal(value, self.priors.dct[parameter][1])
                    tranformed_values.append(transformed)
                else:
                    continue
            return tuple(tranformed_values)
            
    def get_rv_model(self,dct_params,inst):
        bjd = self.data.x[inst]
        P = dct_params["per_p1"]
        t0 = dct_params["t0_p1"]
        e = dct_params["e_p1"]    
        w = dct_params["omega_p1"]
        K = dct_params["K_p1"]
        gamma = dct_params["gamma_"+inst]
        gammadot = dct_params["gammadot"]
        gammadotdot = dct_params["gammadotdot"]
        rv_trend = gammadot*(bjd-t0) + gammadotdot*(bjd-t0)**2.0
        rv_model = rmfit.get_rv_curve(bjd,P,t0,e,w,K,plot=False,verbose=False)+gamma+rv_trend
        return rv_model
    
    def get_lc_model(self,dct_params,inst):
        bjd = self.data.x[inst]
        tr_model = batman.TransitParams()
        tr_model.t0 = dct_params["t0_p1"]                 
        tr_model.per =  dct_params["per_p1"]                 
        tr_model.rp = dct_params["p_p1"]                 
        tr_model.a = dct_params["aRs_p1"]             
        tr_model.inc = dct_params["inc_p1"]           
        tr_model.ecc = dct_params["e_p1"]                     
        tr_model.w = dct_params["omega_p1"]                       
        tr_model.u = [dct_params["u1_"+inst],dct_params["u2_"+inst]]              
        tr_model.limb_dark = "quadratic"       
        if self.data.exp_times[inst] != False:
            m = batman.TransitModel(tr_model, bjd, supersample_factor = 10, exp_time = float(self.data.exp_times[inst]))    #initializes model
        else:
            m = batman.TransitModel(tr_model, bjd)
        flux_lc = m.light_curve(tr_model)          
        return flux_lc
    
    def get_rm_model(self,dct_params,inst):
        bjd = self.data.x[inst]
        lam = dct_params["lam_p1"]
        vsini = dct_params["vsini_star"]
        P =  dct_params["per_p1"]
        t0 = dct_params["t0_p1"]
        b = dct_params["b_p1"]
        RpRs = dct_params["p_p1"] 
        e = dct_params["e_p1"]
        w = dct_params["omega_p1"]
        K = dct_params["K_p1"]
        beta = dct_params["beta_"+inst]
        gamma = dct_params["gamma_"+inst]
        u1 = dct_params["u1_"+inst]
        u2 = dct_params["u2_"+inst]
        sma = dct_params["aRs_p1"]
        inc = dct_params["inc_p1"]
        gammadot = dct_params["gammadot"]
        gammadotdot = dct_params["gammadotdot"]
        if "gammadot_"+inst in dct_params:
            gammadot = dct_params["gammadot_"+inst]
        if "gammadotdot_"+inst in dct_params:
            gammadotdot = dct_params["gammadotdot_"+inst]
        rv_trend = gammadot*(bjd-t0) + gammadotdot*(bjd-t0)**2.0
        if self.data.exp_times[inst] != False:
            RM = rmfit.RMHirano(lam,vsini,P,t0,sma,inc,RpRs,e,w,[u1,u2],beta,vsini/1.31,limb_dark="quadratic",supersample_factor=10,exp_time=float(self.data.exp_times[inst]))
            RM_model = RM.evaluate(bjd) + rmfit.get_rv_curve(bjd,P,t0,e,w,K,plot=False,verbose=False) + gamma + rv_trend
        else:
            print("No exp_time for instrument:", inst)
            return False
        return RM_model
    
    def get_model_for_inst(self,inst,dcti):
        if inst in self.data.lc_instruments:
            model_inst = self.get_lc_model(dcti,inst)
        elif inst in self.data.rv_instruments:
            model_inst = self.get_rv_model(dcti,inst)
        elif inst in self.data.rm_instruments:
            model_inst = self.get_rm_model(dcti,inst)
        return model_inst
                
    def LogLike(self, params):
        chi2 = 0
        n_fixed = 0
        dct_i = {}
        for index,parameter in enumerate(self.priors.parameters):
            if self.priors.dct[parameter][0] != "FIXED":
                val = params[int(index-n_fixed)]
                dct_i[parameter] = val
            elif self.priors.dct[parameter][0] == "FIXED": 
                n_fixed += 1
                val = float(self.priors.dct[parameter][1][0])
                dct_i[parameter] = val
                
        for inst in np.concatenate((self.data.lc_instruments,self.data.rm_instruments)):
            print(inst)
            search_key = "_"+inst
            inst_params = [val for key, val in dct_i.items() if search_key in key]
            dct_i["q1_"+inst] = float(inst_params[0])
            dct_i["q2_"+inst] = float(inst_params[1])
            dct_i["u1_"+inst], dct_i["u2_"+inst] = u1_u2_from_q1_q2(float(dct_i["q1_"+inst]),float(dct_i["q2_"+inst]))
            dct_i["sigma_"+inst] = float(inst_params[2])
            if inst in self.data.rm_instruments:
                dct_i["beta_"+inst] = float(inst_params[2])
                dct_i["gamma_"+inst] = float(inst_params[3])
                dct_i["sigma_"+inst] = float(inst_params[4])
                if len(inst_params) == 6:
                    if "gammadot_"+inst in dct_i:
                        dct_i["gammadot_"+inst] = float(inst_params[5])
                    else:
                        dct_i["gammadotdot_"+inst] = float(inst_params[5])
                elif len(inst_params) > 6:
                    dct_i["gammadot_"+inst] = float(inst_params[5])
                    dct_i["gammadotdot_"+inst] = float(inst_params[6])

        if self.priors.m_star_param:
            volume = 4.0/3.0*np.pi*(dct_i["r_star"]**3.0)*(u.Rsun**3.0)
            dct_i["rho_star"] = (dct_i["m_star"]*u.Msun/volume).to(u.kg/u.m/u.m/u.m).value
        if self.priors.rho_param:     
            dct_i["aRs_p1"] = ((c.G*((dct_i["per_p1"]*u.d)**2.0)*(dct_i["rho_star"]*u.kg/u.m/u.m/u.m)/3.0/np.pi)**(1./3.)).cgs.value
        if self.priors.b_param:    
            dct_i["inc_p1"] = np.arccos(dct_i["b_p1"]/dct_i["aRs_p1"]*((1.0+dct_i["e_p1"]*np.sin(dct_i["omega_p1"]*np.pi/180.0))/(1.0 - dct_i["e_p1"]**2.0)))*180.0/np.pi
        if self.priors.true_obliquity_param:
            veq = (2.0*np.pi*dct_i["r_star"]*u.Rsun/dct_i["Prot_star"]/u.d).to(u.km/u.s).value
            dct_i["vsini_star"] = veq*np.sqrt(1.0-dct_i["cosi_star"]**2.0)
        
        for inst in self.data.x:
            jitter = float(dct_i["sigma_"+inst])
            inst_data = self.data.y[inst]
            inst_err = self.data.yerr[inst]
            inst_model = self.get_model_for_inst(inst,dct_i)
            inst_err = np.sqrt(inst_err**2.0 + jitter**2.0)
            chi2 += np.nansum(((inst_data-inst_model)/inst_err)**2.0) + np.nansum(np.log((inst_err**2.0)))
        return -0.5*chi2
    
    def run(self, n_live = 600, bound = 'multi', sample = 'rwalk', nthreads = 2):
        if self.data.verbose:
            print("Running dynesty with {0} nlive and {1} threads".format(n_live,nthreads))
        with contextlib.closing(Pool(processes= nthreads - 1)) as executor:
            sampler = DynamicNestedSampler(self.LogLike, self.priors_transform, self.ndim, bound=bound, sample=sample , nlive = n_live, pool=executor, queue_size=nthreads, bootstrap = 0)
            sampler.run_nested()
            res = sampler.results
            weights = np.exp(res['logwt'] - res['logz'][-1])
            self.postsamples = resample_equal(res.samples, weights)
        return self.postsamples

class Post:
    def __init__(self,data,priors,fit,chain,print_params=False):
        self.data = data
        self.priors = priors
        self.chain = chain
        self.fit = fit
        vals, err_up, err_down = {}, {}, {}
        for i,parameter in enumerate(self.priors.varyin_parameters):
            val, mi, ma = get_vals(np.sort(self.chain[:,i]))
            vals[parameter], err_up[parameter], err_down[parameter] = val, ma, mi
            print(parameter, val, mi, ma)
        self.chain = pd.DataFrame(data = self.chain, columns = self.priors.varyin_parameters)
        for i,parameter in enumerate(self.priors.fixed_parameters):
            vals[parameter], err_up[parameter], err_down[parameter] = float(self.priors.dct[parameter][1][0]), np.nan, np.nan,
            vals_to_chain = np.full(len(self.chain),float(self.priors.dct[parameter][1][0]))
            self.chain[parameter] = vals_to_chain
        self.vals = vals
        self.err_up = err_up
        self.err_down = err_down
    
        for inst in np.concatenate((self.data.lc_instruments,self.data.rm_instruments)):
            search_key = "_"+inst
            inst_params = [s for s in self.chain.columns if search_key in s]
            self.chain["u1_"+inst],self.chain["u2_"+inst] = u1_u2_from_q1_q2(self.chain[inst_params[0]].values,self.chain[inst_params[1]].values)
            self.vals["u1_"+inst], self.err_down["u1_"+inst], self.err_up["u1_"+inst] = get_vals(self.chain["u1_"+inst].values)
            self.vals["u2_"+inst], self.err_down["u2_"+inst], self.err_up["u2_"+inst] = get_vals(self.chain["u2_"+inst].values)
            if inst in self.data.rm_instruments:
                self.chain["beta_"+inst] = self.chain[inst_params[2]].values
                self.chain["gamma_"+inst] = self.chain[inst_params[3]].values
                self.vals["beta_"+inst], self.err_down["beta_"+inst], self.err_up["beta_"+inst] = get_vals(self.chain["beta_"+inst].values)
                self.vals["gamma_"+inst], self.err_down["gamma_"+inst], self.err_up["gamma_"+inst] = get_vals(self.chain["gamma_"+inst].values)
                
        if self.priors.m_star_param:
            volume = 4.0/3.0*np.pi*(self.chain["r_star"].values**3.0)*(u.Rsun**3.0)
            self.chain["rho_star"] = (self.chain["m_star"].values*u.Msun/volume).to(u.kg/u.m/u.m/u.m).value
            val, mi, ma = get_vals(self.chain["rho_star"].values)
            self.vals["rho_star"], self.err_up["rho_star"], self.err_down["rho_star"] = val, ma, mi
            print("rho_star",val, mi, ma)
        if self.priors.rho_param:     
            self.chain["aRs_p1"] = ((c.G*((self.chain["per_p1"].values*u.d)**2.0)*(self.chain["rho_star"].values*u.kg/u.m/u.m/u.m)/3.0/np.pi)**(1./3.)).cgs.value
            val, mi, ma = get_vals(self.chain["aRs_p1"].values)
            self.vals["aRs_p1"], self.err_up["aRs_p1"], self.err_down["aRs_p1"] = val, ma, mi
            print("aRs_p1",val, mi, ma)
        if self.priors.b_param:    
            self.chain["inc_p1"] = np.arccos(self.chain["b_p1"].values/self.chain["aRs_p1"].values*((1.0+self.chain["e_p1"].values*np.sin(self.chain["omega_p1"].values*np.pi/180.0))/(1.0 - self.chain["e_p1"].values**2.0)))*180.0/np.pi
            val, mi, ma = get_vals(self.chain["inc_p1"].values)
            self.vals["inc_p1"], self.err_up["inc_p1"], self.err_down["inc_p1"] = val, ma, mi
            print("inc_p1",val, mi, ma)
        if self.priors.true_obliquity_param:
            self.chain["veq_star"] = (2.0*np.pi*self.chain["r_star"].values*u.Rsun/self.chain["Prot_star"].values/u.d).to(u.km/u.s).value
            self.chain["vsini_star"] = self.chain["veq_star"].values*np.sqrt(1.0-self.chain["cosi_star"].values**2.0)
            self.chain["psi_p1"] = np.arccos(self.chain["cosi_star"].values*np.cos(self.chain["inc_p1"].values*np.pi/180.0) + np.sqrt(1.0-self.chain["cosi_star"].values**2.0)*np.sin(self.chain["inc_p1"].values*np.pi/180.0)*np.cos(self.chain["lam_p1"].values*np.pi/180.0))*180.0/np.pi
            val, mi, ma = get_vals(self.chain["veq_star"].values)
            self.vals["veq_star"], self.err_up["veq_star"], self.err_down["veq_star"] = val, ma, mi
            print("veq_star",val, mi, ma)
            val, mi, ma = get_vals(self.chain["vsini_star"].values)
            self.vals["vsini_star"], self.err_up["vsini_star"], self.err_down["vsini_star"] = val, ma, mi
            print("vsini_star",val, mi, ma)
            val, mi, ma = get_vals(self.chain["psi_p1"].values)
            self.vals["psi_p1"], self.err_up["psi_p1"], self.err_down["psi_p1"] = val, ma, mi
            print("psi_p1",val, mi, ma)
        
    def print_mass_radius_rho_sma_planet(self,rstar=(1.0,0.1),mstar=(1.0,0.1),r_units=u.Rjup,sma_units=u.AU,m_units=u.Mjup):
        Rs = np.random.normal(rstar[0],rstar[1],len(self.chain))*u.Rsun
        Ms = np.random.normal(mstar[0],mstar[1],len(self.chain))*u.Msun
        rp = self.chain["p_p1"].values*Rs
        rp = rp.to(r_units)
        print("R_planet",get_vals(rp.value))
        aRs = self.chain["aRs_p1"].values*Rs
        sma = aRs.to(sma_units)
        print("Sma_planet:",get_vals(sma.value))
        K = self.chain["K_p1"].values*u.m/u.s
        e = self.chain["e_p1"].values
        inc = self.chain["inc_p1"].values
        mp = K*np.sqrt(Ms*sma*(1.0-e**2.0)/c.G)/np.sin(inc*np.pi/180.0)
        mp = mp.to(m_units)
        print("M_planet:",get_vals(mp.value))
        dens = 3.0*mp/4.0/np.pi/rp**3.0
        dens = dens.to(u.g/u.cm/u.cm/u.cm)
        print("Rho_planet (gr/cm3):",get_vals(dens.value))

    def evaluate_RV_model(self,times1,instrument,models=False,number_models=5000):
        necessary_params = {k: self.vals[k] for k in ('per_p1','t0_p1','e_p1','omega_p1','K_p1','gamma_'+instrument,'gammadot','gammadotdot')}
        P, t0, e, w, K, gamma, gammadot, gammadotdot = necessary_params.values()
        if 'gammadot_'+instrument in self.vals:
            gammadot = self.vals['gammadot_'+instrument]
        if 'gammadotdot_'+instrument in self.vals:
            gammadotdot = self.vals['gammadotdot_'+instrument]
        rv_trend = gammadot*(times1-t0) + gammadotdot*(times1-t0)**2.0
        rv_model = rmfit.get_rv_curve(times1,P,t0,e,w,K,plot=False,verbose=False)
        if models:
            mmodel1 = []
            for i in range(number_models):
                if i%100 == 0: print("Sampling i =",i,end="\r")
                idx = np.random.randint(0,len(self.chain))
                chain_models = self.chain[['per_p1','t0_p1','e_p1','omega_p1','K_p1','gamma_'+instrument]]
                P, t0, e, w, K, gamma = chain_models.values[idx]
                m1 = rmfit.get_rv_curve(times1,P,t0,e,w,K,plot=False,verbose=False)
                mmodel1.append(m1)
            mmodel1 = np.array(mmodel1)
            return rv_model, rv_trend, gamma, mmodel1
        return rv_model, rv_trend, gamma

    def evaluate_RM_model(self,times1,instrument,models=False,number_models=5000):
        necessary_params = {k: self.vals[k] for k in ('lam_p1','vsini_star','per_p1','t0_p1','p_p1','e_p1','omega_p1','K_p1','aRs_p1','inc_p1','u1_'+instrument,'u2_'+instrument,'beta_'+instrument,'gamma_'+instrument, 'gammadot', 'gammadotdot','sigma_'+instrument)}
        lam, vsini, P, t0, RpRs, e, w, K, aRs, inc, u1, u2, beta, gamma, gammadot, gammadotdot, jitter = necessary_params.values()
        if 'gammadot_'+instrument in self.vals:
            gammadot = self.vals['gammadot_'+instrument]
        if 'gammadotdot_'+instrument in self.vals:
            gammadotdot = self.vals['gammadotdot_'+instrument]
        error = np.sqrt(self.data.yerr[instrument]**2.0 + jitter**2.0)
        RM = rmfit.RMHirano(lam,vsini,P,t0,aRs,inc,RpRs,e,w,[u1,u2],beta,vsini/1.31,limb_dark="quadratic")
        rv_trend = gammadot*(times1-t0) + gammadotdot*(times1-t0)**2.0
        model = RM.evaluate(times1) + rmfit.get_rv_curve(times1,P,t0,e,w,K,plot=False,verbose=False) + gamma + rv_trend
        if models:
            mmodel1 = []
            for i in range(number_models):
                if i%100 == 0: print("Sampling i =",i,end="\r")
                idx = np.random.randint(0,len(self.chain))
                chain_models = self.chain[['lam_p1','vsini_star','per_p1','t0_p1','p_p1','e_p1','omega_p1','K_p1','aRs_p1','inc_p1','u1_'+instrument,'u2_'+instrument,'beta_'+instrument,'gamma_'+instrument, 'gammadot', 'gammadotdot']]
                lam, vsini, P, t0, RpRs, e, w, K, aRs, inc, u1, u2, beta, gamma, gammadot, gammadotdot = chain_models.values[idx]
                rv_trend = gammadot*(times1-t0) + gammadotdot*(times1-t0)**2.0
                RM = rmfit.RMHirano(lam,vsini,P,t0,aRs,inc,RpRs,e,w,[u1,u2],beta,vsini/1.31,limb_dark="quadratic")
                m1 = RM.evaluate(times1) + rmfit.get_rv_curve(times1,P,t0,e,w,K,plot=False,verbose=False) + gamma + rv_trend
                mmodel1.append(m1)
            mmodel1 = np.array(mmodel1)
            return model, mmodel1
        return model

    def evaluate_LC_model(self,times1,instrument):
        necessary_params = {k: self.vals[k] for k in ('per_p1','t0_p1','p_p1','e_p1','omega_p1','K_p1','aRs_p1','inc_p1','u1_'+instrument,'u2_'+instrument)}
        P, t0, RpRs, e, w, K, sma, inc, u1, u2 = necessary_params.values()
        model = Transit(times1, t0, P, RpRs, sma, inc, e, w, u1, u2)
        return model
