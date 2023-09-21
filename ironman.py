import numpy as np
from scipy.stats import gamma, norm, beta, truncnorm, lognorm
import batman
import rmfit
from dynesty import NestedSampler, DynamicNestedSampler
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
                hyp = tuple(line[2].split(","))
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
                lower = float(self.priors.dct[parameter][1][0])
                upper = float(self.priors.dct[parameter][1][1])
                priors_values = (lower,upper)
                if dist == "uniform":
                    transformed = transform_uniform(value, priors_values)
                if dist == "normal":
                    transformed = transform_normal(value, priors_values)
                if dist == "loguniform":
                    transformed = transform_loguniform(value, priors_values)
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
        #phase = ((bjd-t0 + 0.5*P) % P)/P
        rv_model = rmfit.get_rv_curve(bjd,P,t0,e,w,K,plot=False,verbose=False)+gamma
        return rv_model
    
    def get_lc_model(self,dct_params,inst):
        bjd = self.data.x[inst]
        tr_model = batman.TransitParams()
        tr_model.t0 = dct_params["t0_p1"]                  #epoch of mid-transit
        tr_model.per =  dct_params["per_p1"]                 #orbital period
        tr_model.rp = dct_params["p_p1"]                 #planet radius (in units of stellar radii)
        tr_model.a = dct_params["sma_p1"]              #semi-major axis (in units of stellar radii)
        tr_model.inc = dct_params["inc_p1"]            #orbital inclination (in degrees)
        tr_model.ecc = dct_params["e_p1"]                     #eccentricity
        tr_model.w = dct_params["omega_p1"]                       #longitude of periastron (in degrees)
        tr_model.u = [dct_params["u1_"+inst],dct_params["u2_"+inst]]              #limb darkening coefficients [u1, u2]
        tr_model.limb_dark = "quadratic"       #limb darkening model
        if self.data.exp_times[inst] != False:
            m = batman.TransitModel(tr_model, bjd, supersample_factor = 10, exp_time = float(self.data.exp_times[inst]))    #initializes model
        else:
            m = batman.TransitModel(tr_model, bjd)
        flux_lc = m.light_curve(tr_model)          #calculates light curve
        return flux_lc
    
    def get_rm_model(self,dct_params,inst):
        bjd = self.data.x[inst]
        lam = dct_params["lam_p1"]
        vsini = dct_params["vsini_star"]
        P =  dct_params["per_p1"]
        t0 = dct_params["t0_p1"]
        rho = dct_params["rho_star"]
        b = dct_params["b_p1"]
        RpRs = dct_params["p_p1"] 
        e = dct_params["e_p1"]
        w = dct_params["omega_p1"]
        K = dct_params["K_p1"]
        beta = dct_params["beta_"+inst]
        gamma = dct_params["gamma_"+inst]
        u1 = dct_params["u1_"+inst]
        u2 = dct_params["u2_"+inst]
        sma = dct_params["sma_p1"]
        inc = dct_params["inc_p1"]
        if self.data.exp_times[inst] != False:
            RM = rmfit.RMHirano(lam,vsini,1,0.5,sma,inc,RpRs,e,w,[u1,u2],beta,vsini/1.31,limb_dark="quadratic",supersample_factor=10,exp_time=float(self.data.exp_times[inst]))
            RM_model = RM.evaluate(bjd) + rmfit.get_rv_curve(bjd,1,0.5,e,w,K,plot=False,verbose=False)+gamma
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
        
        for inst in self.data.lc_instruments:
            search_key = "_"+inst
            inst_params = [val for key, val in dct_i.items() if search_key in key]
            dct_i["q1_"+inst] = inst_params[0]
            dct_i["q2_"+inst] = inst_params[1]
            dct_i["u1_"+inst], dct_i["u2_"+inst] = u1_u2_from_q1_q2(float(dct_i["q1_"+inst]),float(dct_i["q2_"+inst]))
        for inst in self.data.rm_instruments:
            search_key = "_"+inst
            inst_params = [val for key, val in dct_i.items() if search_key in key]
            dct_i["q1_"+inst] = inst_params[0]
            dct_i["q2_"+inst] = inst_params[1]
            dct_i["u1_"+inst], dct_i["u2_"+inst] = u1_u2_from_q1_q2(float(dct_i["q1_"+inst]),float(dct_i["q2_"+inst]))
        dct_i["sma_p1"] = ((c.G*((dct_i["per_p1"]*u.d)**2.0)*(dct_i["rho_star"]*u.kg/u.m/u.m/u.m)/3.0/np.pi)**(1./3.)).cgs.value
        dct_i["inc_p1"] = np.arccos(dct_i["b_p1"]/dct_i["sma_p1"]*((1.0+dct_i["e_p1"]*np.sin(dct_i["omega_p1"]*np.pi/180.0))/(1.0 - dct_i["e_p1"]**2.0)))*180.0/np.pi
        
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
            sampler.run_nested()#dlogz=0.01)
            res = sampler.results
            weights = np.exp(res['logwt'] - res['logz'][-1])
            self.postsamples = resample_equal(res.samples, weights)
        return self.postsamples
