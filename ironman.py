import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["axes.formatter.useoffset"] = False
rcParams['axes.formatter.limits'] = -15,15
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
rcParams['xtick.direction']='in'
rcParams['ytick.direction']='in'
import rmfit
import batman
import matplotlib.gridspec as gridspec
from astropy import units as u
from astropy import constants as c
import corner
import lightkurve as lk
from scipy.stats import gamma, norm, beta, truncnorm, lognorm
from dynesty import NestedSampler, DynamicNestedSampler
from dynesty.utils import resample_equal
import contextlib
from multiprocessing import Pool
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
        rv_model = rmfit.get_rv_curve(bjd,P,t0,e,w,K,plot=False,verbose=False)+gamma
        return rv_model
    
    def get_lc_model(self,dct_params,inst):
        bjd = self.data.x[inst]
        tr_model = batman.TransitParams()
        tr_model.t0 = dct_params["t0_p1"]                 #epoch of mid-transit
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
            RM = rmfit.RMHirano(lam,vsini,P,t0,sma,inc,RpRs,e,w,[u1,u2],beta,vsini/1.31,limb_dark="quadratic",supersample_factor=10,exp_time=float(self.data.exp_times[inst]))
            RM_model = RM.evaluate(bjd) + rmfit.get_rv_curve(bjd,P,t0,e,w,K,plot=False,verbose=False)+gamma
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
            search_key = "_"+inst
            inst_params = [val for key, val in dct_i.items() if search_key in key]
            dct_i["q1_"+inst] = inst_params[0]
            dct_i["q2_"+inst] = inst_params[1]
            #print(inst_params)
            dct_i["u1_"+inst], dct_i["u2_"+inst] = u1_u2_from_q1_q2(float(dct_i["q1_"+inst]),float(dct_i["q2_"+inst]))
            if inst in self.data.rm_instruments:
                dct_i["beta_"+inst] = inst_params[2]
                #print(inst_params)
                #print(dct_i)
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

class Post:
    def __init__(self,data,priors,chain):
        self.data = data
        self.priors = priors
        self.chain = chain
        vals, err_up, err_down = {}, {}, {}
        for i,parameter in enumerate(self.priors.varyin_parameters):
            val, mi, ma = get_vals(np.sort(self.chain[:,i]))
            vals[parameter], err_up[parameter], err_down[parameter] = val, ma, mi
            print(parameter, val, mi, ma)
        self.chain = pd.DataFrame(data = self.chain, columns = self.priors.varyin_parameters)
        for i,parameter in enumerate(self.priors.fixed_parameters):
            vals[parameter] = float(self.priors.dct[parameter][1][0])
            vals_to_chain = np.full(len(self.chain),float(self.priors.dct[parameter][1][0]))
            self.chain[parameter] = vals_to_chain
        self.vals = vals
        self.err_up = err_up
        self.err_down = err_down
        
        for inst in np.concatenate((self.data.lc_instruments,self.data.rm_instruments)):
            search_key = "_"+inst
            inst_params = [s for s in self.chain.columns if search_key in s]
            self.chain["u1_"+inst],self.chain["u2_"+inst] = u1_u2_from_q1_q2(self.chain[inst_params[0]].values,self.chain[inst_params[1]].values)
            self.vals["u1_"+inst] = get_vals(self.chain["u1_"+inst].values)[0]
            self.vals["u2_"+inst] = get_vals(self.chain["u2_"+inst].values)[0]
            if inst in self.data.rm_instruments:
                self.chain["beta_"+inst] = self.chain[inst_params[2]].values
                self.vals["beta_"+inst] = get_vals(self.chain["beta_"+inst].values)[0]
        
    def find_fixed_or_chain(self,parameter):
        if parameter in self.chain.columns:
            return self.chain[parameter].values
        else:
            return float(self.priors.dct[parameter][1][0])
    
    def derive_sma_and_inc(self):
        rho = self.find_fixed_or_chain("rho_star")
        P = self.find_fixed_or_chain("per_p1")
        b = self.find_fixed_or_chain("b_p1")
        e = self.find_fixed_or_chain("e_p1")
        w = self.find_fixed_or_chain("omega_p1")
        aRs = ((c.G*((P*u.d)**2.0)*(rho*u.kg/u.m/u.m/u.m)/3.0/np.pi)**(1./3.)).cgs.value
        inc = np.arccos(b/aRs*((1+e*np.sin(w*np.pi/180.0))/(1.0 - e**2.0)))*180.0/np.pi
        val, mi, ma = get_vals(aRs)
        print("aRs_p1:",val, mi, ma)
        self.vals["sma_p1"], self.err_up["sma_p1"], self.err_down["sma_p1"] = val, ma, mi
        val, mi, ma = get_vals(inc)
        print("inc_p1 (deg):",val, mi, ma)
        self.vals["inc_p1"], self.err_up["inc_p1"], self.err_down["inc_p1"] = val, ma, mi
        self.chain["sma_p1"] = aRs
        self.chain["inc_p1"] = inc
        
    def print_mass_radius_rho_sma_planet(self,rstar,mstar):
        Rs = np.random.normal(rstar[0],rstar[1],len(self.chain))*u.Rsun
        Ms = np.random.normal(mstar[0],mstar[1],len(self.chain))*u.Msun
        rp = self.find_fixed_or_chain("p_p1")*Rs
        rp = rp.to(u.Rjup)
        print("R_planet (Rjup):",get_vals(rp.value))
        aRs = self.find_fixed_or_chain("sma_p1")
        sma = (aRs*Rs).to(u.AU)
        print("Sma_planet (au):",get_vals(sma.value))
        K = self.find_fixed_or_chain("K_p1")*u.m/u.s
        e = self.find_fixed_or_chain("e_p1")
        inc = self.find_fixed_or_chain("inc_p1")
        mp = K*np.sqrt(Ms*sma*(1.0-e**2.0)/c.G)/np.sin(inc*np.pi/180.0)
        mp = mp.to(u.Mjup)
        print("M_planet (Mjup):",get_vals(mp.value))
        dens = 3.0*mp/4.0/np.pi/rp**3.0
        dens = dens.to(u.kg/u.m/u.m/u.m)
        print("Rho_planet (gr/cm3):",get_vals(dens.value/1000.0))
        
    def Plot_RM_all(self,colors,ms = 15,mw = 1, number_models = 5000):
        rcParams["figure.figsize"] = (24,18)
        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(4, 3)
        ax1 = fig.add_subplot(gs[:3, :])
        ax2 = fig.add_subplot(gs[3, :], sharex=ax1)
        for instrument in self.data.rm_instruments:
            necessary_params = {k: self.vals[k] for k in ('lam_p1','vsini_star','per_p1','t0_p1','p_p1','e_p1','omega_p1','K_p1','sma_p1','inc_p1','u1_'+instrument,'u2_'+instrument,'beta_'+instrument,'gamma_'+instrument, 'sigma_'+instrument)}
            lam, vsini, P, t0, RpRs, e, w, K, aRs, inc, u1, u2, beta, gamma, jitter = necessary_params.values()
            times1 = np.linspace(self.data.x[instrument][0]-0.02,self.data.x[instrument][-1]+0.02,5000)
            error = np.sqrt(self.data.yerr[instrument]**2.0 + jitter**2.0)
            phase = ((self.data.x[instrument]-t0 + 0.5*P) % P)/P
            ax1.errorbar(phase,self.data.y[instrument]- rmfit.get_rv_curve(self.data.x[instrument],P,t0,e,w,K,plot=False,verbose=False) - gamma,error,label = instrument,mfc=colors[instrument],mec='k',ecolor='k',marker='o',elinewidth=1,capsize=4,lw=0,mew=mw,markersize=ms,zorder=10)
            RM = rmfit.RMHirano(lam,vsini,P,t0,aRs,inc,RpRs,e,w,[u1,u2],beta,vsini/1.31,limb_dark="quadratic")
            res = self.data.y[instrument]-(RM.evaluate(self.data.x[instrument]) + rmfit.get_rv_curve(self.data.x[instrument],P,t0,e,w,K,plot=False,verbose=False) + gamma)    
            ax2.errorbar(phase,res,error,mfc=colors[instrument],mec='k',ecolor='k',marker='o',elinewidth=1,capsize=4,lw=0,mew=mw,markersize=ms,zorder=10)
            model = RM.evaluate(times1)    
            phase1 = ((times1-t0 + 0.5*P) % P)/P
            ax1.plot(phase1,model,color="crimson", linewidth = 3)
            ax2.axhline(0.0, color="crimson", linewidth=3)
        
        keplerian = rmfit.get_rv_curve(times1,P,t0,e,w,K,plot=False,verbose=False) + gamma 
        mmodel1 = []
        for i in range(number_models):
            if i%100 == 0: print("Sampling i =",i,end="\r")
            idx = np.random.randint(0,len(self.chain))
            chain_models = self.chain[['lam_p1','vsini_star','per_p1','t0_p1','p_p1','e_p1','omega_p1','K_p1','sma_p1','inc_p1','u1_'+instrument,'u2_'+instrument,'beta_'+instrument,'gamma_'+instrument]]
            lam, vsini, P, t0, RpRs, e, w, K, aRs, inc, u1, u2, beta, gamma = chain_models.values[idx]
            RM = rmfit.RMHirano(lam,vsini,P,t0,aRs,inc,RpRs,e,w,[u1,u2],beta,vsini/1.31,limb_dark="quadratic")
            m1 = RM.evaluate(times1) + rmfit.get_rv_curve(times1,P,t0,e,w,K,plot=False,verbose=False) + gamma
            mmodel1.append(m1-keplerian)
        mmodel1 = np.array(mmodel1)
        ax1.fill_between(phase1,np.quantile(mmodel1,0.16,axis=0),np.quantile(mmodel1,0.84,axis=0),alpha=0.1,color="r",lw=0,zorder=-1)
        ax1.fill_between(phase1,np.quantile(mmodel1,0.02,axis=0),np.quantile(mmodel1,0.98,axis=0),alpha=0.1,color="r",lw=0,zorder=-1)
        ax1.fill_between(phase1,np.quantile(mmodel1,0.0015,axis=0),np.quantile(mmodel1,0.9985,axis=0),alpha=0.1,color="r",lw=0,zorder=-1)
        ax1.set_ylabel('RV [m/s]',labelpad=10,size=45)
        ax1.tick_params(axis="both",direction="in",length=15,width=1)
        ax1.tick_params(axis="x",which="minor",direction="in",length=5,width=1)
        ax1.tick_params(axis="y",which="minor",direction="in",length=5,width=1)
        rmfit.utils.ax_apply_settings(ax1,ticksize=35)
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax2.set_xlabel('Phase',labelpad=10,size=45)
        ax2.set_ylabel('O - C',labelpad=10,size=45)
        ax2.tick_params(axis="both",direction="in",length=15,width=1)
        ax2.tick_params(axis="x",which="minor",direction="in",length=5,width=1)
        ax2.tick_params(axis="y",which="minor",direction="in",length=5,width=1)
        rmfit.utils.ax_apply_settings(ax2,ticksize=35)
        plt.tight_layout()
        fig.legend(loc="upper center",fancybox=True,bbox_to_anchor=(0.5, 1.09),shadow=False,fontsize=45,ncol=len(self.data.rm_instruments))
        plt.show()
        
    def Plot_RM_ind(self,instrument,tr_number,color = "dimgrey",ms = 15,mw = 1, number_models = 5000):
        rcParams["figure.figsize"] = (24,18)
        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(4, 3)
        ax1 = fig.add_subplot(gs[:3, :])
        ax2 = fig.add_subplot(gs[3, :], sharex=ax1)
        necessary_params = {k: self.vals[k] for k in ('lam_p1','vsini_star','per_p1','t0_p1','p_p1','e_p1','omega_p1','K_p1','sma_p1','inc_p1','u1_'+instrument,'u2_'+instrument,'beta_'+instrument,'gamma_'+instrument, 'sigma_'+instrument)}
        lam, vsini, P, t0, RpRs, e, w, K, aRs, inc, u1, u2, beta, gamma, jitter = necessary_params.values()
        times1 = np.linspace(self.data.x[instrument][0]-0.02,self.data.x[instrument][-1]+0.02,1000)
        error = np.sqrt(self.data.yerr[instrument]**2.0 + jitter**2.0)
        #phase = ((self.data.x[instrument]-t0 + 0.5*P) % P)/P
        ax1.errorbar((self.data.x[instrument]-t0-tr_number*P)*24.0,self.data.y[instrument]- rmfit.get_rv_curve(self.data.x[instrument],P,t0,e,w,K,plot=False,verbose=False) - gamma,error,label = instrument,mfc=color,mec='k',ecolor='k',marker='o',elinewidth=1,capsize=4,lw=0,mew=mw,markersize=ms,zorder=10)
        RM = rmfit.RMHirano(lam,vsini,P,t0,aRs,inc,RpRs,e,w,[u1,u2],beta,vsini/1.31,limb_dark="quadratic")
        res = self.data.y[instrument]-(RM.evaluate(self.data.x[instrument]) + rmfit.get_rv_curve(self.data.x[instrument],P,t0,e,w,K,plot=False,verbose=False) + gamma)    
        ax2.errorbar((self.data.x[instrument]-t0-tr_number*P)*24.0,res,error,mfc=color,mec='k',ecolor='k',marker='o',elinewidth=1,capsize=4,lw=0,mew=mw,markersize=ms,zorder=10)
        model = RM.evaluate(times1)    
        #phase1 = ((times1-t0 + 0.5*P) % P)/P
        ax1.plot((times1-t0-tr_number*P)*24.0,model,color="crimson", linewidth = 3)
        ax2.axhline(0.0, color="crimson", linewidth=3)
        
        keplerian = rmfit.get_rv_curve(times1,P,t0,e,w,K,plot=False,verbose=False) + gamma 
        mmodel1 = []
        for i in range(number_models):
            if i%100 == 0: print("Sampling i =",i,end="\r")
            idx = np.random.randint(0,len(self.chain))
            chain_models = self.chain[['lam_p1','vsini_star','per_p1','t0_p1','p_p1','e_p1','omega_p1','K_p1','sma_p1','inc_p1','u1_'+instrument,'u2_'+instrument,'beta_'+instrument,'gamma_'+instrument]]
            lam, vsini, P, t0, RpRs, e, w, K, aRs, inc, u1, u2, beta, gamma = chain_models.values[idx]
            RM = rmfit.RMHirano(lam,vsini,P,t0,aRs,inc,RpRs,e,w,[u1,u2],beta,vsini/1.31,limb_dark="quadratic")
            m1 = RM.evaluate(times1) + rmfit.get_rv_curve(times1,P,t0,e,w,K,plot=False,verbose=False) + gamma
            mmodel1.append(m1-keplerian)
        mmodel1 = np.array(mmodel1)
        ax1.fill_between((times1-t0-tr_number*P)*24.0,np.quantile(mmodel1,0.16,axis=0),np.quantile(mmodel1,0.84,axis=0),alpha=0.1,color="r",lw=0,zorder=-1)
        ax1.fill_between((times1-t0-tr_number*P)*24.0,np.quantile(mmodel1,0.02,axis=0),np.quantile(mmodel1,0.98,axis=0),alpha=0.1,color="r",lw=0,zorder=-1)
        ax1.fill_between((times1-t0-tr_number*P)*24.0,np.quantile(mmodel1,0.0015,axis=0),np.quantile(mmodel1,0.9985,axis=0),alpha=0.1,color="r",lw=0,zorder=-1)
        ax1.set_ylabel('RV [m/s]',labelpad=10,size=45)
        ax1.tick_params(axis="both",direction="in",length=15,width=1)
        ax1.tick_params(axis="x",which="minor",direction="in",length=5,width=1)
        ax1.tick_params(axis="y",which="minor",direction="in",length=5,width=1)
        rmfit.utils.ax_apply_settings(ax1,ticksize=35)
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax2.set_xlabel('Hours from mid-transit',labelpad=10,size=45)
        ax2.set_ylabel('O - C',labelpad=10,size=45)
        ax2.tick_params(axis="both",direction="in",length=15,width=1)
        ax2.tick_params(axis="x",which="minor",direction="in",length=5,width=1)
        ax2.tick_params(axis="y",which="minor",direction="in",length=5,width=1)
        rmfit.utils.ax_apply_settings(ax2,ticksize=35)
        plt.tight_layout()
        plt.show()
    
    def Plot_RVs_phase(self,colors,ms = 15, mw = 1, number_models = 5000):
        rcParams["figure.figsize"] = (24,18)
        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(4, 3)
        ax1 = fig.add_subplot(gs[:3, :])
        ax2 = fig.add_subplot(gs[3, :], sharex=ax1)
        for instrument in self.data.rv_instruments:
            necessary_params = {k: self.vals[k] for k in ('per_p1','t0_p1','e_p1','omega_p1','K_p1','gamma_'+instrument,'sigma_'+instrument)}
            P, t0, e, w, K, gamma, jitter = necessary_params.values()
            phase = ((self.data.x[instrument]-t0 + 0.5*P) % P)/P
            data = self.data.y[instrument] - gamma
            error = np.sqrt(self.data.yerr[instrument]**2.0 + jitter**2.0)
            ax1.errorbar(phase,data,error,label = instrument,mfc=colors[instrument],mec='k',ecolor='k',marker='o',elinewidth=1,capsize=4,lw=0,mew=mw,markersize=ms,zorder=10)
            res = data - rmfit.get_rv_curve(phase,1,0.5,e,w,K,plot=False,verbose=False)
            ax2.errorbar(phase,res,error,mfc=colors[instrument],mec='k',ecolor='k',marker='o',elinewidth=1,capsize=4,lw=0,mew=mw,markersize=ms,zorder=10)
        times1 = np.linspace(0,1,5000)
        rv_model = rmfit.get_rv_curve(times1,1,0.5,e,w,K,plot=False,verbose=False)
        ax1.plot(times1,rv_model,color="crimson",linewidth = 3)
        ax2.axhline(0,color="crimson",linewidth = 3)
        
        mmodel1 = []
        for i in range(number_models):
            if i%100 == 0: print("Sampling i =",i,end="\r")
            idx = np.random.randint(0,len(self.chain))
            chain_models = self.chain[['per_p1','t0_p1','e_p1','omega_p1','K_p1','gamma_'+instrument,'sigma_'+instrument]]
            P, t0, e, w, K, gamma, jitter = chain_models.values[idx]
            m1 = rmfit.get_rv_curve(times1,1,0.5,e,w,K,plot=False,verbose=False)
            mmodel1.append(m1)
        mmodel1 = np.array(mmodel1)
        ax1.fill_between(times1,np.quantile(mmodel1,0.16,axis=0),np.quantile(mmodel1,0.84,axis=0),alpha=0.1,color="r",lw=0,zorder=10)
        ax1.fill_between(times1,np.quantile(mmodel1,0.02,axis=0),np.quantile(mmodel1,0.98,axis=0),alpha=0.1,color="r",lw=0,zorder=10)
        ax1.fill_between(times1,np.quantile(mmodel1,0.0015,axis=0),np.quantile(mmodel1,0.9985,axis=0),alpha=0.1,color="r",lw=0,zorder=10)
        
        ax1.set_ylabel('RV [m/s]',labelpad=10,size=45)
        ax1.tick_params(axis="both",direction="in",length=15,width=1)
        ax1.tick_params(axis="x",which="minor",direction="in",length=5,width=1)
        ax1.tick_params(axis="y",which="minor",direction="in",length=5,width=1)
        rmfit.utils.ax_apply_settings(ax1,ticksize=35)
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax2.set_xlim(0,1)
        ax2.set_xticks([0.2,0.4,0.6,0.8])
        ax2.set_xlabel('Phase',labelpad=10,size=45)
        ax2.set_ylabel('O - C',labelpad=10,size=45)
        ax2.tick_params(axis="both",direction="in",length=15,width=1)
        ax2.tick_params(axis="x",which="minor",direction="in",length=5,width=1)
        ax2.tick_params(axis="y",which="minor",direction="in",length=5,width=1)
        rmfit.utils.ax_apply_settings(ax2,ticksize=35)
        plt.tight_layout()
        fig.legend(loc="upper center",fancybox=True,bbox_to_anchor=(0.5, 1.09),shadow=False,fontsize=45,ncol=len(self.data.rv_instruments))
        plt.show()
    
    def Plot_RVs_time(self,colors,ms = 15, mw = 1, number_models = 5000):
        rcParams["figure.figsize"] = (24,18)
        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(4, 3)
        ax1 = fig.add_subplot(gs[:3, :])
        ax2 = fig.add_subplot(gs[3, :], sharex=ax1)
        for instrument in self.data.rv_instruments:
            necessary_params = {k: self.vals[k] for k in ('per_p1','t0_p1','e_p1','omega_p1','K_p1','gamma_'+instrument,'sigma_'+instrument)}
            P, t0, e, w, K, gamma, jitter = necessary_params.values()
            data = self.data.y[instrument] - gamma
            error = np.sqrt(self.data.yerr[instrument]**2.0 + jitter**2.0)
            ax1.errorbar(self.data.x[instrument],data,error,label = instrument,mfc=colors[instrument],mec='k',ecolor='k',marker='o',elinewidth=1,capsize=4,lw=0,mew=mw,markersize=ms,zorder=10)
            res = data - rmfit.get_rv_curve(self.data.x[instrument],P,t0,e,w,K,plot=False,verbose=False)
            ax2.errorbar(self.data.x[instrument],res,error,mfc=colors[instrument],mec='k',ecolor='k',marker='o',elinewidth=1,capsize=4,lw=0,mew=mw,markersize=ms,zorder=10)
        
        bjds = np.array([])
        for instrument in self.data.rv_instruments:
            bjds = np.concatenate((bjds,self.data.x[instrument]))
        rv_times = np.linspace(min(bjds)-10,max(bjds)+10,5000)
        rv_model = rmfit.get_rv_curve(rv_times,P,t0,e,w,K,plot=False,verbose=False)
        ax1.plot(rv_times,rv_model,color="crimson", linewidth = 3)
        ax2.axhline(0,color="crimson",linewidth = 3)
        
        ax1.set_ylabel('RV [m/s]',labelpad=10,size=45)
        ax1.tick_params(axis="both",direction="in",length=15,width=1)
        ax1.tick_params(axis="x",which="minor",direction="in",length=5,width=1)
        ax1.tick_params(axis="y",which="minor",direction="in",length=5,width=1)
        rmfit.utils.ax_apply_settings(ax1,ticksize=35)
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax2.set_xlabel('Time [BJD]',labelpad=10,size=45)
        ax2.set_ylabel('O - C',labelpad=10,size=45)
        ax2.tick_params(axis="both",direction="in",length=15,width=1)
        ax2.tick_params(axis="x",which="minor",direction="in",length=5,width=1)
        ax2.tick_params(axis="y",which="minor",direction="in",length=5,width=1)
        rmfit.utils.ax_apply_settings(ax2,ticksize=35)
        plt.tight_layout()
        fig.legend(loc="upper center",fancybox=True,bbox_to_anchor=(0.5, 1.09),shadow=False,fontsize=45,ncol=len(self.data.rv_instruments))
        plt.show() 
        
    def Plot_LC(self,instrument,color = "cornflowerblue",ms = 15, mw = 1,xlim=(0.45,0.55),ylim1=(0.98,1.02),ylim2=(-0.01,0.01),title = "",xticks=np.linspace(0.485,0.515,7)):
        rcParams["figure.figsize"] = (16,9)
        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(3, 1)
        ax1 = fig.add_subplot(gs[:2, 0])
        ax2 = fig.add_subplot(gs[2, 0], sharex = ax1)
        necessary_params = {k: self.vals[k] for k in ('per_p1','t0_p1','p_p1','e_p1','omega_p1','K_p1','sma_p1','inc_p1','u1_'+instrument,'u2_'+instrument, 'sigma_'+instrument)}
        P, t0, RpRs, e, w, K, sma, inc, u1, u2, jitter = necessary_params.values()
        phase = ((self.data.x[instrument]-t0 + 0.5*P) % P)/P
        times1 = np.linspace(self.data.x[instrument][0],self.data.x[instrument][-1],2000)
        error = np.sqrt(self.data.yerr[instrument]**2.0 + jitter**2.0)
        ax1.errorbar(phase,self.data.y[instrument],error,marker='o',elinewidth=2,capsize=4,lw=0,mew=0.8,color=color,markersize=ms,alpha=1,zorder=5)
        model = Transit(phase, 0.5, 1, RpRs, sma, inc, e, w, u1, u2)
        ax2.errorbar(phase,self.data.y[instrument]-model,error,marker='o',elinewidth=1,capsize=2,lw=0,mew=0.5,color=color,markersize=ms,alpha=1,zorder=5)
        model_phase = np.linspace(0,1,100000)
        model = Transit(model_phase, 0.5, 1, RpRs, sma, inc, e, w, u1, u2)
        ax1.plot(model_phase,model,color = "crimson", zorder = 10, linewidth=3)
        ax2.axhline(0.0,color="crimson",zorder=10,linewidth=3)
        
        ax1.set_ylim(ylim1[0],ylim1[1])
        ax1.set_title(title, fontsize = 40)
        ax1.set_ylabel('Normalized Flux',labelpad=40,size=35)
        ax1.tick_params(axis="both",direction="in",length=15,width=1)
        ax1.tick_params(axis="x",which="minor",direction="in",length=5,width=1)
        ax1.tick_params(axis="y",which="minor",direction="in",length=5,width=1)
        rmfit.utils.ax_apply_settings(ax1,ticksize=25)
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax2.set_ylim(ylim2[0],ylim2[1])
        ax2.set_xlim(xlim[0],xlim[1])
        ax2.set_xticks(xticks)
        ax2.tick_params(axis="both",direction="in",length=15,width=1)
        ax2.tick_params(axis="x",which="minor",direction="in",length=5,width=1)
        ax2.tick_params(axis="y",which="minor",direction="in",length=5,width=1)
        rmfit.utils.ax_apply_settings(ax2,ticksize=25)
        ax2.set_xlabel('Phase',labelpad=10,size=35)
        ax2.set_ylabel('O - C ',labelpad=10,size=35)
        plt.show()
