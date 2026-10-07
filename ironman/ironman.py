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
import os
import json
import re

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

def get_vmacro(Teff):
    vmacro = 3.98 + (Teff - 5770) / 650  #Valenti & Fisher (205)
    if Teff >= 6000:
        vmacro = 3.95 * Teff / 1000 - 19.25 # Gray (1984)
    return vmacro

def create_RM_data(bjd,dct_params):
        lam = dct_params["lam_p1"]
        vsini = dct_params["vsini_star"]
        P =  dct_params["per_p1"]
        t0 = dct_params["t0_p1"]
        RpRs = dct_params["p_p1"]
        e = dct_params["e_p1"]
        w = dct_params["omega_p1"]
        teff = dct_params["teff_star"]
        R = dct_params["R_spec"]
        u1 = dct_params["u"][0]
        u2 = dct_params["u"][1]
        sma = dct_params["aRs_p1"]
        inc = dct_params["inc_p1"]
        exp_time = dct_params["exp_time"]/60./60./24.
        rv_prec = dct_params["rv_prec"]
        errors = np.ones(len(bjd))*rv_prec
        b_macro = get_vmacro(teff)
        b_inst = 300000./R
        beta = (np.sqrt(b_macro**2.0 + b_inst **2.0))
        RM = rmfit.RMHirano(lam,vsini,P,t0,sma,inc,RpRs,e,w,[u1,u2],beta,vsini/1.31,limb_dark="quadratic",supersample_factor=10,exp_time=exp_time)
        RM_model = RM.evaluate(bjd,base_error=rv_prec)
        return RM_model, errors

def get_vals(vec):
    fvec   = np.sort(vec)

    fval  = np.median(fvec)
    nn = int(np.around(len(fvec)*0.15865))

    vali,valf = fval - fvec[nn],fvec[-nn] - fval
    return fval,vali,valf

class DataOrganizer:
    def __init__(self, output, lc_time=None, lc_data=None, lc_err=None, rv_time=None, rv_data=None, rv_err=None, rm_time=None, rm_data=None, rm_err=None, verbose=True, exp_times=None, save=True):
        self.verbose = verbose
        self.exp_times = exp_times
        self.output = output

        self.lc_time = lc_time or {}
        self.lc_data = lc_data or {}
        self.lc_err = lc_err or {}
        self.rv_time = rv_time or {}
        self.rv_data = rv_data or {}
        self.rv_err = rv_err or {}
        self.rm_time = rm_time or {}
        self.rm_data = rm_data or {}
        self.rm_err = rm_err or {}

        if self.verbose:
            print("Reading data...")

        self.lc_instruments = list(self.lc_time.keys())
        self.rv_instruments = list(self.rv_time.keys())
        self.rm_instruments = list(self.rm_time.keys())

        self.x = {**{k: np.array(v) for k, v in self.lc_time.items()},
                  **{k: np.array(v) for k, v in self.rv_time.items()},
                  **{k: np.array(v) for k, v in self.rm_time.items()}}
        self.y = {**{k: np.array(v) for k, v in self.lc_data.items()},
                  **{k: np.array(v) for k, v in self.rv_data.items()},
                  **{k: np.array(v) for k, v in self.rm_data.items()}}
        self.yerr = {**{k: np.array(v) for k, v in self.lc_err.items()},
                     **{k: np.array(v) for k, v in self.rv_err.items()},
                     **{k: np.array(v) for k, v in self.rm_err.items()}}

        if self.verbose:
            print("Data Ready!")

        if save:
            self.save_data()
            self.save_exp_times()

    def save_data(self):
        """
        Saves the data in a CSV file within the output directory.
        """
        data_entries = []

        for instrument in self.lc_time:
            times = self.lc_time[instrument]
            data = self.lc_data[instrument]
            errors = self.lc_err[instrument]
            for t, d, e in zip(times, data, errors):
                data_entries.append([t, d, e, instrument, "LC"])

        for instrument in self.rv_time:
            times = self.rv_time[instrument]
            data = self.rv_data[instrument]
            errors = self.rv_err[instrument]
            for t, d, e in zip(times, data, errors):
                data_entries.append([t, d, e, instrument, "RV"])

        for instrument in self.rm_time:
            times = self.rm_time[instrument]
            data = self.rm_data[instrument]
            errors = self.rm_err[instrument]
            for t, d, e in zip(times, data, errors):
                data_entries.append([t, d, e, instrument, "RM"])

        df = pd.DataFrame(data_entries, columns=["time", "data", "error", "instrument", "type"])
        if not os.path.exists(self.output):
            os.makedirs(self.output)
        csv_path = os.path.join(self.output, 'data.csv')
        df.to_csv(csv_path, index=False)
        if self.verbose:
            print("Data saved ...")

    def save_exp_times(self):
        """
        Saves the exp_times dictionary to a JSON file within the output directory.
        """
        if self.exp_times:
            json_path = os.path.join(self.output, 'exp_times.json')
            with open(json_path, 'w') as file:
                json.dump(self.exp_times, file, indent=4)
            if self.verbose:
                print("exp_times saved ...")

    @classmethod
    def from_input_directory(cls, input, verbose=True, save=False):
        """
        Initializes the DataOrganizer class from an input directory.

        Parameters:
        input (str): Path to the input directory containing 'data.csv' and 'exp_times.json'.

        Returns:
        DataOrganizer: An instance of the DataOrganizer class initialized with the data from the input directory.
        """
        csv_path = os.path.join(input, 'data.csv')
        df = pd.read_csv(csv_path)

        json_path = os.path.join(input, 'exp_times.json')
        with open(json_path, 'r') as file:
            exp_times = json.load(file)

        lc_time, lc_data, lc_err = {}, {}, {}
        rv_time, rv_data, rv_err = {}, {}, {}
        rm_time, rm_data, rm_err = {}, {}, {}

        grouped = df.groupby(['type', 'instrument'])

        for (data_type, instrument), group in grouped:
            if data_type == 'LC':
                lc_time[instrument] = group['time'].tolist()
                lc_data[instrument] = group['data'].tolist()
                lc_err[instrument] = group['error'].tolist()
            elif data_type == 'RV':
                rv_time[instrument] = group['time'].tolist()
                rv_data[instrument] = group['data'].tolist()
                rv_err[instrument] = group['error'].tolist()
            elif data_type == 'RM':
                rm_time[instrument] = group['time'].tolist()
                rm_data[instrument] = group['data'].tolist()
                rm_err[instrument] = group['error'].tolist()

        return cls(
            output=input,
            lc_time=lc_time,
            lc_data=lc_data,
            lc_err=lc_err,
            rv_time=rv_time,
            rv_data=rv_data,
            rv_err=rv_err,
            rm_time=rm_time,
            rm_data=rm_data,
            rm_err=rm_err,
            verbose=verbose,
            exp_times=exp_times,
            save=save
        )

class Priors:
    def __init__(self, file, data, save=True, verbose=True):
        """
        Initialize the Priors class.

        Parameters:
        file (str): Path to the priors file.
        data (object): DataOrganizer object associated with the priors.
        save (bool): Flag to enable saving the priors to a file.
        verbose (bool): Flag to enable verbose output.
        """
        self.data = data
        self.verbose = verbose
        self.save = save
        self.dct = {}
        self.parameters = []
        self.fixed_parameters = []
        self.varying_parameters = []

        self._read_and_classify_priors(file)

        if self.verbose:
            self._print_fixed_parameters()

        self._check_special_parameters()

        if self.save:
            self.save_priors()

    def _read_and_classify_priors(self, file):
        with open(file) as priors_file:
            for line_number, line in enumerate(priors_file, start=1):
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue

                elements = line.split()
                if len(elements) != 3:
                    raise ValueError(
                        f"Invalid prior on line {line_number}: {line}"
                    )

                parameter, prior_type, specification = elements
                if parameter in self.dct:
                    raise ValueError(f"Duplicate prior: {parameter}")

                hyperparameters = tuple(
                    map(float, specification.split(","))
                )
                if prior_type == "FIXED" and len(hyperparameters) != 1:
                    raise ValueError(
                        f"FIXED prior requires one value: {parameter}"
                    )

                self.dct[parameter] = [prior_type, hyperparameters]
                self.parameters.append(parameter)

                if prior_type == "FIXED":
                    self.fixed_parameters.append(parameter)
                else:
                    self.varying_parameters.append(parameter)

    def _print_fixed_parameters(self):
        """
        Prints the fixed parameters if verbose is enabled.
        """
        print("Priors dictionary ready...")
        print("Detecting fixed parameters...")
        for param in self.fixed_parameters:
            print(f"Fixed {param} detected!")

    def _check_special_parameters(self):
        """
        Checks for the presence of specific parameters or combinations of parameters.
        """
        param_set = set(self.parameters)

        self.m_star_param = "r_star" in param_set and "m_star" in param_set
        self.rho_param = self.m_star_param or "rho_star" in param_set
        self.b_param = "b_p1" in param_set

        if "e_p1" in param_set and "omega_p1" in param_set:
            print("e, omega parametrization detected")
            self.secosw_sesinw_param = False
        elif "secosomega_p1" in param_set and "sesinomega_p1" in param_set:
            print("secosomega, sesinomega parametrization detected")
            self.secosw_sesinw_param = True
        else:
            raise ValueError("Something went wrong with the eccentricity parametrization. ")
        
        #if "lam_p1" in param_set:
        if self.data.rm_data != {}:    

            required_true_obliquity_params = {"cosi_star", "Prot_star", "r_star"}
            required_cospsi_obliquity_params = {"cospsi_p1", "Prot_star", "r_star"}

            if "vsini_star" in param_set and "lam_p1" in param_set:
                self.true_obliquity_param = False
                self.cospsi_param = False
            elif required_true_obliquity_params <= param_set:
                self.true_obliquity_param = True
                self.cospsi_param = False
            elif required_cospsi_obliquity_params <= param_set and "cospsi_p1" in param_set:
                self.true_obliquity_param = False
                self.cospsi_param = True
            else:
                self.true_obliquity_param = False
                self.cospsi_param = False
                raise ValueError(
                    "Something went wrong with the obliquity parametrization. "
                    "Please check the parameters. For deriving the true obliquity include cosi_star, Prot_star, r_star, and lam_p1"
                    "Or cospsi_p1, Prot_star, r_star and vsini"
                    "For the sky-projected obliquity include vsini_star and lam_p1"
                )
        else:
            self.rm_param = False
            self.true_obliquity_param = False
            self.cospsi_param = False

        #if "inc_p1" in param_set:
        if self.data.lc_data != {}:
            self.transit_param = True
        else:
            self.transit_param = False
    
    def save_priors(self):
        """
        Saves the priors dictionary to a file in the output folder.
        """
        output_folder = self.data.output
        os.makedirs(output_folder, exist_ok=True)

        output_file = os.path.join(output_folder, 'priors.txt')

        with open(output_file, 'w') as file:
            for param, (prior_type, hyperparameters) in self.dct.items():
                hyp_str = ",".join(map(str, hyperparameters))
                file.write(f"{param}\t{prior_type}\t{hyp_str}\n")

        if self.verbose:
            print("Priors saved ...")

class Fit:
    def __init__(
        self, input=None, data=None, priors=None, ta=None,
        verbose=True, ecclim=0.95
    ):
        """Initialize a fit; new fits default to the mean RV/RM epoch."""
        self.verbose = verbose
        self.ecclim = ecclim

        if self.verbose:
            print(
                f"The code is working with an eccentricity "
                f"limit of {ecclim}"
            )

        if input:
            self._init_from_input(input)
        elif data and priors:
            self.data = data
            self.priors = priors
        else:
            raise ValueError(
                "Either 'input' must be provided or both "
                "'data' and 'priors' must be provided."
            )

        loaded_chain = hasattr(self, "chain")
        saved_ta = self.results.get("ta") if loaded_chain else None

        if ta is not None:
            self.ta = float(ta)

            if saved_ta is not None and self.ta != float(saved_ta):
                raise ValueError(
                    "The loaded posterior uses a different ta. "
                    "Changing its reference epoch requires "
                    "transforming the trend parameters."
                )

        elif saved_ta is not None:
            self.ta = float(saved_ta)

        elif loaded_chain:
            # Legacy outputs used the old default and did not save ta.
            self.ta = self._get_min_rv_time()

        else:
            self.ta = self._get_reference_time()

        if not np.isfinite(self.ta):
            raise ValueError("ta must be finite")

        self.ndim = len(self.priors.varying_parameters)
        self._build_instrument_parameter_map()

        if self.verbose:
            print("Fit class initialized.")

    def _init_from_input(self, input):
        """
        Initialize the Fit class from input folder.
    
        Parameters:
        input (str): Path to the input folder containing data.csv, exp_times.json, and priors.txt.
        """
        data_file = os.path.join(input, 'data.csv')
        exp_times_file = os.path.join(input, 'exp_times.json')
        priors_file = os.path.join(input, 'priors.txt')
        flatchain_file = os.path.join(input, 'flatchain.csv')
        posteriors_file = os.path.join(input, 'posteriors.txt')
        results_file = os.path.join(input, 'results.json')
    
        if not os.path.exists(data_file) or not os.path.exists(exp_times_file) or not os.path.exists(priors_file):
            raise FileNotFoundError("The input folder must contain data.csv, exp_times.json, and priors.txt files.")
    
        self.data = DataOrganizer.from_input_directory(input, verbose=False,save=False)
        self.priors = Priors(priors_file, self.data, verbose=False,save=False)
        if self.verbose:
            print("DataOrganizer and Priors initialized from input folder.")
    
        if os.path.exists(flatchain_file):
            self.chain = pd.read_csv(flatchain_file)
            with open(results_file, 'r') as file:
                self.results = json.load(file)
            if self.verbose:
                print(f"Loaded flatchain from {flatchain_file}")
        else:
            print("No flatchain.csv file detected")
    
        if os.path.exists(posteriors_file):
            posteriors_df = pd.read_csv(posteriors_file, sep='\t')
            self.vals = posteriors_df.set_index('parameter')['median'].to_dict()
            self.err_up = posteriors_df.set_index('parameter')['err_up'].to_dict()
            self.err_down = posteriors_df.set_index('parameter')['err_down'].to_dict()
            if self.verbose:
                print(f"Loaded posteriors from {posteriors_file}")
        else:
            print("No posteriors.txt file detected")

    def _get_reference_time(self):
        """Mean observation time over all active RV and RM instruments."""
        instruments = dict.fromkeys(
            self.data.rv_instruments + self.data.rm_instruments
        )

        arrays = []
        for instrument in instruments:
            times = np.asarray(
                self.data.x[instrument], dtype=float
            ).ravel()
            if times.size:
                arrays.append(times)

        if not arrays:
            if self.verbose:
                print("No RV/RM time data available in DataOrganizer.")
            return 2450000.0

        times = np.concatenate(arrays)
        if not np.all(np.isfinite(times)):
            raise ValueError(
                "RV/RM observation times must be finite"
            )

        return float(np.mean(times))
    
    def _get_min_rv_time(self):
        """
        Get the minimum RV time from the DataOrganizer object.

        Returns:
        float: The minimum time for the RV measurements.
        """
        if self.data.rv_time:
            min_rv_time = min(min(times) for times in self.data.rv_time.values())
        elif self.data.rm_time:
            min_rv_time = min(min(times) for times in self.data.rm_time.values())
        else:
            min_rv_time = 2450000.0
            print("No RV/RM time data available in DataOrganizer.")
        return min_rv_time

    def _build_instrument_parameter_map(self):
        families = {
            "q1", "q2", "beta", "gamma", "sigma",
            "gammadot", "gammadotdot",
        }
        instruments = list(self.data.x)
        mapping = {instrument: {} for instrument in instruments}

        for source in self.priors.parameters:
            family, separator, suffix = source.partition("_")
            if not separator or family not in families:
                continue

            # A complete instrument name takes precedence.
            if suffix in mapping:
                members = [suffix]
            else:
                matches = []
                for instrument in instruments:
                    pattern = (
                        r"(?:^|_)("
                        + re.escape(instrument)
                        + r")(?=_|$)"
                    )
                    occurrences = list(re.finditer(pattern, suffix))

                    if len(occurrences) > 1:
                        raise ValueError(
                            f"Repeated instrument {instrument} in {source}"
                        )

                    if occurrences:
                        start, end = occurrences[0].span(1)
                        matches.append((start, end, instrument))

                matches.sort()
                for left, right in zip(matches, matches[1:]):
                    if left[1] > right[0]:
                        raise ValueError(
                            f"Ambiguous instrument names in {source}"
                        )

                members = [
                    instrument for _, _, instrument in matches
                ]

            for instrument in members:
                previous = mapping[instrument].get(family)
                if previous is not None:
                    raise ValueError(
                        f"Multiple priors for {family}_{instrument}: "
                        f"{previous}, {source}"
                    )
                mapping[instrument][family] = source

        required = {
            "lc": ("q1", "q2", "sigma"),
            "rv": ("gamma", "sigma"),
            "rm": ("q1", "q2", "beta", "gamma", "sigma"),
        }
        for category, families in required.items():
            for instrument in getattr(
                self.data, category + "_instruments"
            ):
                missing = [
                    family for family in families
                    if family not in mapping[instrument]
                ]
                if missing:
                    raise ValueError(
                        f"Missing priors for {instrument}: "
                        f"{', '.join(missing)}"
                    )

        self.instrument_parameter_map = mapping

    def _expand_instrument_parameters(self, values):
        expanded = dict(values)

        for instrument, families in (
            self.instrument_parameter_map.items()
        ):
            for family, source in families.items():
                expanded[f"{family}_{instrument}"] = values[source]

        instruments = dict.fromkeys(
            self.data.lc_instruments + self.data.rm_instruments
        )
        for instrument in instruments:
            (
                expanded[f"u1_{instrument}"],
                expanded[f"u2_{instrument}"],
            ) = u1_u2_from_q1_q2(
                expanded[f"q1_{instrument}"],
                expanded[f"q2_{instrument}"],
            )

        return expanded
        
    def priors_transform(self, params):
        """
        Transforms parameters according to their prior distributions.
    
        Inputs:
        params: Tuple of parameters to be transformed.
    
        Outputs:
        tuple: Tuple of Transformed parameters.
        """
        transformed_values = []
        for index, parameter in enumerate(self.priors.varying_parameters):
            dist = self.priors.dct[parameter][0]
            value = params[index]
    
            if dist == "uniform":
                transformed = transform_uniform(value, self.priors.dct[parameter][1])
            elif dist == "loguniform":
                transformed = transform_loguniform(value, self.priors.dct[parameter][1])
            elif dist == "normal":
                transformed = transform_normal(value, self.priors.dct[parameter][1])
            elif dist == "beta":
                transformed = transform_beta(value, self.priors.dct[parameter][1])
            elif dist == "exponential":
                transformed = transform_exponential(value, self.priors.dct[parameter][1])
            elif dist == "truncatednormal":
                transformed = transform_truncated_normal(value, self.priors.dct[parameter][1])
            elif dist == "modifiedjeffreys":
                transformed = transform_modifiedjeffreys(value, self.priors.dct[parameter][1])
            else:
                raise ValueError(f"Unsupported distribution type: {dist}")
    
            transformed_values.append(transformed)
    
        return tuple(transformed_values)
            
    def get_rv_model(self, dct_params, inst):
        """
        Gets the model RV curve using rmfit/radvel
    
        Inputs:
        dct_params (dictionary): Dictionary with the values of the parameters.
        inst (string): Name of the instrument to model the RV
    
        Outputs:
        rv_model (array): RV model
        """
        bjd = self.data.x[inst]
        P = dct_params["per_p1"]
        t0 = dct_params["t0_p1"]
        e = dct_params["e_p1"]
        w = dct_params["omega_p1"]
        K = dct_params["K_p1"]
        gamma = dct_params["gamma_" + inst]
        gammadot = dct_params.get("gammadot_" + inst, dct_params.get("gammadot", 0))
        gammadotdot = dct_params.get("gammadotdot_" + inst, dct_params.get("gammadotdot", 0))
    
        rv_trend = gammadot * (bjd - self.ta) + gammadotdot * (bjd - self.ta) ** 2.0
        rv_model = rmfit.get_rv_curve(bjd, P, t0, e, w, K, plot=False, verbose=False) + gamma + rv_trend
    
        return rv_model

    def get_lc_model(self, dct_params, inst):
        """
        Gets the model light curve using batman
    
        Inputs:
        dct_params (dictionary): Dictionary with the values of the parameters.
        inst (string): Name of the instrument to model the light curve
    
        Outputs:
        flux_lc (array): Light curve model
        """
        bjd = self.data.x[inst]
        tr_model = batman.TransitParams()
    
        tr_model.t0 = dct_params["t0_p1"]
        tr_model.per = dct_params["per_p1"]
        tr_model.rp = dct_params["p_p1"]
        tr_model.a = dct_params["aRs_p1"]
        tr_model.inc = dct_params["inc_p1"]
        tr_model.ecc = dct_params["e_p1"]
        tr_model.w = dct_params["omega_p1"]
        tr_model.u = [dct_params["u1_" + inst], dct_params["u2_" + inst]]
        tr_model.limb_dark = "quadratic"
        
        exp_time = self.data.exp_times.get(inst, False)
        if exp_time != False:
            m = batman.TransitModel(tr_model, bjd, supersample_factor=10, exp_time=float(exp_time), transittype='primary')
        else:
            m = batman.TransitModel(tr_model, bjd, transittype='primary')
    
        flux_lc = m.light_curve(tr_model)
        return flux_lc

    def get_rm_model(self, dct_params, inst):
        """
        Gets the model RM effect using rmfit/Hirano et al. 2010
    
        Inputs:
        dct_params (dictionary): Dictionary with the values of the parameters.
        inst (string): Name of the instrument to model the RM effect
    
        Outputs:
        RM_model (array): Model RM effect
        """
        bjd = self.data.x[inst]
        lam = dct_params["lam_p1"]
        vsini = dct_params["vsini_star"]
        P = dct_params["per_p1"]
        t0 = dct_params["t0_p1"]
        RpRs = dct_params["p_p1"]
        e = dct_params["e_p1"]
        w = dct_params["omega_p1"]
        K = dct_params["K_p1"]
        beta = dct_params["beta_" + inst]
        gamma = dct_params["gamma_" + inst]
        u1 = dct_params["u1_" + inst]
        u2 = dct_params["u2_" + inst]
        sma = dct_params["aRs_p1"]
        inc = dct_params["inc_p1"]
    
        gammadot = dct_params.get("gammadot_" + inst, dct_params.get("gammadot", 0))
        gammadotdot = dct_params.get("gammadotdot_" + inst, dct_params.get("gammadotdot", 0))
    
        rv_trend = gammadot * (bjd - self.ta) + gammadotdot * (bjd - self.ta) ** 2.0
    
        exp_time = self.data.exp_times.get(inst)
        if exp_time is None:
            raise ValueError(f"No exp_time for instrument: {inst}")
    
        RM = rmfit.RMHirano(lam, vsini, P, t0, sma, inc, RpRs, e, w, [u1, u2], beta, vsini / 1.31,limb_dark="quadratic", supersample_factor=10, exp_time=float(exp_time))
        RM_model = RM.evaluate(bjd) + rmfit.get_rv_curve(bjd, P, t0, e, w, K, plot=False, verbose=False) + gamma + rv_trend
    
        return RM_model

    def get_model_for_inst(self, inst, dcti):
        """
        Call the right model function for the instrument (LC, RV, RM)
    
        Inputs:
        dcti (dictionary): Dictionary with the values of the parameters.
        inst (string): Name of the instrument
    
        Outputs:
        model_func (array): Model for the instrument
        """
        model_functions = {
            "lc": self.get_lc_model,
            "rv": self.get_rv_model,
            "rm": self.get_rm_model
        }
    
        for inst_type, model_func in model_functions.items():
            if inst in getattr(self.data, f"{inst_type}_instruments"):
                return model_func(dcti, inst)
    
        raise ValueError(f"Instrument {inst} not found in any instrument category.")

    def _get_parameter_values(self, params):
        if len(params) != len(self.priors.varying_parameters):
            raise ValueError(
                "Sampler vector does not match the varying parameters"
            )

        values = dict(zip(self.priors.varying_parameters, params))

        for parameter in self.priors.fixed_parameters:
            values[parameter] = float(
                self.priors.dct[parameter][1][0]
            )

        return self._expand_instrument_parameters(values)
                
    def LogLikelihood(self, params):
        """
        Estimates the LogLikelihood
    
        Inputs:
        params (tuple): Valus for the parameters from the Sampler
    
        Outputs:
        ll (float): LogLikelihood value of the model
        """
        ll = 0
        dct_i = self._get_parameter_values(params)

        if self.priors.secosw_sesinw_param:
            dct_i["e_p1"] = (dct_i["secosomega_p1"]**2.0) + (dct_i["sesinomega_p1"]**2.0)
            dct_i["omega_p1"] = np.arctan2(dct_i["sesinomega_p1"], dct_i['secosomega_p1'])*180.0/np.pi
        if dct_i["e_p1"] > self.ecclim or dct_i["e_p1"] < 0.0:
            return -np.inf
        if self.priors.m_star_param:
            volume = 4.0/3.0*np.pi*(dct_i["r_star"]**3.0)*(u.Rsun**3.0)
            dct_i["rho_star"] = (dct_i["m_star"]*u.Msun/volume).to(u.kg/u.m/u.m/u.m).value
        if self.priors.rho_param:     
            dct_i["aRs_p1"] = ((c.G*((dct_i["per_p1"]*u.d)**2.0)*(dct_i["rho_star"]*u.kg/u.m/u.m/u.m)/3.0/np.pi)**(1./3.)).cgs.value
        if self.priors.b_param:    
            dct_i["inc_p1"] = np.arccos(dct_i["b_p1"]/dct_i["aRs_p1"]*((1.0+dct_i["e_p1"]*np.sin(dct_i["omega_p1"]*np.pi/180.0))/(1.0 - dct_i["e_p1"]**2.0)))*180.0/np.pi
        if self.priors.transit_param and np.isnan(dct_i["inc_p1"]):
            return -np.inf
        if self.priors.true_obliquity_param:
            veq = (2.0*np.pi*dct_i["r_star"]*u.Rsun/dct_i["Prot_star"]/u.d).to(u.km/u.s).value
            dct_i["vsini_star"] = veq*np.sqrt(1.0-dct_i["cosi_star"]**2.0)
        if self.priors.cospsi_param:
            veq = (2.0*np.pi*dct_i["r_star"]*u.Rsun/dct_i["Prot_star"]/u.d).to(u.km/u.s).value
            if veq < dct_i["vsini_star"]:
                return -np.inf
            sini_star = dct_i["vsini_star"]/veq
            cosi_star = np.sqrt(1.0 - sini_star**2.0)
            coslambda = (dct_i["cospsi_p1"] - cosi_star * np.cos(np.deg2rad(dct_i["inc_p1"]))) / (sini_star * np.sin(np.deg2rad(dct_i["inc_p1"])))
            sinlambda = ((cosi_star * np.sin(np.deg2rad(dct_i["inc_p1"]))) - (np.cos(np.deg2rad(dct_i["inc_p1"])) * sini_star * dct_i["cospsi_p1"])) / (np.sin(np.deg2rad(dct_i["inc_p1"])) * sini_star)
            if (coslambda < -1. or coslambda > 1.) or (sinlambda < -1. or sinlambda > 1.):
                return -np.inf
            dct_i["lam_p1"] = np.arctan2(sinlambda, coslambda)*180.0/np.pi

        for inst in self.data.x:
            jitter = float(dct_i["sigma_"+inst])
            inst_data = self.data.y[inst]
            inst_err = self.data.yerr[inst]
            inst_model = self.get_model_for_inst(inst,dct_i)
            inst_err = np.sqrt(inst_err**2.0 + jitter**2.0)
            ll += rmfit.likelihood.ll_normal_ev_py(inst_data,inst_model,inst_err)
        return ll

    def _initialize_posterior(self, samples):
        samples = np.asarray(samples)
        if (
            samples.ndim != 2
            or samples.shape[1]
            != len(self.priors.varying_parameters)
        ):
            raise ValueError(
                "Posterior samples do not match the varying parameters"
            )

        values = {
            parameter: samples[:, index].copy()
            for index, parameter in enumerate(
                self.priors.varying_parameters
            )
        }
        for parameter in self.priors.fixed_parameters:
            values[parameter] = np.full(
                len(samples),
                self.priors.dct[parameter][1][0],
            )

        self.chain = pd.DataFrame(
            self._expand_instrument_parameters(values)
        )

        fixed = set(self.priors.fixed_parameters)
        for instrument, families in (
            self.instrument_parameter_map.items()
        ):
            for family, source in families.items():
                if source in fixed:
                    fixed.add(f"{family}_{instrument}")

        instruments = dict.fromkeys(
            self.data.lc_instruments + self.data.rm_instruments
        )
        for instrument in instruments:
            if (
                f"q1_{instrument}" in fixed
                and f"q2_{instrument}" in fixed
            ):
                fixed.update(
                    (f"u1_{instrument}", f"u2_{instrument}")
                )

        self.vals, self.err_down, self.err_up = {}, {}, {}

        for parameter in self.chain.columns:
            if parameter in fixed:
                self.vals[parameter] = float(
                    self.chain[parameter].iloc[0]
                )
                self.err_down[parameter] = np.nan
                self.err_up[parameter] = np.nan
            else:
                median, lower, upper = get_vals(
                    self.chain[parameter].to_numpy()
                )
                self.vals[parameter] = median
                self.err_down[parameter] = lower
                self.err_up[parameter] = upper
    
    def run(self, n_live=600, bound='multi', sample='rwalk', nthreads=2):
        """
        Run the dynamic nested sampling and save the results.

        Inputs:
        n_live (int): Number of live points.
        bound (str): Method used to bound the live points.
        sample (str): Method used to sample new points.
        nthreads (int): Number of threads to use for parallel processing.

        Outputs:
        Saves results.json, flatchain.csv, and posteriors.txt.
        Returns the posterior chain.
        """
        if self.data.verbose:
            print(
                f"Running dynesty with {n_live} nlive "
                f"and {nthreads} threads"
            )

        try:
            with Pool(processes=nthreads - 1) as executor:
                sampler = DynamicNestedSampler(
                    self.LogLikelihood,
                    self.priors_transform,
                    self.ndim,
                    bound=bound,
                    sample=sample,
                    nlive=n_live,
                    pool=executor,
                    queue_size=nthreads,
                    bootstrap=0,
                )
                sampler.run_nested()
                res = sampler.results

                weights = np.exp(res['logwt'] - res['logz'][-1])
                self.chain = resample_equal(res.samples, weights)

                logZ = res.logz[-1]
                logZ_err = res.logzerr[-1]
                n_eff = res.eff

                N = sum(len(times) for times in self.data.x.values())
                k = self.ndim
                max_loglike = np.max(res.logl)

                BIC = k * np.log(N) - 2. * max_loglike
                AIC = 2. * k - 2. * max_loglike

                chi2 = -2 * max_loglike
                dof = N - k
                reduced_chi2 = chi2 / dof

                self.results = {
                    'logZ': logZ,
                    'logZ_err': logZ_err,
                    'BIC': BIC,
                    'AIC': AIC,
                    'reduced_chi2': reduced_chi2,
                    'n_eff': n_eff,
                    'ta': self.ta,
                }

                output_path = os.path.join(
                    self.data.output, 'results.json'
                )
                with open(output_path, 'w') as f:
                    json.dump(self.results, f, indent=4)

                if self.data.verbose:
                    print(f"Saved fit results to {output_path}")

                # New: build the chain and instrumental parameters by name.
                self._initialize_posterior(self.chain)

                if self.priors.secosw_sesinw_param:
                    self.chain["e_p1"] = (
                        self.chain["secosomega_p1"].values**2.0
                        + self.chain["sesinomega_p1"].values**2.0
                    )
                    val, mi, ma = get_vals(self.chain["e_p1"].values)
                    (
                        self.vals["e_p1"],
                        self.err_up["e_p1"],
                        self.err_down["e_p1"],
                    ) = val, ma, mi

                    self.chain["omega_p1"] = (
                        np.arctan2(
                            self.chain["sesinomega_p1"].values,
                            self.chain["secosomega_p1"].values,
                        )
                        * 180.0 / np.pi
                    )
                    val, mi, ma = get_vals(
                        self.chain["omega_p1"].values
                    )
                    (
                        self.vals["omega_p1"],
                        self.err_up["omega_p1"],
                        self.err_down["omega_p1"],
                    ) = val, ma, mi

                if self.priors.m_star_param:
                    volume = (
                        4.0 / 3.0 * np.pi
                        * self.chain["r_star"].values**3.0
                        * u.Rsun**3.0
                    )
                    self.chain["rho_star"] = (
                        self.chain["m_star"].values * u.Msun / volume
                    ).to(u.kg / u.m / u.m / u.m).value

                    val, mi, ma = get_vals(
                        self.chain["rho_star"].values
                    )
                    (
                        self.vals["rho_star"],
                        self.err_up["rho_star"],
                        self.err_down["rho_star"],
                    ) = val, ma, mi

                if self.priors.rho_param:
                    self.chain["aRs_p1"] = (
                        (
                            c.G
                            * (self.chain["per_p1"].values * u.d)**2.0
                            * (
                                self.chain["rho_star"].values
                                * u.kg / u.m / u.m / u.m
                            )
                            / 3.0 / np.pi
                        )**(1. / 3.)
                    ).cgs.value

                    val, mi, ma = get_vals(
                        self.chain["aRs_p1"].values
                    )
                    (
                        self.vals["aRs_p1"],
                        self.err_up["aRs_p1"],
                        self.err_down["aRs_p1"],
                    ) = val, ma, mi

                if self.priors.b_param:
                    self.chain["inc_p1"] = (
                        np.arccos(
                            self.chain["b_p1"].values
                            / self.chain["aRs_p1"].values
                            * (
                                (
                                    1.0
                                    + self.chain["e_p1"].values
                                    * np.sin(
                                        self.chain["omega_p1"].values
                                        * np.pi / 180.0
                                    )
                                )
                                / (
                                    1.0
                                    - self.chain["e_p1"].values**2.0
                                )
                            )
                        )
                        * 180.0 / np.pi
                    )
                    val, mi, ma = get_vals(
                        self.chain["inc_p1"].values
                    )
                    (
                        self.vals["inc_p1"],
                        self.err_up["inc_p1"],
                        self.err_down["inc_p1"],
                    ) = val, ma, mi

                if self.priors.true_obliquity_param:
                    self.chain["veq_star"] = (
                        2.0 * np.pi
                        * self.chain["r_star"].values * u.Rsun
                        / self.chain["Prot_star"].values / u.d
                    ).to(u.km / u.s).value

                    self.chain["vsini_star"] = (
                        self.chain["veq_star"].values
                        * np.sqrt(
                            1.0 - self.chain["cosi_star"].values**2.0
                        )
                    )

                    self.chain["psi_p1"] = (
                        np.arccos(
                            self.chain["cosi_star"].values
                            * np.cos(
                                self.chain["inc_p1"].values
                                * np.pi / 180.0
                            )
                            + np.sqrt(
                                1.0
                                - self.chain["cosi_star"].values**2.0
                            )
                            * np.sin(
                                self.chain["inc_p1"].values
                                * np.pi / 180.0
                            )
                            * np.cos(
                                self.chain["lam_p1"].values
                                * np.pi / 180.0
                            )
                        )
                        * 180.0 / np.pi
                    )

                    val, mi, ma = get_vals(
                        self.chain["veq_star"].values
                    )
                    (
                        self.vals["veq_star"],
                        self.err_up["veq_star"],
                        self.err_down["veq_star"],
                    ) = val, ma, mi

                    val, mi, ma = get_vals(
                        self.chain["vsini_star"].values
                    )
                    (
                        self.vals["vsini_star"],
                        self.err_up["vsini_star"],
                        self.err_down["vsini_star"],
                    ) = val, ma, mi

                    val, mi, ma = get_vals(
                        self.chain["psi_p1"].values
                    )
                    (
                        self.vals["psi_p1"],
                        self.err_up["psi_p1"],
                        self.err_down["psi_p1"],
                    ) = val, ma, mi

                    self.chain["inc_star"] = (
                        np.arccos(self.chain["cosi_star"])
                        * 180.0 / np.pi
                    )
                    val, mi, ma = get_vals(
                        self.chain["inc_star"].values
                    )
                    (
                        self.vals["inc_star"],
                        self.err_up["inc_star"],
                        self.err_down["inc_star"],
                    ) = val, ma, mi

                if self.priors.cospsi_param:
                    self.chain["veq_star"] = (
                        2.0 * np.pi
                        * self.chain["r_star"].values * u.Rsun
                        / self.chain["Prot_star"].values / u.d
                    ).to(u.km / u.s).value

                    self.chain["psi_p1"] = (
                        np.arccos(self.chain["cospsi_p1"].values)
                        * 180.0 / np.pi
                    )

                    val, mi, ma = get_vals(
                        self.chain["veq_star"].values
                    )
                    (
                        self.vals["veq_star"],
                        self.err_up["veq_star"],
                        self.err_down["veq_star"],
                    ) = val, ma, mi

                    val, mi, ma = get_vals(
                        self.chain["psi_p1"].values
                    )
                    (
                        self.vals["psi_p1"],
                        self.err_up["psi_p1"],
                        self.err_down["psi_p1"],
                    ) = val, ma, mi

                    sini_star = (
                        self.chain["vsini_star"].values
                        / self.chain["veq_star"].values
                    )
                    cosi_star = np.sqrt(1.0 - sini_star**2.0)

                    self.chain["cosi_star"] = cosi_star
                    self.chain["inc_star"] = (
                        np.arccos(cosi_star) * 180.0 / np.pi
                    )

                    val, mi, ma = get_vals(
                        self.chain["cosi_star"].values
                    )
                    (
                        self.vals["cosi_star"],
                        self.err_up["cosi_star"],
                        self.err_down["cosi_star"],
                    ) = val, ma, mi

                    val, mi, ma = get_vals(
                        self.chain["inc_star"].values
                    )
                    (
                        self.vals["inc_star"],
                        self.err_up["inc_star"],
                        self.err_down["inc_star"],
                    ) = val, ma, mi

                    coslambda = (
                        (
                            self.chain["cospsi_p1"].values
                            - cosi_star
                            * np.cos(
                                self.chain["inc_p1"].values
                                * np.pi / 180.0
                            )
                        )
                        / (
                            sini_star
                            * np.sin(
                                self.chain["inc_p1"].values
                                * np.pi / 180.0
                            )
                        )
                    )
                    sinlambda = (
                        (
                            cosi_star
                            * np.sin(
                                self.chain["inc_p1"].values
                                * np.pi / 180.0
                            )
                            - np.cos(
                                self.chain["inc_p1"].values
                                * np.pi / 180.0
                            )
                            * sini_star
                            * self.chain["cospsi_p1"].values
                        )
                        / (
                            sini_star
                            * np.sin(
                                self.chain["inc_p1"].values
                                * np.pi / 180.0
                            )
                        )
                    )
                    self.chain["lam_p1"] = (
                        np.arctan2(sinlambda, coslambda)
                        * 180.0 / np.pi
                    )
                    val, mi, ma = get_vals(
                        self.chain["lam_p1"].values
                    )
                    (
                        self.vals["lam_p1"],
                        self.err_up["lam_p1"],
                        self.err_down["lam_p1"],
                    ) = val, ma, mi

                output_path = os.path.join(
                    self.data.output, 'flatchain.csv'
                )
                self.chain.to_csv(output_path, index=False)

                combined_data = []
                for parameter in self.vals:
                    combined_data.append({
                        'parameter': parameter,
                        'median': self.vals[parameter],
                        'err_up': self.err_up[parameter],
                        'err_down': self.err_down[parameter],
                    })

                output_path = os.path.join(
                    self.data.output, 'posteriors.txt'
                )
                df = pd.DataFrame(combined_data)
                df.to_csv(output_path, index=False, sep="\t")

                if self.data.verbose:
                    print(f"Saved posteriors to {output_path}")

                return self.chain

        except Exception as e:
            print(f"An error occurred during the fitting process: {e}")
            return None

class Results:
    def __init__(self, fit):
        """
        Initialize the Results class.

        Parameters:
        fit (Fit): Fit object containing the fitting results.
        """
        self.fit = fit

        if not hasattr(self.fit, 'chain'):
            raise ValueError("The Fit object does not have a .chain attribute. Please run the fitting process first.")

        self.chain = self.fit.chain
        self.vals = self.fit.vals
        self.err_down = self.fit.err_down
        self.err_up = self.fit.err_up
        self.data = self.fit.data
        self.priors = self.fit.priors
        self.results = self.fit.results
        
        if self.fit.verbose:
            print("Results class initialized.")

    def print_mass_radius_rho_sma_planet(self, rstar=1.0, rstar_err=0.1, mstar=1.0, mstar_err=0.1, r_units=u.Rjup, sma_units=u.AU, m_units=u.Mjup):
        """
        Calculate and print the planet's mass, radius, density, and semi-major axis. If r_star and/or m_star are not sampled they must be added as inputs
    
        Parameters:
        rstar (float) optional: Mean stellar radius in solar radii. Only if r_star not sampled
        rstar_err (float) optional: Error in stellar radius. Only if r_star not sampled
        mstar (float) optional: Mean stellar mass in solar masses. Only if m_star not sampled
        mstar_err (float) optional: Error in stellar mass. Only if m_star not sampled
        r_units (Astropy Unit): Desired output units for the planet's radius.
        sma_units (Astropy Unit): Desired output units for the semi-major axis.
        m_units (Astropy Unit): Desired output units for the planet's mass.
        """
        chain = self.fit.chain
        Rs = chain['r_star'].values * u.Rsun if 'r_star' in chain.columns else np.random.normal(rstar, rstar_err, len(chain)) * u.Rsun
        Ms = chain['m_star'].values * u.Msun if 'm_star' in chain.columns else np.random.normal(mstar, mstar_err, len(chain)) * u.Msun

        rp = chain["p_p1"].values * Rs
        rp = rp.to(r_units)
        r_planet_val, r_planet_down, r_planet_up = get_vals(rp.value)
        print(f"R_planet: {r_planet_val:.4f} +{r_planet_up:.4f} -{r_planet_down:.4f} {r_units}")
    
        aRs = chain["aRs_p1"].values * Rs
        sma = aRs.to(sma_units)
        sma_val, sma_down, sma_up = get_vals(sma.value)
        print(f"Sma_planet: {sma_val:.4f} +{sma_up:.4f} -{sma_down:.4f} {sma_units}")
    
        K = chain["K_p1"].values * u.m / u.s
        e = chain["e_p1"].values
        inc = chain["inc_p1"].values * np.pi / 180.0  # Convert to radians
        mp = K * np.sqrt(Ms * sma * (1 - e**2) / c.G) / np.sin(inc)
        mp = mp.to(m_units)
        m_planet_val, m_planet_down, m_planet_up = get_vals(mp.value)
        print(f"M_planet: {m_planet_val:.4f} +{m_planet_up:.4f} -{m_planet_down:.4f} {m_units}")
    
        dens = (3 * mp / (4 * np.pi * rp**3)).to(u.g / u.cm**3)
        dens_val, dens_down, dens_up = get_vals(dens.value)
        print(f"Rho_planet: {dens_val:.4f} +{dens_up:.4f} -{dens_down:.4f} g/cm^3")

    def evaluate_LC_model(self, times, instrument, n_models=False, n=5000):
        """
        Evaluate the light curve model for the given instrument and times.
    
        Parameters:
        times (array): Array of times to evaluate the light curve.
        instrument (str): Name of the instrument.
        n_models (bool): If True, returns multiple models sampled from the chain. Default is False.
        n (int): The number of models to sample if n_models is True. Default is 5000.
    
        Returns:
        flux_lc (array): Evaluated light curve model.
        """
        necessary_params = ['per_p1', 't0_p1', 'p_p1', 'e_p1', 'omega_p1', 'aRs_p1', 'inc_p1', 'u1_' + instrument, 'u2_' + instrument]

        if not n_models:
            dct_params = {k: self.fit.vals[k] for k in necessary_params}
            P, t0, RpRs, e, w, sma, inc, u1, u2 = dct_params.values()
        
            params = batman.TransitParams()
            params.t0 = t0
            params.per = P
            params.rp = RpRs
            params.a = sma
            params.inc = inc
            params.ecc = e
            params.w = w
            params.u = [u1, u2]
            params.limb_dark = "quadratic"
            m = batman.TransitModel(params, times)
            flux_lc = m.light_curve(params)
    
            return flux_lc

        else:
            models = []
            chain_samples = self.fit.chain.sample(n)
            for i, (_, sample) in enumerate(chain_samples.iterrows()):
                if i % 100 == 0:
                    print(f"Sampling i = {i}", end="\r")
                dct_params = {param: sample[param] for param in necessary_params}
                P, t0, RpRs, e, w, sma, inc, u1, u2 = dct_params.values()

                params = batman.TransitParams()
                params.t0 = t0
                params.per = P
                params.rp = RpRs
                params.a = sma
                params.inc = inc
                params.ecc = e
                params.w = w
                params.u = [u1, u2]
                params.limb_dark = "quadratic"
                m = batman.TransitModel(params, times)
                flux_lc = m.light_curve(params)
                
                models.append(flux_lc)
            return np.array(models)

    def evaluate_RV_model(self, times, instrument, n_models=False, n=5000):
        """
        Evaluates the RV model for the given times and instrument.
        
        Parameters:
        times (array): Array of times at which to evaluate the model.
        instrument (str): The instrument name.
        n_models (bool): If True, returns multiple models sampled from the chain. Default is False.
        n (int): The number of models to sample if n_models is True. Default is 5000.
        
        Returns:
        array: The evaluated RV model(s).
        """
        necessary_params = ['per_p1', 't0_p1', 'e_p1', 'omega_p1', 'K_p1', 'gamma_' + instrument]
        
        if not n_models:
            dct_params = {k: self.fit.vals[k] for k in necessary_params}
            P, t0, e, w, K, gamma = dct_params.values()

            gammadot = self.fit.vals.get('gammadot_' + instrument, self.fit.vals.get('gammadot', 0))
            gammadotdot = self.fit.vals.get('gammadotdot_' + instrument, self.fit.vals.get('gammadotdot', 0))

            rv_trend = gammadot * (times - self.fit.ta) + gammadotdot * (times - self.fit.ta) ** 2.0
            rv_model = rmfit.get_rv_curve(times, P, t0, e, w, K, plot=False, verbose=False) + gamma + rv_trend

            return rv_model
        else:
            models = []
            chain_samples = self.fit.chain.sample(n)
            
            for i, (_, sample) in enumerate(chain_samples.iterrows()):
                if i % 100 == 0:
                    print(f"Sampling i = {i}", end="\r")
                dct_params = {param: sample[param] for param in necessary_params}
                P, t0, e, w, K, gamma = dct_params.values()

                gammadot = sample.get('gammadot_' + instrument, sample.get('gammadot', 0))
                gammadotdot = sample.get('gammadotdot_' + instrument, sample.get('gammadotdot', 0))

                rv_trend = gammadot * (times - self.fit.ta) + gammadotdot * (times - self.fit.ta) ** 2.0
                rv_model = rmfit.get_rv_curve(times, P, t0, e, w, K, plot=False, verbose=False) #+ gamma + rv_trend

                models.append(rv_model)

            return np.array(models)

    def evaluate_RM_model(self, times, instrument, n_models=False, n=5000):
        """
        Evaluates the RM effect model for the given times and instrument.
        
        Parameters:
        times (array): Array of times at which to evaluate the model.
        instrument (str): The instrument name.
        n_models (bool): If True, returns multiple models sampled from the chain. Default is False.
        n (int): The number of models to sample if n_models is True. Default is 5000.
        
        Returns:
        array: The evaluated RM effect model(s).
        """
        necessary_params = ['per_p1', 't0_p1', 'e_p1', 'omega_p1', 'K_p1', 'lam_p1', 'vsini_star', 'p_p1', 
                            'aRs_p1', 'inc_p1', 'beta_' + instrument, 'gamma_' + instrument, 'u1_' + instrument, 'u2_' + instrument]

        if not n_models:
            dct_params = {k: self.fit.vals[k] for k in necessary_params}
            P, t0, e, w, K, lam, vsini, RpRs, sma, inc, beta, gamma, u1, u2 = dct_params.values()

            gammadot = self.fit.vals.get('gammadot_' + instrument, self.fit.vals.get('gammadot', 0))
            gammadotdot = self.fit.vals.get('gammadotdot_' + instrument, self.fit.vals.get('gammadotdot', 0))

            rv_trend = gammadot * (times - self.fit.ta) + gammadotdot * (times - self.fit.ta) ** 2.0

            RM = rmfit.RMHirano(lam, vsini, P, t0, sma, inc, RpRs, e, w, [u1, u2], beta, vsini / 1.31, limb_dark="quadratic")
            RM_model = RM.evaluate(times) + rmfit.get_rv_curve(times, P, t0, e, w, K, plot=False, verbose=False) + gamma + rv_trend

            return RM_model
        else:
            models = []
            chain_samples = self.fit.chain.sample(n)
            
            for i, (_, sample) in enumerate(chain_samples.iterrows()):
                if i % 100 == 0:
                    print(f"Sampling i = {i}", end="\r")
                dct_params = {param: sample[param] for param in necessary_params}
                P, t0, e, w, K, lam, vsini, RpRs, sma, inc, beta, gamma, u1, u2 = dct_params.values()

                gammadot = sample.get('gammadot_' + instrument, sample.get('gammadot', 0))
                gammadotdot = sample.get('gammadotdot_' + instrument, sample.get('gammadotdot', 0))

                rv_trend = gammadot * (times - self.fit.ta) + gammadotdot * (times - self.fit.ta) ** 2.0

                RM = rmfit.RMHirano(lam, vsini, P, t0, sma, inc, RpRs, e, w, [u1, u2], beta, vsini / 1.31, limb_dark="quadratic")
                RM_model = RM.evaluate(times) + rmfit.get_rv_curve(times, P, t0, e, w, K, plot=False, verbose=False) + gamma + rv_trend

                models.append(RM_model)

            return np.array(models)
