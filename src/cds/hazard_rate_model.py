"""
Hazard rate models and survival probability calculations.
"""

import numpy as np
from datetime import date, datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import math
from scipy import optimize, integrate

from ..models.cds import CDSCurve


class HazardRateModel(Enum):
    """Types of hazard rate models."""
    CONSTANT = "constant"
    EXPONENTIAL = "exponential"
    WEIBULL = "weibull"
    COX_INGERSOLL_ROSS = "cir"
    JUMP_DIFFUSION = "jump_diffusion"
    PIECEWISE_CONSTANT = "piecewise_constant"


@dataclass
class SurvivalProbability:
    """Survival probability calculation result."""
    time_horizon: float
    survival_probability: float
    default_probability: float
    hazard_rate: float
    model_type: HazardRateModel
    parameters: Dict[str, float]
    
    def __str__(self) -> str:
        return (f"Survival Probability (t={self.time_horizon}): {self.survival_probability:.4f}\n"
                f"Default Probability: {self.default_probability:.4f}\n"
                f"Hazard Rate: {self.hazard_rate:.4f}\n"
                f"Model: {self.model_type.value}")


@dataclass
class HazardRateParameters:
    """Parameters for hazard rate models."""
    model_type: HazardRateModel
    parameters: Dict[str, float]
    calibration_date: date
    calibration_rmse: Optional[float] = None
    
    def get_parameter(self, name: str, default: float = 0.0) -> float:
        """Get parameter value with default."""
        return self.parameters.get(name, default)


class HazardRateModelEngine:
    """
    Engine for hazard rate modeling and survival probability calculations.
    """
    
    def __init__(self, model_type: HazardRateModel = HazardRateModel.PIECEWISE_CONSTANT):
        """
        Initialize hazard rate model engine.
        
        Args:
            model_type: Type of hazard rate model to use
        """
        self.model_type = model_type
        self.parameters = None
    
    def calibrate(self, cds_curve: CDSCurve, 
                 model_type: Optional[HazardRateModel] = None) -> HazardRateParameters:
        """
        Calibrate hazard rate model to CDS curve.
        
        Args:
            cds_curve: CDS curve to calibrate against
            model_type: Override model type for calibration
            
        Returns:
            Calibrated model parameters
        """
        if model_type:
            self.model_type = model_type
        
        if self.model_type == HazardRateModel.CONSTANT:
            return self._calibrate_constant_hazard(cds_curve)
        elif self.model_type == HazardRateModel.EXPONENTIAL:
            return self._calibrate_exponential_hazard(cds_curve)
        elif self.model_type == HazardRateModel.WEIBULL:
            return self._calibrate_weibull_hazard(cds_curve)
        elif self.model_type == HazardRateModel.COX_INGERSOLL_ROSS:
            return self._calibrate_cir_hazard(cds_curve)
        elif self.model_type == HazardRateModel.JUMP_DIFFUSION:
            return self._calibrate_jump_diffusion_hazard(cds_curve)
        elif self.model_type == HazardRateModel.PIECEWISE_CONSTANT:
            return self._calibrate_piecewise_constant_hazard(cds_curve)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def calculate_survival_probability(self, time_horizon: float,
                                     parameters: Optional[HazardRateParameters] = None) -> SurvivalProbability:
        """
        Calculate survival probability for given time horizon.
        
        Args:
            time_horizon: Time horizon in years
            parameters: Model parameters (uses calibrated if None)
            
        Returns:
            SurvivalProbability object
        """
        if parameters is None:
            parameters = self.parameters
        
        if parameters is None:
            raise ValueError("Model must be calibrated before calculating survival probabilities")
        
        # Calculate survival probability based on model type
        if parameters.model_type == HazardRateModel.CONSTANT:
            survival_prob = self._constant_survival_probability(time_horizon, parameters)
        elif parameters.model_type == HazardRateModel.EXPONENTIAL:
            survival_prob = self._exponential_survival_probability(time_horizon, parameters)
        elif parameters.model_type == HazardRateModel.WEIBULL:
            survival_prob = self._weibull_survival_probability(time_horizon, parameters)
        elif parameters.model_type == HazardRateModel.COX_INGERSOLL_ROSS:
            survival_prob = self._cir_survival_probability(time_horizon, parameters)
        elif parameters.model_type == HazardRateModel.JUMP_DIFFUSION:
            survival_prob = self._jump_diffusion_survival_probability(time_horizon, parameters)
        elif parameters.model_type == HazardRateModel.PIECEWISE_CONSTANT:
            survival_prob = self._piecewise_constant_survival_probability(time_horizon, parameters)
        else:
            raise ValueError(f"Unsupported model type: {parameters.model_type}")
        
        # Calculate hazard rate at time horizon
        hazard_rate = self.get_hazard_rate(time_horizon, parameters)
        
        return SurvivalProbability(
            time_horizon=time_horizon,
            survival_probability=survival_prob,
            default_probability=1.0 - survival_prob,
            hazard_rate=hazard_rate,
            model_type=parameters.model_type,
            parameters=parameters.parameters
        )
    
    def get_hazard_rate(self, time: float, 
                       parameters: Optional[HazardRateParameters] = None) -> float:
        """
        Get instantaneous hazard rate at given time.
        
        Args:
            time: Time point in years
            parameters: Model parameters
            
        Returns:
            Hazard rate at time t
        """
        if parameters is None:
            parameters = self.parameters
        
        if parameters is None:
            raise ValueError("Model must be calibrated")
        
        if parameters.model_type == HazardRateModel.CONSTANT:
            return parameters.get_parameter('lambda')
        
        elif parameters.model_type == HazardRateModel.EXPONENTIAL:
            lambda0 = parameters.get_parameter('lambda0')
            alpha = parameters.get_parameter('alpha')
            return lambda0 * math.exp(alpha * time)
        
        elif parameters.model_type == HazardRateModel.WEIBULL:
            alpha = parameters.get_parameter('alpha')
            beta = parameters.get_parameter('beta')
            return alpha * beta * (time ** (beta - 1))
        
        elif parameters.model_type == HazardRateModel.COX_INGERSOLL_ROSS:
            # Simplified - would need full CIR solution
            lambda0 = parameters.get_parameter('lambda0')
            return max(lambda0, 1e-6)
        
        elif parameters.model_type == HazardRateModel.JUMP_DIFFUSION:
            lambda_base = parameters.get_parameter('lambda_base')
            return lambda_base  # Simplified
        
        elif parameters.model_type == HazardRateModel.PIECEWISE_CONSTANT:
            # Find the appropriate piece
            times = sorted([float(k.split('_')[1]) for k in parameters.parameters.keys() if k.startswith('lambda_')])
            
            for i, t in enumerate(times):
                if time <= t:
                    return parameters.get_parameter(f'lambda_{t}')
            
            # If beyond last time point, use last hazard rate
            if times:
                return parameters.get_parameter(f'lambda_{times[-1]}')
            else:
                return 0.01  # Default
        
        else:
            return 0.01  # Default hazard rate
    
    def _calibrate_constant_hazard(self, cds_curve: CDSCurve) -> HazardRateParameters:
        """Calibrate constant hazard rate model."""
        # Use first available quote to estimate constant hazard rate
        if not cds_curve.quotes:
            lambda_param = 0.01
        else:
            # Use first quote
            quote = cds_curve.quotes[0]
            maturity = quote.maturity_years
            
            if quote.quote_type == "spread":
                # Approximate: spread ≈ hazard_rate * (1 - recovery_rate)
                spread_decimal = quote.value / 10000.0
                lambda_param = spread_decimal / (1 - cds_curve.recovery_rate)
            else:
                lambda_param = 0.01
        
        parameters = {'lambda': lambda_param}
        
        return HazardRateParameters(
            model_type=HazardRateModel.CONSTANT,
            parameters=parameters,
            calibration_date=cds_curve.curve_date
        )
    
    def _calibrate_exponential_hazard(self, cds_curve: CDSCurve) -> HazardRateParameters:
        """Calibrate exponential hazard rate model: λ(t) = λ₀ * exp(αt)."""
        if len(cds_curve.quotes) < 2:
            # Fallback to constant
            return self._calibrate_constant_hazard(cds_curve)
        
        # Extract market data
        maturities = []
        spreads = []
        
        for quote in cds_curve.quotes:
            if quote.quote_type == "spread":
                maturities.append(quote.maturity_years)
                spreads.append(quote.value / 10000.0)  # Convert to decimal
        
        if len(maturities) < 2:
            return self._calibrate_constant_hazard(cds_curve)
        
        maturities = np.array(maturities)
        spreads = np.array(spreads)
        
        def objective(params):
            lambda0, alpha = params
            
            # Calculate model spreads
            model_spreads = []
            for t in maturities:
                if alpha == 0:
                    survival_prob = math.exp(-lambda0 * t)
                else:
                    integral = lambda0 * (math.exp(alpha * t) - 1) / alpha
                    survival_prob = math.exp(-integral)
                
                # Convert to spread (simplified)
                model_spread = -math.log(survival_prob) / t * (1 - cds_curve.recovery_rate)
                model_spreads.append(model_spread)
            
            model_spreads = np.array(model_spreads)
            return np.sum((spreads - model_spreads) ** 2)
        
        # Initial guess
        initial_guess = [spreads[0] / (1 - cds_curve.recovery_rate), 0.0]
        
        # Bounds
        bounds = [(1e-6, 1.0), (-0.5, 0.5)]
        
        try:
            result = optimize.minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
            if result.success:
                lambda0, alpha = result.x
            else:
                lambda0, alpha = initial_guess
        except:
            lambda0, alpha = initial_guess
        
        parameters = {'lambda0': lambda0, 'alpha': alpha}
        
        return HazardRateParameters(
            model_type=HazardRateModel.EXPONENTIAL,
            parameters=parameters,
            calibration_date=cds_curve.curve_date
        )
    
    def _calibrate_weibull_hazard(self, cds_curve: CDSCurve) -> HazardRateParameters:
        """Calibrate Weibull hazard rate model: λ(t) = αβt^(β-1)."""
        if len(cds_curve.quotes) < 2:
            return self._calibrate_constant_hazard(cds_curve)
        
        # Extract market data
        maturities = []
        spreads = []
        
        for quote in cds_curve.quotes:
            if quote.quote_type == "spread":
                maturities.append(quote.maturity_years)
                spreads.append(quote.value / 10000.0)
        
        if len(maturities) < 2:
            return self._calibrate_constant_hazard(cds_curve)
        
        maturities = np.array(maturities)
        spreads = np.array(spreads)
        
        def objective(params):
            alpha, beta = params
            
            model_spreads = []
            for t in maturities:
                # Weibull survival: S(t) = exp(-α * t^β)
                survival_prob = math.exp(-alpha * (t ** beta))
                
                # Convert to spread
                model_spread = -math.log(survival_prob) / t * (1 - cds_curve.recovery_rate)
                model_spreads.append(model_spread)
            
            model_spreads = np.array(model_spreads)
            return np.sum((spreads - model_spreads) ** 2)
        
        # Initial guess
        initial_guess = [0.01, 1.0]
        
        # Bounds
        bounds = [(1e-6, 1.0), (0.1, 3.0)]
        
        try:
            result = optimize.minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
            if result.success:
                alpha, beta = result.x
            else:
                alpha, beta = initial_guess
        except:
            alpha, beta = initial_guess
        
        parameters = {'alpha': alpha, 'beta': beta}
        
        return HazardRateParameters(
            model_type=HazardRateModel.WEIBULL,
            parameters=parameters,
            calibration_date=cds_curve.curve_date
        )
    
    def _calibrate_cir_hazard(self, cds_curve: CDSCurve) -> HazardRateParameters:
        """Calibrate Cox-Ingersoll-Ross hazard rate model."""
        # Simplified CIR calibration - in practice would be more complex
        constant_params = self._calibrate_constant_hazard(cds_curve)
        
        parameters = {
            'lambda0': constant_params.parameters['lambda'],
            'kappa': 0.1,  # Mean reversion speed
            'theta': constant_params.parameters['lambda'],  # Long-term mean
            'sigma': 0.05  # Volatility
        }
        
        return HazardRateParameters(
            model_type=HazardRateModel.COX_INGERSOLL_ROSS,
            parameters=parameters,
            calibration_date=cds_curve.curve_date
        )
    
    def _calibrate_jump_diffusion_hazard(self, cds_curve: CDSCurve) -> HazardRateParameters:
        """Calibrate jump-diffusion hazard rate model."""
        # Simplified jump-diffusion calibration
        constant_params = self._calibrate_constant_hazard(cds_curve)
        
        parameters = {
            'lambda_base': constant_params.parameters['lambda'],
            'jump_intensity': 0.1,  # Jump frequency
            'jump_size_mean': 0.02,  # Average jump size
            'jump_size_std': 0.01   # Jump size volatility
        }
        
        return HazardRateParameters(
            model_type=HazardRateModel.JUMP_DIFFUSION,
            parameters=parameters,
            calibration_date=cds_curve.curve_date
        )
    
    def _calibrate_piecewise_constant_hazard(self, cds_curve: CDSCurve) -> HazardRateParameters:
        """Calibrate piecewise constant hazard rate model."""
        parameters = {}
        
        # Use CDS curve's built-in hazard rates if available
        if hasattr(cds_curve, '_hazard_rates') and cds_curve._hazard_rates:
            for maturity, hazard_rate in cds_curve._hazard_rates.items():
                parameters[f'lambda_{maturity}'] = hazard_rate
        else:
            # Bootstrap from quotes
            for quote in cds_curve.quotes:
                if quote.quote_type == "spread":
                    maturity = quote.maturity_years
                    spread_decimal = quote.value / 10000.0
                    hazard_rate = spread_decimal / (1 - cds_curve.recovery_rate)
                    parameters[f'lambda_{maturity}'] = hazard_rate
        
        return HazardRateParameters(
            model_type=HazardRateModel.PIECEWISE_CONSTANT,
            parameters=parameters,
            calibration_date=cds_curve.curve_date
        )
    
    def _constant_survival_probability(self, time: float, 
                                     parameters: HazardRateParameters) -> float:
        """Calculate survival probability for constant hazard rate."""
        lambda_param = parameters.get_parameter('lambda')
        return math.exp(-lambda_param * time)
    
    def _exponential_survival_probability(self, time: float,
                                        parameters: HazardRateParameters) -> float:
        """Calculate survival probability for exponential hazard rate."""
        lambda0 = parameters.get_parameter('lambda0')
        alpha = parameters.get_parameter('alpha')
        
        if abs(alpha) < 1e-10:
            # Constant case
            return math.exp(-lambda0 * time)
        else:
            integral = lambda0 * (math.exp(alpha * time) - 1) / alpha
            return math.exp(-integral)
    
    def _weibull_survival_probability(self, time: float,
                                    parameters: HazardRateParameters) -> float:
        """Calculate survival probability for Weibull hazard rate."""
        alpha = parameters.get_parameter('alpha')
        beta = parameters.get_parameter('beta')
        
        return math.exp(-alpha * (time ** beta))
    
    def _cir_survival_probability(self, time: float,
                                parameters: HazardRateParameters) -> float:
        """Calculate survival probability for CIR hazard rate."""
        # Simplified CIR - in practice would use full analytical solution
        lambda0 = parameters.get_parameter('lambda0')
        kappa = parameters.get_parameter('kappa', 0.1)
        theta = parameters.get_parameter('theta', lambda0)
        
        # Approximate with mean hazard rate
        mean_hazard = theta + (lambda0 - theta) * math.exp(-kappa * time)
        return math.exp(-mean_hazard * time)
    
    def _jump_diffusion_survival_probability(self, time: float,
                                           parameters: HazardRateParameters) -> float:
        """Calculate survival probability for jump-diffusion hazard rate."""
        # Simplified jump-diffusion
        lambda_base = parameters.get_parameter('lambda_base')
        jump_intensity = parameters.get_parameter('jump_intensity', 0.1)
        jump_size_mean = parameters.get_parameter('jump_size_mean', 0.02)
        
        # Approximate with adjusted hazard rate
        adjusted_hazard = lambda_base + jump_intensity * jump_size_mean
        return math.exp(-adjusted_hazard * time)
    
    def _piecewise_constant_survival_probability(self, time: float,
                                               parameters: HazardRateParameters) -> float:
        """Calculate survival probability for piecewise constant hazard rate."""
        # Get all time points
        times = sorted([float(k.split('_')[1]) for k in parameters.parameters.keys() if k.startswith('lambda_')])
        
        if not times:
            return 1.0
        
        integral = 0.0
        prev_time = 0.0
        
        for t in times:
            if time <= t:
                # Integrate up to target time
                hazard_rate = parameters.get_parameter(f'lambda_{t}')
                integral += hazard_rate * (time - prev_time)
                break
            else:
                # Integrate full segment
                hazard_rate = parameters.get_parameter(f'lambda_{t}')
                integral += hazard_rate * (t - prev_time)
                prev_time = t
        
        # If time is beyond last point, use last hazard rate
        if time > times[-1]:
            last_hazard = parameters.get_parameter(f'lambda_{times[-1]}')
            integral += last_hazard * (time - times[-1])
        
        return math.exp(-integral)


class SurvivalProbabilityCalculator:
    """Utility class for survival probability calculations."""
    
    @staticmethod
    def calculate_conditional_survival(survival_prob_t1: float,
                                     survival_prob_t2: float) -> float:
        """
        Calculate conditional survival probability P(T > t2 | T > t1).
        
        Args:
            survival_prob_t1: Survival probability to time t1
            survival_prob_t2: Survival probability to time t2 (t2 > t1)
            
        Returns:
            Conditional survival probability
        """
        if survival_prob_t1 <= 0:
            return 0.0
        
        return survival_prob_t2 / survival_prob_t1
    
    @staticmethod
    def calculate_forward_default_probability(survival_prob_t1: float,
                                            survival_prob_t2: float) -> float:
        """
        Calculate forward default probability P(t1 < T ≤ t2).
        
        Args:
            survival_prob_t1: Survival probability to time t1
            survival_prob_t2: Survival probability to time t2 (t2 > t1)
            
        Returns:
            Forward default probability
        """
        return survival_prob_t1 - survival_prob_t2
    
    @staticmethod
    def calculate_marginal_default_probability(hazard_rate: float, dt: float) -> float:
        """
        Calculate marginal default probability for small time interval.
        
        Args:
            hazard_rate: Instantaneous hazard rate
            dt: Time interval
            
        Returns:
            Marginal default probability
        """
        return 1.0 - math.exp(-hazard_rate * dt)
    
    @staticmethod
    def bootstrap_survival_curve(market_quotes: List[Tuple[float, float]],
                               recovery_rate: float = 0.4) -> Dict[float, float]:
        """
        Bootstrap survival probabilities from market CDS quotes.
        
        Args:
            market_quotes: List of (maturity, spread) tuples
            recovery_rate: Recovery rate assumption
            
        Returns:
            Dictionary of maturity to survival probability
        """
        survival_probs = {0.0: 1.0}
        
        for maturity, spread in sorted(market_quotes):
            # Simplified bootstrapping
            spread_decimal = spread / 10000.0
            hazard_rate = spread_decimal / (1 - recovery_rate)
            
            survival_prob = math.exp(-hazard_rate * maturity)
            survival_probs[maturity] = survival_prob
        
        return survival_probs