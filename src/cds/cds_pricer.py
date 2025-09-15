"""
CDS pricing and valuation engine.
"""

import numpy as np
from datetime import date, datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import math

from ..models.cds import CDSCurve, CDSQuote
from .hazard_rate_model import HazardRateModelEngine, HazardRateParameters
from ..pricing.yield_curve import YieldCurve


class CDSPricingModel(Enum):
    """CDS pricing models."""
    STANDARD = "standard"
    REDUCED_FORM = "reduced_form"
    STRUCTURAL = "structural"
    COPULA = "copula"


@dataclass
class CDSPricingResult:
    """CDS pricing calculation result."""
    fair_spread: float
    present_value: float
    protection_leg_pv: float
    premium_leg_pv: float
    cs01: float  # Credit spread sensitivity
    ir01: float  # Interest rate sensitivity
    recovery_sensitivity: float
    default_probability: float
    survival_probability: float
    
    def __str__(self) -> str:
        return (f"CDS Pricing Result:\n"
                f"Fair Spread: {self.fair_spread:.2f} bps\n"
                f"Present Value: ${self.present_value:,.2f}\n"
                f"Protection Leg PV: ${self.protection_leg_pv:,.2f}\n"
                f"Premium Leg PV: ${self.premium_leg_pv:,.2f}\n"
                f"CS01: ${self.cs01:,.2f}\n"
                f"IR01: ${self.ir01:,.2f}\n"
                f"Default Probability: {self.default_probability:.4f}")


@dataclass
class CDSCashFlow:
    """CDS cash flow representation."""
    payment_date: date
    accrual_start: date
    accrual_end: date
    day_count_fraction: float
    discount_factor: float
    survival_probability: float
    premium_payment: float
    protection_payment: float
    
    @property
    def net_cash_flow(self) -> float:
        """Net cash flow (protection - premium)."""
        return self.protection_payment - self.premium_payment


class CDSPricer:
    """
    Comprehensive CDS pricing engine.
    """
    
    def __init__(self, 
                 yield_curve: YieldCurve,
                 hazard_model: Optional[HazardRateModelEngine] = None,
                 pricing_model: CDSPricingModel = CDSPricingModel.STANDARD):
        """
        Initialize CDS pricer.
        
        Args:
            yield_curve: Risk-free yield curve
            hazard_model: Hazard rate model engine
            pricing_model: CDS pricing model to use
        """
        self.yield_curve = yield_curve
        self.hazard_model = hazard_model or HazardRateModelEngine()
        self.pricing_model = pricing_model
    
    def price_cds(self, 
                  cds_curve: CDSCurve,
                  notional: float = 10_000_000,
                  contract_spread: Optional[float] = None,
                  maturity_years: float = 5.0,
                  valuation_date: Optional[date] = None) -> CDSPricingResult:
        """
        Price a CDS contract.
        
        Args:
            cds_curve: CDS curve for credit risk
            notional: Contract notional amount
            contract_spread: Contract spread in bps (if None, calculates fair spread)
            maturity_years: Contract maturity in years
            valuation_date: Valuation date
            
        Returns:
            CDSPricingResult object
        """
        if valuation_date is None:
            valuation_date = date.today()
        
        # Calibrate hazard rate model
        hazard_params = self.hazard_model.calibrate(cds_curve)
        
        # Generate cash flows
        cash_flows = self._generate_cash_flows(
            valuation_date, maturity_years, cds_curve.recovery_rate
        )
        
        # Calculate protection leg PV
        protection_pv = self._calculate_protection_leg_pv(
            cash_flows, hazard_params, notional, cds_curve.recovery_rate
        )
        
        # Calculate premium leg PV
        if contract_spread is None:
            # Calculate fair spread
            premium_pv_per_bp = self._calculate_premium_leg_pv_per_bp(
                cash_flows, hazard_params, notional
            )
            fair_spread = protection_pv / premium_pv_per_bp if premium_pv_per_bp > 0 else 0.0
            premium_pv = protection_pv  # At fair spread, PVs are equal
        else:
            fair_spread = contract_spread
            premium_pv = self._calculate_premium_leg_pv(
                cash_flows, hazard_params, notional, contract_spread
            )
        
        # Calculate present value (from protection buyer's perspective)
        present_value = protection_pv - premium_pv
        
        # Calculate risk sensitivities
        cs01 = self._calculate_cs01(cash_flows, hazard_params, notional)
        ir01 = self._calculate_ir01(cash_flows, hazard_params, notional, fair_spread)
        recovery_sensitivity = self._calculate_recovery_sensitivity(
            cash_flows, hazard_params, notional, cds_curve.recovery_rate
        )
        
        # Calculate default and survival probabilities
        survival_prob = self.hazard_model.calculate_survival_probability(
            maturity_years, hazard_params
        )
        
        return CDSPricingResult(
            fair_spread=fair_spread,
            present_value=present_value,
            protection_leg_pv=protection_pv,
            premium_leg_pv=premium_pv,
            cs01=cs01,
            ir01=ir01,
            recovery_sensitivity=recovery_sensitivity,
            default_probability=survival_prob.default_probability,
            survival_probability=survival_prob.survival_probability
        )
    
    def calculate_spread_curve(self, 
                              cds_curve: CDSCurve,
                              maturities: List[float]) -> Dict[float, float]:
        """
        Calculate fair spreads for multiple maturities.
        
        Args:
            cds_curve: CDS curve for calibration
            maturities: List of maturities in years
            
        Returns:
            Dictionary of maturity to fair spread
        """
        # Calibrate hazard rate model
        hazard_params = self.hazard_model.calibrate(cds_curve)
        
        spreads = {}
        
        for maturity in maturities:
            # Generate cash flows for this maturity
            cash_flows = self._generate_cash_flows(
                date.today(), maturity, cds_curve.recovery_rate
            )
            
            # Calculate protection and premium leg PVs
            protection_pv = self._calculate_protection_leg_pv(
                cash_flows, hazard_params, 10_000_000, cds_curve.recovery_rate
            )
            
            premium_pv_per_bp = self._calculate_premium_leg_pv_per_bp(
                cash_flows, hazard_params, 10_000_000
            )
            
            # Calculate fair spread
            fair_spread = protection_pv / premium_pv_per_bp if premium_pv_per_bp > 0 else 0.0
            spreads[maturity] = fair_spread
        
        return spreads
    
    def calculate_portfolio_cds_risk(self, 
                                   positions: List[Tuple[CDSCurve, float, float]]) -> Dict[str, float]:
        """
        Calculate portfolio-level CDS risk metrics.
        
        Args:
            positions: List of (cds_curve, notional, spread) tuples
            
        Returns:
            Dictionary of risk metrics
        """
        total_cs01 = 0.0
        total_ir01 = 0.0
        total_pv = 0.0
        total_notional = 0.0
        
        for cds_curve, notional, spread in positions:
            pricing_result = self.price_cds(cds_curve, notional, spread)
            
            total_cs01 += pricing_result.cs01
            total_ir01 += pricing_result.ir01
            total_pv += pricing_result.present_value
            total_notional += abs(notional)
        
        return {
            'total_cs01': total_cs01,
            'total_ir01': total_ir01,
            'total_pv': total_pv,
            'total_notional': total_notional,
            'average_cs01': total_cs01 / len(positions) if positions else 0.0,
            'average_ir01': total_ir01 / len(positions) if positions else 0.0,
            'pv_percentage': (total_pv / total_notional * 100) if total_notional > 0 else 0.0
        }
    
    def _generate_cash_flows(self, 
                           valuation_date: date,
                           maturity_years: float,
                           recovery_rate: float) -> List[CDSCashFlow]:
        """Generate CDS cash flows."""
        cash_flows = []
        
        # Standard quarterly payments
        payment_frequency = 4  # Quarterly
        num_payments = int(maturity_years * payment_frequency)
        
        for i in range(1, num_payments + 1):
            # Calculate payment date
            payment_date = valuation_date + timedelta(days=int(365.25 * i / payment_frequency))
            
            # Calculate accrual period
            if i == 1:
                accrual_start = valuation_date
            else:
                accrual_start = valuation_date + timedelta(days=int(365.25 * (i-1) / payment_frequency))
            
            accrual_end = payment_date
            
            # Day count fraction (30/360 approximation)
            day_count_fraction = (accrual_end - accrual_start).days / 360.0
            
            # Time to payment
            time_to_payment = (payment_date - valuation_date).days / 365.25
            
            # Discount factor
            discount_factor = self.yield_curve.get_discount_factor(time_to_payment)
            
            # Survival probability (will be calculated later with hazard model)
            survival_probability = 1.0  # Placeholder
            
            cash_flow = CDSCashFlow(
                payment_date=payment_date,
                accrual_start=accrual_start,
                accrual_end=accrual_end,
                day_count_fraction=day_count_fraction,
                discount_factor=discount_factor,
                survival_probability=survival_probability,
                premium_payment=0.0,  # Will be calculated
                protection_payment=0.0  # Will be calculated
            )
            
            cash_flows.append(cash_flow)
        
        return cash_flows
    
    def _calculate_protection_leg_pv(self, 
                                   cash_flows: List[CDSCashFlow],
                                   hazard_params: HazardRateParameters,
                                   notional: float,
                                   recovery_rate: float) -> float:
        """Calculate protection leg present value."""
        protection_pv = 0.0
        
        prev_time = 0.0
        
        for cf in cash_flows:
            # Time to payment
            time_to_payment = (cf.payment_date - cash_flows[0].accrual_start).days / 365.25
            
            # Survival probabilities
            survival_start = self.hazard_model.calculate_survival_probability(
                prev_time, hazard_params
            ).survival_probability
            
            survival_end = self.hazard_model.calculate_survival_probability(
                time_to_payment, hazard_params
            ).survival_probability
            
            # Default probability in this period
            default_prob = survival_start - survival_end
            
            # Protection payment (loss given default)
            loss_given_default = notional * (1 - recovery_rate)
            
            # Present value of protection for this period
            protection_pv += default_prob * loss_given_default * cf.discount_factor
            
            prev_time = time_to_payment
        
        return protection_pv
    
    def _calculate_premium_leg_pv(self, 
                                cash_flows: List[CDSCashFlow],
                                hazard_params: HazardRateParameters,
                                notional: float,
                                spread_bps: float) -> float:
        """Calculate premium leg present value."""
        spread_decimal = spread_bps / 10000.0
        premium_pv = 0.0
        
        for cf in cash_flows:
            # Time to payment
            time_to_payment = (cf.payment_date - cash_flows[0].accrual_start).days / 365.25
            
            # Survival probability
            survival_prob = self.hazard_model.calculate_survival_probability(
                time_to_payment, hazard_params
            ).survival_probability
            
            # Premium payment
            premium_payment = notional * spread_decimal * cf.day_count_fraction
            
            # Present value
            premium_pv += premium_payment * survival_prob * cf.discount_factor
        
        return premium_pv
    
    def _calculate_premium_leg_pv_per_bp(self, 
                                       cash_flows: List[CDSCashFlow],
                                       hazard_params: HazardRateParameters,
                                       notional: float) -> float:
        """Calculate premium leg PV per basis point."""
        return self._calculate_premium_leg_pv(cash_flows, hazard_params, notional, 1.0)
    
    def _calculate_cs01(self, 
                       cash_flows: List[CDSCashFlow],
                       hazard_params: HazardRateParameters,
                       notional: float) -> float:
        """Calculate CS01 (credit spread sensitivity)."""
        # Approximate CS01 as premium leg PV per bp
        return self._calculate_premium_leg_pv_per_bp(cash_flows, hazard_params, notional)
    
    def _calculate_ir01(self, 
                       cash_flows: List[CDSCashFlow],
                       hazard_params: HazardRateParameters,
                       notional: float,
                       spread_bps: float) -> float:
        """Calculate IR01 (interest rate sensitivity)."""
        # Simplified IR01 calculation
        # In practice, would shift yield curve and recalculate
        
        base_pv = self._calculate_premium_leg_pv(cash_flows, hazard_params, notional, spread_bps)
        
        # Approximate by calculating duration-like measure
        total_pv_time_weighted = 0.0
        
        for cf in cash_flows:
            time_to_payment = (cf.payment_date - cash_flows[0].accrual_start).days / 365.25
            
            survival_prob = self.hazard_model.calculate_survival_probability(
                time_to_payment, hazard_params
            ).survival_probability
            
            premium_payment = notional * (spread_bps / 10000.0) * cf.day_count_fraction
            pv_contribution = premium_payment * survival_prob * cf.discount_factor
            
            total_pv_time_weighted += pv_contribution * time_to_payment
        
        # Approximate IR01 as modified duration * PV * 0.0001
        if base_pv > 0:
            modified_duration = total_pv_time_weighted / base_pv
            ir01 = modified_duration * base_pv * 0.0001
        else:
            ir01 = 0.0
        
        return ir01
    
    def _calculate_recovery_sensitivity(self, 
                                      cash_flows: List[CDSCashFlow],
                                      hazard_params: HazardRateParameters,
                                      notional: float,
                                      recovery_rate: float) -> float:
        """Calculate sensitivity to recovery rate."""
        # Calculate protection leg PV with recovery rate shifted by 1%
        base_protection_pv = self._calculate_protection_leg_pv(
            cash_flows, hazard_params, notional, recovery_rate
        )
        
        shifted_protection_pv = self._calculate_protection_leg_pv(
            cash_flows, hazard_params, notional, recovery_rate + 0.01
        )
        
        return shifted_protection_pv - base_protection_pv


class CDSPortfolioAnalyzer:
    """Analyzer for CDS portfolio risk and performance."""
    
    def __init__(self, pricer: CDSPricer):
        """Initialize with CDS pricer."""
        self.pricer = pricer
    
    def analyze_basis_risk(self, 
                          single_name_cds: CDSCurve,
                          index_cds: CDSCurve,
                          maturity: float = 5.0) -> Dict[str, float]:
        """
        Analyze basis risk between single-name and index CDS.
        
        Args:
            single_name_cds: Single-name CDS curve
            index_cds: Index CDS curve
            maturity: Analysis maturity
            
        Returns:
            Basis risk metrics
        """
        # Price both CDS
        sn_result = self.pricer.price_cds(single_name_cds, maturity_years=maturity)
        idx_result = self.pricer.price_cds(index_cds, maturity_years=maturity)
        
        # Calculate basis
        spread_basis = sn_result.fair_spread - idx_result.fair_spread
        
        return {
            'spread_basis_bps': spread_basis,
            'single_name_spread': sn_result.fair_spread,
            'index_spread': idx_result.fair_spread,
            'basis_percentage': (spread_basis / idx_result.fair_spread * 100) if idx_result.fair_spread > 0 else 0.0,
            'single_name_cs01': sn_result.cs01,
            'index_cs01': idx_result.cs01,
            'hedge_ratio': sn_result.cs01 / idx_result.cs01 if idx_result.cs01 != 0 else 1.0
        }
    
    def calculate_correlation_adjusted_risk(self, 
                                          cds_curves: List[CDSCurve],
                                          correlation_matrix: np.ndarray,
                                          notionals: List[float]) -> Dict[str, float]:
        """
        Calculate correlation-adjusted portfolio risk.
        
        Args:
            cds_curves: List of CDS curves
            correlation_matrix: Credit correlation matrix
            notionals: Position notionals
            
        Returns:
            Risk metrics
        """
        n = len(cds_curves)
        
        # Calculate individual CS01s
        cs01s = []
        for i, (curve, notional) in enumerate(zip(cds_curves, notionals)):
            result = self.pricer.price_cds(curve, notional)
            cs01s.append(result.cs01)
        
        cs01s = np.array(cs01s)
        
        # Calculate portfolio CS01 with correlations
        portfolio_cs01_squared = np.dot(cs01s, np.dot(correlation_matrix, cs01s))
        portfolio_cs01 = math.sqrt(max(0, portfolio_cs01_squared))
        
        # Diversification benefit
        undiversified_cs01 = np.sum(np.abs(cs01s))
        diversification_benefit = undiversified_cs01 - portfolio_cs01
        
        return {
            'portfolio_cs01': portfolio_cs01,
            'undiversified_cs01': undiversified_cs01,
            'diversification_benefit': diversification_benefit,
            'diversification_ratio': diversification_benefit / undiversified_cs01 if undiversified_cs01 > 0 else 0.0,
            'individual_cs01s': cs01s.tolist()
        }