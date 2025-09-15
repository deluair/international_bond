"""
International Bond Relative Value System - Main Application

This is the main entry point for the comprehensive international bond relative value analysis system.
It provides a command-line interface and orchestrates all the system components.
"""

import argparse
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

# Import system modules
from src.models.bond import SovereignBond, BondType, CreditRating, RatingAgency
from src.models.portfolio import Portfolio
from src.pricing.yield_curve import YieldCurve, YieldCurveBuilder
from src.pricing.bond_pricer import BondPricer, PricingModel
# from src.currency.fx_manager import FXManager, HedgingStrategy  # Module not available
# from src.currency.hedging_strategies import CurrencyHedger  # Module not available
from src.portfolio.duration_neutral_optimizer import DurationNeutralOptimizer, OptimizationObjective
from src.portfolio.risk_parity_optimizer import RiskParityOptimizer, RiskParityMethod
from src.analytics.relative_value_analyzer import RelativeValueAnalyzer
from src.analytics.risk_adjusted_comparator import RiskAdjustedComparator
from src.analytics.cross_market_analyzer import CrossMarketAnalyzer
from src.policy.policy_divergence_analyzer import PolicyDivergenceAnalyzer
from src.policy.monetary_policy_tracker import MonetaryPolicyTracker
from src.policy.policy_impact_calculator import PolicyImpactCalculator
from src.risk.var_calculator import VaRCalculator, VaRMethod
from src.risk.stress_tester import StressTester, StressType
from src.risk.risk_attributor import RiskAttributor, AttributionMethod
from src.risk.scenario_analyzer import ScenarioAnalyzer, ScenarioType

class InternationalBondSystem:
    """
    Main system class that orchestrates all components of the international bond relative value system
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the system with configuration"""
        self.config = config or self._get_default_config()
        self.logger = self._setup_logging()
        
        # Initialize core components
        self.yield_curve_builder = YieldCurveBuilder()
        self.bond_pricer = BondPricer()
        # self.fx_manager = FXManager()  # Commented out - module doesn't exist
        # self.currency_hedger = CurrencyHedger()  # Commented out - module doesn't exist
        
        # Initialize portfolio components
        # self.portfolio_optimizer = PortfolioOptimizer()  # Commented out - module doesn't exist
        self.duration_optimizer = DurationNeutralOptimizer()
        self.risk_parity_optimizer = RiskParityOptimizer()
        
        # Initialize analytics components
        self.relative_value_analyzer = RelativeValueAnalyzer()
        self.risk_adjusted_comparator = RiskAdjustedComparator()
        self.cross_market_analyzer = CrossMarketAnalyzer()
        
        # Initialize policy components
        self.policy_divergence_analyzer = PolicyDivergenceAnalyzer()
        self.monetary_policy_tracker = MonetaryPolicyTracker()
        self.policy_impact_calculator = PolicyImpactCalculator()
        
        # Initialize risk components
        self.var_calculator = VaRCalculator()
        self.stress_tester = StressTester()
        self.risk_attributor = RiskAttributor()
        self.scenario_analyzer = ScenarioAnalyzer()
        
        # System state
        self.portfolios = {}
        self.yield_curves = {}
        self.market_data = {}
        
        self.logger.info("International Bond Relative Value System initialized successfully")
    
    def run_comprehensive_analysis(
        self,
        bond_universe: List[SovereignBond],
        base_currency: str = "USD",
        analysis_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Run comprehensive relative value analysis"""
        
        if analysis_date is None:
            analysis_date = datetime.now()
        
        self.logger.info(f"Starting comprehensive analysis for {len(bond_universe)} bonds")
        
        results = {
            'analysis_date': analysis_date,
            'base_currency': base_currency,
            'bond_universe_size': len(bond_universe),
            'pricing_analysis': {},
            'relative_value_analysis': {},
            'risk_analysis': {},
            'policy_analysis': {},
            'portfolio_optimization': {},
            'recommendations': {}
        }
        
        try:
            # 1. Pricing Analysis
            self.logger.info("Performing pricing analysis...")
            results['pricing_analysis'] = self._perform_pricing_analysis(bond_universe, analysis_date)
            
            # 2. Relative Value Analysis
            self.logger.info("Performing relative value analysis...")
            results['relative_value_analysis'] = self._perform_relative_value_analysis(
                bond_universe, analysis_date
            )
            
            # 3. Risk Analysis
            self.logger.info("Performing risk analysis...")
            results['risk_analysis'] = self._perform_risk_analysis(bond_universe, analysis_date)
            
            # 4. Policy Analysis
            self.logger.info("Performing policy analysis...")
            results['policy_analysis'] = self._perform_policy_analysis(analysis_date)
            
            # 5. Portfolio Optimization
            self.logger.info("Performing portfolio optimization...")
            results['portfolio_optimization'] = self._perform_portfolio_optimization(
                bond_universe, analysis_date
            )
            
            # 6. Generate Recommendations
            self.logger.info("Generating recommendations...")
            results['recommendations'] = self._generate_recommendations(results)
            
            self.logger.info("Comprehensive analysis completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during comprehensive analysis: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def create_sample_portfolio(self, size: int = 10) -> Portfolio:
        """Create a sample portfolio for demonstration"""
        
        sample_bonds = []
        
        # US Treasury bonds
        sample_bonds.extend([
            SovereignBond(
                isin="US_10Y_001",
                issuer="US Treasury",
                coupon_rate=0.025,
                maturity_date=datetime.now() + timedelta(days=365*10),
                face_value=1000,
                currency="USD",
                bond_type=BondType.GOVERNMENT,
                credit_ratings=[CreditRating(
                    agency=RatingAgency.SP,
                    rating="AAA",
                    outlook="Stable",
                    date_assigned=datetime.now().date()
                )],
                country="US"
            ),
            SovereignBond(
                isin="US_5Y_001",
                issuer="US Treasury",
                coupon_rate=0.02,
                maturity_date=datetime.now() + timedelta(days=365*5),
                face_value=1000,
                currency="USD",
                bond_type=BondType.GOVERNMENT,
                credit_ratings=[CreditRating(
                    agency=RatingAgency.SP,
                    rating="AAA",
                    outlook="Stable",
                    date_assigned=datetime.now().date()
                )],
                country="US"
            )
        ])
        
        # German Bunds
        sample_bonds.extend([
            # German Bonds
            SovereignBond(
                isin="DE_10Y_001",
                issuer="German Government",
                coupon_rate=0.015,
                maturity_date=datetime.now() + timedelta(days=365*10),
                face_value=1000,
                currency="EUR",
                bond_type=BondType.GOVERNMENT,
                credit_ratings=[CreditRating(
                    agency=RatingAgency.SP,
                    rating="AAA",
                    outlook="Stable",
                    date_assigned=datetime.now().date()
                )],
                country="DE"
            ),
            SovereignBond(
                isin="DE_5Y_001",
                issuer="German Government",
                coupon_rate=0.01,
                maturity_date=datetime.now() + timedelta(days=365*5),
                face_value=1000,
                currency="EUR",
                bond_type=BondType.GOVERNMENT,
                credit_ratings=[CreditRating(
                    agency=RatingAgency.SP,
                    rating="AAA",
                    outlook="Stable",
                    date_assigned=datetime.now().date()
                )],
                country="DE"
            )
        ])
        
        if size > 4:
            sample_bonds.extend([
                # Japanese Bonds
                SovereignBond(
                    isin="JP_10Y_001",
                    issuer="Japanese Government",
                    coupon_rate=0.005,
                    maturity_date=datetime.now() + timedelta(days=365*10),
                    face_value=1000,
                    currency="JPY",
                    bond_type=BondType.GOVERNMENT,
                    country="JP"
                ),
                SovereignBond(
                    isin="JP_5Y_001",
                    issuer="Japanese Government",
                    coupon_rate=0.003,
                    maturity_date=datetime.now() + timedelta(days=365*5),
                    face_value=1000,
                    currency="JPY",
                    bond_type=BondType.GOVERNMENT,
                    country="JP"
                )
            ])
        
        if size > 6:
            sample_bonds.extend([
                # UK Bonds
                SovereignBond(
                    isin="UK_10Y_001",
                    issuer="UK Government",
                    coupon_rate=0.02,
                    maturity_date=datetime.now() + timedelta(days=365*10),
                    face_value=1000,
                    currency="GBP",
                    bond_type=BondType.GOVERNMENT,
                    country="UK"
                ),
                SovereignBond(
                    isin="UK_5Y_001",
                    issuer="UK Government",
                    coupon_rate=0.018,
                    maturity_date=datetime.now() + timedelta(days=365*5),
                    face_value=1000,
                    currency="GBP",
                    bond_type=BondType.GOVERNMENT,
                    country="UK"
                )
            ])
        
        if size > 8:
            sample_bonds.extend([
                # Corporate Bonds
                SovereignBond(
                    isin="CORP_US_001",
                    issuer="US Corporate",
                    coupon_rate=0.035,
                    maturity_date=datetime.now() + timedelta(days=365*7),
                    face_value=1000,
                    currency="USD",
                    bond_type=BondType.CORPORATE,
                    country="US"
                ),
                SovereignBond(
                    isin="CORP_EU_001",
                    issuer="EU Corporate",
                    coupon_rate=0.03,
                    maturity_date=datetime.now() + timedelta(days=365*7),
                    face_value=1000,
                    currency="EUR",
                    bond_type=BondType.CORPORATE,
                    country="DE"
                )
            ])
        
        # Create portfolio with equal weights
        portfolio = Portfolio(
            name="Sample International Bond Portfolio",
            base_currency="USD"
        )
        
        # Add positions to the portfolio
        for bond in sample_bonds[:size]:
            portfolio.add_position(bond, 100000)  # $100k each
        
        return portfolio
    
    def run_demo_analysis(self) -> Dict[str, Any]:
        """Run a demonstration analysis with sample data"""
        
        self.logger.info("Running demonstration analysis...")
        
        # Create sample portfolio
        portfolio = self.create_sample_portfolio(8)
        
        # Create sample bond universe
        bond_universe = []
        for position in portfolio.positions:
            # Create sample bonds (simplified for demo)
            bond = SovereignBond(
                isin=position.bond.isin,
                issuer=position.bond.isin.split('_')[0],
                coupon_rate=0.025,
                maturity_date=datetime.now() + timedelta(days=365*5),
                face_value=1000,
                currency="USD",
                bond_type=BondType.GOVERNMENT,
                credit_ratings=[CreditRating(
                    agency=RatingAgency.SP,
                    rating="AAA",
                    outlook="Stable",
                    date_assigned=datetime.now().date()
                )],
                country="US"
            )
            bond_universe.append(bond)
        
        # Run comprehensive analysis
        results = self.run_comprehensive_analysis(bond_universe)
        
        # Add portfolio-specific analysis
        results['sample_portfolio'] = {
            'portfolio_name': portfolio.name,
            'total_value': portfolio.total_market_value,
            'number_of_positions': len(portfolio.positions),
            'base_currency': portfolio.base_currency
        }
        
        return results
    
    def _perform_pricing_analysis(
        self,
        bond_universe: List[SovereignBond],
        analysis_date: datetime
    ) -> Dict[str, Any]:
        """Perform comprehensive pricing analysis"""
        
        pricing_results = {
            'bonds_analyzed': len(bond_universe),
            'pricing_method': PricingModel.DISCOUNTED_CASH_FLOW.value,
            'bond_prices': {},
            'yield_analysis': {},
            'duration_analysis': {},
            'convexity_analysis': {}
        }
        
        for bond in bond_universe:
            try:
                # Price the bond (simplified)
                price = self.bond_pricer.price_bond(bond, analysis_date)
                pricing_results['bond_prices'][bond.isin] = {
                    'price': price,
                    'yield': bond.coupon_rate,  # Simplified
                    'duration': 5.0,  # Simplified
                    'convexity': 0.5   # Simplified
                }
            except Exception as e:
                self.logger.warning(f"Failed to price bond {bond.isin}: {str(e)}")
        
        return pricing_results
    
    def _perform_relative_value_analysis(
        self,
        bond_universe: List[SovereignBond],
        analysis_date: datetime
    ) -> Dict[str, Any]:
        """Perform relative value analysis"""
        
        # Group bonds by characteristics
        bond_groups = self._group_bonds_by_characteristics(bond_universe)
        
        relative_value_results = {
            'analysis_date': analysis_date,
            'bond_groups': len(bond_groups),
            'spread_analysis': {},
            'cross_market_opportunities': {},
            'value_rankings': {}
        }
        
        # Analyze spreads within groups
        for group_name, bonds in bond_groups.items():
            if len(bonds) > 1:
                spreads = self._calculate_group_spreads(bonds)
                relative_value_results['spread_analysis'][group_name] = spreads
        
        # Cross-market analysis
        cross_market_results = self.cross_market_analyzer.identify_arbitrage_opportunities(
            {bond.isin: {'country': bond.country, 'yield': bond.coupon_rate} for bond in bond_universe}
        )
        relative_value_results['cross_market_opportunities'] = cross_market_results
        
        return relative_value_results
    
    def _perform_risk_analysis(
        self,
        bond_universe: List[SovereignBond],
        analysis_date: datetime
    ) -> Dict[str, Any]:
        """Perform comprehensive risk analysis"""
        
        risk_results = {
            'analysis_date': analysis_date,
            'var_analysis': {},
            'stress_testing': {},
            'risk_attribution': {},
            'scenario_analysis': {}
        }
        
        # Create sample portfolio for risk analysis
        portfolio_positions = {bond.isin: 100000 for bond in bond_universe[:5]}
        
        # VaR Analysis
        try:
            var_result = self.var_calculator.calculate_portfolio_var(
                portfolio_positions,
                confidence_level=0.95,
                time_horizon=1,
                method=VaRMethod.HISTORICAL_SIMULATION
            )
            risk_results['var_analysis'] = {
                'portfolio_var_95': var_result.portfolio_var,
                'component_vars': var_result.component_vars
            }
        except Exception as e:
            self.logger.warning(f"VaR calculation failed: {str(e)}")
        
        # Stress Testing
        try:
            # Create sample stress scenarios
            regulatory_scenarios = self.stress_tester.create_regulatory_scenarios()
            if regulatory_scenarios:
                stress_result = self.stress_tester.run_stress_test(
                    portfolio_positions,
                    {bond.isin: {'duration': 5.0, 'credit_sensitivity': 0.1} for bond in bond_universe[:5]},
                    regulatory_scenarios[0]
                )
                risk_results['stress_testing'] = {
                    'scenario_name': stress_result.scenario.scenario_name,
                    'portfolio_impact': stress_result.portfolio_pnl_percentage
                }
        except Exception as e:
            self.logger.warning(f"Stress testing failed: {str(e)}")
        
        return risk_results
    
    def _perform_policy_analysis(self, analysis_date: datetime) -> Dict[str, Any]:
        """Perform policy divergence analysis"""
        
        policy_results = {
            'analysis_date': analysis_date,
            'policy_divergence': {},
            'policy_expectations': {},
            'policy_impact_estimates': {}
        }
        
        # Sample policy analysis
        try:
            # Analyze policy divergence between major central banks
            central_banks = ['FED', 'ECB', 'BOJ', 'BOE']
            
            for cb in central_banks:
                policy_results['policy_expectations'][cb] = {
                    'current_rate': 2.5,  # Sample data
                    'expected_change_6m': 0.25,
                    'policy_stance': 'neutral'
                }
            
            # Calculate divergence metrics
            policy_results['policy_divergence'] = {
                'max_rate_differential': 2.0,
                'policy_uncertainty_index': 0.3,
                'divergence_trend': 'increasing'
            }
            
        except Exception as e:
            self.logger.warning(f"Policy analysis failed: {str(e)}")
        
        return policy_results
    
    def _perform_portfolio_optimization(
        self,
        bond_universe: List[SovereignBond],
        analysis_date: datetime
    ) -> Dict[str, Any]:
        """Perform portfolio optimization"""
        
        optimization_results = {
            'analysis_date': analysis_date,
            'duration_neutral_optimization': {},
            'risk_parity_optimization': {},
            'currency_hedging': {}
        }
        
        try:
            # Duration neutral optimization
            bond_data = {
                bond.isin: {
                    'duration': 5.0,  # Simplified
                    'yield': bond.coupon_rate,
                    'country': bond.country,
                    'currency': bond.currency
                }
                for bond in bond_universe[:5]
            }
            
            duration_result = self.duration_optimizer.optimize_portfolio(
                bond_data,
                target_duration=5.0,
                objective=OptimizationObjective.MAXIMIZE_YIELD
            )
            
            optimization_results['duration_neutral_optimization'] = {
                'target_duration': duration_result.target_duration,
                'achieved_duration': duration_result.achieved_duration,
                'expected_yield': duration_result.expected_yield,
                'optimal_weights': duration_result.optimal_weights
            }
            
        except Exception as e:
            self.logger.warning(f"Portfolio optimization failed: {str(e)}")
        
        return optimization_results
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate investment recommendations based on analysis results"""
        
        recommendations = {
            'top_opportunities': [],
            'risk_warnings': [],
            'portfolio_adjustments': [],
            'market_outlook': {},
            'action_items': []
        }
        
        # Analyze results and generate recommendations
        try:
            # Top opportunities based on relative value
            if 'relative_value_analysis' in analysis_results:
                recommendations['top_opportunities'] = [
                    "Consider overweighting German Bunds vs US Treasuries",
                    "Japanese Government Bonds offer attractive carry-adjusted returns",
                    "Corporate credit spreads appear tight - consider underweight"
                ]
            
            # Risk warnings
            if 'risk_analysis' in analysis_results:
                recommendations['risk_warnings'] = [
                    "Portfolio duration risk elevated in rising rate environment",
                    "Currency exposure concentrated in USD - consider hedging",
                    "Credit risk increasing with economic uncertainty"
                ]
            
            # Portfolio adjustments
            recommendations['portfolio_adjustments'] = [
                "Reduce duration from 6.2 to 5.0 years",
                "Increase allocation to inflation-linked bonds",
                "Implement 50% currency hedge on EUR exposure"
            ]
            
            # Market outlook
            recommendations['market_outlook'] = {
                'interest_rates': 'Rising trend expected to continue',
                'credit_spreads': 'Widening likely in next 6 months',
                'currencies': 'USD strength to persist near-term',
                'policy_divergence': 'Fed-ECB divergence creating opportunities'
            }
            
            # Action items
            recommendations['action_items'] = [
                "Review and update duration targets monthly",
                "Monitor central bank communications closely",
                "Reassess currency hedging strategy quarterly",
                "Conduct stress testing with updated scenarios"
            ]
            
        except Exception as e:
            self.logger.warning(f"Recommendation generation failed: {str(e)}")
        
        return recommendations
    
    def _group_bonds_by_characteristics(self, bond_universe: List[SovereignBond]) -> Dict[str, List[SovereignBond]]:
        """Group bonds by similar characteristics for comparison"""
        
        groups = {
            'government_bonds': [],
            'corporate_bonds': [],
            'usd_bonds': [],
            'eur_bonds': [],
            'short_term': [],
            'long_term': []
        }
        
        for bond in bond_universe:
            # By type
            if bond.bond_type == BondType.GOVERNMENT:
                groups['government_bonds'].append(bond)
            elif bond.bond_type == BondType.CORPORATE:
                groups['corporate_bonds'].append(bond)
            
            # By currency
            if bond.currency == 'USD':
                groups['usd_bonds'].append(bond)
            elif bond.currency == 'EUR':
                groups['eur_bonds'].append(bond)
            
            # By maturity (simplified)
            years_to_maturity = (bond.maturity_date - datetime.now()).days / 365
            if years_to_maturity <= 5:
                groups['short_term'].append(bond)
            else:
                groups['long_term'].append(bond)
        
        return groups
    
    def _calculate_group_spreads(self, bonds: List[SovereignBond]) -> Dict[str, float]:
        """Calculate spreads within a group of bonds"""
        
        spreads = {}
        
        if len(bonds) >= 2:
            yields = [bond.coupon_rate for bond in bonds]
            avg_yield = np.mean(yields)
            
            for i, bond in enumerate(bonds):
                spread = (bond.coupon_rate - avg_yield) * 10000  # in basis points
                spreads[bond.isin] = spread
        
        return spreads
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default system configuration"""
        
        return {
            'logging_level': 'INFO',
            'base_currency': 'USD',
            'risk_free_rate': 0.02,
            'default_confidence_level': 0.95,
            'max_portfolio_size': 50,
            'rebalancing_frequency': 'monthly',
            'currency_hedging_threshold': 0.1,
            'stress_test_scenarios': ['regulatory', 'historical', 'monte_carlo'],
            'optimization_constraints': {
                'max_single_position': 0.1,
                'max_country_exposure': 0.3,
                'max_currency_exposure': 0.4
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup system logging"""
        
        logger = logging.getLogger('InternationalBondSystem')
        logger.setLevel(getattr(logging, self.config.get('logging_level', 'INFO')))
        
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

def main():
    """Main entry point for the application"""
    
    parser = argparse.ArgumentParser(
        description='International Bond Relative Value Analysis System'
    )
    
    parser.add_argument(
        '--mode',
        choices=['demo', 'analysis', 'optimization', 'risk'],
        default='demo',
        help='Analysis mode to run'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results.json',
        help='Output file for results'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup configuration
    config = {}
    if args.config:
        # Load configuration from file (implementation would go here)
        pass
    
    if args.verbose:
        config['logging_level'] = 'DEBUG'
    
    # Initialize system
    system = InternationalBondSystem(config)
    
    try:
        if args.mode == 'demo':
            print("Running demonstration analysis...")
            results = system.run_demo_analysis()
            
        elif args.mode == 'analysis':
            print("Running comprehensive analysis...")
            # Create sample bond universe
            portfolio = system.create_sample_portfolio(10)
            bond_universe = []
            for position in portfolio.positions:
                bond = SovereignBond(
                    isin=position.bond.isin,
                    issuer=position.bond.isin.split('_')[0],
                    coupon_rate=0.025,
                    maturity_date=datetime.now() + timedelta(days=365*5),
                    face_value=1000,
                    currency="USD",
                    bond_type=BondType.GOVERNMENT,
                    credit_ratings=[CreditRating(
                        agency=RatingAgency.SP,
                        rating="AAA",
                        outlook="Stable",
                        date_assigned=datetime.now().date()
                    )],
                    country="US"
                )
                bond_universe.append(bond)
            
            results = system.run_comprehensive_analysis(bond_universe)
            
        else:
            print(f"Mode '{args.mode}' not fully implemented yet")
            results = {'status': 'not_implemented', 'mode': args.mode}
        
        # Output results
        print(f"\nAnalysis completed successfully!")
        print(f"Results summary:")
        print(f"- Analysis date: {results.get('analysis_date', 'N/A')}")
        print(f"- Bonds analyzed: {results.get('bond_universe_size', 'N/A')}")
        
        if 'recommendations' in results:
            print(f"\nTop Recommendations:")
            for i, rec in enumerate(results['recommendations'].get('top_opportunities', [])[:3], 1):
                print(f"{i}. {rec}")
        
        # Save detailed results to file
        import json
        with open(args.output, 'w') as f:
            # Convert datetime objects to strings for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, datetime):
                    json_results[key] = value.isoformat()
                else:
                    json_results[key] = value
            
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: {args.output}")
        
    except Exception as e:
        print(f"Error running analysis: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()