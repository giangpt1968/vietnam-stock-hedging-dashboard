#!/usr/bin/env python3
"""
Simple Vietnam Stock Hedging Demo
================================

A simplified version that demonstrates the core concepts without requiring
advanced statistical libraries.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import the data module
from data import get_prices

class SimpleHedgingDemo:
    """Simplified hedging demonstration"""
    
    def __init__(self):
        print("🚀 Simple Vietnam Stock Hedging Demo")
        print("=" * 50)
        
    def calculate_simple_beta(self, stock_returns, market_returns):
        """Calculate beta using simple correlation method"""
        try:
            # Calculate covariance and variance
            covariance = np.cov(stock_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            
            if market_variance == 0:
                return 0, 0, stock_returns
            
            beta = covariance / market_variance
            correlation = np.corrcoef(stock_returns, market_returns)[0, 1]
            r_squared = correlation ** 2
            
            # Calculate residuals
            residuals = stock_returns - beta * market_returns
            
            return beta, r_squared, residuals
            
        except Exception as e:
            print(f"Error calculating beta: {e}")
            return 0, 0, stock_returns
    
    def calculate_effective_breadth(self, correlation_matrix):
        """Calculate effective breadth using Buckle's formula"""
        # Get upper triangular correlations (excluding diagonal)
        n = len(correlation_matrix)
        upper_indices = np.triu_indices(n, k=1)
        correlations = correlation_matrix.values[upper_indices]
        
        avg_correlation = np.mean(correlations)
        effective_breadth = n / (1 + avg_correlation * (n - 1))
        
        return effective_breadth, avg_correlation
    
    def run_demo(self):
        """Run the hedging demonstration"""
        print("\n📡 Loading Vietnam Stock Data...")
        
        # Demo portfolio
        symbols = ['BID', 'CTG', 'ACB', 'VHM', 'VIC', 'VRE', 'VNINDEX']
        
        try:
            # Get data
            prices = get_prices(*symbols, start_date='2024-01-01', end_date='2024-12-31')
            
            if prices is None:
                print("❌ Could not load data")
                return
            
            print(f"✅ Loaded {len(prices)} days of data")
            print(f"📈 Symbols: {list(prices.columns)}")
            
            # Calculate returns
            returns = prices.pct_change().fillna(0)
            
            # Separate market and stocks
            market_returns = returns['VNINDEX']
            stock_returns = returns.drop('VNINDEX', axis=1)
            
            print(f"\n🔍 Analyzing {len(stock_returns.columns)} stocks vs VN-Index")
            
            # 1. Original correlation analysis
            print(f"\n1️⃣ Original Correlation Analysis (No Hedge)")
            original_corr = stock_returns.corr()
            orig_breadth, orig_avg_corr = self.calculate_effective_breadth(original_corr)
            
            print(f"   📊 Average correlation: {orig_avg_corr:.3f}")
            print(f"   📊 Effective breadth: {orig_breadth:.1f}/{len(stock_returns.columns)}")
            
            # 2. Market hedging
            print(f"\n2️⃣ Market Hedging Analysis")
            hedged_residuals = stock_returns.copy() * 0
            beta_info = {}
            
            for stock in stock_returns.columns:
                beta, r_squared, residuals = self.calculate_simple_beta(
                    stock_returns[stock], market_returns
                )
                
                beta_info[stock] = {
                    'beta': beta,
                    'r_squared': r_squared
                }
                
                hedged_residuals[stock] = residuals
                print(f"   📊 {stock}: Beta={beta:.3f}, R²={r_squared:.3f}")
            
            # Calculate hedged correlation
            hedged_corr = hedged_residuals.corr()
            hedged_breadth, hedged_avg_corr = self.calculate_effective_breadth(hedged_corr)
            
            print(f"\n   📊 Hedged average correlation: {hedged_avg_corr:.3f}")
            print(f"   📊 Hedged effective breadth: {hedged_breadth:.1f}/{len(stock_returns.columns)}")
            
            # 3. Results comparison
            print(f"\n📋 RESULTS COMPARISON")
            print("=" * 50)
            
            print(f"{'Method':<15} {'Avg Corr':<10} {'Breadth':<10} {'Improvement':<12}")
            print("-" * 50)
            print(f"{'No Hedge':<15} {orig_avg_corr:<10.3f} {orig_breadth:<10.1f} {'Baseline':<12}")
            print(f"{'Market Hedge':<15} {hedged_avg_corr:<10.3f} {hedged_breadth:<10.1f} {((hedged_breadth/orig_breadth-1)*100):+.1f}%")
            
            # 4. Key insights
            print(f"\n💡 KEY INSIGHTS")
            print("=" * 30)
            
            correlation_reduction = orig_avg_corr - hedged_avg_corr
            breadth_improvement = hedged_breadth - orig_breadth
            
            print(f"✅ Correlation reduced by: {correlation_reduction:.3f}")
            print(f"✅ Effective breadth increased by: {breadth_improvement:.1f}")
            print(f"✅ Breadth improvement: {((hedged_breadth/orig_breadth-1)*100):+.1f}%")
            
            # 5. Recommendations
            print(f"\n🎯 RECOMMENDATIONS")
            print("=" * 30)
            
            if hedged_breadth > orig_breadth * 1.5:
                print("✅ Market hedging is HIGHLY EFFECTIVE")
                print("   → Implement VN-Index hedging immediately")
            elif hedged_breadth > orig_breadth * 1.2:
                print("👍 Market hedging is EFFECTIVE")
                print("   → Consider implementing VN-Index hedging")
            else:
                print("⚠️ Market hedging provides LIMITED benefit")
                print("   → Evaluate costs vs. benefits")
            
            # 6. Implementation guide
            print(f"\n🛠️ IMPLEMENTATION GUIDE")
            print("=" * 30)
            print("1. Use VN-Index ETF or futures for hedging")
            print("2. Apply hedge ratios based on beta analysis:")
            
            for stock, info in beta_info.items():
                hedge_ratio = info['beta']
                print(f"   • {stock}: Hedge {hedge_ratio:.1f}% of position")
            
            print("3. Monitor correlations and rebalance monthly")
            print("4. Track performance vs. unhedged portfolio")
            
            return {
                'original': {
                    'correlation': orig_avg_corr,
                    'breadth': orig_breadth
                },
                'hedged': {
                    'correlation': hedged_avg_corr,
                    'breadth': hedged_breadth
                },
                'beta_info': beta_info,
                'improvement': {
                    'correlation_reduction': correlation_reduction,
                    'breadth_improvement': breadth_improvement,
                    'breadth_improvement_pct': (hedged_breadth/orig_breadth-1)*100
                }
            }
            
        except Exception as e:
            print(f"❌ Error in demo: {str(e)}")
            return None

def main():
    """Main demo function"""
    print("🇻🇳 Vietnam Stock Hedging - Simple Demo")
    print("=" * 50)
    print("Demonstrating core hedging concepts from AI Cafe notebooks")
    print("Using simplified calculations for compatibility")
    print()
    
    demo = SimpleHedgingDemo()
    results = demo.run_demo()
    
    if results:
        print(f"\n✅ Demo completed successfully!")
        print(f"📊 Results show the power of hedging for Vietnam stocks")
        print(f"🎯 This demonstrates the concepts from the AI Cafe notebooks")
    else:
        print(f"\n❌ Demo encountered issues")
        print(f"💡 Check data connection and try again")

if __name__ == "__main__":
    main() 