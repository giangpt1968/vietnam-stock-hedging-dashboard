import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.covariance import LedoitWolf
from vietnam_hedge_pipeline import VietnamStockDataPipeline
import warnings
warnings.filterwarnings('ignore')

class VietnamHedgingEngine:
    """Core hedging calculation engine for Vietnam stock market"""
    
    def __init__(self, data_pipeline=None):
        """Initialize the hedging engine"""
        self.data_pipeline = data_pipeline or VietnamStockDataPipeline()
        self.hedge_results = {}
        
    def calculate_beta(self, stock_returns, market_returns, method='OLS'):
        """Calculate beta using various methods"""
        try:
            if method == 'OLS':
                # Standard OLS regression
                model = sm.OLS(stock_returns, market_returns)
                results = model.fit()
                beta = results.params.iloc[0] if len(results.params) > 0 else 0
                r_squared = results.rsquared
                residuals = results.resid
                
                return {
                    'beta': beta,
                    'r_squared': r_squared,
                    'residuals': residuals,
                    'std_error': results.bse.iloc[0] if len(results.bse) > 0 else 0,
                    'p_value': results.pvalues.iloc[0] if len(results.pvalues) > 0 else 1
                }
            
            elif method == 'correlation':
                # Simple correlation-based beta
                covariance = np.cov(stock_returns, market_returns)[0, 1]
                market_variance = np.var(market_returns)
                beta = covariance / market_variance if market_variance > 0 else 0
                correlation = np.corrcoef(stock_returns, market_returns)[0, 1]
                
                return {
                    'beta': beta,
                    'r_squared': correlation**2,
                    'residuals': stock_returns - beta * market_returns,
                    'correlation': correlation
                }
                
        except Exception as e:
            print(f"⚠️ Error calculating beta: {str(e)}")
            return {
                'beta': 0,
                'r_squared': 0,
                'residuals': stock_returns * 0,
                'error': str(e)
            }
    
    def market_hedge(self, returns_data, market_col='VNINDEX', method='OLS'):
        """Perform market hedging (beta hedging)"""
        print(f"\n🔄 Performing Market Hedging...")
        
        if market_col not in returns_data.columns:
            print(f"❌ Market index '{market_col}' not found in data")
            return None
        
        market_returns = returns_data[market_col]
        stock_returns = returns_data.drop(market_col, axis=1)
        
        # Calculate market hedged residuals
        market_hedged_residuals = stock_returns.copy() * 0
        beta_info = {}
        
        for stock in stock_returns.columns:
            print(f"   📊 Calculating beta for {stock}...")
            
            beta_result = self.calculate_beta(stock_returns[stock], market_returns, method)
            beta_info[stock] = beta_result
            market_hedged_residuals[stock] = beta_result['residuals']
        
        # Calculate correlation improvement
        original_corr = self.calculate_correlation_matrix(stock_returns)
        hedged_corr = self.calculate_correlation_matrix(market_hedged_residuals)
        
        result = {
            'type': 'market_hedge',
            'original_returns': stock_returns,
            'hedged_residuals': market_hedged_residuals,
            'beta_info': beta_info,
            'original_correlation': original_corr,
            'hedged_correlation': hedged_corr,
            'market_returns': market_returns
        }
        
        self.hedge_results['market_hedge'] = result
        print(f"✅ Market hedging completed for {len(stock_returns.columns)} stocks")
        
        return result
    
    def sector_hedge(self, market_hedge_result, sector_mapping=None):
        """Perform sector hedging on market-hedged residuals"""
        print(f"\n🔄 Performing Sector Hedging...")
        
        if market_hedge_result is None:
            print("❌ Market hedge result required for sector hedging")
            return None
        
        market_residuals = market_hedge_result['hedged_residuals']
        market_returns = market_hedge_result['market_returns']
        
        # Auto-detect sectors if not provided
        if sector_mapping is None:
            sector_mapping = self._auto_detect_sectors(market_residuals)
        
        # Calculate sector benchmark residuals first
        sector_benchmarks = {}
        for sector, info in sector_mapping.items():
            if 'benchmark' in info:
                benchmark = info['benchmark']
                if benchmark in market_hedge_result['original_returns'].columns:
                    # Get benchmark's market-hedged residuals
                    benchmark_residuals = market_hedge_result['hedged_residuals'][benchmark]
                    sector_benchmarks[sector] = benchmark_residuals
                    print(f"   📊 Sector {sector} benchmark: {benchmark}")
        
        # Perform sector hedging
        sector_hedged_residuals = market_residuals.copy()
        sector_beta_info = {}
        
        for sector, info in sector_mapping.items():
            if sector not in sector_benchmarks:
                continue
                
            sector_benchmark_residuals = sector_benchmarks[sector]
            
            for stock in info['stocks']:
                if stock in market_residuals.columns:
                    print(f"   📊 Sector hedging {stock} vs {sector}...")
                    
                    # Calculate sector beta on market-hedged residuals
                    sector_beta_result = self.calculate_beta(
                        market_residuals[stock], 
                        sector_benchmark_residuals
                    )
                    
                    sector_beta_info[stock] = {
                        'sector': sector,
                        'benchmark': info['benchmark'],
                        **sector_beta_result
                    }
                    
                    # Apply sector hedge
                    sector_hedged_residuals[stock] = sector_beta_result['residuals']
        
        # Calculate final correlation matrix
        final_corr = self.calculate_correlation_matrix(sector_hedged_residuals)
        
        result = {
            'type': 'sector_hedge',
            'market_hedged_residuals': market_residuals,
            'sector_hedged_residuals': sector_hedged_residuals,
            'sector_beta_info': sector_beta_info,
            'sector_mapping': sector_mapping,
            'final_correlation': final_corr
        }
        
        self.hedge_results['sector_hedge'] = result
        print(f"✅ Sector hedging completed")
        
        return result
    
    def calculate_correlation_matrix(self, data, method='ledoit_wolf'):
        """Calculate correlation matrix using various methods"""
        try:
            if method == 'ledoit_wolf':
                # Use Ledoit-Wolf shrinkage estimator
                lw = LedoitWolf()
                cov_matrix = lw.fit(data).covariance_
                
                # Convert covariance to correlation
                d = np.diag(np.sqrt(np.diag(cov_matrix)))
                d_inv = np.linalg.inv(d)
                corr_matrix = d_inv @ cov_matrix @ d_inv
                
                return pd.DataFrame(corr_matrix, index=data.columns, columns=data.columns)
            
            elif method == 'pearson':
                return data.corr()
                
        except Exception as e:
            print(f"⚠️ Error calculating correlation matrix: {str(e)}")
            return data.corr()  # Fallback to simple correlation
    
    def calculate_effective_breadth(self, correlation_matrix):
        """Calculate effective breadth using Buckle's formula"""
        # Get upper triangular correlations (excluding diagonal)
        upper_tri = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)]
        avg_correlation = np.mean(upper_tri)
        
        N = len(correlation_matrix)
        effective_breadth = N / (1 + avg_correlation * (N - 1))
        
        return {
            'effective_breadth': effective_breadth,
            'average_correlation': avg_correlation,
            'num_assets': N,
            'theoretical_max': N
        }
    
    def _auto_detect_sectors(self, returns_data, correlation_threshold=0.6):
        """Auto-detect sectors based on correlation clustering"""
        print(f"   🔍 Auto-detecting sectors...")
        
        # Use data pipeline's sector mapping as base
        if hasattr(self.data_pipeline, 'sectors'):
            sector_mapping = {}
            for sector_name, sector_info in self.data_pipeline.sectors.items():
                stocks_in_data = [s for s in sector_info['stocks'] if s in returns_data.columns]
                if len(stocks_in_data) >= 2:  # Need at least 2 stocks for sector
                    sector_mapping[sector_name] = {
                        'stocks': stocks_in_data,
                        'benchmark': sector_info['benchmark']
                    }
            return sector_mapping
        
        # Fallback: simple correlation-based clustering
        corr_matrix = returns_data.corr()
        # This is a simplified version - in practice, you'd use more sophisticated clustering
        return {}
    
    def comprehensive_hedge_analysis(self, returns_data, market_col='VNINDEX'):
        """Perform comprehensive hedging analysis"""
        print(f"\n🚀 Starting Comprehensive Hedge Analysis")
        print(f"📊 Data shape: {returns_data.shape}")
        print(f"📈 Assets: {list(returns_data.columns)}")
        
        results = {}
        
        # 1. Original (no hedge) analysis
        print(f"\n1️⃣ Analyzing Original Data (No Hedge)...")
        original_stocks = returns_data.drop(market_col, axis=1, errors='ignore')
        original_corr = self.calculate_correlation_matrix(original_stocks)
        original_breadth = self.calculate_effective_breadth(original_corr)
        
        results['no_hedge'] = {
            'correlation_matrix': original_corr,
            'breadth_analysis': original_breadth,
            'data': original_stocks
        }
        
        # 2. Market hedge analysis
        print(f"\n2️⃣ Performing Market Hedge Analysis...")
        market_hedge_result = self.market_hedge(returns_data, market_col)
        
        if market_hedge_result:
            market_breadth = self.calculate_effective_breadth(market_hedge_result['hedged_correlation'])
            results['market_hedge'] = {
                **market_hedge_result,
                'breadth_analysis': market_breadth
            }
        
        # 3. Sector hedge analysis
        print(f"\n3️⃣ Performing Sector Hedge Analysis...")
        if market_hedge_result:
            sector_hedge_result = self.sector_hedge(market_hedge_result)
            
            if sector_hedge_result:
                sector_breadth = self.calculate_effective_breadth(sector_hedge_result['final_correlation'])
                results['sector_hedge'] = {
                    **sector_hedge_result,
                    'breadth_analysis': sector_breadth
                }
        
        # 4. Summary comparison
        print(f"\n4️⃣ Generating Summary Report...")
        summary = self._generate_summary_report(results)
        results['summary'] = summary
        
        return results
    
    def _generate_summary_report(self, results):
        """Generate comprehensive summary report"""
        summary = {
            'comparison_table': [],
            'improvements': {},
            'recommendations': []
        }
        
        # Create comparison table
        for method in ['no_hedge', 'market_hedge', 'sector_hedge']:
            if method in results and 'breadth_analysis' in results[method]:
                breadth_info = results[method]['breadth_analysis']
                summary['comparison_table'].append({
                    'method': method.replace('_', ' ').title(),
                    'avg_correlation': breadth_info['average_correlation'],
                    'effective_breadth': breadth_info['effective_breadth'],
                    'num_assets': breadth_info['num_assets']
                })
        
        # Calculate improvements
        if len(summary['comparison_table']) >= 2:
            base = summary['comparison_table'][0]  # no_hedge
            for i in range(1, len(summary['comparison_table'])):
                method = summary['comparison_table'][i]
                improvement = {
                    'method': method['method'],
                    'correlation_change': method['avg_correlation'] - base['avg_correlation'],
                    'breadth_improvement': method['effective_breadth'] - base['effective_breadth'],
                    'breadth_improvement_pct': (method['effective_breadth'] / base['effective_breadth'] - 1) * 100
                }
                summary['improvements'][method['method']] = improvement
        
        # Generate recommendations
        if 'Market Hedge' in summary['improvements']:
            market_improvement = summary['improvements']['Market Hedge']['breadth_improvement_pct']
            if market_improvement > 50:
                summary['recommendations'].append("✅ Market hedging is highly effective - strongly recommended")
            elif market_improvement > 20:
                summary['recommendations'].append("👍 Market hedging provides good improvement - recommended")
            else:
                summary['recommendations'].append("⚠️ Market hedging provides limited benefit - consider costs")
        
        if 'Sector Hedge' in summary['improvements']:
            sector_improvement = summary['improvements']['Sector Hedge']['breadth_improvement_pct']
            if sector_improvement > 100:
                summary['recommendations'].append("✅ Sector hedging is very effective - implement immediately")
            elif sector_improvement > 50:
                summary['recommendations'].append("👍 Sector hedging provides significant benefit - recommended")
            else:
                summary['recommendations'].append("⚠️ Sector hedging provides marginal benefit - evaluate costs")
        
        return summary


# Test the hedging engine
if __name__ == "__main__":
    print("🚀 Testing Vietnam Hedging Engine")
    print("=" * 50)
    
    # Initialize pipeline and engine
    pipeline = VietnamStockDataPipeline()
    engine = VietnamHedgingEngine(pipeline)
    
    # Get test data
    print("\n📡 Loading test data...")
    demo_prices = pipeline.get_demo_portfolio()
    
    if demo_prices is not None:
        demo_returns = pipeline.calculate_returns(demo_prices)
        
        # Run comprehensive analysis
        print("\n🔄 Running comprehensive hedge analysis...")
        results = engine.comprehensive_hedge_analysis(demo_returns)
        
        # Display results
        print("\n📊 HEDGE ANALYSIS RESULTS")
        print("=" * 50)
        
        if 'summary' in results:
            summary = results['summary']
            
            # Show comparison table
            print("\n📋 Comparison Table:")
            for row in summary['comparison_table']:
                print(f"   {row['method']:<15}: "
                      f"Corr={row['avg_correlation']:.3f}, "
                      f"BR={row['effective_breadth']:.1f}/{row['num_assets']}")
            
            # Show improvements
            if summary['improvements']:
                print("\n📈 Improvements:")
                for method, improvement in summary['improvements'].items():
                    print(f"   {method}: "
                          f"BR +{improvement['breadth_improvement']:.1f} "
                          f"({improvement['breadth_improvement_pct']:+.1f}%)")
            
            # Show recommendations
            if summary['recommendations']:
                print("\n💡 Recommendations:")
                for rec in summary['recommendations']:
                    print(f"   {rec}")
        
        print("\n✅ Hedging engine testing completed!")
    else:
        print("❌ Could not load test data") 