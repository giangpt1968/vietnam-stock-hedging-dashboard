#!/usr/bin/env python3
"""
Vietnam Stock Backtesting Engine
================================

Comprehensive backtesting framework for hedge strategies with:
- Multiple hedge strategies (No Hedge, Market Hedge, Sector Hedge)
- Performance metrics and risk analysis
- Transaction costs and slippage modeling
- Walk-forward analysis
- Strategy comparison and optimization

Author: AI Cafe Vietnam
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class VietnamBacktestingEngine:
    """Advanced backtesting engine for Vietnam stock hedge strategies"""
    
    def __init__(self, transaction_cost=0.0015, slippage=0.001):
        """
        Initialize backtesting engine
        
        Parameters:
        -----------
        transaction_cost : float
            Transaction cost as decimal (0.15% = 0.0015)
        slippage : float
            Market impact/slippage as decimal (0.1% = 0.001)
        """
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.results = {}
    
    def run_comprehensive_backtest(self, returns_data, initial_capital=1000000, 
                                 rebalance_frequency='monthly', 
                                 hedge_strategies=['no_hedge', 'market_hedge', 'sector_hedge']):
        """
        Run comprehensive backtest across multiple strategies
        
        Parameters:
        -----------
        returns_data : DataFrame
            Daily returns data with VN-Index
        initial_capital : float
            Starting capital in VND
        rebalance_frequency : str
            'daily', 'weekly', 'monthly', 'quarterly'
        hedge_strategies : list
            List of strategies to test
        
        Returns:
        --------
        dict : Comprehensive backtest results
        """
        print(f"🚀 Starting Comprehensive Backtest")
        print(f"📊 Data shape: {returns_data.shape}")
        print(f"💰 Initial capital: {initial_capital:,.0f} VND")
        print(f"🔄 Rebalance frequency: {rebalance_frequency}")
        
        results = {}
        
        # Get stock symbols (exclude VN-Index)
        stock_columns = [col for col in returns_data.columns if col != 'VNINDEX']
        
        for strategy in hedge_strategies:
            print(f"\n🎯 Testing Strategy: {strategy.upper()}")
            
            if strategy == 'no_hedge':
                strategy_returns = self._backtest_no_hedge(returns_data, stock_columns)
            elif strategy == 'market_hedge':
                strategy_returns = self._backtest_market_hedge(returns_data, stock_columns)
            elif strategy == 'sector_hedge':
                strategy_returns = self._backtest_sector_hedge(returns_data, stock_columns)
            elif strategy == 'dynamic_hedge':
                strategy_returns = self._backtest_dynamic_hedge(returns_data, stock_columns)
            
            # Calculate portfolio value
            portfolio_values = self._calculate_portfolio_values(
                strategy_returns, initial_capital, rebalance_frequency
            )
            
            # Calculate comprehensive metrics
            metrics = self._calculate_performance_metrics(portfolio_values, strategy_returns)
            
            results[strategy] = {
                'returns': strategy_returns,
                'portfolio_values': portfolio_values,
                'metrics': metrics,
                'drawdowns': self._calculate_drawdowns(portfolio_values),
                'rolling_metrics': self._calculate_rolling_metrics(strategy_returns)
            }
        
        # Add benchmark (VN-Index)
        if 'VNINDEX' in returns_data.columns:
            benchmark_values = self._calculate_portfolio_values(
                returns_data[['VNINDEX']], initial_capital, 'daily'
            )
            results['benchmark'] = {
                'returns': returns_data[['VNINDEX']],
                'portfolio_values': benchmark_values,
                'metrics': self._calculate_performance_metrics(benchmark_values, returns_data[['VNINDEX']]),
                'drawdowns': self._calculate_drawdowns(benchmark_values),
                'rolling_metrics': self._calculate_rolling_metrics(returns_data[['VNINDEX']])
            }
        
        self.results = results
        print(f"\n✅ Backtest completed for {len(results)} strategies")
        return results
    
    def _backtest_no_hedge(self, returns_data, stock_columns):
        """Backtest equal-weight no hedge strategy"""
        # Equal weight portfolio
        n_stocks = len(stock_columns)
        portfolio_returns = returns_data[stock_columns].mean(axis=1)
        
        return pd.DataFrame({
            'portfolio': portfolio_returns,
            'strategy': 'no_hedge'
        })
    
    def _backtest_market_hedge(self, returns_data, stock_columns):
        """Backtest market-neutral hedge strategy"""
        # Calculate betas for each stock
        market_returns = returns_data['VNINDEX']
        hedged_returns = []
        
        for stock in stock_columns:
            stock_returns = returns_data[stock]
            
            # Calculate rolling beta (60-day window)
            beta = self._calculate_rolling_beta(stock_returns, market_returns, window=60)
            
            # Market hedge: Stock return - Beta * Market return
            hedged_return = stock_returns - beta * market_returns
            hedged_returns.append(hedged_return)
        
        # Equal weight hedged portfolio
        portfolio_returns = pd.concat(hedged_returns, axis=1).mean(axis=1)
        
        return pd.DataFrame({
            'portfolio': portfolio_returns,
            'strategy': 'market_hedge'
        })
    
    def _backtest_sector_hedge(self, returns_data, stock_columns):
        """Backtest sector-neutral hedge strategy"""
        # Auto-detect sectors and create sector benchmarks
        sectors = self._detect_sectors(stock_columns)
        
        hedged_returns = []
        
        for stock in stock_columns:
            stock_returns = returns_data[stock]
            
            # Find sector for this stock
            stock_sector = None
            for sector, stocks in sectors.items():
                if stock in stocks:
                    stock_sector = sector
                    break
            
            if stock_sector and len(sectors[stock_sector]) > 1:
                # Create sector benchmark (excluding the stock itself)
                sector_stocks = [s for s in sectors[stock_sector] if s != stock and s in stock_columns]
                if sector_stocks:
                    sector_benchmark = returns_data[sector_stocks].mean(axis=1)
                    
                    # Calculate beta vs sector
                    beta = self._calculate_rolling_beta(stock_returns, sector_benchmark, window=60)
                    
                    # Sector hedge: Stock return - Beta * Sector return
                    hedged_return = stock_returns - beta * sector_benchmark
                    hedged_returns.append(hedged_return)
                else:
                    # Fall back to market hedge
                    beta = self._calculate_rolling_beta(stock_returns, returns_data['VNINDEX'], window=60)
                    hedged_return = stock_returns - beta * returns_data['VNINDEX']
                    hedged_returns.append(hedged_return)
            else:
                # Fall back to market hedge
                beta = self._calculate_rolling_beta(stock_returns, returns_data['VNINDEX'], window=60)
                hedged_return = stock_returns - beta * returns_data['VNINDEX']
                hedged_returns.append(hedged_return)
        
        # Equal weight hedged portfolio
        portfolio_returns = pd.concat(hedged_returns, axis=1).mean(axis=1)
        
        return pd.DataFrame({
            'portfolio': portfolio_returns,
            'strategy': 'sector_hedge'
        })
    
    def _backtest_dynamic_hedge(self, returns_data, stock_columns):
        """Backtest dynamic hedge strategy (switches based on market conditions)"""
        market_returns = returns_data['VNINDEX']
        
        # Calculate market volatility (rolling 30-day)
        market_vol = market_returns.rolling(30).std() * np.sqrt(252)
        
        # Dynamic strategy: Use sector hedge in low vol, market hedge in high vol
        vol_threshold = market_vol.median()
        
        hedged_returns = []
        
        for stock in stock_columns:
            stock_returns = returns_data[stock]
            stock_hedged = []
            
            for i, date in enumerate(returns_data.index):
                if i < 60:  # Not enough data for beta calculation
                    stock_hedged.append(stock_returns.iloc[i])
                    continue
                
                current_vol = market_vol.iloc[i]
                
                if pd.isna(current_vol) or current_vol < vol_threshold:
                    # Low volatility: Use sector hedge
                    sectors = self._detect_sectors(stock_columns)
                    stock_sector = None
                    for sector, stocks in sectors.items():
                        if stock in stocks:
                            stock_sector = sector
                            break
                    
                    if stock_sector and len(sectors[stock_sector]) > 1:
                        sector_stocks = [s for s in sectors[stock_sector] if s != stock and s in stock_columns]
                        if sector_stocks:
                            sector_benchmark = returns_data[sector_stocks].iloc[:i+1].mean(axis=1)
                            beta = self._calculate_rolling_beta(
                                stock_returns.iloc[:i+1], sector_benchmark, window=60
                            ).iloc[-1]
                            hedged_ret = stock_returns.iloc[i] - beta * sector_benchmark.iloc[-1]
                        else:
                            beta = self._calculate_rolling_beta(
                                stock_returns.iloc[:i+1], market_returns.iloc[:i+1], window=60
                            ).iloc[-1]
                            hedged_ret = stock_returns.iloc[i] - beta * market_returns.iloc[i]
                    else:
                        beta = self._calculate_rolling_beta(
                            stock_returns.iloc[:i+1], market_returns.iloc[:i+1], window=60
                        ).iloc[-1]
                        hedged_ret = stock_returns.iloc[i] - beta * market_returns.iloc[i]
                else:
                    # High volatility: Use market hedge
                    beta = self._calculate_rolling_beta(
                        stock_returns.iloc[:i+1], market_returns.iloc[:i+1], window=60
                    ).iloc[-1]
                    hedged_ret = stock_returns.iloc[i] - beta * market_returns.iloc[i]
                
                stock_hedged.append(hedged_ret)
            
            hedged_returns.append(pd.Series(stock_hedged, index=returns_data.index))
        
        # Equal weight hedged portfolio
        portfolio_returns = pd.concat(hedged_returns, axis=1).mean(axis=1)
        
        return pd.DataFrame({
            'portfolio': portfolio_returns,
            'strategy': 'dynamic_hedge'
        })
    
    def _calculate_rolling_beta(self, stock_returns, market_returns, window=60):
        """Calculate rolling beta"""
        def calc_beta(x, y):
            if len(x) < 10:  # Need minimum data
                return 1.0
            covariance = np.cov(x, y)[0, 1]
            variance = np.var(y)
            return covariance / variance if variance > 0 else 1.0
        
        rolling_beta = []
        for i in range(len(stock_returns)):
            if i < window:
                rolling_beta.append(1.0)
            else:
                x = stock_returns.iloc[i-window:i].values
                y = market_returns.iloc[i-window:i].values
                
                # Remove NaN values
                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() < 10:
                    rolling_beta.append(1.0)
                else:
                    beta = calc_beta(x[mask], y[mask])
                    rolling_beta.append(beta)
        
        return pd.Series(rolling_beta, index=stock_returns.index)
    
    def _detect_sectors(self, stock_columns):
        """Auto-detect sectors based on stock symbols"""
        sectors = {
            'Banking': ['BID', 'CTG', 'ACB', 'VCB', 'TCB', 'VPB', 'STB', 'TPB', 'MBB', 'SHB'],
            'RealEstate': ['VHM', 'VIC', 'VRE', 'DXG', 'HDG', 'KDH', 'PDR', 'DIG', 'NLG', 'IJC'],
            'Industrial': ['HPG', 'HSG', 'NKG', 'SMC', 'POM', 'CSV', 'TLH', 'DPM', 'TVB', 'PVD'],
            'Technology': ['FPT', 'CMG', 'SBT', 'ITD', 'ELC', 'TNI', 'PTI', 'APS', 'MFS', 'DST'],
            'Energy': ['GAS', 'PLX', 'PVS', 'PVC', 'PVD', 'PSH', 'PVB', 'PGC', 'PVG', 'BSR'],
            'Retail': ['MWG', 'PNJ', 'FRT', 'DGW', 'VGC', 'SFI', 'CRC', 'FTM', 'ASM', 'TNG'],
            'Food': ['VNM', 'MSN', 'SAB', 'KDC', 'BBC', 'LAF', 'TLG', 'ANV', 'VHC', 'QNS']
        }
        
        # Filter sectors to only include stocks present in the data
        filtered_sectors = {}
        for sector, stocks in sectors.items():
            sector_stocks = [stock for stock in stocks if stock in stock_columns]
            if len(sector_stocks) >= 2:  # Need at least 2 stocks for sector
                filtered_sectors[sector] = sector_stocks
        
        return filtered_sectors
    
    def _calculate_portfolio_values(self, returns, initial_capital, rebalance_frequency):
        """Calculate portfolio values with transaction costs"""
        portfolio_values = [initial_capital]
        current_value = initial_capital
        
        # Apply transaction costs based on rebalance frequency
        if rebalance_frequency == 'daily':
            transaction_days = 1
        elif rebalance_frequency == 'weekly':
            transaction_days = 5
        elif rebalance_frequency == 'monthly':
            transaction_days = 22
        elif rebalance_frequency == 'quarterly':
            transaction_days = 66
        else:
            transaction_days = 22  # Default monthly
        
        for i, (date, row) in enumerate(returns.iterrows()):
            if len(row) == 1:
                # Single asset (benchmark)
                daily_return = row.iloc[0]
            else:
                # Portfolio
                daily_return = row['portfolio']
            
            # Apply slippage on every trade
            if not pd.isna(daily_return):
                adjusted_return = daily_return - self.slippage
                current_value *= (1 + adjusted_return)
            
            # Apply transaction costs on rebalance days
            if i % transaction_days == 0 and i > 0:
                current_value *= (1 - self.transaction_cost)
            
            portfolio_values.append(current_value)
        
        return pd.Series(portfolio_values[1:], index=returns.index)
    
    def _calculate_performance_metrics(self, portfolio_values, returns):
        """Calculate comprehensive performance metrics"""
        if len(portfolio_values) < 2:
            return {}
        
        # Calculate returns from portfolio values
        portfolio_returns = portfolio_values.pct_change().dropna()
        
        if len(portfolio_returns) == 0:
            return {}
        
        # Basic metrics
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        
        # Annualized return (CAGR)
        years = len(portfolio_values) / 252
        cagr = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** (1/years) - 1 if years > 0 else 0
        
        # Volatility
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 3% risk-free rate)
        risk_free_rate = 0.03
        sharpe_ratio = (cagr - risk_free_rate) / annual_vol if annual_vol > 0 else 0
        
        # Maximum drawdown
        cummax = portfolio_values.cummax()
        drawdowns = (portfolio_values - cummax) / cummax
        max_drawdown = drawdowns.min()
        
        # Sortino ratio
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (cagr - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar ratio
        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # Win rate
        win_rate = (portfolio_returns > 0).sum() / len(portfolio_returns)
        
        # Value at Risk (95%)
        var_95 = np.percentile(portfolio_returns, 5)
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'var_95': var_95,
            'final_value': portfolio_values.iloc[-1],
            'best_day': portfolio_returns.max(),
            'worst_day': portfolio_returns.min()
        }
    
    def _calculate_drawdowns(self, portfolio_values):
        """Calculate detailed drawdown analysis"""
        cummax = portfolio_values.cummax()
        drawdowns = (portfolio_values - cummax) / cummax
        
        return {
            'drawdown_series': drawdowns,
            'max_drawdown': drawdowns.min(),
            'current_drawdown': drawdowns.iloc[-1],
            'avg_drawdown': drawdowns[drawdowns < 0].mean(),
            'drawdown_duration': self._calculate_drawdown_duration(drawdowns)
        }
    
    def _calculate_drawdown_duration(self, drawdowns):
        """Calculate average drawdown duration"""
        in_drawdown = drawdowns < -0.001  # 0.1% threshold
        
        if not in_drawdown.any():
            return 0
        
        # Find drawdown periods
        drawdown_periods = []
        start = None
        
        for i, is_dd in enumerate(in_drawdown):
            if is_dd and start is None:
                start = i
            elif not is_dd and start is not None:
                drawdown_periods.append(i - start)
                start = None
        
        # Handle case where drawdown continues to the end
        if start is not None:
            drawdown_periods.append(len(in_drawdown) - start)
        
        return np.mean(drawdown_periods) if drawdown_periods else 0
    
    def _calculate_rolling_metrics(self, returns):
        """Calculate rolling performance metrics"""
        if len(returns) < 252:
            return {}
        
        rolling_window = 252  # 1 year
        
        if hasattr(returns, 'iloc') and len(returns.shape) > 1:
            # DataFrame
            portfolio_returns = returns['portfolio'] if 'portfolio' in returns.columns else returns.iloc[:, 0]
        else:
            # Series
            portfolio_returns = returns.iloc[:, 0] if hasattr(returns, 'iloc') else returns
        
        rolling_sharpe = []
        rolling_vol = []
        
        for i in range(rolling_window, len(portfolio_returns)):
            window_returns = portfolio_returns.iloc[i-rolling_window:i]
            
            # Annual volatility
            vol = window_returns.std() * np.sqrt(252)
            rolling_vol.append(vol)
            
            # Sharpe ratio
            annual_return = (1 + window_returns.mean()) ** 252 - 1
            sharpe = (annual_return - 0.03) / vol if vol > 0 else 0
            rolling_sharpe.append(sharpe)
        
        return {
            'rolling_sharpe': pd.Series(rolling_sharpe, index=portfolio_returns.index[rolling_window:]),
            'rolling_volatility': pd.Series(rolling_vol, index=portfolio_returns.index[rolling_window:])
        }
    
    def generate_strategy_comparison(self):
        """Generate comprehensive strategy comparison"""
        if not self.results:
            return None
        
        comparison_data = []
        
        for strategy, data in self.results.items():
            metrics = data['metrics']
            comparison_data.append({
                'Strategy': strategy.replace('_', ' ').title(),
                'Total Return': f"{metrics.get('total_return', 0):.1%}",
                'CAGR': f"{metrics.get('cagr', 0):.1%}",
                'Volatility': f"{metrics.get('annual_volatility', 0):.1%}",
                'Sharpe Ratio': f"{metrics.get('sharpe_ratio', 0):.2f}",
                'Max Drawdown': f"{metrics.get('max_drawdown', 0):.1%}",
                'Win Rate': f"{metrics.get('win_rate', 0):.1%}",
                'Final Value': f"{metrics.get('final_value', 0):,.0f}"
            })
        
        return pd.DataFrame(comparison_data)
    
    def get_monte_carlo_analysis(self, n_simulations=1000):
        """Run Monte Carlo simulation for strategy robustness"""
        if not self.results:
            return None
        
        print(f"🎲 Running Monte Carlo Analysis ({n_simulations} simulations)")
        
        mc_results = {}
        
        for strategy, data in self.results.items():
            if strategy == 'benchmark':
                continue
                
            returns = data['returns']['portfolio']
            
            # Bootstrap simulation
            simulated_results = []
            
            for i in range(n_simulations):
                # Randomly sample returns with replacement
                sampled_returns = np.random.choice(returns.dropna(), size=len(returns), replace=True)
                
                # Calculate cumulative return
                final_return = (1 + pd.Series(sampled_returns)).prod() - 1
                simulated_results.append(final_return)
            
            mc_results[strategy] = {
                'mean_return': np.mean(simulated_results),
                'std_return': np.std(simulated_results),
                'percentile_5': np.percentile(simulated_results, 5),
                'percentile_95': np.percentile(simulated_results, 95),
                'success_rate': np.mean(np.array(simulated_results) > 0)
            }
        
        return mc_results 