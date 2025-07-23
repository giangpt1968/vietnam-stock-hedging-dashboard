import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from vietnam_hedge_pipeline import VietnamStockDataPipeline
from vietnam_hedging_engine import VietnamHedgingEngine
import warnings
warnings.filterwarnings('ignore')

class VietnamHedgeMonitor:
    """Advanced monitoring and alerting system for Vietnam stock hedging"""
    
    def __init__(self, data_pipeline=None, hedging_engine=None):
        """Initialize the monitoring system"""
        self.data_pipeline = data_pipeline or VietnamStockDataPipeline()
        self.hedging_engine = hedging_engine or VietnamHedgingEngine(self.data_pipeline)
        
        # Alert thresholds
        self.alert_thresholds = {
            'high_correlation': 0.7,
            'medium_correlation': 0.5,
            'low_correlation': 0.3,
            'volatility_spike': 0.05,  # 5% daily move
            'breadth_degradation': 0.2  # 20% reduction in effective breadth
        }
        
        # Monitoring history
        self.monitoring_history = []
        
    def advanced_sector_detection(self, returns_data, n_clusters=None, method='kmeans'):
        """Advanced sector detection using machine learning clustering"""
        print(f"\n🔍 Advanced Sector Detection using {method.upper()}...")
        
        # Prepare correlation matrix
        corr_matrix = returns_data.corr()
        
        if method == 'kmeans':
            return self._kmeans_sector_detection(corr_matrix, n_clusters)
        elif method == 'hierarchical':
            return self._hierarchical_sector_detection(corr_matrix)
        elif method == 'correlation_threshold':
            return self._correlation_threshold_detection(corr_matrix)
        else:
            print(f"❌ Unknown method: {method}")
            return None
    
    def _kmeans_sector_detection(self, corr_matrix, n_clusters=None):
        """K-means clustering for sector detection"""
        
        # Auto-determine number of clusters if not specified
        if n_clusters is None:
            n_clusters = min(5, len(corr_matrix) // 3)  # Reasonable default
        
        # Use correlation matrix as features
        features = corr_matrix.values
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Organize results
        sectors = {}
        for i, stock in enumerate(corr_matrix.index):
            cluster_id = cluster_labels[i]
            sector_name = f"Cluster_{cluster_id}"
            
            if sector_name not in sectors:
                sectors[sector_name] = {'stocks': [], 'avg_correlation': 0}
            
            sectors[sector_name]['stocks'].append(stock)
        
        # Calculate average intra-cluster correlation
        for sector_name, sector_info in sectors.items():
            stocks = sector_info['stocks']
            if len(stocks) > 1:
                sector_corr = corr_matrix.loc[stocks, stocks]
                # Get upper triangular correlations
                upper_tri = sector_corr.values[np.triu_indices_from(sector_corr.values, k=1)]
                sector_info['avg_correlation'] = np.mean(upper_tri)
                # Choose most representative stock as benchmark
                sector_info['benchmark'] = stocks[0]  # Simple heuristic
            else:
                sector_info['avg_correlation'] = 0
                sector_info['benchmark'] = stocks[0]
        
        print(f"✅ Detected {len(sectors)} sectors using K-means")
        return sectors
    
    def _correlation_threshold_detection(self, corr_matrix, threshold=0.6):
        """Correlation threshold-based sector detection"""
        
        sectors = {}
        assigned_stocks = set()
        sector_counter = 0
        
        for stock in corr_matrix.index:
            if stock in assigned_stocks:
                continue
                
            # Find highly correlated stocks
            high_corr_stocks = []
            for other_stock in corr_matrix.index:
                if other_stock != stock and other_stock not in assigned_stocks:
                    correlation = corr_matrix.loc[stock, other_stock]
                    if correlation > threshold:
                        high_corr_stocks.append(other_stock)
            
            if high_corr_stocks:
                # Create new sector
                sector_name = f"Sector_{sector_counter}"
                all_stocks = [stock] + high_corr_stocks
                
                sectors[sector_name] = {
                    'stocks': all_stocks,
                    'benchmark': stock,  # First stock as benchmark
                    'avg_correlation': np.mean([corr_matrix.loc[stock, other] 
                                              for other in high_corr_stocks])
                }
                
                # Mark stocks as assigned
                for s in all_stocks:
                    assigned_stocks.add(s)
                
                sector_counter += 1
        
        # Handle unassigned stocks
        unassigned = [s for s in corr_matrix.index if s not in assigned_stocks]
        if unassigned:
            sectors['Miscellaneous'] = {
                'stocks': unassigned,
                'benchmark': unassigned[0],
                'avg_correlation': 0
            }
        
        print(f"✅ Detected {len(sectors)} sectors using correlation threshold")
        return sectors
    
    def real_time_correlation_monitor(self, returns_data, window=30):
        """Real-time correlation monitoring with alerts"""
        print(f"\n📊 Real-time Correlation Monitoring (window={window} days)...")
        
        if len(returns_data) < window:
            print(f"❌ Insufficient data: need at least {window} days")
            return None
        
        # Calculate rolling correlation
        rolling_correlations = []
        dates = []
        
        for i in range(window, len(returns_data)):
            window_data = returns_data.iloc[i-window:i]
            corr_matrix = window_data.corr()
            
            # Calculate average correlation
            upper_tri = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
            avg_corr = np.mean(upper_tri)
            
            rolling_correlations.append(avg_corr)
            dates.append(returns_data.index[i])
        
        # Create monitoring DataFrame
        monitoring_df = pd.DataFrame({
            'date': dates,
            'avg_correlation': rolling_correlations
        }).set_index('date')
        
        # Generate alerts
        alerts = self._generate_correlation_alerts(monitoring_df)
        
        # Calculate effective breadth over time
        n_assets = len(returns_data.columns)
        monitoring_df['effective_breadth'] = n_assets / (1 + monitoring_df['avg_correlation'] * (n_assets - 1))
        
        result = {
            'monitoring_data': monitoring_df,
            'alerts': alerts,
            'current_correlation': rolling_correlations[-1] if rolling_correlations else 0,
            'current_breadth': monitoring_df['effective_breadth'].iloc[-1] if len(monitoring_df) > 0 else 0
        }
        
        print(f"✅ Monitoring completed. Current correlation: {result['current_correlation']:.3f}")
        return result
    
    def _generate_correlation_alerts(self, monitoring_df):
        """Generate correlation-based alerts"""
        alerts = []
        current_corr = monitoring_df['avg_correlation'].iloc[-1]
        
        # Recent trend (last 5 days)
        if len(monitoring_df) >= 5:
            recent_trend = monitoring_df['avg_correlation'].iloc[-5:].mean()
            previous_trend = monitoring_df['avg_correlation'].iloc[-10:-5].mean() if len(monitoring_df) >= 10 else recent_trend
            
            trend_change = recent_trend - previous_trend
            
            if current_corr > self.alert_thresholds['high_correlation']:
                alerts.append({
                    'level': 'CRITICAL',
                    'message': f"🚨 Very high correlation detected: {current_corr:.3f}",
                    'recommendation': "Increase hedging immediately",
                    'timestamp': monitoring_df.index[-1]
                })
            
            elif current_corr > self.alert_thresholds['medium_correlation']:
                alerts.append({
                    'level': 'WARNING',
                    'message': f"⚠️ High correlation detected: {current_corr:.3f}",
                    'recommendation': "Consider increasing hedge positions",
                    'timestamp': monitoring_df.index[-1]
                })
            
            if trend_change > 0.1:
                alerts.append({
                    'level': 'INFO',
                    'message': f"📈 Correlation trending up: +{trend_change:.3f}",
                    'recommendation': "Monitor closely for hedging opportunities",
                    'timestamp': monitoring_df.index[-1]
                })
        
        return alerts
    
    def hedge_effectiveness_tracker(self, returns_data, rebalance_frequency='weekly'):
        """Track hedge effectiveness over time"""
        print(f"\n📈 Tracking Hedge Effectiveness ({rebalance_frequency})...")
        
        # Determine rebalance periods
        if rebalance_frequency == 'daily':
            rebalance_days = 1
        elif rebalance_frequency == 'weekly':
            rebalance_days = 5
        elif rebalance_frequency == 'monthly':
            rebalance_days = 22
        else:
            rebalance_days = 5
        
        effectiveness_history = []
        
        # Rolling hedge effectiveness calculation
        for i in range(60, len(returns_data), rebalance_days):  # Start after 60 days
            window_data = returns_data.iloc[i-60:i]  # 60-day window
            
            try:
                # Run hedge analysis
                hedge_results = self.hedging_engine.comprehensive_hedge_analysis(window_data)
                
                if 'summary' in hedge_results:
                    summary = hedge_results['summary']
                    
                    effectiveness_record = {
                        'date': returns_data.index[i],
                        'no_hedge_breadth': 0,
                        'market_hedge_breadth': 0,
                        'sector_hedge_breadth': 0,
                        'market_improvement': 0,
                        'sector_improvement': 0
                    }
                    
                    # Extract breadth information
                    for row in summary['comparison_table']:
                        if row['method'] == 'No Hedge':
                            effectiveness_record['no_hedge_breadth'] = row['effective_breadth']
                        elif row['method'] == 'Market Hedge':
                            effectiveness_record['market_hedge_breadth'] = row['effective_breadth']
                        elif row['method'] == 'Sector Hedge':
                            effectiveness_record['sector_hedge_breadth'] = row['effective_breadth']
                    
                    # Calculate improvements
                    if effectiveness_record['no_hedge_breadth'] > 0:
                        effectiveness_record['market_improvement'] = (
                            effectiveness_record['market_hedge_breadth'] / 
                            effectiveness_record['no_hedge_breadth'] - 1
                        ) * 100
                        
                        effectiveness_record['sector_improvement'] = (
                            effectiveness_record['sector_hedge_breadth'] / 
                            effectiveness_record['no_hedge_breadth'] - 1
                        ) * 100
                    
                    effectiveness_history.append(effectiveness_record)
                    
            except Exception as e:
                print(f"⚠️ Error in effectiveness calculation: {str(e)}")
                continue
        
        if effectiveness_history:
            effectiveness_df = pd.DataFrame(effectiveness_history).set_index('date')
            print(f"✅ Tracked effectiveness over {len(effectiveness_df)} periods")
            return effectiveness_df
        else:
            print("❌ No effectiveness data generated")
            return None
    
    def generate_hedge_recommendations(self, returns_data, portfolio_value=1000000):
        """Generate specific hedge recommendations"""
        print(f"\n💡 Generating Hedge Recommendations...")
        
        # Run comprehensive analysis
        hedge_results = self.hedging_engine.comprehensive_hedge_analysis(returns_data)
        
        if 'summary' not in hedge_results:
            print("❌ Could not generate recommendations")
            return None
        
        recommendations = {
            'portfolio_value': portfolio_value,
            'recommendations': [],
            'risk_metrics': {},
            'implementation_plan': []
        }
        
        # Extract current metrics
        summary = hedge_results['summary']
        
        # Calculate risk metrics
        if 'market_hedge' in hedge_results:
            market_result = hedge_results['market_hedge']
            for stock, beta_info in market_result['beta_info'].items():
                recommendations['risk_metrics'][stock] = {
                    'beta': beta_info['beta'],
                    'r_squared': beta_info['r_squared'],
                    'hedge_ratio': beta_info['beta']
                }
        
        # Generate recommendations based on improvements
        for method, improvement in summary.get('improvements', {}).items():
            if improvement['breadth_improvement_pct'] > 50:
                recommendations['recommendations'].append({
                    'method': method,
                    'priority': 'HIGH',
                    'expected_improvement': f"{improvement['breadth_improvement_pct']:.1f}%",
                    'action': f"Implement {method.lower()} immediately"
                })
            elif improvement['breadth_improvement_pct'] > 20:
                recommendations['recommendations'].append({
                    'method': method,
                    'priority': 'MEDIUM',
                    'expected_improvement': f"{improvement['breadth_improvement_pct']:.1f}%",
                    'action': f"Consider implementing {method.lower()}"
                })
        
        # Implementation plan
        if recommendations['recommendations']:
            recommendations['implementation_plan'] = [
                "1. Start with market hedging using VN-Index futures/ETF",
                "2. Calculate optimal hedge ratios based on beta analysis",
                "3. Implement sector hedging for concentrated positions",
                "4. Monitor effectiveness and rebalance regularly",
                "5. Track costs vs. benefits continuously"
            ]
        
        print(f"✅ Generated {len(recommendations['recommendations'])} recommendations")
        return recommendations
    
    def create_monitoring_dashboard(self, returns_data, save_path=None):
        """Create comprehensive monitoring dashboard"""
        print(f"\n📊 Creating Monitoring Dashboard...")
        
        # Run all analyses
        correlation_monitor = self.real_time_correlation_monitor(returns_data)
        sector_detection = self.advanced_sector_detection(returns_data)
        hedge_analysis = self.hedging_engine.comprehensive_hedge_analysis(returns_data)
        
        # Create dashboard plots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Vietnam Stock Hedging Monitoring Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Correlation over time
        if correlation_monitor:
            monitoring_data = correlation_monitor['monitoring_data']
            axes[0, 0].plot(monitoring_data.index, monitoring_data['avg_correlation'], 
                          color='blue', linewidth=2)
            axes[0, 0].axhline(y=self.alert_thresholds['high_correlation'], 
                             color='red', linestyle='--', alpha=0.7, label='High Alert')
            axes[0, 0].axhline(y=self.alert_thresholds['medium_correlation'], 
                             color='orange', linestyle='--', alpha=0.7, label='Medium Alert')
            axes[0, 0].set_title('Correlation Over Time')
            axes[0, 0].set_ylabel('Average Correlation')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Effective breadth over time
        if correlation_monitor:
            axes[0, 1].plot(monitoring_data.index, monitoring_data['effective_breadth'], 
                          color='green', linewidth=2)
            axes[0, 1].axhline(y=len(returns_data.columns), color='gray', 
                             linestyle='--', alpha=0.7, label='Theoretical Max')
            axes[0, 1].set_title('Effective Breadth Over Time')
            axes[0, 1].set_ylabel('Effective Breadth')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Current correlation heatmap
        current_corr = returns_data.corr()
        sns.heatmap(current_corr, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                   center=0, ax=axes[0, 2])
        axes[0, 2].set_title('Current Correlation Matrix')
        
        # 4. Hedge effectiveness comparison
        if 'summary' in hedge_analysis:
            summary = hedge_analysis['summary']
            methods = [row['method'] for row in summary['comparison_table']]
            breadths = [row['effective_breadth'] for row in summary['comparison_table']]
            
            bars = axes[1, 0].bar(methods, breadths, 
                                color=['red', 'orange', 'green'][:len(methods)])
            axes[1, 0].set_title('Hedge Effectiveness Comparison')
            axes[1, 0].set_ylabel('Effective Breadth')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Add values on bars
            for bar, value in zip(bars, breadths):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                              f'{value:.1f}', ha='center', va='bottom')
        
        # 5. Sector detection visualization
        if sector_detection:
            sector_names = list(sector_detection.keys())
            sector_sizes = [len(info['stocks']) for info in sector_detection.values()]
            
            axes[1, 1].pie(sector_sizes, labels=sector_names, autopct='%1.1f%%')
            axes[1, 1].set_title('Detected Sectors')
        
        # 6. Risk metrics
        if 'market_hedge' in hedge_analysis:
            beta_info = hedge_analysis['market_hedge']['beta_info']
            stocks = list(beta_info.keys())
            betas = [info['beta'] for info in beta_info.values()]
            r_squareds = [info['r_squared'] for info in beta_info.values()]
            
            scatter = axes[1, 2].scatter(betas, r_squareds, s=100, alpha=0.7)
            axes[1, 2].set_xlabel('Beta')
            axes[1, 2].set_ylabel('R-squared')
            axes[1, 2].set_title('Risk Characteristics')
            axes[1, 2].grid(True, alpha=0.3)
            
            # Add stock labels
            for i, stock in enumerate(stocks):
                axes[1, 2].annotate(stock, (betas[i], r_squareds[i]), 
                                  xytext=(5, 5), textcoords='offset points')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Dashboard saved to {save_path}")
        
        plt.show()
        
        return fig


# Test the monitoring system
if __name__ == "__main__":
    print("🚀 Testing Vietnam Hedge Monitor")
    print("=" * 50)
    
    # Initialize components
    pipeline = VietnamStockDataPipeline()
    engine = VietnamHedgingEngine(pipeline)
    monitor = VietnamHedgeMonitor(pipeline, engine)
    
    # Get test data
    print("\n📡 Loading test data...")
    demo_prices = pipeline.get_demo_portfolio()
    
    if demo_prices is not None:
        demo_returns = pipeline.calculate_returns(demo_prices)
        
        # Test sector detection
        print("\n🔍 Testing Advanced Sector Detection...")
        sectors = monitor.advanced_sector_detection(demo_returns, method='correlation_threshold')
        if sectors:
            print(f"✅ Detected sectors:")
            for sector_name, sector_info in sectors.items():
                print(f"   {sector_name}: {sector_info['stocks']} (benchmark: {sector_info['benchmark']})")
        
        # Test correlation monitoring
        print("\n📊 Testing Correlation Monitoring...")
        correlation_monitor = monitor.real_time_correlation_monitor(demo_returns)
        if correlation_monitor:
            print(f"✅ Current correlation: {correlation_monitor['current_correlation']:.3f}")
            print(f"✅ Current breadth: {correlation_monitor['current_breadth']:.1f}")
            
            if correlation_monitor['alerts']:
                print("🚨 Alerts:")
                for alert in correlation_monitor['alerts']:
                    print(f"   {alert['level']}: {alert['message']}")
        
        # Test recommendations
        print("\n💡 Testing Hedge Recommendations...")
        recommendations = monitor.generate_hedge_recommendations(demo_returns)
        if recommendations:
            print(f"✅ Generated {len(recommendations['recommendations'])} recommendations")
            for rec in recommendations['recommendations']:
                print(f"   {rec['priority']}: {rec['action']} (Expected: {rec['expected_improvement']})")
        
        # Create dashboard
        print("\n📊 Creating Dashboard...")
        dashboard = monitor.create_monitoring_dashboard(demo_returns)
        
        print("\n✅ Monitor testing completed!")
    else:
        print("❌ Could not load test data") 