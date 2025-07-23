#!/usr/bin/env python3
"""
Vietnam Stock Hedging Application
=================================

A comprehensive hedging analysis system for Vietnam stock market based on the concepts
from the AI Cafe notebooks on beta and sector hedging.

Features:
- Real-time Vietnam stock data pipeline
- Beta and sector hedging calculations
- Automatic sector detection
- Correlation monitoring and alerts
- Hedge effectiveness tracking
- Interactive dashboard

Author: AI Assistant
Based on: AI Cafe notebooks on hedging beta and sector risk
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from vietnam_hedge_pipeline import VietnamStockDataPipeline
from vietnam_hedging_engine import VietnamHedgingEngine
from vietnam_hedge_monitor import VietnamHedgeMonitor

class VietnamHedgeApp:
    """Main application class for Vietnam stock hedging system"""
    
    def __init__(self):
        """Initialize the application"""
        self.pipeline = VietnamStockDataPipeline()
        self.engine = VietnamHedgingEngine(self.pipeline)
        self.monitor = VietnamHedgeMonitor(self.pipeline, self.engine)
        
        print("🚀 Vietnam Stock Hedging Application")
        print("=" * 50)
        print("📊 Initialized with Vietnam market data pipeline")
        print("🔧 Hedging engine ready")
        print("📈 Monitoring system active")
        
    def run_complete_analysis(self, symbols=None, start_date='2024-01-01', end_date=None):
        """Run complete hedging analysis"""
        print(f"\n🎯 Running Complete Hedging Analysis")
        print("=" * 50)
        
        # 1. Data Loading
        print(f"\n1️⃣ Loading Data...")
        if symbols is None:
            # Use demo portfolio
            prices = self.pipeline.get_demo_portfolio()
            symbols = list(prices.columns) if prices is not None else []
        else:
            prices = self.pipeline.get_market_data(symbols, start_date, end_date)
        
        if prices is None:
            print("❌ Failed to load data")
            return None
        
        # Validate data quality
        self.pipeline.validate_data_quality(prices)
        
        # Calculate returns
        returns = self.pipeline.calculate_returns(prices)
        
        # 2. Comprehensive Hedge Analysis
        print(f"\n2️⃣ Comprehensive Hedge Analysis...")
        hedge_results = self.engine.comprehensive_hedge_analysis(returns)
        
        # 3. Advanced Sector Detection
        print(f"\n3️⃣ Advanced Sector Detection...")
        detected_sectors = self.monitor.advanced_sector_detection(returns, method='correlation_threshold')
        
        # 4. Real-time Monitoring
        print(f"\n4️⃣ Real-time Correlation Monitoring...")
        correlation_monitor = self.monitor.real_time_correlation_monitor(returns)
        
        # 5. Generate Recommendations
        print(f"\n5️⃣ Generating Hedge Recommendations...")
        recommendations = self.monitor.generate_hedge_recommendations(returns)
        
        # 6. Create Dashboard
        print(f"\n6️⃣ Creating Monitoring Dashboard...")
        dashboard = self.monitor.create_monitoring_dashboard(returns)
        
        # 7. Compile Results
        complete_results = {
            'data': {
                'prices': prices,
                'returns': returns,
                'symbols': symbols,
                'date_range': (prices.index.min(), prices.index.max())
            },
            'hedge_analysis': hedge_results,
            'sector_detection': detected_sectors,
            'correlation_monitor': correlation_monitor,
            'recommendations': recommendations,
            'dashboard': dashboard
        }
        
        # 8. Generate Report
        self._generate_executive_report(complete_results)
        
        return complete_results
    
    def _generate_executive_report(self, results):
        """Generate executive summary report"""
        print(f"\n📋 EXECUTIVE SUMMARY REPORT")
        print("=" * 50)
        
        # Data Summary
        data_info = results['data']
        print(f"\n📊 Data Summary:")
        print(f"   • Assets analyzed: {len(data_info['symbols'])}")
        print(f"   • Symbols: {', '.join(data_info['symbols'])}")
        print(f"   • Date range: {data_info['date_range'][0].strftime('%Y-%m-%d')} to {data_info['date_range'][1].strftime('%Y-%m-%d')}")
        print(f"   • Total observations: {len(data_info['returns'])}")
        
        # Hedge Analysis Summary
        if 'summary' in results['hedge_analysis']:
            summary = results['hedge_analysis']['summary']
            print(f"\n🔄 Hedge Analysis Summary:")
            
            # Comparison table
            print(f"   📋 Effectiveness Comparison:")
            for row in summary['comparison_table']:
                print(f"      {row['method']:<15}: Correlation={row['avg_correlation']:.3f}, Breadth={row['effective_breadth']:.1f}")
            
            # Improvements
            if summary['improvements']:
                print(f"   📈 Key Improvements:")
                for method, improvement in summary['improvements'].items():
                    print(f"      {method}: +{improvement['breadth_improvement']:.1f} breadth ({improvement['breadth_improvement_pct']:+.1f}%)")
        
        # Sector Detection Summary
        if results['sector_detection']:
            print(f"\n🔍 Sector Detection Summary:")
            print(f"   • Sectors detected: {len(results['sector_detection'])}")
            for sector_name, sector_info in results['sector_detection'].items():
                print(f"      {sector_name}: {len(sector_info['stocks'])} stocks (avg corr: {sector_info.get('avg_correlation', 0):.3f})")
        
        # Current Risk Status
        if results['correlation_monitor']:
            monitor_data = results['correlation_monitor']
            print(f"\n📊 Current Risk Status:")
            print(f"   • Current correlation: {monitor_data['current_correlation']:.3f}")
            print(f"   • Current breadth: {monitor_data['current_breadth']:.1f}")
            
            if monitor_data['alerts']:
                print(f"   🚨 Active Alerts:")
                for alert in monitor_data['alerts']:
                    print(f"      {alert['level']}: {alert['message']}")
            else:
                print(f"   ✅ No active alerts")
        
        # Recommendations
        if results['recommendations'] and results['recommendations']['recommendations']:
            print(f"\n💡 Key Recommendations:")
            for rec in results['recommendations']['recommendations']:
                print(f"   • {rec['priority']} PRIORITY: {rec['action']}")
                print(f"     Expected improvement: {rec['expected_improvement']}")
        
        # Implementation Plan
        if results['recommendations'] and results['recommendations']['implementation_plan']:
            print(f"\n🛠️ Implementation Plan:")
            for i, step in enumerate(results['recommendations']['implementation_plan'], 1):
                print(f"   {i}. {step}")
        
        print(f"\n" + "=" * 50)
        print(f"✅ Analysis completed successfully!")
        print(f"📊 Dashboard generated for detailed visualization")
        print(f"💾 All results available in returned object")
    
    def run_quick_analysis(self, symbols=None):
        """Run quick analysis for immediate insights"""
        print(f"\n⚡ Quick Analysis Mode")
        print("=" * 30)
        
        # Load data
        if symbols is None:
            prices = self.pipeline.get_demo_portfolio()
        else:
            prices = self.pipeline.get_market_data(symbols, start_date='2024-01-01')
        
        if prices is None:
            print("❌ Failed to load data")
            return None
        
        returns = self.pipeline.calculate_returns(prices)
        
        # Quick hedge analysis
        hedge_results = self.engine.comprehensive_hedge_analysis(returns)
        
        # Quick summary
        if 'summary' in hedge_results:
            summary = hedge_results['summary']
            print(f"\n📊 Quick Results:")
            
            for row in summary['comparison_table']:
                print(f"   {row['method']}: BR={row['effective_breadth']:.1f}, Corr={row['avg_correlation']:.3f}")
            
            # Best recommendation
            if summary['improvements']:
                best_method = max(summary['improvements'].items(), 
                                key=lambda x: x[1]['breadth_improvement_pct'])
                print(f"\n💡 Best Option: {best_method[0]}")
                print(f"   Improvement: +{best_method[1]['breadth_improvement_pct']:.1f}% breadth")
        
        return hedge_results
    
    def monitor_portfolio(self, symbols, days_back=60):
        """Monitor specific portfolio"""
        print(f"\n👁️ Portfolio Monitoring")
        print("=" * 30)
        
        # Load recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        prices = self.pipeline.get_market_data(
            symbols, 
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        if prices is None:
            print("❌ Failed to load portfolio data")
            return None
        
        returns = self.pipeline.calculate_returns(prices)
        
        # Real-time monitoring
        monitor_result = self.monitor.real_time_correlation_monitor(returns)
        
        if monitor_result:
            print(f"📊 Portfolio Status:")
            print(f"   • Assets: {len(symbols)}")
            print(f"   • Current correlation: {monitor_result['current_correlation']:.3f}")
            print(f"   • Effective breadth: {monitor_result['current_breadth']:.1f}/{len(symbols)}")
            
            if monitor_result['alerts']:
                print(f"   🚨 Alerts:")
                for alert in monitor_result['alerts']:
                    print(f"      {alert['level']}: {alert['message']}")
        
        return monitor_result
    
    def demo_vietnam_banks(self):
        """Demo analysis for Vietnam banking sector"""
        print(f"\n🏦 Vietnam Banking Sector Demo")
        print("=" * 40)
        
        banking_stocks = ['BID', 'CTG', 'ACB', 'VCB', 'TCB', 'VPB']
        return self.run_complete_analysis(banking_stocks)
    
    def demo_vietnam_realestate(self):
        """Demo analysis for Vietnam real estate sector"""
        print(f"\n🏢 Vietnam Real Estate Sector Demo")
        print("=" * 40)
        
        realestate_stocks = ['VHM', 'VIC', 'VRE', 'HDG', 'KDH', 'DXG']
        return self.run_complete_analysis(realestate_stocks)
    
    def demo_diversified_portfolio(self):
        """Demo analysis for diversified Vietnam portfolio"""
        print(f"\n🌟 Diversified Vietnam Portfolio Demo")
        print("=" * 40)
        
        diversified_stocks = ['BID', 'VCB', 'VHM', 'VIC', 'HPG', 'MWG', 'GAS', 'PLX']
        return self.run_complete_analysis(diversified_stocks)


def main():
    """Main function to demonstrate the application"""
    
    print("🇻🇳 Vietnam Stock Hedging Application")
    print("=" * 50)
    print("Based on AI Cafe notebooks on hedging beta and sector risk")
    print("Demonstrates practical implementation for Vietnam stock market")
    print()
    
    # Initialize application
    app = VietnamHedgeApp()
    
    # Demo menu
    print("\n📋 Available Demos:")
    print("1. Complete Analysis (Demo Portfolio)")
    print("2. Quick Analysis (Demo Portfolio)")
    print("3. Vietnam Banking Sector")
    print("4. Vietnam Real Estate Sector")
    print("5. Diversified Portfolio")
    print("6. Custom Portfolio Monitoring")
    
    try:
        choice = input("\nSelect demo (1-6) or press Enter for default: ").strip()
        
        if choice == '1' or choice == '':
            print("\n🎯 Running Complete Analysis Demo...")
            results = app.run_complete_analysis()
            
        elif choice == '2':
            print("\n⚡ Running Quick Analysis Demo...")
            results = app.run_quick_analysis()
            
        elif choice == '3':
            print("\n🏦 Running Banking Sector Demo...")
            results = app.demo_vietnam_banks()
            
        elif choice == '4':
            print("\n🏢 Running Real Estate Sector Demo...")
            results = app.demo_vietnam_realestate()
            
        elif choice == '5':
            print("\n🌟 Running Diversified Portfolio Demo...")
            results = app.demo_diversified_portfolio()
            
        elif choice == '6':
            symbols = input("Enter stock symbols (comma-separated): ").strip().split(',')
            symbols = [s.strip().upper() for s in symbols if s.strip()]
            if symbols:
                results = app.monitor_portfolio(symbols)
            else:
                print("❌ No valid symbols provided")
                return
                
        else:
            print("❌ Invalid choice")
            return
        
        print(f"\n✅ Demo completed successfully!")
        print(f"📊 Results available in the returned object")
        
        # Optional: Save results
        save_choice = input("\nSave results to file? (y/n): ").strip().lower()
        if save_choice == 'y':
            filename = f"vietnam_hedge_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            try:
                import pickle
                with open(filename, 'wb') as f:
                    pickle.dump(results, f)
                print(f"💾 Results saved to {filename}")
            except Exception as e:
                print(f"❌ Error saving results: {str(e)}")
        
    except KeyboardInterrupt:
        print("\n\n👋 Application interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Please check your data connection and try again")


if __name__ == "__main__":
    main() 