#!/usr/bin/env python3
"""
Vietnam Stock Real-time Hedging System
=====================================

Real-time system for live hedge monitoring and automated trading signals.

Features:
- Live data streaming simulation
- Real-time correlation monitoring
- Automated hedge alerts
- Position management
- Risk monitoring
- Performance tracking

Usage:
    python vietnam_realtime_system.py
"""

import asyncio
import threading
import time
import json
from datetime import datetime, timedelta
from collections import deque
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from vietnam_hedge_pipeline import VietnamStockDataPipeline
from vietnam_hedging_engine import VietnamHedgingEngine
from vietnam_hedge_monitor import VietnamHedgeMonitor
from vietnam_advanced_analytics import VietnamAdvancedAnalytics

class VietnamRealTimeSystem:
    """Real-time hedging system for Vietnam stocks"""
    
    def __init__(self, update_interval=30):
        """Initialize real-time system"""
        self.update_interval = update_interval  # seconds
        self.is_running = False
        
        # Initialize components
        self.pipeline = VietnamStockDataPipeline()
        self.engine = VietnamHedgingEngine(self.pipeline)
        self.monitor = VietnamHedgeMonitor(self.pipeline, self.engine)
        self.analytics = VietnamAdvancedAnalytics(self.pipeline, self.engine)
        
        # Real-time data storage
        self.live_data = {}
        self.data_buffer = deque(maxlen=1000)  # Keep last 1000 data points
        
        # Alert system
        self.active_alerts = []
        self.alert_history = deque(maxlen=100)
        
        # Position tracking
        self.current_positions = {}
        self.hedge_positions = {}
        
        # Performance tracking
        self.performance_history = deque(maxlen=500)
        
        # Configuration
        self.config = {
            'symbols': ['BID', 'CTG', 'ACB', 'VHM', 'VIC', 'VRE'],
            'correlation_threshold_high': 0.7,
            'correlation_threshold_medium': 0.5,
            'volatility_threshold': 0.05,
            'hedge_rebalance_threshold': 0.1,
            'max_position_size': 0.2,
            'auto_hedge_enabled': True
        }
        
        print("🚀 Real-time Hedging System Initialized")
        print(f"📊 Update interval: {update_interval} seconds")
        print(f"📈 Monitoring symbols: {self.config['symbols']}")
    
    def start_real_time_monitoring(self):
        """Start real-time monitoring"""
        if self.is_running:
            print("⚠️ System already running")
            return
        
        self.is_running = True
        print("🟢 Starting real-time monitoring...")
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        # Start alert processing thread
        self.alert_thread = threading.Thread(target=self._alert_processing_loop)
        self.alert_thread.daemon = True
        self.alert_thread.start()
        
        print("✅ Real-time monitoring started")
    
    def stop_real_time_monitoring(self):
        """Stop real-time monitoring"""
        if not self.is_running:
            print("⚠️ System not running")
            return
        
        self.is_running = False
        print("🔴 Stopping real-time monitoring...")
        
        # Wait for threads to finish
        if hasattr(self, 'monitoring_thread'):
            self.monitoring_thread.join(timeout=5)
        if hasattr(self, 'alert_thread'):
            self.alert_thread.join(timeout=5)
        
        print("✅ Real-time monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Get current timestamp
                current_time = datetime.now()
                
                # Simulate live data update (in real system, this would fetch from API)
                live_data = self._fetch_live_data()
                
                if live_data:
                    # Add to buffer
                    self.data_buffer.append({
                        'timestamp': current_time,
                        'data': live_data
                    })
                    
                    # Update live data
                    self.live_data = live_data
                    
                    # Run real-time analysis
                    self._run_real_time_analysis()
                    
                    # Check for alerts
                    self._check_alerts()
                    
                    # Update hedge positions if needed
                    if self.config['auto_hedge_enabled']:
                        self._update_hedge_positions()
                    
                    # Track performance
                    self._track_performance()
                
                # Sleep until next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"❌ Error in monitoring loop: {str(e)}")
                time.sleep(self.update_interval)
    
    def _fetch_live_data(self):
        """Fetch live market data (simulated)"""
        try:
            # In real system, this would fetch from live API
            # For demo, we'll simulate with recent data
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            prices = self.pipeline.get_market_data(
                self.config['symbols'] + ['VNINDEX'],
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            if prices is not None and len(prices) > 0:
                # Get latest prices and simulate small random changes
                latest_prices = prices.iloc[-1].copy()
                
                # Add small random movements (±0.5%)
                for symbol in latest_prices.index:
                    change = np.random.normal(0, 0.005)  # 0.5% volatility
                    latest_prices[symbol] *= (1 + change)
                
                return latest_prices.to_dict()
            
        except Exception as e:
            print(f"❌ Error fetching live data: {str(e)}")
            return None
    
    def _run_real_time_analysis(self):
        """Run real-time analysis on current data"""
        if len(self.data_buffer) < 30:  # Need minimum data
            return
        
        try:
            # Convert buffer to DataFrame
            recent_data = []
            for item in list(self.data_buffer)[-30:]:  # Last 30 data points
                recent_data.append(item['data'])
            
            df = pd.DataFrame(recent_data)
            
            # Calculate returns
            returns = df.pct_change().fillna(0)
            
            # Calculate current correlation
            if 'VNINDEX' in returns.columns:
                stock_returns = returns.drop('VNINDEX', axis=1)
                corr_matrix = stock_returns.corr()
                
                # Get average correlation
                upper_tri = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
                current_correlation = np.mean(upper_tri)
                
                # Calculate effective breadth
                n_stocks = len(stock_returns.columns)
                effective_breadth = n_stocks / (1 + current_correlation * (n_stocks - 1))
                
                # Store current metrics
                self.current_metrics = {
                    'correlation': current_correlation,
                    'effective_breadth': effective_breadth,
                    'volatility': stock_returns.std().mean(),
                    'timestamp': datetime.now()
                }
            
        except Exception as e:
            print(f"❌ Error in real-time analysis: {str(e)}")
    
    def _check_alerts(self):
        """Check for alert conditions"""
        if not hasattr(self, 'current_metrics'):
            return
        
        current_time = datetime.now()
        metrics = self.current_metrics
        
        # Check correlation alerts
        if metrics['correlation'] > self.config['correlation_threshold_high']:
            self._create_alert(
                'HIGH_CORRELATION',
                f"Very high correlation detected: {metrics['correlation']:.3f}",
                'CRITICAL',
                "Increase hedging immediately"
            )
        elif metrics['correlation'] > self.config['correlation_threshold_medium']:
            self._create_alert(
                'MEDIUM_CORRELATION',
                f"High correlation detected: {metrics['correlation']:.3f}",
                'WARNING',
                "Consider increasing hedge positions"
            )
        
        # Check volatility alerts
        if metrics['volatility'] > self.config['volatility_threshold']:
            self._create_alert(
                'HIGH_VOLATILITY',
                f"High volatility detected: {metrics['volatility']:.3f}",
                'WARNING',
                "Monitor positions closely"
            )
        
        # Check breadth degradation
        if metrics['effective_breadth'] < 2.0:
            self._create_alert(
                'LOW_BREADTH',
                f"Low effective breadth: {metrics['effective_breadth']:.1f}",
                'WARNING',
                "Diversification compromised"
            )
    
    def _create_alert(self, alert_type, message, severity, recommendation):
        """Create new alert"""
        alert = {
            'id': f"{alert_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'type': alert_type,
            'message': message,
            'severity': severity,
            'recommendation': recommendation,
            'timestamp': datetime.now(),
            'status': 'ACTIVE'
        }
        
        # Check if similar alert already exists
        similar_exists = any(
            a['type'] == alert_type and a['status'] == 'ACTIVE'
            for a in self.active_alerts
        )
        
        if not similar_exists:
            self.active_alerts.append(alert)
            self.alert_history.append(alert)
            print(f"🚨 {severity} ALERT: {message}")
    
    def _alert_processing_loop(self):
        """Process alerts and manage lifecycle"""
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Auto-resolve old alerts (after 5 minutes)
                for alert in self.active_alerts[:]:
                    if (current_time - alert['timestamp']).seconds > 300:
                        alert['status'] = 'RESOLVED'
                        self.active_alerts.remove(alert)
                        print(f"✅ Auto-resolved alert: {alert['type']}")
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                print(f"❌ Error in alert processing: {str(e)}")
                time.sleep(30)
    
    def _update_hedge_positions(self):
        """Update hedge positions based on current analysis"""
        if not hasattr(self, 'current_metrics'):
            return
        
        try:
            # Calculate required hedge ratios
            if len(self.data_buffer) >= 30:
                # Get recent returns data
                recent_data = []
                for item in list(self.data_buffer)[-30:]:
                    recent_data.append(item['data'])
                
                df = pd.DataFrame(recent_data)
                returns = df.pct_change().fillna(0)
                
                if 'VNINDEX' in returns.columns:
                    market_returns = returns['VNINDEX']
                    stock_returns = returns.drop('VNINDEX', axis=1)
                    
                    # Calculate current hedge ratios
                    new_hedge_ratios = {}
                    
                    for stock in stock_returns.columns:
                        # Calculate beta
                        covariance = np.cov(stock_returns[stock], market_returns)[0, 1]
                        market_variance = np.var(market_returns)
                        
                        if market_variance > 0:
                            beta = covariance / market_variance
                            new_hedge_ratios[stock] = beta
                    
                    # Check if rebalancing is needed
                    rebalance_needed = False
                    
                    for stock, new_ratio in new_hedge_ratios.items():
                        current_ratio = self.hedge_positions.get(stock, 0)
                        
                        if abs(new_ratio - current_ratio) > self.config['hedge_rebalance_threshold']:
                            rebalance_needed = True
                            break
                    
                    if rebalance_needed:
                        self.hedge_positions.update(new_hedge_ratios)
                        print(f"🔄 Hedge positions updated: {datetime.now().strftime('%H:%M:%S')}")
                        
                        # Create rebalance alert
                        self._create_alert(
                            'HEDGE_REBALANCE',
                            "Hedge positions rebalanced",
                            'INFO',
                            "Review new hedge ratios"
                        )
        
        except Exception as e:
            print(f"❌ Error updating hedge positions: {str(e)}")
    
    def _track_performance(self):
        """Track system performance"""
        if not hasattr(self, 'current_metrics'):
            return
        
        try:
            # Calculate portfolio performance metrics
            performance_record = {
                'timestamp': datetime.now(),
                'correlation': self.current_metrics['correlation'],
                'effective_breadth': self.current_metrics['effective_breadth'],
                'volatility': self.current_metrics['volatility'],
                'active_alerts': len(self.active_alerts),
                'hedge_positions': len(self.hedge_positions)
            }
            
            self.performance_history.append(performance_record)
            
        except Exception as e:
            print(f"❌ Error tracking performance: {str(e)}")
    
    def get_real_time_status(self):
        """Get current system status"""
        status = {
            'is_running': self.is_running,
            'last_update': datetime.now(),
            'data_points': len(self.data_buffer),
            'active_alerts': len(self.active_alerts),
            'hedge_positions': len(self.hedge_positions),
            'performance_records': len(self.performance_history)
        }
        
        if hasattr(self, 'current_metrics'):
            status['current_metrics'] = self.current_metrics
        
        return status
    
    def get_active_alerts(self):
        """Get current active alerts"""
        return self.active_alerts.copy()
    
    def get_hedge_positions(self):
        """Get current hedge positions"""
        return self.hedge_positions.copy()
    
    def get_performance_summary(self):
        """Get performance summary"""
        if not self.performance_history:
            return {}
        
        recent_performance = list(self.performance_history)[-50:]  # Last 50 records
        
        df = pd.DataFrame(recent_performance)
        
        summary = {
            'avg_correlation': df['correlation'].mean(),
            'avg_effective_breadth': df['effective_breadth'].mean(),
            'avg_volatility': df['volatility'].mean(),
            'correlation_trend': df['correlation'].iloc[-10:].mean() - df['correlation'].iloc[-20:-10].mean(),
            'alert_frequency': df['active_alerts'].mean(),
            'data_points': len(recent_performance)
        }
        
        return summary
    
    def export_real_time_data(self, filename=None):
        """Export real-time data to file"""
        if not filename:
            filename = f"realtime_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            'system_status': self.get_real_time_status(),
            'active_alerts': self.get_active_alerts(),
            'hedge_positions': self.get_hedge_positions(),
            'performance_summary': self.get_performance_summary(),
            'configuration': self.config,
            'export_timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"💾 Real-time data exported to {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Error exporting data: {str(e)}")
            return None
    
    def update_configuration(self, new_config):
        """Update system configuration"""
        self.config.update(new_config)
        print(f"⚙️ Configuration updated: {new_config}")
    
    def manual_hedge_signal(self, action, symbol=None, ratio=None):
        """Send manual hedge signal"""
        signal = {
            'timestamp': datetime.now(),
            'action': action,  # 'BUY', 'SELL', 'REBALANCE'
            'symbol': symbol,
            'ratio': ratio,
            'source': 'MANUAL'
        }
        
        print(f"📡 Manual hedge signal: {action} {symbol} ({ratio})")
        
        # Process signal
        if action == 'REBALANCE' and symbol and ratio:
            self.hedge_positions[symbol] = ratio
            
            self._create_alert(
                'MANUAL_HEDGE',
                f"Manual hedge: {action} {symbol} @ {ratio}",
                'INFO',
                "Manual hedge position updated"
            )
    
    def run_demo(self, duration_minutes=5):
        """Run system demo for specified duration"""
        print(f"🎬 Starting {duration_minutes}-minute demo...")
        
        # Start monitoring
        self.start_real_time_monitoring()
        
        try:
            # Run for specified duration
            start_time = time.time()
            
            while time.time() - start_time < duration_minutes * 60:
                # Print status every 30 seconds
                if int(time.time() - start_time) % 30 == 0:
                    status = self.get_real_time_status()
                    print(f"📊 Status: {status['data_points']} data points, "
                          f"{status['active_alerts']} alerts, "
                          f"{status['hedge_positions']} hedge positions")
                
                time.sleep(1)
            
            # Final summary
            print(f"\n📋 Demo Summary:")
            summary = self.get_performance_summary()
            if summary:
                print(f"   Average correlation: {summary['avg_correlation']:.3f}")
                print(f"   Average effective breadth: {summary['avg_effective_breadth']:.1f}")
                print(f"   Alert frequency: {summary['alert_frequency']:.1f}")
            
            # Export data
            self.export_real_time_data()
            
        except KeyboardInterrupt:
            print("\n👋 Demo interrupted by user")
        
        finally:
            # Stop monitoring
            self.stop_real_time_monitoring()
            print("✅ Demo completed")


# Demo function
def run_real_time_demo():
    """Run real-time system demo"""
    print("🚀 Vietnam Stock Real-time Hedging System Demo")
    print("=" * 60)
    
    # Initialize system
    system = VietnamRealTimeSystem(update_interval=10)  # 10-second updates for demo
    
    # Run demo
    system.run_demo(duration_minutes=2)  # 2-minute demo


if __name__ == "__main__":
    run_real_time_demo() 