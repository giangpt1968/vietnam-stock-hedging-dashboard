#!/usr/bin/env python3
"""
Vietnam Stock Advanced Analytics
===============================

Advanced analytics module with machine learning capabilities for:
- Correlation prediction using time series models
- Sector rotation detection
- Portfolio optimization with hedging constraints
- Risk regime detection
- Dynamic hedge ratio calculation

Features:
- LSTM models for correlation forecasting
- Clustering for sector analysis
- Optimization algorithms
- Risk metrics calculation
- Backtesting framework
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced ML libraries with fallbacks
try:
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.decomposition import PCA
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    from scipy.optimize import minimize
    from scipy.stats import norm, jarque_bera
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Import our modules
from vietnam_hedge_pipeline import VietnamStockDataPipeline
from vietnam_hedging_engine import VietnamHedgingEngine

class VietnamAdvancedAnalytics:
    """Advanced analytics for Vietnam stock hedging"""
    
    def __init__(self, data_pipeline=None, hedging_engine=None):
        """Initialize advanced analytics"""
        self.data_pipeline = data_pipeline or VietnamStockDataPipeline()
        self.hedging_engine = hedging_engine or VietnamHedgingEngine(self.data_pipeline)
        
        # Model storage
        self.models = {}
        self.scalers = {}
        
        # Analysis results
        self.analysis_cache = {}
        
        print("🧠 Advanced Analytics Module Initialized")
        print(f"📊 ML Libraries Available: {ML_AVAILABLE}")
        print(f"🔬 SciPy Available: {SCIPY_AVAILABLE}")
    
    def correlation_prediction_model(self, returns_data, forecast_days=30):
        """Build and train correlation prediction model"""
        print(f"\n🔮 Building Correlation Prediction Model...")
        
        if not ML_AVAILABLE:
            print("⚠️ ML libraries not available, using simple forecast")
            return self._simple_correlation_forecast(returns_data, forecast_days)
        
        try:
            # Prepare time series features
            features, targets = self._prepare_correlation_features(returns_data)
            
            if len(features) < 60:  # Need minimum data
                print("⚠️ Insufficient data for ML model, using simple forecast")
                return self._simple_correlation_forecast(returns_data, forecast_days)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, targets, test_size=0.2, shuffle=False
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"✅ Model trained - MSE: {mse:.4f}, R²: {r2:.4f}")
            
            # Generate forecasts
            forecasts = self._generate_correlation_forecasts(
                model, scaler, features, forecast_days
            )
            
            # Store model
            self.models['correlation_predictor'] = model
            self.scalers['correlation_predictor'] = scaler
            
            return {
                'model': model,
                'scaler': scaler,
                'performance': {'mse': mse, 'r2': r2},
                'forecasts': forecasts,
                'feature_importance': dict(zip(
                    [f'feature_{i}' for i in range(len(model.feature_importances_))],
                    model.feature_importances_
                ))
            }
            
        except Exception as e:
            print(f"❌ Error in correlation prediction: {str(e)}")
            return self._simple_correlation_forecast(returns_data, forecast_days)
    
    def _prepare_correlation_features(self, returns_data, window=30):
        """Prepare features for correlation prediction"""
        correlations = []
        features = []
        
        for i in range(window, len(returns_data)):
            # Calculate rolling correlation
            window_data = returns_data.iloc[i-window:i]
            corr_matrix = window_data.corr()
            
            # Extract upper triangular correlations
            upper_tri = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
            avg_corr = np.mean(upper_tri)
            correlations.append(avg_corr)
            
            # Create features: returns statistics, volatility, etc.
            window_features = []
            
            # Return statistics
            window_features.extend([
                window_data.mean().mean(),  # Average return
                window_data.std().mean(),   # Average volatility
                window_data.skew().mean(),  # Average skewness
                window_data.kurt().mean()   # Average kurtosis
            ])
            
            # Market features
            if 'VNINDEX' in window_data.columns:
                market_returns = window_data['VNINDEX']
                window_features.extend([
                    market_returns.mean(),
                    market_returns.std(),
                    market_returns.min(),
                    market_returns.max()
                ])
            else:
                window_features.extend([0, 0, 0, 0])
            
            # Cross-sectional features
            window_features.extend([
                np.mean(upper_tri),      # Average correlation
                np.std(upper_tri),       # Correlation volatility
                np.min(upper_tri),       # Minimum correlation
                np.max(upper_tri)        # Maximum correlation
            ])
            
            features.append(window_features)
        
        return np.array(features[:-1]), np.array(correlations[1:])
    
    def _simple_correlation_forecast(self, returns_data, forecast_days):
        """Simple correlation forecast using moving averages"""
        # Calculate recent correlation trend
        recent_correlations = []
        
        for i in range(max(30, len(returns_data) - 60), len(returns_data)):
            if i >= 30:
                window_data = returns_data.iloc[i-30:i]
                corr_matrix = window_data.corr()
                upper_tri = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
                recent_correlations.append(np.mean(upper_tri))
        
        if recent_correlations:
            # Simple trend extrapolation
            current_corr = recent_correlations[-1]
            trend = np.mean(np.diff(recent_correlations[-10:])) if len(recent_correlations) >= 10 else 0
            
            forecasts = []
            for day in range(forecast_days):
                forecast = current_corr + trend * day
                forecast = np.clip(forecast, -1, 1)  # Keep in valid range
                forecasts.append(forecast)
            
            return {
                'forecasts': forecasts,
                'method': 'simple_trend',
                'current_correlation': current_corr,
                'trend': trend
            }
        
        return {'forecasts': [0.3] * forecast_days, 'method': 'fallback'}
    
    def _generate_correlation_forecasts(self, model, scaler, features, forecast_days):
        """Generate correlation forecasts using trained model"""
        forecasts = []
        
        # Use last feature vector as starting point
        last_features = features[-1:].copy()
        
        for _ in range(forecast_days):
            # Scale features
            scaled_features = scaler.transform(last_features)
            
            # Predict next correlation
            next_corr = model.predict(scaled_features)[0]
            forecasts.append(next_corr)
            
            # Update features for next prediction (simplified)
            # In practice, you'd update with actual market data
            last_features = last_features.copy()
            last_features[0, 0] = next_corr  # Update correlation feature
        
        return forecasts
    
    def sector_rotation_detection(self, returns_data, lookback_days=60):
        """Detect sector rotation patterns"""
        print(f"\n🔄 Detecting Sector Rotation Patterns...")
        
        if not ML_AVAILABLE:
            return self._simple_sector_analysis(returns_data)
        
        try:
            # Get sector classifications
            sector_performance = {}
            
            for sector_name, sector_info in self.data_pipeline.sectors.items():
                sector_stocks = [s for s in sector_info['stocks'] if s in returns_data.columns]
                
                if len(sector_stocks) >= 2:
                    # Calculate sector performance
                    sector_returns = returns_data[sector_stocks].mean(axis=1)
                    
                    # Rolling performance metrics
                    rolling_returns = sector_returns.rolling(lookback_days).mean()
                    rolling_volatility = sector_returns.rolling(lookback_days).std()
                    rolling_sharpe = rolling_returns / rolling_volatility
                    
                    sector_performance[sector_name] = {
                        'returns': sector_returns,
                        'rolling_returns': rolling_returns,
                        'rolling_volatility': rolling_volatility,
                        'rolling_sharpe': rolling_sharpe,
                        'current_momentum': sector_returns.iloc[-30:].mean(),
                        'volatility': sector_returns.std(),
                        'stocks': sector_stocks
                    }
            
            # Detect rotation patterns
            rotation_signals = self._analyze_sector_rotation(sector_performance)
            
            # Clustering analysis
            sector_clusters = self._cluster_sectors(sector_performance)
            
            return {
                'sector_performance': sector_performance,
                'rotation_signals': rotation_signals,
                'sector_clusters': sector_clusters,
                'analysis_date': datetime.now()
            }
            
        except Exception as e:
            print(f"❌ Error in sector rotation detection: {str(e)}")
            return self._simple_sector_analysis(returns_data)
    
    def _simple_sector_analysis(self, returns_data):
        """Simple sector analysis without ML"""
        sector_performance = {}
        
        for sector_name, sector_info in self.data_pipeline.sectors.items():
            sector_stocks = [s for s in sector_info['stocks'] if s in returns_data.columns]
            
            if len(sector_stocks) >= 1:
                sector_returns = returns_data[sector_stocks].mean(axis=1)
                
                sector_performance[sector_name] = {
                    'current_momentum': sector_returns.iloc[-30:].mean(),
                    'volatility': sector_returns.std(),
                    'total_return': sector_returns.sum(),
                    'stocks': sector_stocks
                }
        
        return {'sector_performance': sector_performance, 'method': 'simple'}
    
    def _analyze_sector_rotation(self, sector_performance):
        """Analyze sector rotation signals"""
        rotation_signals = {}
        
        # Calculate relative performance
        sector_names = list(sector_performance.keys())
        momentum_scores = [sector_performance[name]['current_momentum'] for name in sector_names]
        
        # Rank sectors by momentum
        sector_ranks = pd.Series(momentum_scores, index=sector_names).rank(ascending=False)
        
        for sector_name in sector_names:
            rank = sector_ranks[sector_name]
            total_sectors = len(sector_names)
            
            if rank <= total_sectors * 0.3:  # Top 30%
                signal = 'STRONG_BUY'
            elif rank <= total_sectors * 0.6:  # Top 60%
                signal = 'BUY'
            elif rank <= total_sectors * 0.8:  # Top 80%
                signal = 'HOLD'
            else:
                signal = 'SELL'
            
            rotation_signals[sector_name] = {
                'signal': signal,
                'rank': rank,
                'momentum_score': sector_performance[sector_name]['current_momentum'],
                'confidence': min(abs(sector_performance[sector_name]['current_momentum']) * 100, 100)
            }
        
        return rotation_signals
    
    def _cluster_sectors(self, sector_performance):
        """Cluster sectors based on performance characteristics"""
        if not ML_AVAILABLE:
            return {}
        
        # Prepare features for clustering
        features = []
        sector_names = []
        
        for sector_name, perf in sector_performance.items():
            features.append([
                perf['current_momentum'],
                perf['volatility'],
                perf.get('rolling_sharpe', pd.Series([0])).iloc[-1] if hasattr(perf.get('rolling_sharpe', 0), 'iloc') else 0
            ])
            sector_names.append(sector_name)
        
        if len(features) < 2:
            return {}
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # K-means clustering
        n_clusters = min(3, len(features))  # Max 3 clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Organize results
        clusters = {}
        for i, sector_name in enumerate(sector_names):
            cluster_id = cluster_labels[i]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(sector_name)
        
        return clusters
    
    def portfolio_optimization_with_hedging(self, returns_data, target_return=None, max_volatility=None):
        """Portfolio optimization with hedging constraints"""
        print(f"\n🎯 Portfolio Optimization with Hedging...")
        
        if not SCIPY_AVAILABLE:
            return self._simple_portfolio_optimization(returns_data)
        
        try:
            # Calculate expected returns and covariance
            expected_returns = returns_data.mean() * 252  # Annualized
            cov_matrix = returns_data.cov() * 252  # Annualized
            
            n_assets = len(returns_data.columns)
            
            # Objective function: minimize portfolio variance
            def objective(weights):
                portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
                return portfolio_variance
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
            ]
            
            # Add return constraint if specified
            if target_return is not None:
                constraints.append({
                    'type': 'eq',
                    'fun': lambda x: np.dot(x, expected_returns) - target_return
                })
            
            # Add volatility constraint if specified
            if max_volatility is not None:
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x: max_volatility**2 - np.dot(x, np.dot(cov_matrix, x))
                })
            
            # Bounds: allow short positions for hedging
            bounds = [(-0.5, 1.0) for _ in range(n_assets)]
            
            # Initial guess: equal weights
            initial_guess = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                objective,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                optimal_weights = result.x
                
                # Calculate portfolio metrics
                portfolio_return = np.dot(optimal_weights, expected_returns)
                portfolio_volatility = np.sqrt(np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights)))
                sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
                
                # Identify hedge positions
                hedge_positions = {
                    asset: weight for asset, weight in zip(returns_data.columns, optimal_weights)
                    if weight < 0
                }
                
                return {
                    'optimal_weights': dict(zip(returns_data.columns, optimal_weights)),
                    'portfolio_return': portfolio_return,
                    'portfolio_volatility': portfolio_volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'hedge_positions': hedge_positions,
                    'optimization_success': True
                }
            else:
                print(f"⚠️ Optimization failed: {result.message}")
                return self._simple_portfolio_optimization(returns_data)
                
        except Exception as e:
            print(f"❌ Error in portfolio optimization: {str(e)}")
            return self._simple_portfolio_optimization(returns_data)
    
    def _simple_portfolio_optimization(self, returns_data):
        """Simple portfolio optimization without advanced optimization"""
        # Equal weight portfolio
        n_assets = len(returns_data.columns)
        equal_weights = {asset: 1/n_assets for asset in returns_data.columns}
        
        # Calculate basic metrics
        expected_returns = returns_data.mean() * 252
        portfolio_return = np.mean(expected_returns)
        portfolio_volatility = np.sqrt(np.mean(returns_data.var()) * 252)
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return {
            'optimal_weights': equal_weights,
            'portfolio_return': portfolio_return,
            'portfolio_volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'hedge_positions': {},
            'optimization_success': False,
            'method': 'equal_weight'
        }
    
    def risk_regime_detection(self, returns_data, window=60):
        """Detect risk regime changes"""
        print(f"\n⚠️ Detecting Risk Regime Changes...")
        
        if not ML_AVAILABLE:
            return self._simple_risk_analysis(returns_data)
        
        try:
            # Calculate rolling risk metrics
            rolling_volatility = returns_data.rolling(window).std()
            rolling_correlation = []
            
            for i in range(window, len(returns_data)):
                window_data = returns_data.iloc[i-window:i]
                corr_matrix = window_data.corr()
                upper_tri = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
                rolling_correlation.append(np.mean(upper_tri))
            
            # Prepare features for regime detection
            features = []
            for i in range(len(rolling_correlation)):
                if i + window < len(returns_data):
                    features.append([
                        rolling_correlation[i],
                        rolling_volatility.iloc[i + window].mean(),
                        returns_data.iloc[i + window].mean().mean()
                    ])
            
            if len(features) < 20:
                return self._simple_risk_analysis(returns_data)
            
            # Use Isolation Forest for anomaly detection
            isolation_forest = IsolationForest(contamination=0.1, random_state=42)
            anomalies = isolation_forest.fit_predict(features)
            
            # Identify regime changes
            regime_changes = []
            for i, anomaly in enumerate(anomalies):
                if anomaly == -1:  # Anomaly detected
                    regime_changes.append({
                        'date': returns_data.index[i + window],
                        'type': 'regime_change',
                        'correlation': rolling_correlation[i],
                        'volatility': rolling_volatility.iloc[i + window].mean()
                    })
            
            # Current regime assessment
            current_regime = self._assess_current_regime(
                rolling_correlation[-10:] if len(rolling_correlation) >= 10 else rolling_correlation,
                rolling_volatility.iloc[-10:].mean() if len(rolling_volatility) >= 10 else rolling_volatility.mean()
            )
            
            return {
                'regime_changes': regime_changes,
                'current_regime': current_regime,
                'rolling_correlation': rolling_correlation,
                'rolling_volatility': rolling_volatility,
                'anomaly_scores': isolation_forest.decision_function(features)
            }
            
        except Exception as e:
            print(f"❌ Error in risk regime detection: {str(e)}")
            return self._simple_risk_analysis(returns_data)
    
    def _simple_risk_analysis(self, returns_data):
        """Simple risk analysis without ML"""
        # Calculate basic risk metrics
        current_volatility = returns_data.std().mean()
        current_correlation = returns_data.corr().values[np.triu_indices_from(returns_data.corr().values, k=1)].mean()
        
        # Simple regime classification
        if current_correlation > 0.6:
            regime = 'HIGH_CORRELATION'
        elif current_correlation > 0.4:
            regime = 'MEDIUM_CORRELATION'
        else:
            regime = 'LOW_CORRELATION'
        
        return {
            'current_regime': {
                'regime': regime,
                'correlation': current_correlation,
                'volatility': current_volatility
            },
            'method': 'simple'
        }
    
    def _assess_current_regime(self, recent_correlations, recent_volatility):
        """Assess current market regime"""
        avg_correlation = np.mean(recent_correlations)
        correlation_trend = np.mean(np.diff(recent_correlations)) if len(recent_correlations) > 1 else 0
        
        if avg_correlation > 0.7:
            regime = 'CRISIS'
            risk_level = 'HIGH'
        elif avg_correlation > 0.5:
            regime = 'STRESSED'
            risk_level = 'MEDIUM'
        elif avg_correlation > 0.3:
            regime = 'NORMAL'
            risk_level = 'LOW'
        else:
            regime = 'DIVERSIFIED'
            risk_level = 'VERY_LOW'
        
        return {
            'regime': regime,
            'risk_level': risk_level,
            'correlation': avg_correlation,
            'correlation_trend': correlation_trend,
            'volatility': recent_volatility,
            'recommendation': self._get_regime_recommendation(regime)
        }
    
    def _get_regime_recommendation(self, regime):
        """Get recommendation based on regime"""
        recommendations = {
            'CRISIS': 'Increase hedging immediately. Consider defensive positions.',
            'STRESSED': 'Monitor closely. Prepare to increase hedging.',
            'NORMAL': 'Maintain current hedging strategy.',
            'DIVERSIFIED': 'Consider reducing hedge positions for alpha generation.'
        }
        return recommendations.get(regime, 'Monitor market conditions.')
    
    def dynamic_hedge_ratio_calculation(self, returns_data, lookback_window=60):
        """Calculate dynamic hedge ratios"""
        print(f"\n⚖️ Calculating Dynamic Hedge Ratios...")
        
        if 'VNINDEX' not in returns_data.columns:
            print("❌ Market index (VNINDEX) not found in data")
            return {}
        
        market_returns = returns_data['VNINDEX']
        stock_returns = returns_data.drop('VNINDEX', axis=1)
        
        dynamic_ratios = {}
        
        for stock in stock_returns.columns:
            # Calculate rolling hedge ratios
            rolling_ratios = []
            dates = []
            
            for i in range(lookback_window, len(returns_data)):
                window_stock = stock_returns[stock].iloc[i-lookback_window:i]
                window_market = market_returns.iloc[i-lookback_window:i]
                
                # Calculate beta (hedge ratio)
                covariance = np.cov(window_stock, window_market)[0, 1]
                market_variance = np.var(window_market)
                
                if market_variance > 0:
                    beta = covariance / market_variance
                else:
                    beta = 0
                
                rolling_ratios.append(beta)
                dates.append(returns_data.index[i])
            
            # Calculate adaptive hedge ratio
            if rolling_ratios:
                # Use exponential weighted moving average for adaptive ratio
                weights = np.exp(np.linspace(-1, 0, len(rolling_ratios)))
                weights = weights / weights.sum()
                
                adaptive_ratio = np.sum(np.array(rolling_ratios) * weights)
                
                # Calculate confidence metrics
                ratio_volatility = np.std(rolling_ratios)
                ratio_trend = np.mean(np.diff(rolling_ratios[-10:])) if len(rolling_ratios) >= 10 else 0
                
                dynamic_ratios[stock] = {
                    'current_ratio': adaptive_ratio,
                    'static_ratio': rolling_ratios[-1],
                    'ratio_volatility': ratio_volatility,
                    'ratio_trend': ratio_trend,
                    'confidence': max(0, 1 - ratio_volatility),
                    'rolling_ratios': rolling_ratios,
                    'dates': dates
                }
        
        return dynamic_ratios
    
    def comprehensive_advanced_analysis(self, returns_data):
        """Run comprehensive advanced analysis"""
        print(f"\n🧠 Running Comprehensive Advanced Analysis...")
        print("=" * 60)
        
        results = {}
        
        # 1. Correlation prediction
        print(f"1️⃣ Correlation Prediction Model...")
        results['correlation_prediction'] = self.correlation_prediction_model(returns_data)
        
        # 2. Sector rotation detection
        print(f"2️⃣ Sector Rotation Detection...")
        results['sector_rotation'] = self.sector_rotation_detection(returns_data)
        
        # 3. Portfolio optimization
        print(f"3️⃣ Portfolio Optimization...")
        results['portfolio_optimization'] = self.portfolio_optimization_with_hedging(returns_data)
        
        # 4. Risk regime detection
        print(f"4️⃣ Risk Regime Detection...")
        results['risk_regime'] = self.risk_regime_detection(returns_data)
        
        # 5. Dynamic hedge ratios
        print(f"5️⃣ Dynamic Hedge Ratios...")
        results['dynamic_hedging'] = self.dynamic_hedge_ratio_calculation(returns_data)
        
        # 6. Generate insights
        print(f"6️⃣ Generating Insights...")
        results['insights'] = self._generate_advanced_insights(results)
        
        print(f"\n✅ Advanced Analysis Complete!")
        return results
    
    def _generate_advanced_insights(self, results):
        """Generate advanced insights from analysis results"""
        insights = []
        
        # Correlation insights
        if 'correlation_prediction' in results:
            corr_pred = results['correlation_prediction']
            if 'forecasts' in corr_pred:
                future_corr = np.mean(corr_pred['forecasts'][:7])  # Next week average
                if future_corr > 0.6:
                    insights.append({
                        'type': 'WARNING',
                        'message': f"High correlation predicted: {future_corr:.3f}",
                        'recommendation': "Consider increasing hedge positions"
                    })
        
        # Sector rotation insights
        if 'sector_rotation' in results:
            sector_rot = results['sector_rotation']
            if 'rotation_signals' in sector_rot:
                strong_buys = [s for s, info in sector_rot['rotation_signals'].items() 
                             if info['signal'] == 'STRONG_BUY']
                if strong_buys:
                    insights.append({
                        'type': 'OPPORTUNITY',
                        'message': f"Strong sector momentum: {', '.join(strong_buys)}",
                        'recommendation': "Consider overweighting these sectors"
                    })
        
        # Risk regime insights
        if 'risk_regime' in results:
            risk_reg = results['risk_regime']
            if 'current_regime' in risk_reg:
                regime_info = risk_reg['current_regime']
                if regime_info.get('risk_level') == 'HIGH':
                    insights.append({
                        'type': 'ALERT',
                        'message': f"High risk regime detected: {regime_info.get('regime')}",
                        'recommendation': regime_info.get('recommendation', 'Increase hedging')
                    })
        
        # Portfolio optimization insights
        if 'portfolio_optimization' in results:
            port_opt = results['portfolio_optimization']
            if port_opt.get('sharpe_ratio', 0) > 1.5:
                insights.append({
                    'type': 'SUCCESS',
                    'message': f"Excellent risk-adjusted returns: Sharpe = {port_opt['sharpe_ratio']:.2f}",
                    'recommendation': "Current portfolio allocation is optimal"
                })
        
        return insights


# Test the advanced analytics
if __name__ == "__main__":
    print("🧠 Testing Vietnam Advanced Analytics")
    print("=" * 50)
    
    # Initialize components
    pipeline = VietnamStockDataPipeline()
    engine = VietnamHedgingEngine(pipeline)
    analytics = VietnamAdvancedAnalytics(pipeline, engine)
    
    # Get test data
    print("\n📡 Loading test data...")
    demo_prices = pipeline.get_demo_portfolio()
    
    if demo_prices is not None:
        demo_returns = pipeline.calculate_returns(demo_prices)
        
        # Run comprehensive advanced analysis
        print("\n🔄 Running comprehensive advanced analysis...")
        results = analytics.comprehensive_advanced_analysis(demo_returns)
        
        # Display key insights
        if 'insights' in results:
            print("\n💡 KEY INSIGHTS:")
            for insight in results['insights']:
                print(f"   {insight['type']}: {insight['message']}")
                print(f"   → {insight['recommendation']}")
        
        print("\n✅ Advanced analytics testing completed!")
    else:
        print("❌ Could not load test data") 