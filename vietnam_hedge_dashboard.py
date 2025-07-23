#!/usr/bin/env python3
"""
Vietnam Stock Hedging Web Dashboard
==================================

Interactive web dashboard for real-time hedging analysis and monitoring
using Streamlit framework.

Features:
- Real-time data visualization
- Interactive hedge analysis
- Portfolio monitoring
- Alert system
- Export capabilities

Usage:
    streamlit run vietnam_hedge_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Import our modules
try:
    from vietnam_hedge_pipeline import VietnamStockDataPipeline
    from vietnam_hedging_engine import VietnamHedgingEngine
    from vietnam_hedge_monitor import VietnamHedgeMonitor
    from vietnam_backtesting_engine import VietnamBacktestingEngine
    FULL_SYSTEM = True
except ImportError:
    # Fallback to simple demo if advanced libraries not available
    FULL_SYSTEM = False

# Import SimpleHedgingDemo if needed
try:
    from simple_demo import SimpleHedgingDemo
except ImportError:
    SimpleHedgingDemo = None

class VietnamHedgeDashboard:
    """Interactive web dashboard for Vietnam stock hedging"""
    
    def __init__(self):
        """Initialize the dashboard"""
        self.setup_page_config()
        self.initialize_session_state()
        
        if FULL_SYSTEM:
            self.pipeline = VietnamStockDataPipeline()
            self.engine = VietnamHedgingEngine(self.pipeline)
            self.monitor = VietnamHedgeMonitor(self.pipeline, self.engine)
            self.backtest_engine = VietnamBacktestingEngine()
        else:
            self.demo = SimpleHedgingDemo() if SimpleHedgingDemo else None
    
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="Vietnam Stock Hedging Dashboard",
            page_icon="🇻🇳",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'last_update' not in st.session_state:
            st.session_state.last_update = None
        if 'selected_stocks' not in st.session_state:
            st.session_state.selected_stocks = ['BID', 'CTG', 'ACB', 'VHM', 'VIC', 'VRE']
        if 'alert_history' not in st.session_state:
            st.session_state.alert_history = []
    
    def render_header(self):
        """Render dashboard header with enhanced layout"""
        # Enhanced title with subtitle
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="color: #2E86AB; margin-bottom: 0.5rem;">🇻🇳 Vietnam Stock Hedging Dashboard</h1>
            <p style="color: #6C757D; font-size: 1.1rem; margin-top: 0;">
                Professional Portfolio Management & Risk Analysis Platform
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Enhanced status indicators with better proportions
        # Using unequal column widths for better visual balance
        col1, col2, col3, col4, col5 = st.columns([1.2, 1.2, 1.2, 1.2, 0.2])
        
        with col1:
            # Enhanced metric with background
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                padding: 1rem;
                border-radius: 10px;
                text-align: center;
                color: white;
                margin-bottom: 1rem;
            ">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🟢</div>
                <div style="font-size: 1.2rem; font-weight: bold;">Online</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">System Status</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            last_update_text = (
                st.session_state.last_update.strftime("%H:%M:%S") 
                if st.session_state.last_update else "Never"
            )
            update_color = "#17a2b8" if st.session_state.last_update else "#6c757d"
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {update_color} 0%, #20c997 100%);
                padding: 1rem;
                border-radius: 10px;
                text-align: center;
                color: white;
                margin-bottom: 1rem;
            ">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🕒</div>
                <div style="font-size: 1.2rem; font-weight: bold;">{last_update_text}</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Last Update</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            portfolio_size = (
                len(st.session_state.selected_stocks) 
                if st.session_state.analysis_results else 0
            )
            portfolio_color = "#2E86AB" if portfolio_size > 0 else "#6c757d"
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {portfolio_color} 0%, #4ecdc4 100%);
                padding: 1rem;
                border-radius: 10px;
                text-align: center;
                color: white;
                margin-bottom: 1rem;
            ">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📊</div>
                <div style="font-size: 1.2rem; font-weight: bold;">{portfolio_size} Stocks</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Portfolio Size</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            alert_count = len(st.session_state.alert_history)
            alert_color = "#dc3545" if alert_count > 0 else "#28a745"
            alert_icon = "🚨" if alert_count > 0 else "✅"
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {alert_color} 0%, #fd7e14 100%);
                padding: 1rem;
                border-radius: 10px;
                text-align: center;
                color: white;
                margin-bottom: 1rem;
            ">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{alert_icon}</div>
                <div style="font-size: 1.2rem; font-weight: bold;">{alert_count} Alerts</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Active Alerts</div>
            </div>
            """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.title("⚙️ Controls")
        
        # Stock selection
        st.sidebar.subheader("📈 Stock Selection")
        
        # Predefined portfolios
        portfolio_options = {
            "Demo Portfolio": ['BID', 'CTG', 'ACB', 'VHM', 'VIC', 'VRE'],
            "Banking Sector": ['BID', 'CTG', 'ACB', 'VCB', 'TCB', 'VPB'],
            "Real Estate": ['VHM', 'VIC', 'VRE', 'HDG', 'KDH', 'DXG'],
            "Diversified": ['BID', 'VCB', 'VHM', 'VIC', 'HPG', 'MWG', 'GAS', 'PLX']
        }
        
        selected_portfolio = st.sidebar.selectbox(
            "Select Portfolio Template",
            options=list(portfolio_options.keys()),
            index=0
        )
        
        # Update selected stocks
        if selected_portfolio:
            st.session_state.selected_stocks = portfolio_options[selected_portfolio]
        
        # Custom stock selection
        st.sidebar.subheader("🎯 Custom Selection")
        custom_stocks = st.sidebar.text_input(
            "Enter stock symbols (comma-separated)",
            value=",".join(st.session_state.selected_stocks),
            help="Enter Vietnamese stock symbols separated by commas"
        )
        
        if custom_stocks:
            st.session_state.selected_stocks = [s.strip().upper() for s in custom_stocks.split(",")]
        
        # Analysis parameters
        st.sidebar.subheader("📊 Analysis Parameters")
        
        analysis_period = st.sidebar.selectbox(
            "Analysis Period",
            options=["1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "5 Years", "10 Years"],
            index=6  # Default to 10 Years
        )
        
        # Convert to days
        period_days = {
            "1 Month": 30,
            "3 Months": 90,
            "6 Months": 180,
            "1 Year": 365,
            "2 Years": 730,
            "5 Years": 1825,
            "10 Years": 3650
        }
        
        correlation_window = st.sidebar.slider(
            "Correlation Window (days)",
            min_value=10,
            max_value=60,
            value=30,
            help="Rolling window for correlation calculation"
        )
        
        # Alert thresholds
        st.sidebar.subheader("🚨 Alert Settings")
        
        high_corr_threshold = st.sidebar.slider(
            "High Correlation Alert",
            min_value=0.5,
            max_value=0.9,
            value=0.7,
            step=0.05,
            help="Trigger alert when correlation exceeds this level"
        )
        
        medium_corr_threshold = st.sidebar.slider(
            "Medium Correlation Alert",
            min_value=0.3,
            max_value=0.7,
            value=0.5,
            step=0.05,
            help="Trigger warning when correlation exceeds this level"
        )
        
        # Analysis controls
        st.sidebar.subheader("🔄 Analysis Controls")
        
        if st.sidebar.button("🚀 Run Analysis", type="primary"):
            self.run_analysis(period_days[analysis_period], correlation_window)
        
        if st.sidebar.button("🔄 Refresh Data"):
            self.refresh_data()
        
        auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=False)
        
        if auto_refresh:
            time.sleep(30)
            st.rerun()
        
        # Export options
        st.sidebar.subheader("💾 Export Options")
        
        if st.session_state.analysis_results:
            if st.sidebar.button("📊 Export Results"):
                self.export_results()
        
        return {
            'analysis_period': period_days[analysis_period],
            'correlation_window': correlation_window,
            'high_corr_threshold': high_corr_threshold,
            'medium_corr_threshold': medium_corr_threshold
        }
    
    def run_analysis(self, period_days, correlation_window):
        """Run hedging analysis"""
        with st.spinner("🔄 Running hedging analysis..."):
            try:
                if FULL_SYSTEM:
                    # Use full system
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=period_days)
                    
                    # Get data
                    prices = self.pipeline.get_market_data(
                        st.session_state.selected_stocks,
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d')
                    )
                    
                    if prices is not None:
                        returns = self.pipeline.calculate_returns(prices)
                        
                        # Run comprehensive analysis
                        hedge_results = self.engine.comprehensive_hedge_analysis(returns)
                        
                        # Get monitoring data
                        correlation_monitor = self.monitor.real_time_correlation_monitor(
                            returns, window=correlation_window
                        )
                        
                        # Get recommendations
                        recommendations = self.monitor.generate_hedge_recommendations(returns)
                        
                        st.session_state.analysis_results = {
                            'hedge_results': hedge_results,
                            'correlation_monitor': correlation_monitor,
                            'recommendations': recommendations,
                            'prices': prices,
                            'returns': returns
                        }
                        
                        st.session_state.last_update = datetime.now()
                        st.success("✅ Analysis completed successfully!")
                    else:
                        st.error("❌ Failed to load data")
                
                else:
                    # Use simple demo
                    results = self.demo.run_demo()
                    if results:
                        st.session_state.analysis_results = {'simple_results': results}
                        st.session_state.last_update = datetime.now()
                        st.success("✅ Demo analysis completed!")
                    else:
                        st.error("❌ Demo analysis failed")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
    
    def refresh_data(self):
        """Refresh data without full analysis"""
        st.session_state.analysis_results = None
        st.session_state.last_update = None
        st.rerun()
    
    def render_main_dashboard(self):
        """Render main dashboard content with tabs"""
        if not st.session_state.analysis_results:
            st.info("👆 Select stocks and click 'Run Analysis' to get started")
            
            # Show sample visualization
            self.render_sample_charts()
            return
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Hedge Analysis", 
            "📈 Price Charts",
            "📊 Cumulative Returns", 
            "🔍 Monitoring", 
            "💡 Recommendations",
            "🎯 Backtesting"
        ])
        
        with tab1:
            # Render analysis results
            if FULL_SYSTEM and 'hedge_results' in st.session_state.analysis_results:
                self.render_analysis_tab()
            elif 'simple_results' in st.session_state.analysis_results:
                self.render_simple_analysis()
        
        with tab2:
            self.render_price_charts_tab()
        
        with tab3:
            self.render_cumulative_returns_tab()
        
        with tab4:
            self.render_monitoring_tab()
        
        with tab5:
            self.render_recommendations_tab()
        
        with tab6:
            self.render_backtesting_tab()
    
    def render_sample_charts(self):
        """Render sample charts when no data is available"""
        st.subheader("📊 Sample Vietnam Stock Hedging Analysis")
        
        # Sample correlation matrix
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔗 Correlation Matrix")
            sample_corr = np.array([
                [1.0, 0.65, 0.45, 0.35, 0.25, 0.30],
                [0.65, 1.0, 0.55, 0.40, 0.30, 0.35],
                [0.45, 0.55, 1.0, 0.25, 0.20, 0.25],
                [0.35, 0.40, 0.25, 1.0, 0.60, 0.55],
                [0.25, 0.30, 0.20, 0.60, 1.0, 0.45],
                [0.30, 0.35, 0.25, 0.55, 0.45, 1.0]
            ])
            
            fig = px.imshow(
                sample_corr,
                x=['BID', 'CTG', 'ACB', 'VHM', 'VIC', 'VRE'],
                y=['BID', 'CTG', 'ACB', 'VHM', 'VIC', 'VRE'],
                color_continuous_scale='RdYlBu_r',
                title="Sample Correlation Matrix"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Effective Breadth Comparison")
            methods = ['No Hedge', 'Market Hedge', 'Sector Hedge']
            breadths = [2.4, 7.3, 7.0]
            
            fig = go.Figure(data=[
                go.Bar(x=methods, y=breadths, 
                      marker_color=['red', 'orange', 'green'])
            ])
            fig.update_layout(
                title="Effective Breadth Comparison",
                yaxis_title="Effective Breadth",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_analysis_tab(self):
        """Render analysis tab content"""
        results = st.session_state.analysis_results
        hedge_results = results['hedge_results']
        
        # Summary metrics
        st.subheader("📋 Analysis Summary")
        
        if 'summary' in hedge_results:
            summary = hedge_results['summary']
            
            # Metrics row
            cols = st.columns(len(summary['comparison_table']))
            
            for i, row in enumerate(summary['comparison_table']):
                with cols[i]:
                    st.metric(
                        f"{row['method']}",
                        f"BR: {row['effective_breadth']:.1f}",
                        f"Corr: {row['avg_correlation']:.3f}"
                    )
        
        # Detailed charts
        self.render_correlation_charts(results)
        self.render_hedge_effectiveness_charts(results)
    
    def render_price_charts_tab(self):
        """Render price charts and technical analysis tab"""
        results = st.session_state.analysis_results
        
        st.subheader("📈 Price Charts & Technical Analysis")
        
        if FULL_SYSTEM and 'prices' in results and 'returns' in results:
            prices = results['prices']
            returns = results['returns']
            
            # Chart configuration controls
            st.subheader("⚙️ Chart Configuration")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                chart_type = st.selectbox(
                    "Chart Type",
                    ["Line Chart", "Candlestick", "OHLC"],
                    help="Select chart visualization type"
                )
            
            with col2:
                timeframe = st.selectbox(
                    "Timeframe",
                    ["All Data", "10 Years", "5 Years", "2 Years", "1 Year", "6 Months", "3 Months", "1 Month"],
                    help="Select time period to display"
                )
            
            with col3:
                normalize_prices = st.checkbox(
                    "Normalize Prices",
                    value=False,
                    help="Show all stocks starting from 100 for comparison"
                )
            
            with col4:
                show_volume = st.checkbox(
                    "Show Volume",
                    value=False,
                    help="Display volume overlay (simulated)"
                )
            
            # Filter data based on timeframe
            if timeframe != "All Data":
                days_map = {
                    "1 Month": 30,
                    "3 Months": 90, 
                    "6 Months": 180,
                    "1 Year": 365,
                    "2 Years": 730,
                    "5 Years": 1825,
                    "10 Years": 3650
                }
                days = days_map[timeframe]
                filtered_prices = prices.tail(days)
                filtered_returns = returns.tail(days)
            else:
                filtered_prices = prices
                filtered_returns = returns
            
            # Stock selection for individual charts
            st.subheader("📊 Individual Stock Charts")
            
            stock_columns = [col for col in filtered_prices.columns if col != 'VNINDEX']
            
            # Multi-stock comparison chart
            st.subheader("📈 Multi-Stock Price Comparison")
            
            selected_stocks = st.multiselect(
                "Select stocks to compare",
                options=stock_columns + ['VNINDEX'],
                default=stock_columns[:4] + ['VNINDEX'],
                help="Choose stocks for comparison chart"
            )
            
            if selected_stocks:
                # Prepare price data
                comparison_data = filtered_prices[selected_stocks].copy()
                
                if normalize_prices:
                    # Normalize to start at 100
                    comparison_data = (comparison_data / comparison_data.iloc[0]) * 100
                    y_title = "Normalized Price (Base=100)"
                else:
                    y_title = "Price (VND)"
                
                # Create comparison chart
                fig = go.Figure()
                
                # Add price lines
                for stock in selected_stocks:
                    line_style = dict(width=3, dash='dash') if stock == 'VNINDEX' else dict(width=2)
                    color = 'red' if stock == 'VNINDEX' else None
                    
                    fig.add_trace(go.Scatter(
                        x=comparison_data.index,
                        y=comparison_data[stock],
                        mode='lines',
                        name=stock,
                        line=line_style,
                        marker_color=color
                    ))
                
                fig.update_layout(
                    title=f"Price Comparison - {timeframe}",
                    xaxis_title="Date",
                    yaxis_title=y_title,
                    hovermode='x unified',
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    ),
                    # Enhanced interactivity with range selectors
                    xaxis=dict(
                        rangeselector=dict(
                            buttons=list([
                                dict(count=1, label="1M", step="month", stepmode="backward"),
                                dict(count=3, label="3M", step="month", stepmode="backward"),
                                dict(count=6, label="6M", step="month", stepmode="backward"),
                                dict(count=1, label="1Y", step="year", stepmode="backward"),
                                dict(count=2, label="2Y", step="year", stepmode="backward"),
                                dict(step="all", label="All")
                            ]),
                            bgcolor="rgba(150, 150, 150, 0.1)",
                            bordercolor="rgba(150, 150, 150, 0.2)",
                            activecolor="rgba(46, 134, 171, 0.8)",
                            x=0.02,
                            y=1.05
                        ),
                        rangeslider=dict(
                            visible=True,
                            bgcolor="rgba(150, 150, 150, 0.1)",
                            bordercolor="rgba(150, 150, 150, 0.2)",
                            thickness=0.05
                        ),
                        type="date"
                    ),
                    height=500,
                    # Professional styling
                    plot_bgcolor='rgba(248, 249, 250, 0.8)',
                    paper_bgcolor='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Technical Analysis Section with Expandable Panel
            with st.expander("🔍 Advanced Technical Analysis", expanded=False):
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                    padding: 1rem;
                    border-radius: 8px;
                    border-left: 4px solid #2E86AB;
                    margin-bottom: 1rem;
                ">
                    <h4 style="color: #2E86AB; margin-bottom: 0.5rem;">📊 Technical Indicators & Analysis</h4>
                    <p style="color: #6c757d; margin-bottom: 0;">
                        Configure moving averages, support/resistance levels, and advanced technical indicators
                        for individual stock analysis.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Moving averages controls with better spacing
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    show_ma = st.checkbox("Show Moving Averages", value=True)
                    ma_periods = st.multiselect(
                        "MA Periods",
                        [5, 10, 20, 50, 100, 200],
                        default=[20, 50],
                        help="Select moving average periods"
                    ) if show_ma else []
                
                with col2:
                    selected_stock_ta = st.selectbox(
                        "Stock for Technical Analysis",
                        options=stock_columns + ['VNINDEX'],
                        help="Select stock for detailed technical analysis"
                    )
                
                with col3:
                    show_support_resistance = st.checkbox(
                        "Show Support/Resistance",
                        value=False,
                        help="Identify key price levels"
                    )
                
                # Individual stock technical chart
                if selected_stock_ta:
                    stock_data = filtered_prices[selected_stock_ta].copy()
                    
                    # Create technical analysis chart
                    tech_fig = go.Figure()
                    
                    # Main price line/candlestick
                    if chart_type == "Line Chart":
                        tech_fig.add_trace(go.Scatter(
                            x=stock_data.index,
                            y=stock_data,
                            mode='lines',
                            name=f"{selected_stock_ta} Price",
                            line=dict(width=2, color='blue')
                        ))
                    elif chart_type == "Candlestick":
                        # Simulate OHLC data from price
                        ohlc_data = self.simulate_ohlc_from_price(stock_data, filtered_returns[selected_stock_ta])
                        
                        tech_fig.add_trace(go.Candlestick(
                            x=ohlc_data.index,
                            open=ohlc_data['Open'],
                            high=ohlc_data['High'],
                            low=ohlc_data['Low'],
                            close=ohlc_data['Close'],
                            name=f"{selected_stock_ta}"
                        ))
                    
                    # Add moving averages
                    if show_ma and ma_periods:
                        colors = ['orange', 'purple', 'green', 'brown', 'pink', 'gray']
                        for i, period in enumerate(ma_periods):
                            if len(stock_data) >= period:
                                ma_data = stock_data.rolling(window=period).mean()
                                
                                tech_fig.add_trace(go.Scatter(
                                    x=ma_data.index,
                                    y=ma_data,
                                    mode='lines',
                                    name=f"MA{period}",
                                    line=dict(width=1, color=colors[i % len(colors)])
                                ))
                    
                    # Add support and resistance levels
                    if show_support_resistance:
                        # Simple support/resistance based on recent highs/lows
                        recent_data = stock_data.tail(60)  # Last 60 periods
                        resistance = recent_data.max()
                        support = recent_data.min()
                        
                        tech_fig.add_hline(
                            y=resistance, 
                            line_dash="dash", 
                            line_color="green",
                            annotation_text="Resistance"
                        )
                        tech_fig.add_hline(
                            y=support, 
                            line_dash="dash", 
                            line_color="red",
                            annotation_text="Support"
                        )
                    
                    tech_fig.update_layout(
                        title=f"{selected_stock_ta} - Technical Analysis",
                        xaxis_title="Date",
                        yaxis_title="Price (VND)",
                        hovermode='x unified',
                        height=500
                    )
                    
                    st.plotly_chart(tech_fig, use_container_width=True)
            
            # Volume analysis (simulated)
            if show_volume:
                st.subheader("📊 Volume Analysis")
                
                # Simulate volume data based on volatility
                volume_data = self.simulate_volume_data(filtered_returns, stock_columns)
                
                vol_fig = go.Figure()
                
                for stock in selected_stocks[:3]:  # Limit to 3 stocks for clarity
                    if stock in volume_data.columns:
                        vol_fig.add_trace(go.Bar(
                            x=volume_data.index,
                            y=volume_data[stock],
                            name=f"{stock} Volume",
                            opacity=0.7
                        ))
                
                vol_fig.update_layout(
                    title="Trading Volume Analysis (Simulated)",
                    xaxis_title="Date",
                    yaxis_title="Volume",
                    hovermode='x unified'
                )
                
                st.plotly_chart(vol_fig, use_container_width=True)
            
            # Price statistics table
            st.subheader("📋 Price Statistics")
            
            price_stats = []
            for stock in stock_columns + ['VNINDEX']:
                if stock in filtered_prices.columns:
                    current_price = filtered_prices[stock].iloc[-1]
                    high_52w = filtered_prices[stock].max()
                    low_52w = filtered_prices[stock].min()
                    avg_price = filtered_prices[stock].mean()
                    
                    # Calculate relative position
                    position_pct = (current_price - low_52w) / (high_52w - low_52w) * 100
                    
                    price_stats.append({
                        'Stock': stock,
                        'Current Price': f"{current_price:,.0f}",
                        'Period High': f"{high_52w:,.0f}",
                        'Period Low': f"{low_52w:,.0f}",
                        'Average': f"{avg_price:,.0f}",
                        'Position %': f"{position_pct:.1f}%",
                        'vs High': f"{(current_price/high_52w-1)*100:+.1f}%",
                        'vs Low': f"{(current_price/low_52w-1)*100:+.1f}%"
                    })
            
            price_stats_df = pd.DataFrame(price_stats)
            st.dataframe(price_stats_df, use_container_width=True)
            
            # Price correlation heatmap
            st.subheader("🔗 Price Correlation Matrix")
            
            price_corr = filtered_prices.corr()
            
            corr_fig = px.imshow(
                price_corr,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title="Price Correlation Matrix"
            )
            
            corr_fig.update_layout(height=400)
            st.plotly_chart(corr_fig, use_container_width=True)
            
        else:
            st.info("📊 Price charts require analysis data. Please run the hedge analysis first.")
    
    def simulate_ohlc_from_price(self, price_series, return_series):
        """Simulate OHLC data from price series"""
        ohlc_data = []
        
        for i in range(len(price_series)):
            close_price = price_series.iloc[i]
            
            if i == 0:
                open_price = close_price
            else:
                open_price = price_series.iloc[i-1]
            
            # Simulate intraday volatility
            daily_vol = abs(return_series.iloc[i]) if i < len(return_series) else 0.01
            price_range = close_price * daily_vol * 2
            
            high_price = max(open_price, close_price) + price_range * 0.5
            low_price = min(open_price, close_price) - price_range * 0.5
            
            ohlc_data.append({
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price
            })
        
        return pd.DataFrame(ohlc_data, index=price_series.index)
    
    def calculate_support_resistance(self, price_series):
        """Calculate basic support and resistance levels"""
        # Simple method: use recent lows/highs
        recent_data = price_series.tail(60)  # Last 60 days
        
        support = recent_data.min()
        resistance = recent_data.max()
        
        return support, resistance
    
    def simulate_volume_data(self, return_series, stock_columns):
        """Simulate volume data based on volatility"""
        volume_data = pd.DataFrame(index=return_series.index)
        
        for stock in stock_columns:
            if stock in return_series.columns:
                # Higher volatility = higher volume
                base_volume = 1000000  # Base volume
                vol_multiplier = abs(return_series[stock]) * 10 + 1
                volume_data[stock] = (base_volume * vol_multiplier).astype(int)
        
        return volume_data
    
    def calculate_drawdowns(self, cumulative_returns, portfolio_cumulative, stock_columns):
        """Calculate comprehensive drawdown analysis for portfolio and individual stocks"""
        
        def calculate_drawdown_series(prices):
            """Calculate drawdown series for a price/value series"""
            # Calculate running maximum (peak)
            running_max = prices.expanding().max()
            # Calculate drawdown as (current - peak) / peak
            drawdown = (prices - running_max) / running_max
            return drawdown
        
        def calculate_drawdown_stats(drawdown_series):
            """Calculate drawdown statistics"""
            # Find drawdown periods (when drawdown < 0)
            in_drawdown = drawdown_series < 0
            
            # Calculate basic stats
            max_drawdown = drawdown_series.min()
            avg_drawdown = drawdown_series[in_drawdown].mean() if in_drawdown.any() else 0
            
            # Count drawdown periods
            drawdown_periods = []
            in_period = False
            start_idx = None
            
            for i, is_dd in enumerate(in_drawdown):
                if is_dd and not in_period:
                    # Start of new drawdown period
                    in_period = True
                    start_idx = i
                elif not is_dd and in_period:
                    # End of drawdown period
                    in_period = False
                    if start_idx is not None:
                        drawdown_periods.append((start_idx, i-1))
            
            # Handle case where we end in a drawdown
            if in_period and start_idx is not None:
                drawdown_periods.append((start_idx, len(in_drawdown)-1))
            
            # Calculate average recovery time
            recovery_times = []
            for start, end in drawdown_periods:
                recovery_times.append(end - start + 1)
            
            avg_recovery_days = np.mean(recovery_times) if recovery_times else 0
            drawdown_count = len(drawdown_periods)
            
            return {
                'max_drawdown': max_drawdown,
                'avg_drawdown': avg_drawdown,
                'avg_recovery_days': avg_recovery_days,
                'drawdown_count': drawdown_count,
                'drawdown_periods': drawdown_periods
            }
        
        # Calculate portfolio drawdown
        portfolio_drawdown = calculate_drawdown_series(portfolio_cumulative)
        portfolio_stats = calculate_drawdown_stats(portfolio_drawdown)
        
        # Calculate individual stock drawdowns
        individual_drawdowns = {}
        individual_stats = {}
        
        for stock in stock_columns + (['VNINDEX'] if 'VNINDEX' in cumulative_returns.columns else []):
            if stock in cumulative_returns.columns:
                stock_drawdown = calculate_drawdown_series(cumulative_returns[stock])
                individual_drawdowns[stock] = stock_drawdown
                individual_stats[stock] = calculate_drawdown_stats(stock_drawdown)
        
        return {
            'portfolio_drawdown': portfolio_drawdown,
            'portfolio_stats': portfolio_stats,
            'individual_drawdowns': individual_drawdowns,
            'individual_stats': individual_stats
        }
    
    def render_cumulative_returns_tab(self):
        """Render cumulative returns analysis tab"""
        results = st.session_state.analysis_results
        
        st.subheader("📈 Cumulative Returns Analysis")
        
        if FULL_SYSTEM and 'prices' in results and 'returns' in results:
            prices = results['prices']
            returns = results['returns']
            
            # Calculate cumulative returns
            cumulative_returns = (1 + returns).cumprod()
            
            # Portfolio weight configuration
            st.subheader("⚖️ Portfolio Configuration")
            
            stock_columns = [col for col in returns.columns if col != 'VNINDEX']
            n_stocks = len(stock_columns)
            
            # Weight selection method
            weight_method = st.selectbox(
                "Portfolio Weighting Method",
                ["Equal Weight", "Market Cap Weight", "Inverse Volatility", "Custom Weights"],
                help="Choose how to weight stocks in the portfolio"
            )
            
            if weight_method == "Equal Weight":
                portfolio_weights = np.ones(n_stocks) / n_stocks
                vol_period = 30  # Default for other calculations
                st.info(f"📊 Using equal weights: {1/n_stocks:.1%} per stock")
                # Reset volatility scaling
                st.session_state['vol_scaling_factor'] = 1.0
                st.session_state['target_vol'] = None
            
            elif weight_method == "Market Cap Weight":
                # Simple market cap approximation based on recent price levels
                recent_prices = prices[stock_columns].iloc[-1]
                market_caps = recent_prices / recent_prices.sum()
                portfolio_weights = market_caps.values
                vol_period = 30  # Default for other calculations
                
                weight_df = pd.DataFrame({
                    'Stock': stock_columns,
                    'Weight': [f"{w:.1%}" for w in portfolio_weights]
                })
                st.dataframe(weight_df, use_container_width=True)
                # Reset volatility scaling
                st.session_state['vol_scaling_factor'] = 1.0
                st.session_state['target_vol'] = None
            
            elif weight_method == "Inverse Volatility":
                # Volatility period selection
                col1, col2, col3 = st.columns(3)
                with col1:
                    vol_period = st.selectbox(
                        "Volatility Calculation Period",
                        [20, 30, 60, 90, 120],
                        index=1,
                        help="Number of days to calculate volatility"
                    )
                
                with col2:
                    min_weight = st.slider(
                        "Minimum Weight (%)",
                        min_value=1.0,
                        max_value=20.0,
                        value=5.0,
                        step=1.0,
                        help="Minimum weight for any single stock"
                    ) / 100.0
                
                with col3:
                    vol_targeting = st.checkbox(
                        "Volatility Targeting",
                        value=False,
                        help="Scale portfolio to achieve target volatility"
                    )
                
                # Volatility target setting
                if vol_targeting:
                    target_vol = st.slider(
                        "Target Portfolio Volatility (%)",
                        min_value=5.0,
                        max_value=30.0,
                        value=15.0,
                        step=0.5,
                        help="Desired annual portfolio volatility"
                    ) / 100.0
                else:
                    target_vol = None
                
                # Calculate rolling volatility for each stock
                stock_returns = returns[stock_columns]
                volatilities = stock_returns.rolling(window=vol_period).std().iloc[-1] * np.sqrt(252)  # Annualized
                
                # Inverse volatility weights
                inverse_vol = 1.0 / volatilities
                raw_weights = inverse_vol / inverse_vol.sum()
                
                # Apply minimum weight constraint
                portfolio_weights = np.maximum(raw_weights.values, min_weight)
                portfolio_weights = portfolio_weights / portfolio_weights.sum()  # Renormalize
                
                # Calculate portfolio volatility for targeting
                if target_vol is not None:
                    # Calculate covariance matrix (annualized)
                    stock_cov_matrix = stock_returns.rolling(window=vol_period).cov().iloc[-len(stock_columns):] * 252
                    
                    # Calculate portfolio variance
                    portfolio_variance = np.dot(portfolio_weights, np.dot(stock_cov_matrix.values, portfolio_weights))
                    current_portfolio_vol = np.sqrt(portfolio_variance)
                    
                    # Calculate scaling factor to achieve target volatility
                    vol_scaling_factor = target_vol / current_portfolio_vol if current_portfolio_vol > 0 else 1.0
                    
                    # Apply volatility scaling (this affects position sizing, not weights)
                    scaled_weights = portfolio_weights  # Weights remain the same
                    actual_target_vol = target_vol
                    
                    vol_info = f"📊 Volatility Targeting: {current_portfolio_vol:.1%} → {target_vol:.1%} (Scale: {vol_scaling_factor:.2f}x)"
                else:
                    vol_scaling_factor = 1.0
                    scaled_weights = portfolio_weights
                    actual_target_vol = None
                    vol_info = f"📊 Using {vol_period}-day volatility with minimum weight {min_weight:.1%}"
                
                # Display weights and volatilities
                if target_vol is not None:
                    weight_vol_df = pd.DataFrame({
                        'Stock': stock_columns,
                        'Volatility': [f"{v:.1%}" for v in volatilities],
                        'Raw Weight': [f"{w:.1%}" for w in raw_weights],
                        'Final Weight': [f"{w:.1%}" for w in scaled_weights],
                        'Scaled Position': [f"{w * vol_scaling_factor:.1%}" for w in scaled_weights]
                    })
                else:
                    weight_vol_df = pd.DataFrame({
                        'Stock': stock_columns,
                        'Volatility': [f"{v:.1%}" for v in volatilities],
                        'Raw Weight': [f"{w:.1%}" for w in raw_weights],
                        'Final Weight': [f"{w:.1%}" for w in portfolio_weights]
                    })
                
                st.dataframe(weight_vol_df, use_container_width=True)
                st.info(vol_info)
                
                # Store the scaling factor and weights for portfolio calculations
                portfolio_weights = scaled_weights
                # Store volatility scaling info in session state for later use
                if target_vol is not None:
                    st.session_state['vol_scaling_factor'] = vol_scaling_factor
                    st.session_state['target_vol'] = target_vol
                else:
                    st.session_state['vol_scaling_factor'] = 1.0
                    st.session_state['target_vol'] = None
            
            else:  # Custom Weights
                st.write("🎯 Set custom weights for each stock:")
                weights = []
                cols = st.columns(min(4, n_stocks))
                vol_period = 30  # Default for other calculations
                
                for i, stock in enumerate(stock_columns):
                    with cols[i % 4]:
                        weight = st.number_input(
                            f"{stock} (%)",
                            min_value=0.0,
                            max_value=100.0,
                            value=100.0/n_stocks,
                            step=1.0,
                            key=f"weight_{stock}"
                        )
                        weights.append(weight / 100.0)
                
                portfolio_weights = np.array(weights)
                
                # Normalize weights
                if portfolio_weights.sum() > 0:
                    portfolio_weights = portfolio_weights / portfolio_weights.sum()
                
                total_weight = sum(weights)
                if abs(total_weight - 100.0) > 0.1:
                    st.warning(f"⚠️ Weights sum to {total_weight:.1f}%. They will be normalized to 100%.")
                
                # Reset volatility scaling for custom weights
                st.session_state['vol_scaling_factor'] = 1.0
                st.session_state['target_vol'] = None
            
            # Calculate portfolio cumulative returns
            stock_returns = returns[stock_columns]
            portfolio_returns = (stock_returns * portfolio_weights).sum(axis=1)
            
            # Apply volatility scaling if applicable
            vol_scaling_factor = st.session_state.get('vol_scaling_factor', 1.0)
            target_vol = st.session_state.get('target_vol', None)
            
            if weight_method == "Inverse Volatility" and vol_scaling_factor != 1.0:
                portfolio_returns = portfolio_returns * vol_scaling_factor
                portfolio_name = f"Portfolio ({weight_method} - {target_vol:.0%} Vol Target)"
            else:
                portfolio_name = f"Portfolio ({weight_method})"
            
            portfolio_cumulative = (1 + portfolio_returns).cumprod()
            
            # Create the plot
            fig = go.Figure()
            
            # Add individual stock lines with enhanced tooltips
            for col in cumulative_returns.columns:
                if col != 'VNINDEX':
                    # Calculate daily return for tooltip
                    daily_returns = returns[col] * 100  # Convert to percentage
                    
                    fig.add_trace(go.Scatter(
                        x=cumulative_returns.index,
                        y=cumulative_returns[col],
                        mode='lines',
                        name=f"{col}",
                        line=dict(width=1, color='rgba(173, 216, 230, 0.8)'),
                        opacity=0.7,
                        hovertemplate=(
                            f'<b>{col}</b><br>' +
                            'Date: %{x}<br>' +
                            'Cumulative Return: %{y:.2f}x<br>' +
                            'Total Gain: %{customdata:.1f}%' +
                            '<extra></extra>'
                        ),
                        customdata=[(cumulative_returns[col].iloc[i] - 1) * 100 for i in range(len(cumulative_returns))]
                    ))
            
            # Add portfolio line (bold) with enhanced tooltip
            portfolio_name = f"Portfolio ({weight_method})"
            # Calculate portfolio volatility for tooltip
            portfolio_vol = portfolio_returns.std() * np.sqrt(252) * 100
            
            fig.add_trace(go.Scatter(
                x=portfolio_cumulative.index,
                y=portfolio_cumulative,
                mode='lines',
                name=portfolio_name,
                line=dict(width=4, color='rgba(46, 134, 171, 0.9)'),
                opacity=1.0,
                hovertemplate=(
                    f'<b>{portfolio_name}</b><br>' +
                    'Date: %{x}<br>' +
                    'Cumulative Return: %{y:.2f}x<br>' +
                    'Total Gain: %{customdata:.1f}%<br>' +
                    f'Annualized Vol: {portfolio_vol:.1f}%' +
                    '<extra></extra>'
                ),
                customdata=[(portfolio_cumulative.iloc[i] - 1) * 100 for i in range(len(portfolio_cumulative))]
            ))
            
            # Add market benchmark with enhanced tooltip
            if 'VNINDEX' in cumulative_returns.columns:
                vnindex_vol = returns['VNINDEX'].std() * np.sqrt(252) * 100
                fig.add_trace(go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns['VNINDEX'],
                    mode='lines',
                    name="VN-Index",
                    line=dict(width=3, color='rgba(220, 53, 69, 0.8)', dash='dash'),
                    hovertemplate=(
                        '<b>VN-Index (Benchmark)</b><br>' +
                        'Date: %{x}<br>' +
                        'Cumulative Return: %{y:.2f}x<br>' +
                        'Total Gain: %{customdata:.1f}%<br>' +
                        f'Annualized Vol: {vnindex_vol:.1f}%' +
                        '<extra></extra>'
                    ),
                    customdata=[(cumulative_returns['VNINDEX'].iloc[i] - 1) * 100 for i in range(len(cumulative_returns))]
                ))
            
            fig.update_layout(
                title="Cumulative Returns: Individual Stocks vs Portfolio vs Market",
                xaxis_title="Date",
                yaxis_title="Cumulative Return",
                hovermode='x unified',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ),
                # Enhanced interactivity with range selectors
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1M", step="month", stepmode="backward"),
                            dict(count=3, label="3M", step="month", stepmode="backward"),
                            dict(count=6, label="6M", step="month", stepmode="backward"),
                            dict(count=1, label="1Y", step="year", stepmode="backward"),
                            dict(count=2, label="2Y", step="year", stepmode="backward"),
                            dict(step="all", label="All")
                        ]),
                        bgcolor="rgba(150, 150, 150, 0.1)",
                        bordercolor="rgba(150, 150, 150, 0.2)",
                        activecolor="rgba(46, 134, 171, 0.8)",
                        x=0.02,
                        y=1.05
                    ),
                    rangeslider=dict(
                        visible=True,
                        bgcolor="rgba(150, 150, 150, 0.1)",
                        bordercolor="rgba(150, 150, 150, 0.2)",
                        thickness=0.05
                    ),
                    type="date"
                ),
                height=600,
                # Professional styling
                plot_bgcolor='rgba(248, 249, 250, 0.8)',
                paper_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Maximum Drawdown Analysis
            st.subheader("📉 Maximum Drawdown Analysis")
            
            # Calculate drawdowns
            drawdown_data = self.calculate_drawdowns(cumulative_returns, portfolio_cumulative, stock_columns)
            
            # Drawdown controls
            col1, col2 = st.columns(2)
            with col1:
                show_individual_dd = st.checkbox(
                    "Show Individual Stock Drawdowns",
                    value=True,
                    help="Display drawdown for each individual stock"
                )
            
            with col2:
                dd_threshold = st.slider(
                    "Drawdown Alert Threshold (%)",
                    min_value=5,
                    max_value=50,
                    value=20,
                    step=5,
                    help="Highlight drawdowns exceeding this threshold"
                )
            
            # Create drawdown chart
            dd_fig = go.Figure()
            
            # Add individual stock drawdowns (optional)
            if show_individual_dd:
                for stock in stock_columns:
                    if stock in drawdown_data['individual_drawdowns']:
                        dd_fig.add_trace(go.Scatter(
                            x=drawdown_data['individual_drawdowns'][stock].index,
                            y=drawdown_data['individual_drawdowns'][stock] * 100,  # Convert to percentage
                            mode='lines',
                            name=f"{stock} DD",
                            line=dict(width=1),
                            opacity=0.7,
                            fill='tonexty' if stock != stock_columns[0] else None
                        ))
            
            # Add portfolio drawdown (prominent)
            dd_fig.add_trace(go.Scatter(
                x=drawdown_data['portfolio_drawdown'].index,
                y=drawdown_data['portfolio_drawdown'] * 100,
                mode='lines',
                name=f"Portfolio DD ({weight_method})",
                line=dict(width=4, color='red'),
                fill='tozeroy',
                fillcolor='rgba(255,0,0,0.3)'
            ))
            
            # Add market drawdown for comparison
            if 'VNINDEX' in cumulative_returns.columns:
                market_dd = drawdown_data['individual_drawdowns']['VNINDEX']
                dd_fig.add_trace(go.Scatter(
                    x=market_dd.index,
                    y=market_dd * 100,
                    mode='lines',
                    name="VN-Index DD",
                    line=dict(width=3, color='black', dash='dash'),
                    fill='tozeroy',
                    fillcolor='rgba(0,0,0,0.1)'
                ))
            
            # Add threshold line
            dd_fig.add_hline(
                y=-dd_threshold,
                line_dash="dot",
                line_color="orange",
                annotation_text=f"Alert Threshold: -{dd_threshold}%"
            )
            
            dd_fig.update_layout(
                title="Portfolio Drawdown Analysis - Peak to Trough Decline",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                hovermode='x unified',
                yaxis=dict(tickformat='.1f'),
                legend=dict(
                    yanchor="bottom",
                    y=0.02,
                    xanchor="right",
                    x=0.98
                )
            )
            
            st.plotly_chart(dd_fig, use_container_width=True)
            
            # Drawdown statistics
            st.subheader("📊 Drawdown Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                portfolio_max_dd = drawdown_data['portfolio_stats']['max_drawdown']
                st.metric(
                    "Portfolio Max Drawdown",
                    f"{portfolio_max_dd:.2%}",
                    help="Worst peak-to-trough decline"
                )
            
            with col2:
                portfolio_avg_dd = drawdown_data['portfolio_stats']['avg_drawdown']
                st.metric(
                    "Average Drawdown",
                    f"{portfolio_avg_dd:.2%}",
                    help="Average of all drawdown periods"
                )
            
            with col3:
                current_dd = drawdown_data['portfolio_drawdown'].iloc[-1]
                dd_status = "🟢 Healthy" if current_dd > -0.05 else "🟡 Moderate" if current_dd > -0.15 else "🔴 High"
                st.metric(
                    "Current Drawdown",
                    f"{current_dd:.2%}",
                    dd_status
                )
            
            with col4:
                recovery_days = drawdown_data['portfolio_stats']['avg_recovery_days']
                st.metric(
                    "Avg Recovery Time",
                    f"{recovery_days:.0f} days",
                    help="Average time to recover from drawdowns"
                )
            
            # Detailed drawdown table
            st.subheader("📋 Drawdown Comparison Table")
            
            dd_comparison = []
            
            # Portfolio stats
            dd_comparison.append({
                'Asset': f'Portfolio ({weight_method})',
                'Max Drawdown': f"{drawdown_data['portfolio_stats']['max_drawdown']:.2%}",
                'Current Drawdown': f"{current_dd:.2%}",
                'Avg Drawdown': f"{portfolio_avg_dd:.2%}",
                'Recovery Time (Days)': f"{recovery_days:.0f}",
                'Drawdown Periods': drawdown_data['portfolio_stats']['drawdown_count']
            })
            
            # Individual stock stats
            for stock in stock_columns + (['VNINDEX'] if 'VNINDEX' in cumulative_returns.columns else []):
                if stock in drawdown_data['individual_stats']:
                    stats = drawdown_data['individual_stats'][stock]
                    current_stock_dd = drawdown_data['individual_drawdowns'][stock].iloc[-1]
                    
                    dd_comparison.append({
                        'Asset': stock,
                        'Max Drawdown': f"{stats['max_drawdown']:.2%}",
                        'Current Drawdown': f"{current_stock_dd:.2%}",
                        'Avg Drawdown': f"{stats['avg_drawdown']:.2%}",
                        'Recovery Time (Days)': f"{stats['avg_recovery_days']:.0f}",
                        'Drawdown Periods': stats['drawdown_count']
                    })
            
            dd_comparison_df = pd.DataFrame(dd_comparison)
            st.dataframe(dd_comparison_df, use_container_width=True)
            
            # Drawdown insights
            st.subheader("🔍 Drawdown Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📉 Risk Assessment**")
                if portfolio_max_dd > -0.3:
                    risk_level = "🔴 High Risk"
                    risk_desc = "Portfolio experiences significant drawdowns"
                elif portfolio_max_dd > -0.2:
                    risk_level = "🟡 Moderate Risk"
                    risk_desc = "Portfolio has moderate drawdown risk"
                else:
                    risk_level = "🟢 Low Risk"
                    risk_desc = "Portfolio shows good downside protection"
                
                st.write(f"**{risk_level}**: {risk_desc}")
                st.write(f"Max drawdown of {portfolio_max_dd:.1%} vs market {drawdown_data['individual_stats']['VNINDEX']['max_drawdown']:.1%}" if 'VNINDEX' in drawdown_data['individual_stats'] else "")
            
            with col2:
                st.write("**⏱️ Recovery Analysis**")
                if recovery_days < 30:
                    recovery_speed = "🟢 Fast Recovery"
                elif recovery_days < 90:
                    recovery_speed = "🟡 Moderate Recovery"
                else:
                    recovery_speed = "🔴 Slow Recovery"
                
                st.write(f"**{recovery_speed}**: Avg {recovery_days:.0f} days to recover")
                
                # Show worst drawdown period
                max_dd_date = drawdown_data['portfolio_drawdown'].idxmin()
                st.write(f"Worst drawdown occurred on: **{max_dd_date.strftime('%Y-%m-%d')}**")
            
            # Volatility Analysis Chart
            st.subheader("📊 Rolling Volatility Analysis")
            
            # Volatility parameters
            col1, col2 = st.columns(2)
            with col1:
                vol_window = st.selectbox(
                    "Volatility Window",
                    [20, 30, 60, 90],
                    index=1,
                    key="vol_chart_window",
                    help="Rolling window for volatility calculation"
                )
            
            with col2:
                vol_annualize = st.checkbox(
                    "Annualize Volatility",
                    value=True,
                    help="Show annualized volatility (×√252)"
                )
            
            # Calculate rolling volatilities
            multiplier = np.sqrt(252) if vol_annualize else 1.0
            
            # Create volatility figure
            vol_fig = go.Figure()
            
            # Individual stock volatilities
            for col in stock_columns:
                rolling_vol = returns[col].rolling(window=vol_window).std() * multiplier
                vol_fig.add_trace(go.Scatter(
                    x=rolling_vol.index,
                    y=rolling_vol,
                    mode='lines',
                    name=f"{col}",
                    line=dict(width=1),
                    opacity=0.7
                ))
            
            # Portfolio volatility
            portfolio_vol = portfolio_returns.rolling(window=vol_window).std() * multiplier
            vol_fig.add_trace(go.Scatter(
                x=portfolio_vol.index,
                y=portfolio_vol,
                mode='lines',
                name=f"Portfolio ({weight_method})",
                line=dict(width=3, color='blue'),
                opacity=1.0
            ))
            
            # Market volatility
            if 'VNINDEX' in returns.columns:
                market_vol = returns['VNINDEX'].rolling(window=vol_window).std() * multiplier
                vol_fig.add_trace(go.Scatter(
                    x=market_vol.index,
                    y=market_vol,
                    mode='lines',
                    name="VN-Index",
                    line=dict(width=2, color='red', dash='dash')
                ))
            
            vol_title = f"{vol_window}-Day Rolling Volatility"
            if vol_annualize:
                vol_title += " (Annualized)"
            
            vol_fig.update_layout(
                title=vol_title,
                xaxis_title="Date",
                yaxis_title="Volatility",
                hovermode='x unified',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            st.plotly_chart(vol_fig, use_container_width=True)
            
            # Performance statistics
            st.subheader("📊 Performance Statistics")
            
            # Calculate time period metrics
            start_date = cumulative_returns.index[0]
            end_date = cumulative_returns.index[-1]
            time_period_years = (end_date - start_date).days / 365.25
            
            # Calculate key metrics
            total_return_portfolio = portfolio_cumulative.iloc[-1] - 1
            total_return_market = cumulative_returns['VNINDEX'].iloc[-1] - 1 if 'VNINDEX' in cumulative_returns.columns else 0
            
            # CAGR (Compound Annual Growth Rate)
            portfolio_cagr = (portfolio_cumulative.iloc[-1] ** (1/time_period_years)) - 1 if time_period_years > 0 else 0
            market_cagr = (cumulative_returns['VNINDEX'].iloc[-1] ** (1/time_period_years)) - 1 if 'VNINDEX' in cumulative_returns.columns and time_period_years > 0 else 0
            
            # Volatility (annualized)
            portfolio_vol = portfolio_returns.std() * np.sqrt(252)
            market_vol = returns['VNINDEX'].std() * np.sqrt(252) if 'VNINDEX' in returns.columns else 0
            
            # Sharpe ratio (assuming 0% risk-free rate)
            portfolio_sharpe = (portfolio_returns.mean() * 252) / portfolio_vol if portfolio_vol > 0 else 0
            market_sharpe = (returns['VNINDEX'].mean() * 252) / market_vol if market_vol > 0 else 0
            
            # Sortino ratio (using downside deviation)
            downside_returns_portfolio = portfolio_returns[portfolio_returns < 0]
            downside_vol_portfolio = downside_returns_portfolio.std() * np.sqrt(252) if len(downside_returns_portfolio) > 0 else portfolio_vol
            portfolio_sortino = (portfolio_returns.mean() * 252) / downside_vol_portfolio if downside_vol_portfolio > 0 else 0
            
            downside_returns_market = returns['VNINDEX'][returns['VNINDEX'] < 0] if 'VNINDEX' in returns.columns else pd.Series([])
            downside_vol_market = downside_returns_market.std() * np.sqrt(252) if len(downside_returns_market) > 0 else market_vol
            market_sortino = (returns['VNINDEX'].mean() * 252) / downside_vol_market if 'VNINDEX' in returns.columns and downside_vol_market > 0 else 0
            
            # Calmar ratio (CAGR / Max Drawdown)
            max_dd_portfolio = drawdown_data['portfolio_stats']['max_drawdown']
            portfolio_calmar = abs(portfolio_cagr / max_dd_portfolio) if max_dd_portfolio < 0 else 0
            
            market_dd = drawdown_data['individual_stats']['VNINDEX']['max_drawdown'] if 'VNINDEX' in drawdown_data['individual_stats'] else -0.1
            market_calmar = abs(market_cagr / market_dd) if market_dd < 0 else 0
            
            # Information ratio (tracking error adjusted excess return)
            excess_returns = portfolio_returns - (returns['VNINDEX'] if 'VNINDEX' in returns.columns else 0)
            tracking_error = excess_returns.std() * np.sqrt(252)
            information_ratio = (excess_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0
            
            # Win rate (percentage of positive periods)
            portfolio_win_rate = (portfolio_returns > 0).sum() / len(portfolio_returns)
            market_win_rate = (returns['VNINDEX'] > 0).sum() / len(returns['VNINDEX']) if 'VNINDEX' in returns.columns else 0
            
            # Display metrics in two rows
            st.write("**📈 Return Metrics**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Portfolio CAGR",
                    f"{portfolio_cagr:.2%}",
                    f"vs Market: {portfolio_cagr - market_cagr:+.2%}"
                )
            
            with col2:
                st.metric(
                    "Total Return",
                    f"{total_return_portfolio:.2%}",
                    f"vs Market: {total_return_portfolio - total_return_market:+.2%}"
                )
            
            with col3:
                st.metric(
                    "Win Rate",
                    f"{portfolio_win_rate:.1%}",
                    f"vs Market: {portfolio_win_rate - market_win_rate:+.1%}"
                )
            
            with col4:
                st.metric(
                    "Information Ratio",
                    f"{information_ratio:.2f}",
                    "Excess return per unit of tracking error"
                )
            
            st.write("**📊 Risk-Adjusted Metrics**")
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                st.metric(
                    "Sharpe Ratio",
                    f"{portfolio_sharpe:.2f}",
                    f"vs Market: {portfolio_sharpe - market_sharpe:+.2f}"
                )
            
            with col6:
                st.metric(
                    "Sortino Ratio",
                    f"{portfolio_sortino:.2f}",
                    f"vs Market: {portfolio_sortino - market_sortino:+.2f}"
                )
            
            with col7:
                st.metric(
                    "Calmar Ratio",
                    f"{portfolio_calmar:.2f}",
                    f"vs Market: {portfolio_calmar - market_calmar:+.2f}"
                )
            
            with col8:
                st.metric(
                    "Volatility",
                    f"{portfolio_vol:.2%}",
                    f"vs Market: {portfolio_vol - market_vol:+.2%}"
                )
            
            # Additional performance summary
            st.write("**📋 Performance Summary**")
            performance_summary = pd.DataFrame({
                'Metric': [
                    'Analysis Period',
                    'CAGR',
                    'Total Return',
                    'Volatility',
                    'Sharpe Ratio',
                    'Sortino Ratio',
                    'Calmar Ratio',
                    'Max Drawdown',
                    'Win Rate',
                    'Information Ratio',
                    'Tracking Error'
                ],
                'Portfolio': [
                    f"{time_period_years:.1f} years",
                    f"{portfolio_cagr:.2%}",
                    f"{total_return_portfolio:.2%}",
                    f"{portfolio_vol:.2%}",
                    f"{portfolio_sharpe:.2f}",
                    f"{portfolio_sortino:.2f}",
                    f"{portfolio_calmar:.2f}",
                    f"{max_dd_portfolio:.2%}",
                    f"{portfolio_win_rate:.1%}",
                    f"{information_ratio:.2f}",
                    f"{tracking_error:.2%}"
                ],
                'Market (VN-Index)': [
                    f"{time_period_years:.1f} years",
                    f"{market_cagr:.2%}",
                    f"{total_return_market:.2%}",
                    f"{market_vol:.2%}",
                    f"{market_sharpe:.2f}",
                    f"{market_sortino:.2f}",
                    f"{market_calmar:.2f}",
                    f"{market_dd:.2%}",
                    f"{market_win_rate:.1%}",
                    "N/A",
                    "N/A"
                ],
                'Difference': [
                    "Same",
                    f"{portfolio_cagr - market_cagr:+.2%}",
                    f"{total_return_portfolio - total_return_market:+.2%}",
                    f"{portfolio_vol - market_vol:+.2%}",
                    f"{portfolio_sharpe - market_sharpe:+.2f}",
                    f"{portfolio_sortino - market_sortino:+.2f}",
                    f"{portfolio_calmar - market_calmar:+.2f}",
                    f"{max_dd_portfolio - market_dd:+.2%}",
                    f"{portfolio_win_rate - market_win_rate:+.1%}",
                    f"{information_ratio:.2f}",
                    f"{tracking_error:.2%}"
                ]
            })
            
            st.dataframe(performance_summary, use_container_width=True)
            
            # Individual stock performance table
            st.subheader("📋 Individual Stock Performance")
            
            perf_data = []
            for col in cumulative_returns.columns:
                if col != 'VNINDEX':
                    total_ret = cumulative_returns[col].iloc[-1] - 1
                    cagr = (cumulative_returns[col].iloc[-1] ** (1/time_period_years)) - 1 if time_period_years > 0 else 0
                    vol = returns[col].std() * np.sqrt(252)
                    vol_recent = returns[col].rolling(window=vol_period).std().iloc[-1] * np.sqrt(252)
                    sharpe = (returns[col].mean() * 252) / vol if vol > 0 else 0
                    
                    # Calculate win rate for individual stocks
                    win_rate = (returns[col] > 0).sum() / len(returns[col])
                    
                    perf_data.append({
                        'Stock': col,
                        'CAGR': f"{cagr:.2%}",
                        'Total Return': f"{total_ret:.2%}",
                        'Volatility': f"{vol:.2%}",
                        f'{vol_period}D Recent Vol': f"{vol_recent:.2%}",
                        'Sharpe Ratio': f"{sharpe:.2f}",
                        'Win Rate': f"{win_rate:.1%}",
                        'Final Value': f"{cumulative_returns[col].iloc[-1]:.3f}"
                    })
            
            perf_df = pd.DataFrame(perf_data)
            st.dataframe(perf_df, use_container_width=True)
            
            # Volatility insights
            st.subheader("🔍 Volatility Insights")
            
            # Current vs average volatility comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📊 Current Volatility Ranking** (Lower = Less Risky)")
                vol_ranking = []
                for stock in stock_columns:
                    recent_vol = returns[stock].rolling(window=vol_period).std().iloc[-1] * np.sqrt(252)
                    vol_ranking.append((stock, recent_vol))
                
                vol_ranking.sort(key=lambda x: x[1])
                
                for i, (stock, vol) in enumerate(vol_ranking, 1):
                    rank_color = "🟢" if i <= len(vol_ranking)//3 else "🟡" if i <= 2*len(vol_ranking)//3 else "🔴"
                    st.write(f"{rank_color} **{i}.** {stock}: {vol:.2%}")
            
            with col2:
                st.write("**⚖️ Inverse Volatility Weights** (Theoretical)")
                theoretical_weights = []
                vol_values = [vol for _, vol in vol_ranking]
                inverse_vols = [1/vol for _, vol in vol_ranking]
                total_inverse = sum(inverse_vols)
                
                for (stock, vol), inv_vol in zip(vol_ranking, inverse_vols):
                    weight = inv_vol / total_inverse
                    theoretical_weights.append((stock, weight))
                
                theoretical_weights.sort(key=lambda x: x[1], reverse=True)
                
                for stock, weight in theoretical_weights:
                    weight_color = "🔵" if weight >= 0.2 else "🟣" if weight >= 0.15 else "⚪"
                    st.write(f"{weight_color} **{stock}**: {weight:.1%}")
            
        else:
            st.info("📊 Cumulative returns analysis requires price data. Please run a full analysis first.")
    
    def render_monitoring_tab(self):
        """Render monitoring tab content with enhanced layout"""
        results = st.session_state.analysis_results
        
        # Enhanced header with description
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
        ">
            <h2 style="margin-bottom: 0.5rem;">🔍 Real-time Portfolio Monitoring</h2>
            <p style="margin-bottom: 0; opacity: 0.9; font-size: 1.1rem;">
                Advanced monitoring dashboard with real-time alerts, risk metrics, and performance tracking
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if FULL_SYSTEM and 'correlation_monitor' in results:
            # Enhanced tabbed monitoring sections
            monitor_tab1, monitor_tab2, monitor_tab3, monitor_tab4 = st.tabs([
                "📊 Performance Monitor", 
                "⚠️ Risk Alerts", 
                "📈 Correlation Tracking",
                "🎯 Rebalancing Signals"
            ])
            
            with monitor_tab1:
                st.markdown("### 📊 Portfolio Performance Monitoring")
                self.render_performance_monitoring(results)
                
            with monitor_tab2:
                st.markdown("### ⚠️ Risk Alert System")
                self.render_risk_alerts(results)
                
            with monitor_tab3:
                st.markdown("### 📈 Correlation & Volatility Tracking")
                self.render_correlation_monitoring(results)
                
            with monitor_tab4:
                st.markdown("### 🎯 Rebalancing Recommendation Engine")
                self.render_rebalancing_signals(results)
        else:
            st.info("📊 Advanced monitoring requires full system analysis with real-time data feeds.")
    
    def render_recommendations_tab(self):
        """Render recommendations tab content"""
        results = st.session_state.analysis_results
        
        st.subheader("💡 Hedge Recommendations")
        
        if FULL_SYSTEM and 'recommendations' in results:
            self.render_recommendations(results)
        else:
            st.info("📊 Recommendations require full system analysis.")
    
    def render_simple_analysis(self):
        """Render simple analysis results"""
        results = st.session_state.analysis_results['simple_results']
        
        st.subheader("📋 Hedging Analysis Results")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Original Breadth",
                f"{results['original']['breadth']:.1f}",
                f"Corr: {results['original']['correlation']:.3f}"
            )
        
        with col2:
            st.metric(
                "Hedged Breadth",
                f"{results['hedged']['breadth']:.1f}",
                f"Corr: {results['hedged']['correlation']:.3f}"
            )
        
        with col3:
            improvement = results['improvement']['breadth_improvement_pct']
            st.metric(
                "Improvement",
                f"{improvement:+.1f}%",
                f"Δ: {results['improvement']['breadth_improvement']:.1f}"
            )
        
        # Beta analysis chart
        st.subheader("📊 Beta Analysis")
        
        beta_data = results['beta_info']
        stocks = list(beta_data.keys())
        betas = [info['beta'] for info in beta_data.values()]
        r_squareds = [info['r_squared'] for info in beta_data.values()]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=betas,
            y=r_squareds,
            mode='markers+text',
            text=stocks,
            textposition='top center',
            marker=dict(size=12, color='blue'),
            name='Stocks'
        ))
        
        fig.update_layout(
            title="Beta vs R-squared Analysis",
            xaxis_title="Beta (Market Sensitivity)",
            yaxis_title="R-squared (Market Explanation)",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_correlation_charts(self, results):
        """Render correlation analysis charts"""
        st.subheader("🔗 Correlation Analysis")
        
        col1, col2 = st.columns(2)
        
        # Original correlation matrix
        with col1:
            if 'no_hedge' in results['hedge_results']:
                no_hedge = results['hedge_results']['no_hedge']
                corr_matrix = no_hedge['correlation_matrix']
                
                fig = px.imshow(
                    corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    color_continuous_scale='RdYlBu_r',
                    title="Original Correlation Matrix"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Hedged correlation matrix
        with col2:
            if 'market_hedge' in results['hedge_results']:
                market_hedge = results['hedge_results']['market_hedge']
                hedged_corr = market_hedge['hedged_correlation']
                
                fig = px.imshow(
                    hedged_corr.values,
                    x=hedged_corr.columns,
                    y=hedged_corr.index,
                    color_continuous_scale='RdYlBu_r',
                    title="Market Hedged Correlation Matrix"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def render_hedge_effectiveness_charts(self, results):
        """Render hedge effectiveness charts"""
        st.subheader("📈 Hedge Effectiveness")
        
        if 'summary' in results['hedge_results']:
            summary = results['hedge_results']['summary']
            
            # Effectiveness comparison
            methods = [row['method'] for row in summary['comparison_table']]
            breadths = [row['effective_breadth'] for row in summary['comparison_table']]
            correlations = [row['avg_correlation'] for row in summary['comparison_table']]
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Effective Breadth', 'Average Correlation'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Breadth chart
            fig.add_trace(
                go.Bar(x=methods, y=breadths, name='Effective Breadth',
                      marker_color=['red', 'orange', 'green'][:len(methods)]),
                row=1, col=1
            )
            
            # Correlation chart
            fig.add_trace(
                go.Bar(x=methods, y=correlations, name='Avg Correlation',
                      marker_color=['red', 'orange', 'green'][:len(methods)]),
                row=1, col=2
            )
            
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_monitoring_charts(self, results):
        """Render monitoring charts"""
        if 'correlation_monitor' in results and results['correlation_monitor']:
            st.subheader("📊 Real-time Monitoring")
            
            monitor_data = results['correlation_monitor']['monitoring_data']
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Correlation Over Time', 'Effective Breadth Over Time'),
                vertical_spacing=0.1
            )
            
            # Correlation over time
            fig.add_trace(
                go.Scatter(
                    x=monitor_data.index,
                    y=monitor_data['avg_correlation'],
                    mode='lines',
                    name='Correlation',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            # Effective breadth over time
            fig.add_trace(
                go.Scatter(
                    x=monitor_data.index,
                    y=monitor_data['effective_breadth'],
                    mode='lines',
                    name='Effective Breadth',
                    line=dict(color='green')
                ),
                row=2, col=1
            )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_recommendations(self, results):
        """Render recommendations"""
        if 'recommendations' in results and results['recommendations']:
            st.subheader("💡 Hedge Recommendations")
            
            recommendations = results['recommendations']['recommendations']
            
            for rec in recommendations:
                priority_color = {
                    'HIGH': '🔴',
                    'MEDIUM': '🟡',
                    'LOW': '🟢'
                }.get(rec['priority'], '⚪')
                
                st.info(f"{priority_color} **{rec['priority']} PRIORITY**: {rec['action']}")
                st.write(f"Expected improvement: {rec['expected_improvement']}")
            
            # Implementation plan
            if 'implementation_plan' in results['recommendations']:
                st.subheader("🛠️ Implementation Plan")
                
                for i, step in enumerate(results['recommendations']['implementation_plan'], 1):
                    st.write(f"{i}. {step}")
    
    def export_results(self):
        """Export analysis results"""
        if st.session_state.analysis_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"vietnam_hedge_analysis_{timestamp}.json"
            
            # Convert results to JSON-serializable format
            export_data = {
                'timestamp': timestamp,
                'stocks': st.session_state.selected_stocks,
                'analysis_type': 'full' if FULL_SYSTEM else 'simple',
                'results': st.session_state.analysis_results
            }
            
            st.download_button(
                label="📥 Download Analysis Results",
                data=str(export_data),
                file_name=filename,
                mime="application/json"
            )
    
    def run(self):
        """Run the dashboard"""
        self.render_header()
        
        # Sidebar controls
        settings = self.render_sidebar()
        
        # Main content
        self.render_main_dashboard()
        
        # Footer
        st.markdown("---")
        st.markdown("*Vietnam Stock Hedging Dashboard - Built with ❤️ for the Vietnam financial community*")
        st.markdown("*Based on AI Cafe educational materials*")
    
    def render_backtesting_tab(self):
        """Render comprehensive backtesting analysis tab"""
        st.header("🎯 Strategy Backtesting & Performance Analysis")
        
        if not FULL_SYSTEM:
            st.warning("⚠️ Full backtesting requires the complete system. Please install required dependencies.")
            return
        
        if not st.session_state.analysis_results:
            st.warning("⚠️ Please run hedge analysis first to enable backtesting.")
            return
        
        # Backtesting controls
        st.subheader("⚙️ Backtesting Configuration")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            initial_capital = st.number_input(
                "💰 Initial Capital (VND)",
                min_value=100000,
                max_value=10000000000,
                value=1000000,
                step=100000,
                format="%d"
            )
        
        with col2:
            rebalance_freq = st.selectbox(
                "🔄 Rebalance Frequency",
                ["daily", "weekly", "monthly", "quarterly"],
                index=2
            )
        
        with col3:
            transaction_cost = st.slider(
                "💸 Transaction Cost (%)",
                min_value=0.0,
                max_value=1.0,
                value=0.15,
                step=0.01,
                format="%.2f%%"
            ) / 100
        
        with col4:
            slippage = st.slider(
                "📊 Market Slippage (%)",
                min_value=0.0,
                max_value=0.5,
                value=0.1,
                step=0.01,
                format="%.2f%%"
            ) / 100
        
        # Strategy selection
        st.subheader("📋 Strategy Selection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            strategies = st.multiselect(
                "Select strategies to backtest:",
                ["no_hedge", "market_hedge", "sector_hedge", "dynamic_hedge"],
                default=["no_hedge", "market_hedge", "sector_hedge"],
                help="Choose which hedge strategies to include in the backtest"
            )
        
        with col2:
            include_monte_carlo = st.checkbox(
                "🎲 Include Monte Carlo Analysis",
                value=False,
                help="Run probabilistic analysis for strategy robustness (takes longer)"
            )
        
        # Run backtest button
        if st.button("🚀 Run Comprehensive Backtest", type="primary"):
            if not strategies:
                st.error("Please select at least one strategy to backtest.")
                return
            
            # Get returns data from current analysis
            try:
                # Extract returns data from session state
                if 'hedge_results' in st.session_state.analysis_results:
                    prices = st.session_state.analysis_results['prices']
                    returns = self.pipeline.calculate_returns(prices)
                else:
                    st.error("No hedge analysis data found. Please run analysis first.")
                    return
                
                # Initialize backtesting engine with custom parameters
                backtest_engine = VietnamBacktestingEngine(
                    transaction_cost=transaction_cost,
                    slippage=slippage
                )
                
                # Run backtest
                with st.spinner("🔄 Running comprehensive backtesting analysis..."):
                    backtest_results = backtest_engine.run_comprehensive_backtest(
                        returns_data=returns,
                        initial_capital=initial_capital,
                        rebalance_frequency=rebalance_freq,
                        hedge_strategies=strategies
                    )
                
                # Store results in session state
                st.session_state.backtest_results = backtest_results
                st.session_state.backtest_engine = backtest_engine
                
                st.success("✅ Backtesting completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Backtesting failed: {str(e)}")
                return
        
        # Display results if available
        if 'backtest_results' in st.session_state and st.session_state.backtest_results:
            self.display_backtest_results(
                st.session_state.backtest_results,
                st.session_state.backtest_engine,
                include_monte_carlo
            )
    
    def display_backtest_results(self, backtest_results, backtest_engine, include_monte_carlo=False):
        """Display comprehensive backtesting results"""
        st.markdown("---")
        st.subheader("📊 Backtesting Results")
        
        # 1. Strategy Performance Comparison
        st.subheader("🏆 Strategy Performance Comparison")
        
        comparison_df = backtest_engine.generate_strategy_comparison()
        if comparison_df is not None:
            st.dataframe(comparison_df, use_container_width=True)
        
        # 2. Portfolio Value Evolution
        st.subheader("📈 Portfolio Value Evolution")
        
        fig = go.Figure()
        
        colors = {
            'no_hedge': '#FF6B6B',
            'market_hedge': '#4ECDC4', 
            'sector_hedge': '#45B7D1',
            'dynamic_hedge': '#96CEB4',
            'benchmark': '#FFEAA7'
        }
        
        for strategy, data in backtest_results.items():
            portfolio_values = data['portfolio_values']
            
            fig.add_trace(go.Scatter(
                x=portfolio_values.index,
                y=portfolio_values.values,
                mode='lines',
                name=strategy.replace('_', ' ').title(),
                line=dict(color=colors.get(strategy, '#95A5A6'), width=2),
                hovertemplate=f'<b>{strategy.title()}</b><br>' +
                             'Date: %{x}<br>' +
                             'Value: %{y:,.0f} VND<extra></extra>'
            ))
        
        # Dynamic height based on number of strategies
        chart_height = max(500, len(backtest_results) * 50 + 400)
        
        fig.update_layout(
            title="Portfolio Value Evolution Over Time",
            xaxis_title="Date",
            yaxis_title="Portfolio Value (VND)",
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            height=chart_height,
            # Enhanced interactivity with range selectors
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(step="all", label="All")
                    ]),
                    bgcolor="rgba(150, 150, 150, 0.1)",
                    bordercolor="rgba(150, 150, 150, 0.2)",
                    activecolor="rgba(46, 134, 171, 0.8)",
                    x=0.02,
                    y=1.05
                ),
                rangeslider=dict(
                    visible=True,
                    bgcolor="rgba(150, 150, 150, 0.1)",
                    bordercolor="rgba(150, 150, 150, 0.2)",
                    thickness=0.05
                ),
                type="date"
            ),
            # Professional styling
            plot_bgcolor='rgba(248, 249, 250, 0.8)',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 3. Drawdown Analysis
        st.subheader("📉 Maximum Drawdown Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Drawdown comparison chart
            fig_dd = go.Figure()
            
            for strategy, data in backtest_results.items():
                drawdown_data = data['drawdowns']['drawdown_series']
                
                fig_dd.add_trace(go.Scatter(
                    x=drawdown_data.index,
                    y=drawdown_data.values * 100,  # Convert to percentage
                    mode='lines',
                    name=strategy.replace('_', ' ').title(),
                    line=dict(color=colors.get(strategy, '#95A5A6')),
                    fill='tonexty' if strategy == list(backtest_results.keys())[0] else 'tonexty',
                    hovertemplate=f'<b>{strategy.title()}</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Drawdown: %{y:.1f}%<extra></extra>'
                ))
            
            fig_dd.update_layout(
                title="Drawdown Evolution",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig_dd, use_container_width=True)
        
        with col2:
            # Drawdown statistics table
            st.write("**Drawdown Statistics**")
            
            dd_stats = []
            for strategy, data in backtest_results.items():
                dd_data = data['drawdowns']
                dd_stats.append({
                    'Strategy': strategy.replace('_', ' ').title(),
                    'Max Drawdown': f"{dd_data['max_drawdown']:.1%}",
                    'Current Drawdown': f"{dd_data['current_drawdown']:.1%}",
                    'Avg Drawdown': f"{dd_data['avg_drawdown']:.1%}",
                    'Avg Duration (days)': f"{dd_data['drawdown_duration']:.0f}"
                })
            
            dd_df = pd.DataFrame(dd_stats)
            st.dataframe(dd_df, use_container_width=True)
        
        # 4. Risk-Return Scatter Plot
        st.subheader("⚖️ Risk-Return Analysis")
        
        fig_scatter = go.Figure()
        
        for strategy, data in backtest_results.items():
            metrics = data['metrics']
            
            fig_scatter.add_trace(go.Scatter(
                x=[metrics.get('annual_volatility', 0) * 100],
                y=[metrics.get('cagr', 0) * 100],
                mode='markers+text',
                name=strategy.replace('_', ' ').title(),
                marker=dict(
                    size=15,
                    color=colors.get(strategy, '#95A5A6'),
                    symbol='circle'
                ),
                text=[strategy.replace('_', ' ').title()],
                textposition="top center",
                hovertemplate=f'<b>{strategy.title()}</b><br>' +
                             'Volatility: %{x:.1f}%<br>' +
                             'CAGR: %{y:.1f}%<br>' +
                             'Sharpe: ' + f"{metrics.get('sharpe_ratio', 0):.2f}<extra></extra>"
            ))
        
        fig_scatter.update_layout(
            title="Risk-Return Profile",
            xaxis_title="Annual Volatility (%)",
            yaxis_title="CAGR (%)",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # 5. Rolling Performance Metrics
        st.subheader("📊 Rolling Performance Metrics")
        
        # Check if rolling metrics are available
        has_rolling_data = any(
            'rolling_metrics' in data and data['rolling_metrics'] 
            for data in backtest_results.values()
        )
        
        if has_rolling_data:
            tab1, tab2 = st.tabs(["📈 Rolling Sharpe Ratio", "📊 Rolling Volatility"])
            
            with tab1:
                fig_rolling_sharpe = go.Figure()
                
                for strategy, data in backtest_results.items():
                    rolling_metrics = data.get('rolling_metrics', {})
                    if 'rolling_sharpe' in rolling_metrics:
                        rolling_sharpe = rolling_metrics['rolling_sharpe']
                        
                        fig_rolling_sharpe.add_trace(go.Scatter(
                            x=rolling_sharpe.index,
                            y=rolling_sharpe.values,
                            mode='lines',
                            name=strategy.replace('_', ' ').title(),
                            line=dict(color=colors.get(strategy, '#95A5A6'))
                        ))
                
                fig_rolling_sharpe.update_layout(
                    title="Rolling 1-Year Sharpe Ratio",
                    xaxis_title="Date",
                    yaxis_title="Sharpe Ratio",
                    height=400
                )
                
                st.plotly_chart(fig_rolling_sharpe, use_container_width=True)
            
            with tab2:
                fig_rolling_vol = go.Figure()
                
                for strategy, data in backtest_results.items():
                    rolling_metrics = data.get('rolling_metrics', {})
                    if 'rolling_volatility' in rolling_metrics:
                        rolling_vol = rolling_metrics['rolling_volatility']
                        
                        fig_rolling_vol.add_trace(go.Scatter(
                            x=rolling_vol.index,
                            y=rolling_vol.values * 100,  # Convert to percentage
                            mode='lines',
                            name=strategy.replace('_', ' ').title(),
                            line=dict(color=colors.get(strategy, '#95A5A6'))
                        ))
                
                fig_rolling_vol.update_layout(
                    title="Rolling 1-Year Volatility",
                    xaxis_title="Date",
                    yaxis_title="Volatility (%)",
                    height=400
                )
                
                st.plotly_chart(fig_rolling_vol, use_container_width=True)
        else:
            st.info("ℹ️ Rolling metrics require at least 1 year of data for calculation.")
        
        # 6. Monte Carlo Analysis (if requested)
        if include_monte_carlo:
            st.subheader("🎲 Monte Carlo Analysis")
            
            with st.spinner("Running Monte Carlo simulation..."):
                mc_results = backtest_engine.get_monte_carlo_analysis(n_simulations=1000)
            
            if mc_results:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Monte Carlo results table
                    mc_data = []
                    for strategy, results in mc_results.items():
                        mc_data.append({
                            'Strategy': strategy.replace('_', ' ').title(),
                            'Mean Return': f"{results['mean_return']:.1%}",
                            'Std Deviation': f"{results['std_return']:.1%}",
                            '5th Percentile': f"{results['percentile_5']:.1%}",
                            '95th Percentile': f"{results['percentile_95']:.1%}",
                            'Success Rate': f"{results['success_rate']:.1%}"
                        })
                    
                    mc_df = pd.DataFrame(mc_data)
                    st.dataframe(mc_df, use_container_width=True)
                
                with col2:
                    # Success rate chart
                    strategies = list(mc_results.keys())
                    success_rates = [mc_results[s]['success_rate'] * 100 for s in strategies]
                    
                    fig_success = go.Figure(go.Bar(
                        x=[s.replace('_', ' ').title() for s in strategies],
                        y=success_rates,
                        marker_color=[colors.get(s, '#95A5A6') for s in strategies],
                        text=[f"{rate:.1f}%" for rate in success_rates],
                        textposition='auto'
                    ))
                    
                    fig_success.update_layout(
                        title="Strategy Success Rate (Monte Carlo)",
                        xaxis_title="Strategy",
                        yaxis_title="Success Rate (%)",
                        height=400
                    )
                    
                    st.plotly_chart(fig_success, use_container_width=True)
        
        # 7. Export Results
        st.subheader("💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export Performance Summary"):
                csv_data = comparison_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"backtest_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📈 Export Portfolio Values"):
                # Combine all portfolio values
                combined_values = pd.DataFrame({
                    strategy: data['portfolio_values'] 
                    for strategy, data in backtest_results.items()
                })
                csv_data = combined_values.to_csv()
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"portfolio_values_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("📉 Export Drawdown Data"):
                # Combine all drawdown data
                combined_dd = pd.DataFrame({
                    strategy: data['drawdowns']['drawdown_series'] 
                    for strategy, data in backtest_results.items()
                })
                csv_data = combined_dd.to_csv()
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"drawdown_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    def render_performance_monitoring(self, results):
        """Render performance monitoring sub-section"""
        # This is a placeholder that calls the existing monitoring functionality
        if 'correlation_monitor' in results:
            self.render_monitoring_charts(results)
        else:
            st.info("📊 Performance monitoring data not available.")
    
    def render_risk_alerts(self, results):
        """Render risk alerts sub-section"""
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #dc3545 0%, #fd7e14 100%);
            padding: 1rem;
            border-radius: 8px;
            color: white;
            margin-bottom: 1rem;
        ">
            <h4>⚠️ Active Risk Alerts</h4>
            <p style="margin-bottom: 0; opacity: 0.9;">
                Real-time monitoring of portfolio risk levels and threshold breaches
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Mock alert data for demonstration
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("✅ Correlation levels within target range")
            st.warning("⚠️ VHM volatility above 25% threshold")
            
        with col2:
            st.error("🚨 Portfolio concentration > 30% in Banking sector")
            st.info("ℹ️ Suggested rebalancing in 2 days")
    
    def render_correlation_monitoring(self, results):
        """Render correlation monitoring sub-section"""
        st.markdown("Real-time correlation and volatility tracking charts would be displayed here.")
        # Placeholder for additional correlation-specific charts
        if 'correlation_monitor' in results:
            # Could add specific correlation tracking visualizations
            st.info("📈 Advanced correlation tracking features coming soon.")
    
    def render_rebalancing_signals(self, results):
        """Render rebalancing signals sub-section"""
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            padding: 1rem;
            border-radius: 8px;
            color: white;
            margin-bottom: 1rem;
        ">
            <h4>🎯 Smart Rebalancing Engine</h4>
            <p style="margin-bottom: 0; opacity: 0.9;">
                AI-powered rebalancing recommendations based on market conditions
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Mock rebalancing recommendations
        rebalance_data = {
            'Stock': ['BID', 'VHM', 'VIC', 'CTG'],
            'Current Weight': ['25%', '20%', '18%', '15%'],
            'Target Weight': ['22%', '23%', '20%', '12%'],
            'Action': ['REDUCE', 'INCREASE', 'INCREASE', 'REDUCE'],
            'Priority': ['Medium', 'High', 'Medium', 'Low']
        }
        
        rebalance_df = pd.DataFrame(rebalance_data)
        st.dataframe(
            rebalance_df.style.format({}).apply(
                lambda x: ['background-color: #f8d7da' if v == 'REDUCE' 
                          else 'background-color: #d4edda' if v == 'INCREASE' 
                          else '' for v in x], 
                subset=['Action']
            ),
            use_container_width=True
        )
    
    def get_enhanced_chart_config(self, chart_type="time_series", height=500, enable_range_selector=True):
        """Get enhanced chart configuration for consistent styling and interactivity"""
        base_config = {
            'height': height,
            'hovermode': 'x unified',
            'plot_bgcolor': 'rgba(248, 249, 250, 0.8)',
            'paper_bgcolor': 'white',
            'font': {
                'family': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
                'size': 12,
                'color': '#2c3e50'
            },
            'hoverlabel': {
                'bgcolor': 'rgba(255,255,255,0.95)',
                'bordercolor': '#34495e',
                'font': {'size': 11}
            },
            'legend': {
                'yanchor': "top",
                'y': 0.99,
                'xanchor': "left",
                'x': 0.01,
                'bgcolor': 'rgba(255,255,255,0.8)',
                'bordercolor': 'rgba(0,0,0,0.1)',
                'borderwidth': 1
            }
        }
        
        if chart_type == "time_series" and enable_range_selector:
            base_config['xaxis'] = {
                'rangeselector': {
                    'buttons': [
                        {'count': 1, 'label': "1M", 'step': "month", 'stepmode': "backward"},
                        {'count': 3, 'label': "3M", 'step': "month", 'stepmode': "backward"},
                        {'count': 6, 'label': "6M", 'step': "month", 'stepmode': "backward"},
                        {'count': 1, 'label': "1Y", 'step': "year", 'stepmode': "backward"},
                        {'count': 2, 'label': "2Y", 'step': "year", 'stepmode': "backward"},
                        {'step': "all", 'label': "All"}
                    ],
                    'bgcolor': "rgba(150, 150, 150, 0.1)",
                    'bordercolor': "rgba(150, 150, 150, 0.2)",
                    'activecolor': "rgba(46, 134, 171, 0.8)",
                    'x': 0.02,
                    'y': 1.05
                },
                'rangeslider': {
                    'visible': True,
                    'bgcolor': "rgba(150, 150, 150, 0.1)",
                    'bordercolor': "rgba(150, 150, 150, 0.2)",
                    'thickness': 0.05
                },
                'type': "date"
            }
        
        return base_config
    
    def add_chart_interactions(self, fig, data_key=None):
        """Add enhanced interactions to charts for cross-filtering"""
        # Store the current chart state in session state for cross-chart communication
        if 'chart_states' not in st.session_state:
            st.session_state.chart_states = {}
        
        # Add click events and selection capabilities
        fig.update_layout(
            clickmode='event+select',
            selectdirection='horizontal'
        )
        
        # Enable selection and zoom synchronization across charts
        if data_key:
            st.session_state.chart_states[data_key] = {
                'selected_range': None,
                'zoom_level': None
            }
        
        return fig


def main():
    """Main function to run the dashboard"""
    dashboard = VietnamHedgeDashboard()
    dashboard.run()


if __name__ == "__main__":
    main() 