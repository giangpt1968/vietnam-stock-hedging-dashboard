import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from data import get_prices, set_source
import warnings
warnings.filterwarnings('ignore')

class VietnamStockDataPipeline:
    """Enhanced data pipeline for Vietnam stock market hedging analysis"""
    
    def __init__(self, source='CafeF'):
        """Initialize the pipeline with data source"""
        self.source = source
        set_source(source)
        
        # Vietnam market sectors and benchmarks
        self.sectors = {
            'Banking': {
                'stocks': ['BID', 'CTG', 'ACB', 'VCB', 'TCB', 'VPB', 'MBB', 'STB'],
                'benchmark': 'VCB'
            },
            'RealEstate': {
                'stocks': ['VHM', 'VIC', 'VRE', 'HDG', 'KDH', 'DXG', 'BCM'],
                'benchmark': 'VHM'
            },
            'Steel': {
                'stocks': ['HPG', 'HSG', 'NKG', 'TVN', 'POM'],
                'benchmark': 'HPG'
            },
            'Oil_Gas': {
                'stocks': ['GAS', 'PLX', 'PVD', 'PVS', 'BSR'],
                'benchmark': 'GAS'
            },
            'Retail': {
                'stocks': ['MWG', 'FRT', 'PNJ', 'DGW'],
                'benchmark': 'MWG'
            }
        }
        
        self.market_index = 'VNINDEX'
        self.cache = {}
        
    def get_market_data(self, symbols, start_date='2023-01-01', end_date=None, 
                       include_market=True, field='close'):
        """Get market data with automatic market index inclusion"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # Add market index if requested
        if include_market and self.market_index not in symbols:
            symbols = list(symbols) + [self.market_index]
        
        # Create cache key
        cache_key = f"{'-'.join(sorted(symbols))}_{start_date}_{end_date}_{field}"
        
        if cache_key in self.cache:
            print(f"📊 Using cached data for {len(symbols)} symbols")
            return self.cache[cache_key]
        
        try:
            print(f"📡 Fetching data for {len(symbols)} symbols from {start_date} to {end_date}")
            prices = get_prices(*symbols, start_date=start_date, end_date=end_date, field=field)
            
            if prices is not None:
                # Forward fill missing values
                prices = prices.fillna(method='ffill')
                # Drop rows with all NaN
                prices = prices.dropna(how='all')
                
                print(f"✅ Successfully loaded {len(prices)} days of data")
                print(f"📈 Symbols: {list(prices.columns)}")
                
                # Cache the result
                self.cache[cache_key] = prices
                return prices
            else:
                print("❌ No data returned")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching data: {str(e)}")
            return None
    
    def get_sector_data(self, sector_name, start_date='2023-01-01', end_date=None):
        """Get data for a specific sector including benchmark"""
        if sector_name not in self.sectors:
            print(f"❌ Sector '{sector_name}' not found. Available: {list(self.sectors.keys())}")
            return None
        
        sector_info = self.sectors[sector_name]
        symbols = sector_info['stocks'] + [sector_info['benchmark']]
        
        return self.get_market_data(symbols, start_date, end_date, include_market=True)
    
    def get_all_sectors_data(self, start_date='2023-01-01', end_date=None):
        """Get data for all sectors and market index"""
        all_symbols = [self.market_index]
        
        for sector_info in self.sectors.values():
            all_symbols.extend(sector_info['stocks'])
            if sector_info['benchmark'] not in all_symbols:
                all_symbols.append(sector_info['benchmark'])
        
        # Remove duplicates while preserving order
        unique_symbols = []
        for symbol in all_symbols:
            if symbol not in unique_symbols:
                unique_symbols.append(symbol)
        
        return self.get_market_data(unique_symbols, start_date, end_date, include_market=False)
    
    def calculate_returns(self, prices):
        """Calculate returns from prices"""
        if prices is None:
            return None
        
        returns = prices.pct_change().fillna(0)
        print(f"📊 Calculated returns for {len(returns.columns)} assets over {len(returns)} days")
        return returns
    
    def get_sector_classification(self, symbol):
        """Get sector classification for a symbol"""
        for sector_name, sector_info in self.sectors.items():
            if symbol in sector_info['stocks']:
                return sector_name, sector_info['benchmark']
        return None, None
    
    def validate_data_quality(self, data):
        """Validate data quality and report issues"""
        if data is None:
            return False
        
        print(f"\n🔍 Data Quality Report:")
        print(f"📅 Date range: {data.index.min()} to {data.index.max()}")
        print(f"📊 Shape: {data.shape}")
        
        # Check for missing values
        missing_pct = (data.isnull().sum() / len(data)) * 100
        if missing_pct.any():
            print(f"⚠️  Missing data percentage:")
            for col, pct in missing_pct[missing_pct > 0].items():
                print(f"   {col}: {pct:.1f}%")
        else:
            print(f"✅ No missing values")
        
        # Check for zero variance (non-trading days)
        zero_var = data.var() == 0
        if zero_var.any():
            print(f"⚠️  Zero variance assets: {list(zero_var[zero_var].index)}")
        
        # Check for extreme values
        extreme_returns = (data.pct_change().abs() > 0.2).any()
        if extreme_returns.any():
            print(f"⚠️  Assets with >20% daily moves: {list(extreme_returns[extreme_returns].index)}")
        
        return True
    
    def get_demo_portfolio(self):
        """Get a demo portfolio for testing"""
        demo_symbols = ['BID', 'CTG', 'ACB', 'VHM', 'VIC', 'VRE', 'HPG', 'MWG']
        return self.get_market_data(demo_symbols, start_date='2024-01-01')


# Test the pipeline
if __name__ == "__main__":
    print("🚀 Testing Vietnam Stock Data Pipeline")
    print("=" * 50)
    
    pipeline = VietnamStockDataPipeline()
    
    # Test 1: Get demo portfolio
    print("\n1️⃣ Testing Demo Portfolio:")
    demo_prices = pipeline.get_demo_portfolio()
    if demo_prices is not None:
        pipeline.validate_data_quality(demo_prices)
        demo_returns = pipeline.calculate_returns(demo_prices)
        print(f"✅ Demo portfolio ready with {len(demo_returns.columns)} assets")
    
    # Test 2: Get sector data
    print("\n2️⃣ Testing Sector Data:")
    banking_data = pipeline.get_sector_data('Banking', start_date='2024-01-01')
    if banking_data is not None:
        print(f"✅ Banking sector data loaded: {list(banking_data.columns)}")
    
    # Test 3: Sector classification
    print("\n3️⃣ Testing Sector Classification:")
    for symbol in ['BID', 'VHM', 'HPG', 'MWG']:
        sector, benchmark = pipeline.get_sector_classification(symbol)
        print(f"   {symbol}: {sector} (benchmark: {benchmark})")
    
    print("\n✅ Pipeline testing completed!") 