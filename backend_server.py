from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import json
import os
from anthropic import Anthropic

# Initialize FastAPI app
app = FastAPI(title="AI Mastery Investment Backend")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# DATA MODELS
# ============================================================================

class StockData(BaseModel):
    symbol: str
    name: str
    price: float
    change: float
    changePercent: float
    volume: int
    pe_ratio: Optional[float]
    dividend_yield: Optional[float]
    shariaCompliant: bool
    sector: str

class PortfolioHolding(BaseModel):
    symbol: str
    name: str
    shares: int
    avgPrice: float
    current: float
    sector: str
    shariaCompliant: bool

class Portfolio(BaseModel):
    holdings: List[PortfolioHolding]
    monthlyBudget: float
    riskTolerance: str
    currency: str

class OptimizationResult(BaseModel):
    weights: Dict[str, float]
    expectedReturn: float
    expectedRisk: float
    sharpeRatio: float
    recommendation: str

class TradingSignal(BaseModel):
    symbol: str
    signal: str  # BUY, SELL, HOLD
    confidence: float
    reasons: List[str]
    targetPrice: Optional[float]
    stopLoss: Optional[float]
    technicalScore: float
    mlScore: float

# ============================================================================
# REAL EGX DATA FETCHING
# ============================================================================

# EGX stock symbols and details
EGX_STOCKS = {
    'CIHC.CA': {
        'name': 'Commercial International Bank',
        'sector': 'Banking',
        'shariaCompliant': True
    },
    'ETEL.CA': {
        'name': 'Etisalat Misr',
        'sector': 'Telecom',
        'shariaCompliant': True
    },
    'ORWA.CA': {
        'name': 'Orascom Construction',
        'sector': 'Construction',
        'shariaCompliant': True
    },
    'OCDI.CA': {
        'name': 'Orascom Development Holding',
        'sector': 'Real Estate',
        'shariaCompliant': False
    },
    'EMRA.CA': {
        'name': 'Emaar Misr for Development',
        'sector': 'Real Estate',
        'shariaCompliant': False
    },
    'NABE.CA': {
        'name': 'National Bank of Egypt',
        'sector': 'Banking',
        'shariaCompliant': True
    },
    'QNB.CA': {
        'name': 'QNB Alahli',
        'sector': 'Banking',
        'shariaCompliant': True
    },
    'SWDY.CA': {
        'name': 'Swvl Holdings',
        'sector': 'Transportation',
        'shariaCompliant': True
    }
}

@app.get("/api/stock/{symbol}", response_model=StockData)
async def get_stock_data(symbol: str):
    """
    Fetch real-time stock data from Yahoo Finance
    EGX stocks use .CA suffix for Cairo Exchange
    """
    try:
        stock = yf.Ticker(symbol)
        
        # Get historical data for analysis
        hist = stock.history(period="1d")
        
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
        
        current_price = hist['Close'].iloc[-1]
        
        # Get info
        info = stock.info if hasattr(stock, 'info') else {}
        
        # Calculate change
        hist_week = stock.history(period="5d")
        prev_price = hist_week['Close'].iloc[0] if len(hist_week) > 1 else current_price
        change = current_price - prev_price
        change_percent = (change / prev_price * 100) if prev_price != 0 else 0
        
        # Get stock details from our database
        stock_info = EGX_STOCKS.get(symbol, {
            'name': info.get('longName', symbol),
            'sector': info.get('sector', 'Unknown'),
            'shariaCompliant': False
        })
        
        return StockData(
            symbol=symbol,
            name=stock_info['name'],
            price=current_price,
            change=change,
            changePercent=change_percent,
            volume=int(hist['Volume'].iloc[-1]) if 'Volume' in hist else 0,
            pe_ratio=float(info.get('trailingPE', 0)) or None,
            dividend_yield=float(info.get('dividendYield', 0)) or None,
            shariaCompliant=stock_info['shariaCompliant'],
            sector=stock_info['sector']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stocks", response_model=List[StockData])
async def get_all_egx_stocks(sharia_only: bool = False):
    """
    Fetch all EGX stocks with real-time data
    """
    stocks = []
    for symbol, info in EGX_STOCKS.items():
        if sharia_only and not info['shariaCompliant']:
            continue
        
        try:
            stock_data = await get_stock_data(symbol)
            stocks.append(stock_data)
        except Exception as e:
            print(f"Error fetching {symbol}: {str(e)}")
            continue
    
    return stocks

@app.get("/api/stock/{symbol}/history")
async def get_stock_history(symbol: str, period: str = "1y"):
    """
    Get historical stock data for analysis
    Periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    """
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        
        return {
            "symbol": symbol,
            "dates": hist.index.strftime('%Y-%m-%d').tolist(),
            "closes": hist['Close'].tolist(),
            "highs": hist['High'].tolist(),
            "lows": hist['Low'].tolist(),
            "volumes": hist['Volume'].tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# PORTFOLIO OPTIMIZATION (MODERN PORTFOLIO THEORY)
# ============================================================================

@app.post("/api/optimize-portfolio", response_model=OptimizationResult)
async def optimize_portfolio(portfolio: Portfolio):
    """
    Optimize portfolio using Modern Portfolio Theory
    Calculates efficient frontier and optimal weights
    """
    try:
        # Fetch historical data for all holdings
        symbols = [h.symbol for h in portfolio.holdings]
        data = {}
        
        for symbol in symbols:
            hist = yf.Ticker(symbol).history(period="1y")
            data[symbol] = hist['Close']
        
        # Create dataframe
        prices_df = pd.DataFrame(data)
        returns = prices_df.pct_change().dropna()
        
        # Calculate expected returns and covariance
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # Number of assets
        num_assets = len(symbols)
        
        # Objective function: negative Sharpe ratio (we minimize)
        def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.03):
            portfolio_return = np.sum(weights * mean_returns) * 252  # Annualize
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
            sharpe = (portfolio_return - risk_free_rate) / portfolio_std
            return -sharpe
        
        # Portfolio std deviation
        def portfolio_std(weights, mean_returns, cov_matrix):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        
        # Portfolio return
        def portfolio_return(weights, mean_returns):
            return np.sum(weights * mean_returns) * 252
        
        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds: weights between 0 and 1
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Initial guess: equal weight
        init_guess = np.array([1/num_assets] * num_assets)
        
        # Optimize
        result = minimize(
            neg_sharpe_ratio,
            init_guess,
            args=(mean_returns, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_weights = result.x
        
        # Calculate metrics
        opt_return = portfolio_return(optimal_weights, mean_returns)
        opt_std = portfolio_std(optimal_weights, mean_returns)
        opt_sharpe = (opt_return - 0.03) / opt_std
        
        # Create weights dictionary
        weights_dict = {symbols[i]: float(optimal_weights[i]) for i in range(num_assets)}
        
        # Generate recommendation
        recommendation = f"Allocate {', '.join([f'{s}: {w*100:.1f}%' for s, w in weights_dict.items() if w > 0.01])}. "
        recommendation += f"Expected annual return: {opt_return*100:.2f}%, Risk: {opt_std*100:.2f}%, Sharpe: {opt_sharpe:.2f}"
        
        return OptimizationResult(
            weights=weights_dict,
            expectedReturn=opt_return,
            expectedRisk=opt_std,
            sharpeRatio=opt_sharpe,
            recommendation=recommendation
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")

# ============================================================================
# MACHINE LEARNING PRICE PREDICTION
# ============================================================================

def create_features(df, lookback=30):
    """Create features for ML model"""
    features = pd.DataFrame()
    
    # Simple moving averages
    features['sma_5'] = df['Close'].rolling(5).mean()
    features['sma_20'] = df['Close'].rolling(20).mean()
    features['sma_50'] = df['Close'].rolling(50).mean()
    
    # Price momentum
    features['momentum'] = df['Close'].diff(5)
    
    # Volatility
    features['volatility'] = df['Close'].rolling(20).std()
    
    # Volume
    features['volume_ma'] = df['Volume'].rolling(20).mean()
    
    # Price range
    features['high_low_ratio'] = (df['High'] - df['Low']) / df['Close']
    
    # Returns
    features['returns'] = df['Close'].pct_change()
    
    return features.dropna()

@app.get("/api/predict-price/{symbol}")
async def predict_price(symbol: str, days_ahead: int = 30):
    """
    Predict stock price using ML (Random Forest)
    Uses technical indicators + historical patterns
    """
    try:
        # Fetch historical data
        stock = yf.Ticker(symbol)
        hist = stock.history(period="2y")
        
        # Create features
        features = create_features(hist)
        
        # Prepare data
        X = features.drop('returns', axis=1).fillna(method='bfill')
        y = hist['Close'][len(hist) - len(X):]
        
        # Split into train/test
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        # Predict next value
        last_features = X.iloc[-1:].values
        next_price = model.predict(last_features)[0]
        
        # Calculate confidence
        predictions = model.predict(X_test)
        mape = np.mean(np.abs((y_test - predictions) / y_test))
        confidence = max(0, 1 - mape)  # Confidence as inverse of MAPE
        
        current_price = hist['Close'].iloc[-1]
        change_percent = ((next_price - current_price) / current_price) * 100
        
        return {
            "symbol": symbol,
            "currentPrice": float(current_price),
            "predictedPrice": float(next_price),
            "priceChange": float(next_price - current_price),
            "changePercent": float(change_percent),
            "confidence": float(confidence),
            "daysAhead": days_ahead,
            "trainAccuracy": float(train_score),
            "testAccuracy": float(test_score),
            "modelType": "Random Forest (100 estimators)"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ============================================================================
# TECHNICAL ANALYSIS & TRADING SIGNALS
# ============================================================================

def calculate_technical_indicators(hist):
    """Calculate technical indicators"""
    indicators = {}
    
    close = hist['Close']
    high = hist['High']
    low = hist['Low']
    volume = hist['Volume']
    
    # RSI (Relative Strength Index)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    indicators['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = close.ewm(span=12).mean()
    ema_26 = close.ewm(span=26).mean()
    indicators['macd'] = ema_12 - ema_26
    indicators['macd_signal'] = indicators['macd'].ewm(span=9).mean()
    
    # Bollinger Bands
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    indicators['bb_upper'] = sma_20 + (std_20 * 2)
    indicators['bb_lower'] = sma_20 - (std_20 * 2)
    
    # Moving Averages
    indicators['sma_20'] = close.rolling(20).mean()
    indicators['sma_50'] = close.rolling(50).mean()
    indicators['sma_200'] = close.rolling(200).mean()
    
    return indicators

@app.get("/api/trading-signals/{symbol}", response_model=TradingSignal)
async def generate_trading_signal(symbol: str):
    """
    Generate trading signals using:
    1. Technical analysis (RSI, MACD, Bollinger Bands)
    2. ML price predictions
    3. Moving average crossovers
    """
    try:
        # Fetch data
        stock = yf.Ticker(symbol)
        hist = stock.history(period="6mo")
        
        # Calculate indicators
        indicators = calculate_technical_indicators(hist)
        
        current_price = hist['Close'].iloc[-1]
        rsi = indicators['rsi'].iloc[-1]
        macd = indicators['macd'].iloc[-1]
        macd_signal = indicators['macd_signal'].iloc[-1]
        bb_upper = indicators['bb_upper'].iloc[-1]
        bb_lower = indicators['bb_lower'].iloc[-1]
        sma_20 = indicators['sma_20'].iloc[-1]
        sma_50 = indicators['sma_50'].iloc[-1]
        sma_200 = indicators['sma_200'].iloc[-1]
        
        # Technical score (0-1)
        technical_score = 0
        reasons = []
        
        # RSI signals
        if rsi < 30:
            technical_score += 0.3
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > 70:
            technical_score -= 0.3
            reasons.append(f"RSI overbought ({rsi:.1f})")
        else:
            technical_score += 0.1
        
        # MACD signals
        if macd > macd_signal:
            technical_score += 0.2
            reasons.append("MACD bullish crossover")
        else:
            technical_score -= 0.2
            reasons.append("MACD bearish")
        
        # Bollinger Bands
        if current_price < bb_lower:
            technical_score += 0.2
            reasons.append("Price below lower Bollinger Band")
        elif current_price > bb_upper:
            technical_score -= 0.2
            reasons.append("Price above upper Bollinger Band")
        
        # Moving Averages
        if sma_20 > sma_50 > sma_200:
            technical_score += 0.2
            reasons.append("Strong uptrend (20>50>200 SMA)")
        elif sma_20 < sma_50 < sma_200:
            technical_score -= 0.2
            reasons.append("Downtrend (20<50<200 SMA)")
        
        # Get ML prediction
        prediction_data = await predict_price(symbol)
        ml_score = prediction_data['confidence']
        predicted_price = prediction_data['predictedPrice']
        
        # Determine signal
        combined_score = (technical_score + ml_score) / 2
        
        if combined_score > 0.3:
            signal = "BUY"
            confidence = min(0.95, abs(combined_score))
        elif combined_score < -0.3:
            signal = "SELL"
            confidence = min(0.95, abs(combined_score))
        else:
            signal = "HOLD"
            confidence = 0.5
        
        # Calculate target and stop loss
        if signal == "BUY":
            target_price = predicted_price * 1.1  # 10% upside
            stop_loss = current_price * 0.95  # 5% downside
        elif signal == "SELL":
            target_price = current_price * 0.9
            stop_loss = current_price * 1.05
        else:
            target_price = None
            stop_loss = None
        
        reasons.append(f"ML confidence: {ml_score:.1%}")
        
        return TradingSignal(
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            reasons=reasons,
            targetPrice=target_price,
            stopLoss=stop_loss,
            technicalScore=max(-1, min(1, technical_score)),
            mlScore=ml_score
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Signal generation error: {str(e)}")

@app.post("/api/portfolio-signals")
async def generate_portfolio_signals(portfolio: Portfolio):
    """
    Generate trading signals for entire portfolio
    """
    signals = []
    for holding in portfolio.holdings:
        try:
            signal = await generate_trading_signal(holding.symbol)
            signals.append(signal)
        except Exception as e:
            print(f"Error generating signal for {holding.symbol}: {str(e)}")
    
    return {
        "signals": signals,
        "timestamp": datetime.now().isoformat(),
        "buyCount": sum(1 for s in signals if s['signal'] == 'BUY'),
        "sellCount": sum(1 for s in signals if s['signal'] == 'SELL'),
        "holdCount": sum(1 for s in signals if s['signal'] == 'HOLD')
    }

# ============================================================================
# AI-POWERED ANALYSIS WITH CLAUDE
# ============================================================================

@app.post("/api/ai-analysis")
async def get_ai_analysis(request: dict):
    """
    Get AI-powered analysis using Claude
    Input should have: symbol, analysis_type, api_key
    """
    try:
        symbol = request.get('symbol')
        analysis_type = request.get('analysis_type', 'comprehensive')
        api_key = request.get('api_key')
        
        if not api_key:
            raise HTTPException(status_code=400, detail="Claude API key required")
        
        # Get real data
        stock_data = await get_stock_data(symbol)
        prediction = await predict_price(symbol)
        signal = await generate_trading_signal(symbol)
        
        # Prepare prompt
        prompt = f"""
        Analyze {stock_data.name} ({symbol}) with the following data:
        
        Current Price: {stock_data.price}
        Change: {stock_data.change} ({stock_data.changePercent}%)
        Sector: {stock_data.sector}
        Sharia Compliant: {stock_data.shariaCompliant}
        
        ML Prediction: Price will be {prediction['predictedPrice']} ({prediction['changePercent']:+.2f}%)
        Confidence: {prediction['confidence']:.1%}
        
        Trading Signal: {signal.signal}
        Technical Score: {signal.technicalScore:.2f}
        ML Score: {signal.mlScore:.2f}
        
        Key Reasons: {', '.join(signal.reasons)}
        
        Provide a {analysis_type} analysis including:
        1. Current market position
        2. Technical analysis summary
        3. ML prediction interpretation
        4. Investment recommendation
        5. Risk factors
        6. Time horizon for decision
        
        Be specific and actionable for an investor with moderate risk tolerance.
        """
        
        # Call Claude API
        client = Anthropic()
        response = client.messages.create(
            model="claude-opus-4-1",
            max_tokens=1000,
            system="""You are an expert financial analyst specializing in Egyptian and MENA stock markets.
            Provide detailed, actionable analysis combining technical indicators, ML predictions, and fundamental insights.
            Be concise but thorough. Include specific price targets and risk levels.""",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return {
            "symbol": symbol,
            "analysis": response.content[0].text,
            "timestamp": datetime.now().isoformat(),
            "data": {
                "stock": stock_data.dict(),
                "prediction": prediction,
                "signal": signal.dict()
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# PORTFOLIO BACKTESTING
# ============================================================================

@app.post("/api/backtest-strategy")
async def backtest_strategy(request: dict):
    """
    Backtest a trading strategy
    Input: symbols, start_date, end_date, initial_capital, strategy_type
    """
    try:
        symbols = request.get('symbols', [])
        start_date = request.get('start_date', '2023-01-01')
        end_date = request.get('end_date', '2024-01-01')
        initial_capital = request.get('initial_capital', 10000)
        strategy_type = request.get('strategy_type', 'momentum')
        
        results = {}
        total_returns = 0
        win_rate = 0
        
        for symbol in symbols:
            hist = yf.Ticker(symbol).history(start=start_date, end=end_date)
            
            if hist.empty:
                continue
            
            # Calculate returns
            returns = hist['Close'].pct_change()
            
            if strategy_type == 'momentum':
                # Momentum strategy
                signals = (returns.rolling(10).mean() > 0).astype(int)
            elif strategy_type == 'mean_reversion':
                # Mean reversion strategy
                ma_20 = hist['Close'].rolling(20).mean()
                signals = (hist['Close'] < ma_20).astype(int)
            else:
                signals = (returns > 0).astype(int)
            
            # Calculate strategy returns
            strategy_returns = signals.shift(1) * returns
            cumulative_returns = (1 + strategy_returns).cumprod()
            
            total_return = cumulative_returns.iloc[-1] - 1 if len(cumulative_returns) > 0 else 0
            buy_hold_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1
            
            # Sharpe ratio
            annual_return = strategy_returns.mean() * 252
            annual_std = strategy_returns.std() * np.sqrt(252)
            sharpe = annual_return / annual_std if annual_std > 0 else 0
            
            # Win rate
            wins = (strategy_returns > 0).sum()
            total_trades = (strategy_returns != 0).sum()
            win_rate_symbol = (wins / total_trades * 100) if total_trades > 0 else 0
            
            results[symbol] = {
                "strategy_return": float(total_return * 100),
                "buy_hold_return": float(buy_hold_return * 100),
                "sharpe_ratio": float(sharpe),
                "win_rate": float(win_rate_symbol),
                "final_value": float(initial_capital * cumulative_returns.iloc[-1]) if len(cumulative_returns) > 0 else initial_capital
            }
        
        return {
            "strategy": strategy_type,
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": initial_capital,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "AI Investment Analysis Backend"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
