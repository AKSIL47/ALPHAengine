from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import create_engine, Column, String, Integer, Boolean, DateTime, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import yfinance as yf
from openai import OpenAI
import os
import json
import random
import requests  # <-- NEW: For SEC API calls

# ---------- CONFIG ----------
SECRET_KEY = "alphaengine-super-secret-key-change-this-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

# ---------- DATABASE SETUP ----------
SQLALCHEMY_DATABASE_URL = "sqlite:///./alphaengine.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ---------- PASSWORD HASHING ----------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="AlphaEngine API", version="2.0")

# ---------- DATABASE MODELS ----------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_pro = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    portfolios = relationship("Portfolio", back_populates="user")

class Portfolio(Base):
    __tablename__ = "portfolios"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="portfolios")
    holdings = relationship("Holding", back_populates="portfolio")

class Holding(Base):
    __tablename__ = "holdings"
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"))
    ticker = Column(String)
    shares = Column(Float)
    buy_price = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    portfolio = relationship("Portfolio", back_populates="holdings")

Base.metadata.create_all(bind=engine)

# ---------- HELPER FUNCTIONS ----------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def authenticate_user(db: Session, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

# ==========================================
# ========== SEC FILINGS FUNCTION ==========
# ==========================================

def get_sec_data(ticker):
    """
    Fetches latest Revenue and Net Income from SEC EDGAR API.
    Returns a dictionary with revenue and net_income (in millions).
    """
    try:
        # Get CIK from Yahoo Finance
        stock = yf.Ticker(ticker)
        cik = stock.info.get('cik')
        if not cik:
            return {"error": "CIK not found for this ticker"}

        # SEC requires a User-Agent header with contact info
        headers = {
            'User-Agent': 'AlphaEngine (your_email@example.com)'
        }
        
        # SEC API endpoint for company facts
        cik_str = str(cik).zfill(10)  # CIK must be 10 digits with leading zeros
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_str}.json"
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return {"error": "SEC API request failed"}
        
        data = response.json()
        facts = data.get('facts', {}).get('us-gaap', {})
        
        # Helper function to get latest value for a metric
        def get_latest_value(metric_key):
            metric_data = facts.get(metric_key, {})
            if 'units' in metric_data and 'USD' in metric_data['units']:
                entries = metric_data['units']['USD']
                # Sort by end date (most recent first)
                sorted_entries = sorted(entries, key=lambda x: x['end'], reverse=True)
                if sorted_entries:
                    return sorted_entries[0]['val']
            return None

        # Revenue (Annual or Quarterly)
        # Common tags: RevenueFromContractWithCustomerExcludingAssessedTax, RevenueFromContractWithCustomer
        revenue = get_latest_value('RevenueFromContractWithCustomerExcludingAssessedTax')
        if not revenue:
            revenue = get_latest_value('RevenueFromContractWithCustomer')
        
        # Net Income
        net_income = get_latest_value('NetIncomeLoss')
        
        # If values are huge (>1 million), convert to millions and round
        if revenue:
            revenue = round(revenue / 1_000_000, 2)  # Convert to millions
        if net_income:
            net_income = round(net_income / 1_000_000, 2)
        
        return {
            "revenue": revenue,
            "net_income": net_income,
            "success": True
        }
        
    except Exception as e:
        return {"error": str(e), "success": False}

# ==========================================
# ========== AUTH ENDPOINTS ================
# ==========================================

@app.post("/signup")
def signup(username: str, password: str, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.username == username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already taken")
    hashed = get_password_hash(password)
    new_user = User(username=username, hashed_password=hashed)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": f"User {username} created successfully!", "username": username}

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    token = create_access_token(data={"sub": user.username})
    return {
        "access_token": token,
        "token_type": "bearer",
        "username": user.username,
        "is_pro": user.is_pro
    }

# ==========================================
# ========== PORTFOLIO ENDPOINTS ===========
# ==========================================

@app.post("/portfolio")
def create_portfolio(name: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    new_portfolio = Portfolio(user_id=current_user.id, name=name)
    db.add(new_portfolio)
    db.commit()
    db.refresh(new_portfolio)
    return {"message": f"Portfolio '{name}' created!", "id": new_portfolio.id}

@app.get("/portfolios")
def get_portfolios(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    portfolios = db.query(Portfolio).filter(Portfolio.user_id == current_user.id).all()
    return [{"id": p.id, "name": p.name, "created_at": p.created_at} for p in portfolios]

# ==========================================
# ========== HOLDINGS ENDPOINTS ============
# ==========================================

@app.post("/holding")
def add_holding(
    portfolio_id: int, 
    ticker: str, 
    shares: float, 
    buy_price: float,
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    portfolio = db.query(Portfolio).filter(
        Portfolio.id == portfolio_id, 
        Portfolio.user_id == current_user.id
    ).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    new_holding = Holding(
        portfolio_id=portfolio_id,
        ticker=ticker.upper(),
        shares=shares,
        buy_price=buy_price
    )
    db.add(new_holding)
    db.commit()
    db.refresh(new_holding)
    return {"message": f"Added {shares} shares of {ticker.upper()} at ${buy_price}"}

@app.get("/holdings/{portfolio_id}")
def get_holdings(
    portfolio_id: int,
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    portfolio = db.query(Portfolio).filter(
        Portfolio.id == portfolio_id, 
        Portfolio.user_id == current_user.id
    ).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    holdings = db.query(Holding).filter(Holding.portfolio_id == portfolio_id).all()
    result = []
    total_value = 0
    total_cost = 0
    for h in holdings:
        try:
            stock = yf.Ticker(h.ticker)
            hist = stock.history(period="1d")
            if not hist.empty:
                current_price = hist["Close"].iloc[-1]
            else:
                current_price = h.buy_price
        except:
            current_price = h.buy_price
        current_value = current_price * h.shares
        cost_basis = h.buy_price * h.shares
        pnl = current_value - cost_basis
        pnl_pct = ((current_price - h.buy_price) / h.buy_price) * 100 if h.buy_price > 0 else 0
        total_value += current_value
        total_cost += cost_basis
        result.append({
            "id": h.id,
            "ticker": h.ticker,
            "shares": h.shares,
            "buy_price": h.buy_price,
            "current_price": round(current_price, 2),
            "current_value": round(current_value, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2)
        })
    total_pnl = total_value - total_cost
    total_pnl_pct = ((total_value - total_cost) / total_cost) * 100 if total_cost > 0 else 0
    return {
        "holdings": result,
        "summary": {
            "total_value": round(total_value, 2),
            "total_cost": round(total_cost, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 2)
        }
    }

# ==========================================
# ========== UPGRADE TO PRO ================
# ==========================================

@app.post("/upgrade")
def upgrade_to_pro(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    current_user.is_pro = True
    db.commit()
    return {"message": f"🎉 {current_user.username} is now a PRO user!", "is_pro": True}

# ==========================================
# ========== AI ANALYSIS (MOCK MODE + SEC) ==
# ==========================================

@app.post("/analyze")
def analyze_stock(data: dict, current_user: User = Depends(get_current_user)):
    ticker = data.get("ticker")
    if not ticker:
        raise HTTPException(status_code=400, detail="Please provide a ticker like AAPL")
    
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1mo")
    info = stock.info
    
    if hist.empty:
        raise HTTPException(status_code=404, detail="No data found for this ticker")
    
    current_price = hist["Close"].iloc[-1]
    prev_price = hist["Close"].iloc[0]
    change_pct = ((current_price - prev_price) / prev_price) * 100

    # ---------- NEW: Fetch SEC Data ----------
    sec_data = get_sec_data(ticker)
    
    # Build a summary string for the AI
    sec_summary = ""
    if sec_data.get("success") and sec_data.get("revenue") is not None:
        sec_summary = f"Latest Revenue: ${sec_data['revenue']}M, Net Income: ${sec_data['net_income']}M"
    else:
        sec_summary = "No recent SEC filing data available."

    # ---------- Mock AI Reasoning ----------
    try:
        random.seed(hash(ticker))
        
        # Generate Bull/Bear cases based on price and SEC data
        if change_pct > 2:
            confidence = random.randint(75, 92)
            bull = f"{ticker} shows strong upward momentum. Revenue is growing and market sentiment is positive."
            bear = f"Near-term resistance and potential profit-taking could pressure {ticker}."
        elif change_pct < -2:
            confidence = random.randint(60, 85)
            bull = f"{ticker} has solid long-term fundamentals. The dip represents a potential buying opportunity based on SEC filings."
            bear = f"Downward trend and increased selling pressure suggest near-term weakness."
        else:
            confidence = random.randint(50, 75)
            bull = f"{ticker} is trading in a stable range with balanced buying and selling activity."
            bear = f"Lack of clear directional catalyst in the short term."

        risk_pool = [
            "Macroeconomic headwinds", 
            "Sector rotation risk", 
            "Regulatory scrutiny", 
            "Supply chain disruptions", 
            "Valuation concerns", 
            "Interest rate sensitivity",
            "Geopolitical tensions",
            "Currency fluctuation risk"
        ]
        selected_risks = random.sample(risk_pool, 3)
        price_target = round(current_price * random.uniform(0.90, 1.18), 2)

        ai_result = {
            "bull_case": bull,
            "bear_case": bear,
            "confidence_score": confidence,
            "key_risks": selected_risks,
            "price_target": price_target,
            "sec_data": {
                "revenue": sec_data.get("revenue"),
                "net_income": sec_data.get("net_income"),
                "success": sec_data.get("success", False),
                "summary": sec_summary
            }
        }
        
        ai_result["ticker"] = ticker
        ai_result["current_price"] = round(current_price, 2)
        ai_result["change_pct"] = round(change_pct, 2)
        ai_result["user"] = current_user.username
        ai_result["is_pro"] = current_user.is_pro
        
        return ai_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# ==========================================
# ========== CHART ENDPOINT ================
# ==========================================

@app.get("/chart/{ticker}")
def get_chart(ticker: str):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1mo")
    if hist.empty:
        raise HTTPException(status_code=404, detail="No data found")
    return {
        "dates": [str(d.date()) for d in hist.index],
        "prices": [float(p) for p in hist["Close"]]
    }

# ==========================================
# ========== FRONTEND ======================
# ==========================================

@app.get("/")
def home():
    return FileResponse("templates/index.html")

# ==========================================
# ========== FOR DEPLOYMENT ================
# ==========================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
