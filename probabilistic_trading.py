"""
PROBABILISTIC FORECASTING FOR TRADING DECISIONS
Method: LightGBM + Isotonic Calibration (Probabilistic Classification)
"""
from sklearn.metrics import accuracy_score
import warnings; warnings.filterwarnings('ignore')
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich import box
from rich.text import Text
console = Console()

import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (brier_score_loss, log_loss, roc_auc_score,
                             classification_report, precision_recall_curve)
import matplotlib
matplotlib.use('Agg') # Khong hien thi cua so
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import joblib, os

# ── CONFIG ───────────────────────────────────────────────────────────────────
TICKER        = "FPT"
START_DATE    = "2020-01-01"
END_DATE      = datetime.today().strftime("%Y-%m-%d")
FILE_PATH     = r"d:\ck\FPT_data.csv"   # <-- duong dan file local
THRESHOLD_BUY  = 0.6
THRESHOLD_SELL = 0.4
MODEL_FILE    = "lgbm_model.pkl"
SCALER_FILE   = "scaler.pkl"

# ── FEATURES ─────────────────────────────────────────────────────────────────
def rsi(s, p=14):
    d=s.diff(); g=d.clip(lower=0).rolling(p).mean()
    l=(-d.clip(upper=0)).rolling(p).mean()
    return 100-100/(1+g/(l+1e-9))

def macd(s, f=12, sl=26, sg=9):
    ef=s.ewm(span=f,adjust=False).mean(); es=s.ewm(span=sl,adjust=False).mean()
    m=ef-es; sig=m.ewm(span=sg,adjust=False).mean()
    return m, sig, m-sig

def bollinger(s, p=20, k=2):
    mu=s.rolling(p).mean(); std=s.rolling(p).std()
    up=mu+k*std; lo=mu-k*std
    return mu, up, lo, (s-lo)/(up-lo+1e-9)

def add_features(df):
    c,h,l,v = df['Close'],df['High'],df['Low'],df['Volume']
    for w in [5,10,20,50,200]:
        df[f'SMA_{w}']=c.rolling(w).mean()
        df[f'EMA_{w}']=c.ewm(span=w,adjust=False).mean()
        df[f'Ret_{w}d']=c.pct_change(w)
        df[f'PvsSMA_{w}']=(c-df[f'SMA_{w}'])/(df[f'SMA_{w}']+1e-9)
    df['RSI14']=rsi(c,14); df['RSI7']=rsi(c,7); df['RSIdiff']=df['RSI14']-df['RSI7']
    df['MACD'],df['MACDsig'],df['MACDhist']=macd(c)
    df['MACDn']=df['MACD']/(c+1e-9)
    lo14=l.rolling(14).min(); hi14=h.rolling(14).max()
    df['StochK']=100*(c-lo14)/(hi14-lo14+1e-9)
    df['StochD']=df['StochK'].rolling(3).mean()
    df['BBsma'],df['BBup'],df['BBlo'],df['BBpct']=bollinger(c)
    atr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    df['ATR']=atr.rolling(14).mean(); df['ATRpct']=df['ATR']/(c+1e-9)
    df['Vol10']=c.pct_change().rolling(10).std()
    df['Vol20']=c.pct_change().rolling(20).std()
    df['VolMA']=v.rolling(20).mean(); df['VolR']=v/(df['VolMA']+1e-9)
    df['OBV']=(np.sign(c.diff())*v).cumsum()
    df['HLR']=(h-l)/(c+1e-9); df['COR']=(c-df['Open'])/(df['Open']+1e-9)
    ret=c.pct_change()
    for lag in [1,2,3,5,10]: df[f'Lag{lag}']=ret.shift(lag)
    # Regime: above/below 200SMA
    df['Regime']=(c>df['SMA_200']).astype(int)
    # Target
    df['Target']=(c.shift(-1)>c).astype(int)
    return df

EXCLUDE={'Target','Open','High','Low','Close','Volume','Adj Close'}

def load_data():
    print(f"[*] Doc file local: {FILE_PATH} ...")
    raw = pd.read_csv(FILE_PATH, parse_dates=['Date'])
    raw = raw.sort_values('Date').set_index('Date')
    raw.index.name = 'Date'
    # Chuan hoa ten cot neu can
    raw.columns = [c.strip().capitalize() for c in raw.columns]
    raw = raw[['Open','High','Low','Close','Volume']].dropna()
    print(f"    -> {len(raw)} ngay du lieu (local), tu {raw.index[0].date()} den {raw.index[-1].date()}")
    
    # Lay du lieu moi nhat tu yfinance cho FPT.VN (neu TICKER la FPT)
    yf_ticker = f"{TICKER}.VN" if TICKER == "FPT" else TICKER
    last_date = raw.index[-1]
    
    # Tinh toan start date cho yfinance (lay them vai ngay de tranh thieu xot do lech mui gio/ngay nghi)
    yf_start = (last_date - timedelta(days=5)).strftime("%Y-%m-%d")
    print(f"[*] Downloading latest data cho {yf_ticker} tu {yf_start} ...")
    
    try:
        raw_new = yf.download(yf_ticker, start=yf_start, progress=False)
        if not raw_new.empty:
            raw_new.columns = raw_new.columns.get_level_values(0)
            raw_new = raw_new[['Open','High','Low','Close','Volume']].dropna()
            # Loai bo timezone neu co de match voi local data
            raw_new.index = raw_new.index.tz_localize(None)
            
            # Combine 2 dataframe, uu tien du lieu tu yfinance cho cac ngay trung nhau
            raw = pd.concat([raw[~raw.index.isin(raw_new.index)], raw_new])
            raw = raw.sort_index()
            print(f"    -> Cap nhat thanh cong den: {raw.index[-1].date()}")
        else:
            print("    -> Khong co du lieu moi tu yfinance.")
    except Exception as e:
        print(f"    -> Loi khi lay du lieu yfinance: {e}")

    df  = add_features(raw.copy())
    df.dropna(inplace=True)
    feat = [c for c in df.columns if c not in EXCLUDE]
    return df, df[feat], df['Target'], feat

# ── TRAINING ─────────────────────────────────────────────────────────────────
def train(df, X, y, feat):
    print("[*] Training with TimeSeriesSplit (5 folds) ...")
    tscv=TimeSeriesSplit(n_splits=5)
    splits=list(tscv.split(X))
    tr_idx, val_idx = splits[-1]

    X_tr,X_val=X.iloc[tr_idx],X.iloc[val_idx]
    y_tr,y_val=y.iloc[tr_idx],y.iloc[val_idx]

    scaler=StandardScaler()
    X_tr_sc=scaler.fit_transform(X_tr)
    X_val_sc=scaler.transform(X_val)

    base=lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        max_depth=5, min_child_samples=30,
        subsample=0.8, subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=1.0,
        random_state=42, verbose=-1, class_weight='balanced'
    )
    model=CalibratedClassifierCV(base,method='isotonic',cv=3)
    model.fit(X_tr_sc, y_tr.values)

    prob=model.predict_proba(X_val_sc)[:,1]

    # Vi xac suat da duoc Calibrated, nguong 0.5 la nguong tu nhien cho Accuracy tot nhat
    opt_thresh = 0.5
    pred=(prob>=opt_thresh).astype(int)

    auc=roc_auc_score(y_val,prob)
    brier=brier_score_loss(y_val,prob)
    ll=log_loss(y_val,prob)
    acc=accuracy_score(y_val,pred)

    print(f"\n{'='*50}")
    print("  VALIDATION METRICS")
    print(f"{'='*50}")
    print(f"  ROC-AUC          : {auc:.4f}")
    print(f"  Accuracy         : {acc:.2%}")
    print(f"  Brier Score      : {brier:.4f}  (random baseline=0.25)")
    print(f"  Log Loss         : {ll:.4f}  (random baseline=0.693)")
    print(f"  Optimal Threshold: {opt_thresh:.3f}")
    print()
    print(classification_report(y_val,pred,target_names=['DOWN','UP']))

    # Backtest
    vc=df['Close'].iloc[val_idx]; dr=vc.pct_change().fillna(0)
    sig=pd.Series(np.where(prob>=THRESHOLD_BUY,1,np.where(prob<=THRESHOLD_SELL,-1,0)),
                  index=vc.index)
    sr=(1+sig.shift(1).fillna(0)*dr).cumprod()
    mr=(1+dr).cumprod()
    sharpe=dr[sig.shift(1)!=0].mean()/(dr[sig.shift(1)!=0].std()+1e-9)*np.sqrt(252)
    dd=(sr/sr.cummax()-1).min()

    print(f"{'='*50}")
    print("  BACKTEST (validation period)")
    print(f"{'='*50}")
    print(f"  Strategy return : {(sr.iloc[-1]-1)*100:+.2f}%")
    print(f"  Market   return : {(mr.iloc[-1]-1)*100:+.2f}%")
    print(f"  Sharpe Ratio    : {sharpe:.3f}")
    print(f"  Max Drawdown    : {dd*100:.2f}%")
    print(f"{'='*50}\n")

    joblib.dump(model, MODEL_FILE); joblib.dump(scaler, SCALER_FILE)
    print(f"[+] Saved {MODEL_FILE}, {SCALER_FILE}")
    return model, scaler, prob, val_idx, sig

# ── PREDICT ──────────────────────────────────────────────────────────────────
def predict(df, feat, model, scaler):
    row=df[feat].iloc[-1:]; sc=scaler.transform(row)
    p_up=float(model.predict_proba(sc)[0,1]); p_dn=1-p_up
    conf=abs(p_up-0.5)*2
    if p_up>=THRESHOLD_BUY:   dec,tag,dec_color="MUA (BUY)","BUY  ","bold green"
    elif p_up<=THRESHOLD_SELL: dec,tag,dec_color="BAN (SELL)","SELL ","bold red"
    else:                       dec,tag,dec_color="GIU NGUYEN (HOLD)","HOLD ","bold yellow"
    today=df.index[-1].strftime("%Y-%m-%d")
    nday=(df.index[-1]+timedelta(days=1)).strftime("%Y-%m-%d")
    return p_up, p_dn, dec, tag, dec_color, conf, today, nday


def print_report(df, feat, p_up, p_dn, dec, tag, dec_color, conf, today, nday,
                 auc, brier, ll, strat_ret, mkt_ret, sharpe, maxdd, acc):
    lat = df.iloc[-1]

    # ── Header ──────────────────────────────────────────────────────────────
    console.rule("[bold cyan]PROBABILISTIC FORECASTING FOR TRADING DECISIONS[/bold cyan]")
    console.print(f"  Ticker: [bold white]{TICKER}[/]   "
                  f"Period: [dim]{START_DATE} -> {today}[/]", justify="center")
    console.print()

    # ── Signal panel ────────────────────────────────────────────────────────
    open_price = float(lat['Open'])
    high_price = float(lat['High'])
    low_price = float(lat['Low'])
    close_price = float(lat['Close'])
    volume = float(lat['Volume'])
    
    # Format color strings
    c_green = "[bold green]"
    c_red = "[bold red]"
    c_yellow = "[bold yellow]"
    c_reset = "[/]"
    
    if tag.strip() == "BUY":
        color_sig = c_green
        signal_vn = "MUA"
        signal = "Dự đoán giá TĂNG"
    elif tag.strip() == "SELL":
        color_sig = c_red
        signal_vn = "BÁN"
        signal = "Dự đoán giá GIẢM"
    else:
        color_sig = c_yellow
        signal_vn = "GIỮ"
        signal = "Chưa rõ xu hướng"

    # Use rich console to print with colors, but keep the requested exact layout
    # Use standard print for the box to match requested format perfectly without rich panel wrapping
    print()
    console.print("╔══════════════════════════════════════════════════╗")
    console.print("║      KẾT QUẢ DỰ ĐOÁN – NGÀY MAI                  ║")
    console.print(f"║      Mô hình: LightGBM + Isotonic Calibration    ║")
    console.print("╠══════════════════════════════════════════════════╣")
    console.print(f"║  Dữ liệu đầu vào HÔM NAY ({today}):              ║")
    console.print(f"║    Giá mở cửa  (Open)  : {open_price:>10,.0f} VNĐ       ║")
    console.print(f"║    Giá cao nhất (High) : {high_price:>10,.0f} VNĐ       ║")
    console.print(f"║    Giá thấp nhất (Low) : {low_price:>10,.0f} VNĐ       ║")
    console.print(f"║    Giá đóng cửa (Close): {close_price:>10,.0f} VNĐ       ║")
    console.print(f"║    Khối lượng (Volume) : {volume:>10,.0f} cp        ║")
    console.print("╠══════════════════════════════════════════════════╣")
    console.print(f"║  Dự đoán cho NGÀY MAI ({nday}):                  ║")
    console.print(f"║    P(Tăng giá)  = {p_up:>6.2%}                       ║")
    console.print(f"║    P(Giảm giá)  = {p_dn:>6.2%}                       ║")
    console.print(f"║                                                  ║")
    console.print(f"║  ➤  Quyết định: {color_sig}{signal_vn:<33}{c_reset}  ║")
    console.print(f"║     Tín hiệu  : {signal:<35}  ║")
    console.print("╠══════════════════════════════════════════════════╣")
    console.print(f"║  Ngưỡng: BUY nếu P(Up) >= {THRESHOLD_BUY:.0%}  |  SELL nếu P(Up) <= {THRESHOLD_SELL:.0%}  ║")
    console.print("╚══════════════════════════════════════════════════╝")
    print()

    # ── Model metrics table ──────────────────────────────────────────────────
    t1 = Table(title="Model Performance (Validation Set)", box=box.ROUNDED,
               title_style="bold cyan", show_header=True, header_style="bold magenta")
    t1.add_column("Metric",       style="dim",        width=22)
    t1.add_column("Value",        justify="right",    width=12)
    t1.add_column("Benchmark",    justify="right",    width=15, style="dim")
    t1.add_column("Verdict",      justify="center",   width=12)

    def verdict(val, good_if, threshold):
        ok = val < threshold if good_if == "lower" else val > threshold
        return "[green]GOOD[/]" if ok else "[yellow]FAIR[/]"

    t1.add_row("ROC-AUC",        f"{auc:.4f}",    "Random = 0.50", verdict(auc,"higher",0.52))
    t1.add_row("Accuracy",       f"{acc:.2%}",    "Random = 50%",  verdict(acc,"higher",0.50))
    t1.add_row("Brier Score",    f"{brier:.4f}",  "Random = 0.25", verdict(brier,"lower",0.25))
    t1.add_row("Log Loss",       f"{ll:.4f}",     "Random = 0.693",verdict(ll,"lower",0.693))
    t1.add_row("","","","")
    strat_col = "[green]" if strat_ret>=0 else "[red]"
    mkt_col   = "[green]" if mkt_ret>=0   else "[red]"
    t1.add_row("Strategy Return",f"{strat_col}{strat_ret:+.2f}%[/]", "Backtest period","")
    t1.add_row("Market Return",  f"{mkt_col}{mkt_ret:+.2f}%[/]",   "Buy & Hold","")
    t1.add_row("Sharpe Ratio",   f"{sharpe:.3f}",  "> 1.0 = good",  verdict(sharpe,"higher",1.0))
    t1.add_row("Max Drawdown",   f"{maxdd:.2f}%",  "< -20% = risk", "[green]OK[/]" if maxdd>-20 else "[red]HIGH[/]")
    console.print(t1)
    console.print()

    # ── Today indicators table ───────────────────────────────────────────────
    t2 = Table(title=f"Today's Market Snapshot  ({today})", box=box.SIMPLE_HEAVY,
               title_style="bold cyan", show_header=True, header_style="bold magenta")
    t2.add_column("Indicator", style="dim",      width=18)
    t2.add_column("Value",     justify="right",  width=14)
    t2.add_column("Signal",    justify="center", width=16)

    close_val = float(lat['Close'])
    rsi14     = float(lat['RSI14'])
    rsi7      = float(lat['RSI7'])
    macdh     = float(lat['MACDhist'])
    bbpct     = float(lat['BBpct'])*100
    stochk    = float(lat['StochK'])
    volr      = float(lat['VolR'])
    ret5      = float(lat['Ret_5d'])*100
    ret20     = float(lat['Ret_20d'])*100
    atr_val   = float(lat['ATR'])

    def rsi_sig(v):
        if v>=70: return "[red]Overbought[/]"
        if v<=30: return "[green]Oversold[/]"
        return "[dim]Neutral[/]"

    def pct_sig(v):
        return "[green]+" + f"{v:.2f}%[/]" if v>=0 else "[red]" + f"{v:.2f}%[/]"

    t2.add_row("Close Price",  f"{close_val:,.0f} VNĐ",  "")
    t2.add_row("RSI (14)",     f"{rsi14:.1f}",         rsi_sig(rsi14))
    t2.add_row("RSI (7)",      f"{rsi7:.1f}",          rsi_sig(rsi7))
    t2.add_row("MACD Hist",    f"{macdh:.4f}",         "[green]Bullish[/]" if macdh>=0 else "[red]Bearish[/]")
    t2.add_row("BB %B",        f"{bbpct:.1f}%",        "[red]Near Top[/]" if bbpct>80 else ("[green]Near Bot[/]" if bbpct<20 else "[dim]Mid[/]"))
    t2.add_row("Stoch %K",     f"{stochk:.1f}",        "[red]Overbought[/]" if stochk>80 else ("[green]Oversold[/]" if stochk<20 else "[dim]Neutral[/]"))
    t2.add_row("ATR",          f"{atr_val:.2f}",        "[dim]Volatility[/]")
    t2.add_row("Volume Ratio", f"{volr:.2f}x",          "[cyan]High Volume[/]" if volr>1.5 else "[dim]Normal[/]")
    t2.add_row("Return 5d",    pct_sig(ret5),           "")
    t2.add_row("Return 20d",   pct_sig(ret20),          "")
    console.print(t2)
    console.rule(style="dim")

# ── DASHBOARD ────────────────────────────────────────────────────────────────
def plot(df, p_up, p_dn, dec, feat, val_prob=None, val_idx=None, val_sig=None):
    DK='#0d1117'; PN='#161b22'; EG='#30363d'
    BL='#58a6ff'; OR='#f0883e'; PU='#bc8cff'
    GR='#3fb950'; RD='#f85149'; GY='#8b949e'; WH='#e6edf3'

    fig=plt.figure(figsize=(20,13),facecolor=DK)
    fig.suptitle(f'Probabilistic Forecasting for Trading  —  {TICKER}',
                 fontsize=17,fontweight='bold',color=WH,y=0.99)
    gs=fig.add_gridspec(3,3,hspace=0.48,wspace=0.30,
                        left=0.06,right=0.97,top=0.95,bottom=0.05)

    def sty(ax,t=''):
        ax.set_facecolor(PN); ax.tick_params(colors=GY,labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(EG)
        if t: ax.set_title(t,color=WH,fontsize=10,pad=6)

    tail=df.tail(120); tail2=df.tail(60)

    # A: Price
    ax1=fig.add_subplot(gs[0,:2]); sty(ax1,'Price + Bollinger Bands (120 days)')
    ax1.plot(tail.index,tail['Close'],color=BL,lw=1.6,label='Close')
    ax1.plot(tail.index,tail['SMA_20'],color=OR,lw=1.0,ls='--',label='SMA20')
    ax1.plot(tail.index,tail['SMA_50'],color=PU,lw=1.0,ls='--',label='SMA50')
    ax1.fill_between(tail.index,tail['BBlo'],tail['BBup'],alpha=0.12,color=BL)
    ax1.legend(facecolor=PN,labelcolor=WH,fontsize=8)

    # B: RSI
    ax2=fig.add_subplot(gs[1,:2]); sty(ax2,'RSI (14)')
    ax2.plot(tail.index,tail['RSI14'],color=PU,lw=1.4)
    ax2.axhline(70,color=RD,ls='--',lw=0.8,alpha=0.8)
    ax2.axhline(30,color=GR,ls='--',lw=0.8,alpha=0.8)
    ax2.axhline(50,color=GY,ls=':',lw=0.6)
    ax2.fill_between(tail.index,tail['RSI14'],50,where=tail['RSI14']>=50,alpha=0.13,color=GR)
    ax2.fill_between(tail.index,tail['RSI14'],50,where=tail['RSI14']<50, alpha=0.13,color=RD)
    ax2.set_ylim(0,100)

    # C: MACD
    ax3=fig.add_subplot(gs[2,:2]); sty(ax3,'MACD')
    ax3.plot(tail.index,tail['MACD'],color=BL,lw=1.2,label='MACD')
    ax3.plot(tail.index,tail['MACDsig'],color=OR,lw=1.2,label='Signal')
    hc=[GR if v>=0 else RD for v in tail['MACDhist']]
    ax3.bar(tail.index,tail['MACDhist'],color=hc,alpha=0.65,width=0.8)
    ax3.axhline(0,color=WH,lw=0.4,ls='--')
    ax3.legend(facecolor=PN,labelcolor=WH,fontsize=8)

    # D: Probability donut
    ax4=fig.add_subplot(gs[0,2]); sty(ax4)
    dc=GR if 'MUA' in dec else (RD if 'BAN' in dec else OR)
    ax4.pie([p_dn,p_up],colors=[RD,GR],startangle=90,
            wedgeprops=dict(width=0.45,edgecolor=DK,linewidth=2),counterclock=False)
    ax4.text(0, 0.12,f"{p_up*100:.1f}%",ha='center',va='center',
             fontsize=22,fontweight='bold',color=GR)
    ax4.text(0,-0.18,"P(UP)",ha='center',va='center',fontsize=10,color=GY)
    ax4.set_title(f'Signal: {dec}',color=dc,fontsize=11,pad=8)

    # E: Calibration curve
    ax5=fig.add_subplot(gs[1,2]); sty(ax5,'Probability Calibration Curve')
    if val_prob is not None:
        frac_pos,mean_pred=calibration_curve(df['Target'].iloc[val_idx],val_prob,n_bins=10)
        ax5.plot(mean_pred,frac_pos,color=BL,lw=1.4,marker='o',ms=4,label='Model')
        ax5.plot([0,1],[0,1],color=GY,ls='--',lw=1.0,label='Perfect')
        ax5.set_xlim(0,1); ax5.set_ylim(0,1)
        ax5.set_xlabel('Mean predicted prob',color=GY,fontsize=8)
        ax5.set_ylabel('Fraction positives',color=GY,fontsize=8)
        ax5.legend(facecolor=PN,labelcolor=WH,fontsize=8)

    # F: Snapshot
    ax6=fig.add_subplot(gs[2,2]); sty(ax6,"Today's Key Metrics"); ax6.axis('off')
    lat=df.iloc[-1]
    rows=[
        ("Close",    f"{float(lat['Close']):,.0f} VNĐ"),
        ("RSI 14",   f"{float(lat['RSI14']):.1f}"),
        ("RSI 7",    f"{float(lat['RSI7']):.1f}"),
        ("MACD Hist",f"{float(lat['MACDhist']):.4f}"),
        ("BB %B",    f"{float(lat['BBpct'])*100:.1f}%"),
        ("Stoch %K", f"{float(lat['StochK']):.1f}"),
        ("ATR",      f"{float(lat['ATR']):.2f}"),
        ("Vol Ratio",f"{float(lat['VolR']):.2f}x"),
        ("Ret 5d",   f"{float(lat['Ret_5d'])*100:+.2f}%"),
        ("Ret 20d",  f"{float(lat['Ret_20d'])*100:+.2f}%"),
    ]
    for i,(lbl,val) in enumerate(rows):
        yp=0.93-i*0.092
        ax6.text(0.04,yp,lbl,transform=ax6.transAxes,color=GY,fontsize=8.5)
        vc=GR if val.startswith('+') else (RD if val.startswith('-') else WH)
        ax6.text(0.97,yp,val,transform=ax6.transAxes,color=vc,
                 fontsize=8.5,ha='right',fontweight='bold')

    plt.savefig('trading_dashboard.png',dpi=150,bbox_inches='tight',facecolor=DK)
    print("[+] Dashboard saved -> trading_dashboard.png")
    plt.close(fig) # Dong figure giai phong bo nho

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main(retrain=True):
    df,X,y,feat=load_data()
    if retrain or not os.path.exists(MODEL_FILE):
        model,scaler,vp,vi,vs=train(df,X,y,feat)
        # luu lai metrics de in
        from sklearn.metrics import roc_auc_score,brier_score_loss,log_loss,accuracy_score
        from sklearn.calibration import CalibratedClassifierCV
        scaler2=joblib.load(SCALER_FILE)
        X_val_sc2=scaler2.transform(X.iloc[list(TimeSeriesSplit(n_splits=5).split(X))[-1][1]])
        y_val2=y.iloc[list(TimeSeriesSplit(n_splits=5).split(X))[-1][1]]
        prob2=model.predict_proba(X_val_sc2)[:,1]
        auc=roc_auc_score(y_val2,prob2)
        brier=brier_score_loss(y_val2,prob2)
        ll=log_loss(y_val2,prob2)
        
        # Calculate accuracy with natural 0.5 threshold
        opt_thresh=0.5
        pred2=(prob2>=opt_thresh).astype(int)
        acc=accuracy_score(y_val2,pred2)
        val_idx2=list(TimeSeriesSplit(n_splits=5).split(X))[-1][1]
        vc2=df['Close'].iloc[val_idx2]; dr2=vc2.pct_change().fillna(0)
        sig2=pd.Series(np.where(prob2>=THRESHOLD_BUY,1,np.where(prob2<=THRESHOLD_SELL,-1,0)),index=vc2.index)
        sr2=(1+sig2.shift(1).fillna(0)*dr2).cumprod()
        mr2=(1+dr2).cumprod()
        sharpe2=dr2[sig2.shift(1)!=0].mean()/(dr2[sig2.shift(1)!=0].std()+1e-9)*np.sqrt(252)
        dd2=(sr2/sr2.cummax()-1).min()
        strat_ret=(sr2.iloc[-1]-1)*100; mkt_ret=(mr2.iloc[-1]-1)*100
        sharpe=sharpe2; maxdd=dd2*100
    else:
        console.print("[*] Loading saved model ..."); model=joblib.load(MODEL_FILE); scaler=joblib.load(SCALER_FILE)
        vp=vi=vs=None
        auc=brier=ll=strat_ret=mkt_ret=sharpe=maxdd=acc=0.0

    p_up,p_dn,dec,tag,dec_color,conf,today,nday=predict(df,feat,model,scaler)
    print_report(df,feat,p_up,p_dn,dec,tag,dec_color,conf,today,nday,
                 auc,brier,ll,strat_ret,mkt_ret,sharpe,maxdd,acc)
    plot(df,p_up,p_dn,dec,feat,vp,vi,vs)

if __name__=="__main__":
    main(retrain=True)
