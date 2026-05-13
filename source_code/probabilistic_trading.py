"""
PROBABILISTIC FORECASTING FOR TRADING DECISIONS
Method: LightGBM + Isotonic Calibration (Probabilistic Classification)
"""
from joblib import compressor
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
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
FILE_PATH     = r"D:\project_QT_N17\source_code\FPT_data.csv"   # <-- duong dan file local
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
    """Chỉ đọc dữ liệu từ file CSV local để training."""
    print(f"[*] Doc file local: {FILE_PATH} ...")
    raw = pd.read_csv(FILE_PATH, parse_dates=['Date'])
    raw = raw.sort_values('Date').set_index('Date')
    raw.index.name = 'Date'
    raw.columns = [c.strip().capitalize() for c in raw.columns]
    raw = raw[['Open','High','Low','Close','Volume']].dropna()
    print(f"    -> {len(raw)} ngay du lieu (CSV), tu {raw.index[0].date()} den {raw.index[-1].date()}")

    df   = add_features(raw.copy())
    df.dropna(inplace=True)
    feat = [c for c in df.columns if c not in EXCLUDE]
    return df, df[feat], df['Target'], feat


def load_today():
    """
    Fetch dữ liệu ngày hôm nay (hoặc ngày giao dịch gần nhất)
    từ yfinance để dự đoán cho ngày kế tiếp.
    Trả về DataFrame đã có đầy đủ features (1 hàng cuối = hôm nay).
    """
    yf_ticker = f"{TICKER}.VN" if TICKER == "FPT" else TICKER
    # Lấy ~300 ngày để tính đủ các chỉ báo kỹ thuật cần lookback dài (SMA200, ...)
    fetch_start = (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d")
    try:
        raw = yf.download(yf_ticker, start=fetch_start, progress=False)
        if raw.empty:
            print("    -> Khong lay duoc du lieu hom nay tu yfinance.")
            return None
        raw.columns = raw.columns.get_level_values(0)
        raw = raw[['Open','High','Low','Close','Volume']].dropna()
        raw.index = raw.index.tz_localize(None)
        print(f"    -> Du lieu moi nhat: {raw.index[-1].date()}")
        df_today = add_features(raw.copy())
        df_today.dropna(inplace=True)
        return df_today
    except Exception as e:
        print(f"    -> Loi khi lay du lieu hom nay: {e}")
        return None

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
    # sharpe=dr[sig.shift(1)!=0].mean()/(dr[sig.shift(1)!=0].std()+1e-9)*np.sqrt(252)
    strategy_returns = sig.shift(1).fillna(0) * dr
    sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-9) * np.sqrt(252)
    dd=(sr/sr.cummax()-1).min()

    print(f"{'='*50}")
    print("  BACKTEST (validation period)")
    print(f"{'='*50}")
    print(f"  Strategy return : {(sr.iloc[-1]-1)*100:+.2f}%")
    print(f"  Market   return : {(mr.iloc[-1]-1)*100:+.2f}%")
    print(f"  Sharpe Ratio    : {sharpe:.3f}")
    print(f"  Max Drawdown    : {dd*100:.2f}%")
    print(f"{'='*50}\n")

    # So sánh với các phương pháp khác
    comparison_data = compare_methods(
        X_tr_sc, y_tr, X_val_sc, y_val,
        close_val=df['Close'].iloc[val_idx],
        main_prob=prob,
        main_auc=auc, main_acc=acc, main_brier=brier, main_ll=ll,
        main_sharpe=sharpe,
        main_strat_ret=(sr.iloc[-1]-1)*100,
        main_mkt_ret=(mr.iloc[-1]-1)*100,
        main_maxdd=dd*100,
    )

    joblib.dump(model, MODEL_FILE); joblib.dump(scaler, SCALER_FILE)
    print(f"[+] Saved {MODEL_FILE}, {SCALER_FILE}")
    return model, scaler, prob, val_idx, sig, comparison_data

# ── METHOD COMPARISON ───────────────────────────────────────────────────────
def compare_methods(X_tr_sc, y_tr, X_val_sc, y_val,
                    close_val,
                    main_prob, main_auc, main_acc, main_brier,
                    main_ll, main_sharpe, main_strat_ret,
                    main_mkt_ret, main_maxdd):
    """
    Huấn luyện 2 phương pháp baseline, tính metrics và in bảng so sánh đẹp.
    Phương pháp:
      1. LightGBM + Isotonic Calibration  (phương pháp đề xuất — đã huấn luyện)
      2. Quantile Regression proxy        (Gradient Boosting + Platt Scaling)
      3. Logistic Regression              + Hiệu chỉnh xác suất (Isotonic)
    """
    def _backtest(prob, close):
        dr  = close.pct_change().fillna(0)
        sig = pd.Series(
            np.where(prob >= THRESHOLD_BUY, 1,
                     np.where(prob <= THRESHOLD_SELL, -1, 0)),
            index=close.index
        )
        strat = sig.shift(1).fillna(0) * dr
        cum   = (1 + strat).cumprod()
        mkt   = (1 + dr).cumprod()
        return {
            'sharpe'   : strat.mean() / (strat.std() + 1e-9) * np.sqrt(252),
            'strat_ret': (cum.iloc[-1]  - 1) * 100,
            'mkt_ret'  : (mkt.iloc[-1]  - 1) * 100,
            'maxdd'    : (cum / cum.cummax() - 1).min() * 100,
        }

    def _eval(model):
        model.fit(X_tr_sc, y_tr)
        prob = model.predict_proba(X_val_sc)[:, 1]
        pred = (prob >= 0.5).astype(int)
        bt   = _backtest(prob, close_val)
        return {
            'auc'      : roc_auc_score(y_val, prob),
            'acc'      : accuracy_score(y_val, pred),
            'brier'    : brier_score_loss(y_val, prob),
            'll'       : log_loss(y_val, prob),
            'sharpe'   : bt['sharpe'],
            'strat_ret': bt['strat_ret'],
            'mkt_ret'  : bt['mkt_ret'],
            'maxdd'    : bt['maxdd'],
        }

    # Phương pháp 2: Quantile Regression proxy
    print("[*] Comparing — Quantile Regression (GradBoost + Platt Scaling) ...")
    r2 = _eval(CalibratedClassifierCV(
        GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, min_samples_leaf=20, random_state=42
        ), method='sigmoid', cv=3
    ))

    # Phương pháp 3: Logistic Regression + Isotonic Calibration
    print("[*] Comparing — Logistic Regression + Hiệu chỉnh xác suất (Isotonic) ...")
    r3 = _eval(CalibratedClassifierCV(
        LogisticRegression(
            max_iter=2000, class_weight='balanced',
            C=0.5, solver='lbfgs', random_state=42
        ), method='isotonic', cv=3
    ))

    # Kết quả phương pháp đề xuất (đã tính sẵn từ train())
    r1 = {
        'auc': main_auc, 'acc': main_acc, 'brier': main_brier, 'll': main_ll,
        'sharpe': main_sharpe, 'strat_ret': main_strat_ret,
        'mkt_ret': main_mkt_ret, 'maxdd': main_maxdd,
    }

    methods = [
        ("LightGBM + Isotonic Cal. (Đề xuất)", r1),
        ("Quantile Regression + Platt Scaling", r2),
        ("Logistic Reg. + Isotonic Cal.",       r3),
    ]

    keys_higher = ['auc', 'acc', 'sharpe', 'strat_ret', 'maxdd']
    keys_lower  = ['brier', 'll']   # maxdd âm → cao hơn = tốt hơn

    def is_best(key, val, all_res):
        vals = [r[key] for _, r in all_res]
        return abs(val - (max(vals) if key in keys_higher else min(vals))) < 1e-9

    # ── In bảng rich ──────────────────────────────────────────────────────────
    console.print()
    console.rule("[bold cyan]SO SÁNH PHƯƠNG PHÁP  ·  Method Comparison[/bold cyan]")

    tbl = Table(
        title="So sánh LightGBM + Isotonic  vs  Quantile Regression  vs  Logistic Regression",
        box=box.DOUBLE_EDGE,
        title_style="bold cyan",
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
    )
    tbl.add_column("Chỉ số",                     style="dim",       width=22)
    tbl.add_column("LightGBM\n+ Isotonic",       justify="center",  width=18)
    tbl.add_column("Quantile Reg.\n+ Platt",     justify="center",  width=18)
    tbl.add_column("Logistic Reg.\n+ Isotonic",  justify="center",  width=18)
    tbl.add_column("Tốt hơn nếu",               justify="center",  width=14, style="dim")

    def cell(key, val, fmt, higher_is_better=True):
        best = is_best(key, val, methods)
        color = "green" if best else "white"
        star  = "★ " if best else "  "
        return f"[{color}]{star}{fmt.format(val)}[/{color}]"

    rows = [
        ("ROC-AUC",        "auc",       "{:.4f}",  True,  "↑ Cao hơn"),
        ("Accuracy",       "acc",       "{:.2%}",   True,  "↑ Cao hơn"),
        ("Brier Score",    "brier",     "{:.4f}",  False, "↓ Thấp hơn"),
        ("Log Loss",       "ll",        "{:.4f}",  False, "↓ Thấp hơn"),
        ("Sharpe Ratio",   "sharpe",    "{:.3f}",  True,  "↑ Cao hơn"),
        ("Strategy Return","strat_ret", "{:+.2f}%", True, "↑ Cao hơn"),
        ("Market Return",  "mkt_ret",   "{:+.2f}%", True, "(tham khảo)"),
        ("Max Drawdown",   "maxdd",     "{:+.2f}%", True, "↑ Gần 0"),
    ]

    for label, key, fmt, higher, direction in rows:
        tbl.add_row(
            label,
            cell(key, r1[key], fmt, higher),
            cell(key, r2[key], fmt, higher),
            cell(key, r3[key], fmt, higher),
            direction,
        )

    console.print(tbl)
    console.print(
        "  [dim]★ = Tốt nhất trong nhóm  ·  "
        "Baseline: random ROC-AUC=0.50 | Brier=0.25 | LogLoss=0.693[/dim]"
    )
    console.rule(style="dim")
    console.print()

    # Trả về dict để vẽ biểu đồ
    return {
        'methods': [name for name, _ in methods],
        'results': [res  for _, res  in methods],
    }


# ── COMPARISON CHART ─────────────────────────────────────────────────────────
def plot_comparison(comparison_data):
    """
    Vẽ biểu đồ cột so sánh các chỉ số giữa 3 phương pháp.
    Lưu ra file method_comparison.png
    """
    if comparison_data is None:
        return

    DK='#ffffff'; PN='#ffffff'; EG='#d0d7de'
    BL='#0969da'; OR='#0969da'; PU='#0969da'
    GR='#1a7f37'; RD='#d1242f'; GY='#57606a'; WH='#24292f'
    COLORS = [BL, OR, PU]          # màu cho mỗi phương pháp
    STAR   = '★ '

    method_names = comparison_data['methods']
    results      = comparison_data['results']

    # Nhãn ngắn gọn hơn cho trục X
    short_names = [
        'LightGBM\n+ Isotonic',
        'Quantile Reg.\n+ Platt',
        'Logistic Reg.\n+ Isotonic',
    ]

    # ── Các panel cần vẽ ────────────────────────────────────────────────────
    panels = [
        # (title, key, higher_is_better, unit, y_label)
        ('ROC-AUC',        'auc',       True,  '',   'AUC'),
        ('Accuracy',       'acc',       True,  '%',  'Accuracy (%)'),
        ('Log Loss',       'll',        False, '',   'Loss (↓ better)'),
        ('Brier Score',    'brier',     False, '',   'Score (↓ better)'),
        ('Sharpe Ratio',   'sharpe',    True,  '',   'Sharpe'),
        ('Max Drawdown',   'maxdd',     True,  '%',  'Drawdown (%)'),
        ('Strategy Return','strat_ret', True,  '%',  'Return (%)'),
    ]

    ncols = 4
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 10), facecolor=DK)
    fig.suptitle(
        'So Sánh Phương Pháp  ·  Method Comparison',
        fontsize=16, fontweight='bold', color=WH, y=1.01
    )
    fig.patch.set_facecolor(DK)

    axes_flat = axes.flatten()

    x = np.arange(len(short_names))
    bar_w = 0.55

    for idx, (title, key, higher, unit, ylabel) in enumerate(panels):
        ax = axes_flat[idx]
        ax.set_facecolor(PN)
        for sp in ax.spines.values():
            sp.set_edgecolor(EG)
        ax.tick_params(colors=GY, labelsize=8)

        vals = [r[key] for r in results]
        best_val = max(vals) if higher else min(vals)

        # Chuyển acc sang %
        display_vals = [v * 100 if key == 'acc' else v for v in vals]
        display_best = best_val * 100 if key == 'acc' else best_val

        bar_colors = []
        for v in vals:
            if abs(v - best_val) < 1e-9:
                bar_colors.append(GR)
            else:
                bar_colors.append(GY)

        bars = ax.bar(x, display_vals, width=bar_w, color=bar_colors,
                      edgecolor=EG, linewidth=0.8, zorder=3)

        # Đường baseline tham khảo
        baselines = {
            'auc': (0.50, 'Random'),
            'brier': (0.25, 'Random'),
            'll': (0.693, 'Random'),
            'sharpe': (1.0, 'Good≥1'),
        }
        if key in baselines:
            bv, blabel = baselines[key]
            if key == 'acc':
                bv *= 100
            ax.axhline(bv, color=RD, ls='--', lw=1.0, alpha=0.8, zorder=2)
            ax.text(len(short_names) - 0.5, bv * 1.01, blabel,
                    color=RD, fontsize=7, va='bottom', ha='right')

        # Giá trị trên mỗi cột
        for bar, dv, orig_v, base_c in zip(bars, display_vals, vals, bar_colors):
            label_text = f'{dv:.2f}{unit}'
            if key == 'acc':
                label_text = f'{orig_v:.1%}'
            elif key in ('strat_ret', 'maxdd'):
                label_text = f'{dv:+.1f}%'
            is_best = abs(orig_v - best_val) < 1e-9
            txt_color = GR if is_best else WH
            prefix = STAR if is_best else ''
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.003 if dv >= 0 else -0.02) * abs(max(display_vals, key=abs) or 1),
                f'{prefix}{label_text}',
                ha='center', va='bottom' if dv >= 0 else 'top',
                color=txt_color, fontsize=8, fontweight='bold'
            )

        ax.set_title(title, color=WH, fontsize=10, pad=6)
        ax.set_ylabel(ylabel, color=GY, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, color=GY, fontsize=7.5)
        ax.yaxis.grid(True, color=EG, linewidth=0.6, alpha=0.6)
        ax.set_axisbelow(True)

    # Panel cuối: radar / spider chart tổng hợp
    ax_radar = axes_flat[len(panels)]
    ax_radar.set_facecolor(PN)
    for sp in ax_radar.spines.values():
        sp.set_edgecolor(EG)

    radar_keys   = ['auc', 'acc', 'sharpe']
    radar_labels = ['ROC-AUC', 'Accuracy', 'Sharpe']

    # Chuẩn hoá 0-1 cho mỗi chỉ số
    def normalise(vals_list, higher=True):
        mn, mx = min(vals_list), max(vals_list)
        if mx - mn < 1e-9:
            return [0.5] * len(vals_list)
        normed = [(v - mn) / (mx - mn) for v in vals_list]
        return normed if higher else [1 - n for n in normed]

    norm_data = []
    for rk in radar_keys:
        raw = [r[rk] for r in results]
        norm_data.append(normalise(raw, higher=True))

    # Bar-group chart thay radar vì matplotlib radar khó đọc
    ax_radar.set_title('Tổng hợp (chuẩn hoá)', color=WH, fontsize=10, pad=6)
    n_metrics = len(radar_keys)
    grp_w = 0.22
    offsets = np.linspace(-(n_metrics - 1) * grp_w / 2,
                          (n_metrics - 1) * grp_w / 2, n_metrics)
    legend_handles = []
    for mi, (m_name, m_color, m_norm) in enumerate(
            zip(short_names, COLORS, zip(*norm_data))):
        positions = x + offsets[mi]
        b = ax_radar.bar(positions, list(m_norm), width=grp_w * 0.85,
                         color=m_color, edgecolor=EG, linewidth=0.6,
                         label=m_name.replace('\n', ' '), alpha=0.85)
        legend_handles.append(b)

    ax_radar.set_xticks(x)
    ax_radar.set_xticklabels(radar_labels, color=GY, fontsize=8)
    ax_radar.set_ylabel('Normalised score', color=GY, fontsize=8)
    ax_radar.set_ylim(0, 1.15)
    ax_radar.yaxis.grid(True, color=EG, linewidth=0.6, alpha=0.6)
    ax_radar.set_axisbelow(True)
    ax_radar.tick_params(colors=GY, labelsize=8)
    ax_radar.legend(facecolor=PN, labelcolor=WH, fontsize=7, loc='upper right')

    # Ẩn ô thừa nếu có
    for i in range(len(panels) + 1, len(axes_flat)):
        axes_flat[i].set_visible(False)

    # Chú thích chung
    fig.text(
        0.5, -0.01,
        '★ = Tốt nhất trong nhóm  ·  Cột xanh lá = giá trị tốt nhất  ·  '
        'Baseline: ROC-AUC 0.50 | Brier 0.25 | LogLoss 0.693',
        ha='center', color=GY, fontsize=8
    )

    plt.tight_layout(pad=2.0)
    out = 'method_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=DK)
    print(f'[+] Comparison chart saved -> {out}')
    plt.close(fig)


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
                  f"Date: {today}", justify="center")
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
def plot(df, p_up, p_dn, dec, feat, val_prob=None, val_idx=None, val_sig=None, comparison_data=None, df_train=None):
    DK='#ffffff'; PN='#ffffff'; EG='#d0d7de'
    BL='#0969da'; OR='#0969da'; PU='#0969da'
    GR='#1a7f37'; RD='#d1242f'; GY='#57606a'; WH='#24292f'

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
    _df_cal = df_train if df_train is not None else df
    if val_prob is not None:
        frac_pos,mean_pred=calibration_curve(_df_cal['Target'].iloc[val_idx],val_prob,n_bins=10)
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

    # Vẽ biểu đồ so sánh phương pháp
    plot_comparison(comparison_data)

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main(retrain=True):
    df,X,y,feat=load_data()
    comparison_data = None
    if retrain or not os.path.exists(MODEL_FILE):
        model,scaler,vp,vi,vs,comparison_data=train(df,X,y,feat)
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
        strat_returns2 = sig2.shift(1).fillna(0) * dr2
        sharpe2 = strat_returns2.mean() / (strat_returns2.std() + 1e-9) * np.sqrt(252)
        dd2=(sr2/sr2.cummax()-1).min()
        strat_ret=(sr2.iloc[-1]-1)*100; mkt_ret=(mr2.iloc[-1]-1)*100
        sharpe=sharpe2; maxdd=dd2*100
    else:
        console.print("[*] Loading saved model ..."); model=joblib.load(MODEL_FILE); scaler=joblib.load(SCALER_FILE)
        vp=vi=vs=None
        auc=brier=ll=strat_ret=mkt_ret=sharpe=maxdd=acc=0.0

    # Lấy dữ liệu hôm nay từ yfinance để dự đoán
    df_today = load_today()
    df_pred  = df_today if df_today is not None else df  # fallback về CSV nếu lỗi

    p_up,p_dn,dec,tag,dec_color,conf,today,nday=predict(df_pred,feat,model,scaler)
    print_report(df_pred,feat,p_up,p_dn,dec,tag,dec_color,conf,today,nday,
                 auc,brier,ll,strat_ret,mkt_ret,sharpe,maxdd,acc)
    # df_pred = hom nay (de ve chart), df = training data (de ve calibration curve voi val_idx chinh xac)
    plot(df_pred,p_up,p_dn,dec,feat,vp,vi,vs,comparison_data=comparison_data,df_train=df)

if __name__=="__main__":
    main(retrain=True)
