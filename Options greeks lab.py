
"""
Options Greeks Lab — The Mountain Path: World of Finance
Prof. V. Ravichandran | 28+ Years Corporate Finance & Banking | 10+ Years Academic Excellence

Black-Scholes-Merton Framework · 3D Greek Surfaces · P&L Simulation · Sensitivity Analysis

Design: Mountain Path Master Dark Theme (from Portfolio Risk Dashboard)
"""

import streamlit as st
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Options Greeks Lab | Mountain Path",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# MASTER DESIGN — Mountain Path Dark Theme
# (Exact port from Portfolio Risk Dashboard)
# ============================================================================
COLORS = {
    'dark_blue':    '#003366',
    'medium_blue':  '#004d80',
    'accent_gold':  '#FFD700',
    'light_blue':   '#ADD8E6',
    'bg_dark':      '#0a1628',
    'card_bg':      '#112240',
    'text_primary': '#e6f1ff',
    'text_secondary':'#8892b0',
    'text_dark':    '#1a1a2e',
    'success':      '#28a745',
    'danger':       '#dc3545',
}

BRANDING = {
    'name':        'The Mountain Path - World of Finance',
    'instructor':  'Prof. V. Ravichandran',
    'credentials': '28+ Years Corporate Finance & Banking | 10+ Years Academic Excellence',
    'icon':        '🏔️',
    'linkedin':    'https://www.linkedin.com/in/trichyravis',
    'github':      'https://github.com/trichyravis',
}

def apply_styles():
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+Pro:wght@300;400;600;700&display=swap');

        .stApp {{
            background: linear-gradient(135deg, #1a2332 0%, #243447 50%, #2a3f5f 100%);
        }}
        .main {{ color: {COLORS['text_primary']} !important; }}
        .main *, .main p, .main span, .main div, .main li, .main label {{
            color: {COLORS['text_primary']} !important;
        }}
        .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {{
            color: {COLORS['accent_gold']} !important;
            font-family: 'Playfair Display', serif;
        }}

        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {COLORS['bg_dark']} 0%, {COLORS['dark_blue']} 100%);
            border-right: 1px solid rgba(255,215,0,0.2);
        }}
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span {{
            color: {COLORS['text_primary']} !important;
        }}
        section[data-testid="stSidebar"] input {{
            color: {COLORS['text_dark']} !important;
            background-color: #ffffff !important;
        }}

        .header-container {{
            background: linear-gradient(135deg, {COLORS['dark_blue']}, {COLORS['medium_blue']});
            border: 2px solid {COLORS['accent_gold']};
            border-radius: 12px;
            padding: 1.5rem 2rem;
            margin-bottom: 1.5rem;
            text-align: center;
        }}
        .header-container h1 {{
            font-family: 'Playfair Display', serif;
            color: {COLORS['accent_gold']};
            margin: 0; font-size: 2rem;
        }}
        .header-container p {{
            color: {COLORS['text_primary']};
            font-family: 'Source Sans Pro', sans-serif;
            margin: 0.3rem 0 0; font-size: 0.9rem;
        }}

        .metric-card {{
            background: {COLORS['card_bg']};
            border: 1px solid rgba(255,215,0,0.3);
            border-radius: 10px;
            padding: 1.2rem;
            text-align: center;
            margin-bottom: 0.8rem;
        }}
        .metric-card .label {{
            color: {COLORS['text_secondary']};
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .metric-card .value {{
            color: {COLORS['accent_gold']};
            font-size: 1.6rem;
            font-weight: 700;
            font-family: 'Playfair Display', serif;
            margin-top: 0.3rem;
        }}
        .metric-card .sub {{
            color: {COLORS['text_secondary']};
            font-size: 0.78rem;
            margin-top: 0.3rem;
        }}

        .info-box {{
            background: rgba(0,51,102,0.5);
            border: 1px solid {COLORS['accent_gold']};
            border-radius: 8px;
            padding: 1rem 1.5rem;
            color: {COLORS['text_primary']};
            margin: 0.8rem 0;
        }}

        .section-title {{
            font-family: 'Playfair Display', serif;
            color: {COLORS['accent_gold']};
            font-size: 1.3rem;
            border-bottom: 2px solid rgba(255,215,0,0.3);
            padding-bottom: 0.5rem;
            margin: 1.5rem 0 1rem;
        }}

        .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
        .stTabs [data-baseweb="tab"] {{
            background: {COLORS['card_bg']};
            border: 1px solid rgba(255,215,0,0.3);
            border-radius: 8px;
            color: {COLORS['text_primary']};
            padding: 0.5rem 1rem;
        }}
        .stTabs [aria-selected="true"] {{
            background: {COLORS['dark_blue']};
            border: 2px solid {COLORS['accent_gold']};
            color: {COLORS['accent_gold']};
        }}

        .streamlit-expanderHeader {{
            background: {COLORS['card_bg']} !important;
            border: 1px solid {COLORS['accent_gold']} !important;
            border-radius: 8px !important;
        }}
        .streamlit-expanderHeader p,
        .streamlit-expanderHeader span,
        .streamlit-expanderHeader div {{
            color: {COLORS['accent_gold']} !important;
            font-weight: 600 !important;
        }}
        .streamlit-expanderContent {{
            background: rgba(17,34,64,0.5) !important;
            border: 1px solid rgba(255,215,0,0.2) !important;
        }}

        .stButton > button {{
            background: linear-gradient(135deg, {COLORS['medium_blue']}, {COLORS['dark_blue']}) !important;
            color: {COLORS['accent_gold']} !important;
            border: 2px solid {COLORS['accent_gold']} !important;
            border-radius: 8px !important;
            padding: 0.5rem 1rem !important;
            font-weight: 600 !important;
            width: 100% !important;
            transition: all 0.3s ease !important;
        }}
        .stButton > button:hover {{
            background: linear-gradient(135deg, {COLORS['accent_gold']}, #d4af37) !important;
            color: {COLORS['dark_blue']} !important;
            box-shadow: 0 4px 12px rgba(255,215,0,0.4) !important;
            transform: translateY(-2px) !important;
        }}

        .stAlert {{ background-color: rgba(255,255,255,0.95) !important; }}
        .stAlert p, .stAlert span, .stAlert div {{ color: {COLORS['text_dark']} !important; }}

        footer {{ visibility: hidden; }}
    </style>
    """, unsafe_allow_html=True)

apply_styles()

# ============================================================================
# COMPONENT HELPERS
# ============================================================================
def header_container(title, subtitle=None, description=None):
    s_html = f'<p style="font-size:1rem;color:{COLORS["accent_gold"]};font-weight:600;margin:0.5rem 0;">{subtitle}</p>' if subtitle else ""
    d_html = f'<p style="font-size:0.85rem;color:{COLORS["text_primary"]};margin:0.3rem 0;">{description}</p>' if description else ""
    st.markdown(f"""
    <div class="header-container">
        <h1>{BRANDING['icon']} {title}</h1>
        {s_html}{d_html}
        <p>{BRANDING['name']}</p>
        <p style="font-size:0.8rem;color:{COLORS['text_secondary']};">
            {BRANDING['instructor']} | {BRANDING['credentials']}
        </p>
    </div>""", unsafe_allow_html=True)

def metric_card(label, value, sub=None):
    s_html = f'<div class="sub">{sub}</div>' if sub else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {s_html}
    </div>""", unsafe_allow_html=True)

def section_title(t):
    st.markdown(f'<div class="section-title">{t}</div>', unsafe_allow_html=True)

def info_box(content, title=None):
    t_html = f"<h4 style='color:{COLORS['accent_gold']};margin-top:0;'>{title}</h4>" if title else ""
    st.markdown(f'<div class="info-box">{t_html}{content}</div>', unsafe_allow_html=True)

def sidebar_label(text):
    st.sidebar.markdown(f"<p style='color:{COLORS['accent_gold']};font-weight:700;'>{text}</p>", unsafe_allow_html=True)

def footer():
    st.divider()
    st.markdown(f"""
    <div style="text-align:center;padding:1.5rem;">
        <p style="color:{COLORS['accent_gold']};font-family:'Playfair Display',serif;
                  font-weight:700;font-size:1.1rem;margin-bottom:0.5rem;">
            {BRANDING['icon']} {BRANDING['name']}
        </p>
        <p style="color:{COLORS['text_secondary']};font-size:0.85rem;margin:0.3rem 0;">
            {BRANDING['instructor']} | {BRANDING['credentials']}
        </p>
        <div style="margin-top:1rem;padding-top:1rem;border-top:1px solid rgba(255,215,0,0.3);">
            <p style="color:{COLORS['text_primary']};font-size:0.9rem;margin:0.5rem 0;">
                <a href="{BRANDING['linkedin']}" target="_blank"
                   style="color:{COLORS['accent_gold']};text-decoration:none;margin:0 1rem;">
                    🔗 LinkedIn
                </a>
                <a href="{BRANDING['github']}" target="_blank"
                   style="color:{COLORS['accent_gold']};text-decoration:none;margin:0 1rem;">
                    💻 GitHub
                </a>
            </p>
        </div>
    </div>""", unsafe_allow_html=True)

# ============================================================================
# BSM ENGINE
# ============================================================================
def bsm(S, K, T, r, sigma, opt='call'):
    if T <= 0 or sigma <= 0:
        return dict(price=0, delta=0, gamma=0, vega=0, theta=0, rho=0, d1=0, d2=0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    phi = norm.pdf(d1)
    if opt == 'call':
        price = S*norm.cdf(d1)  - K*np.exp(-r*T)*norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (-(S*phi*sigma)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2))/365
        rho   = K*T*np.exp(-r*T)*norm.cdf(d2)/100
    else:
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        theta = (-(S*phi*sigma)/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2))/365
        rho   = -K*T*np.exp(-r*T)*norm.cdf(-d2)/100
    gamma = phi / (S*sigma*np.sqrt(T))
    vega  = S*phi*np.sqrt(T)/100
    return dict(price=price, delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho, d1=d1, d2=d2)

def bsm_vec(S_arr, K, T_arr, r, sigma, opt='call'):
    out = np.zeros(S_arr.shape)
    mask = (T_arr > 0) & (sigma > 0)
    S_m, T_m = S_arr[mask], T_arr[mask]
    d1 = (np.log(S_m/K) + (r + 0.5*sigma**2)*T_m) / (sigma*np.sqrt(T_m))
    d2 = d1 - sigma*np.sqrt(T_m)
    if opt == 'call':
        out[mask] = S_m*norm.cdf(d1) - K*np.exp(-r*T_m)*norm.cdf(d2)
    else:
        out[mask] = K*np.exp(-r*T_m)*norm.cdf(-d2) - S_m*norm.cdf(-d1)
    return out

def greeks_vec(S_arr, K, T_arr, r, sigma, greek):
    out = np.zeros(S_arr.shape)
    mask = (T_arr > 0) & (sigma > 0)
    S_m, T_m = S_arr[mask], T_arr[mask]
    d1 = (np.log(S_m/K) + (r + 0.5*sigma**2)*T_m) / (sigma*np.sqrt(T_m))
    d2 = d1 - sigma*np.sqrt(T_m)
    phi = norm.pdf(d1)
    g = greek.lower()
    if   g == 'delta': out[mask] = norm.cdf(d1)
    elif g == 'gamma': out[mask] = phi/(S_m*sigma*np.sqrt(T_m))
    elif g == 'vega':  out[mask] = S_m*phi*np.sqrt(T_m)/100
    elif g == 'theta': out[mask] = (-(S_m*phi*sigma)/(2*np.sqrt(T_m)) - r*K*np.exp(-r*T_m)*norm.cdf(d2))/365
    elif g == 'rho':   out[mask] = K*T_m*np.exp(-r*T_m)*norm.cdf(d2)/100
    return out

# ============================================================================
# PLOTLY DARK LAYOUT HELPER
# ============================================================================
DARK_LAYOUT = dict(
    paper_bgcolor='#0f1824',
    plot_bgcolor='#0f1824',
    font=dict(color=COLORS['text_primary'], family='Source Sans Pro'),
    title_font=dict(color=COLORS['accent_gold'], family='Playfair Display', size=16),
    xaxis=dict(gridcolor='rgba(255,255,255,0.1)', color=COLORS['text_secondary']),
    yaxis=dict(gridcolor='rgba(255,255,255,0.1)', color=COLORS['text_secondary']),
    legend=dict(bgcolor='rgba(17,34,64,0.8)', bordercolor=COLORS['accent_gold'],
                borderwidth=1, font=dict(color=COLORS['text_primary'])),
)

# ============================================================================
# HEADER
# ============================================================================
header_container(
    title="Options Pricing & Greeks Lab",
    subtitle="Black-Scholes-Merton Framework",
    description="Interactive Greek Profiles · 3D Surfaces · P&L Simulation · Sensitivity Analysis · Theory"
)

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.markdown(f"""
<div style="text-align:center;padding:1.2rem;background:rgba(255,215,0,0.08);
     border-radius:10px;margin-bottom:1.5rem;border:2px solid {COLORS['accent_gold']};">
    <h3 style="color:{COLORS['accent_gold']};margin:0;">🏔️ OPTIONS LAB</h3>
    <p style="color:{COLORS['text_secondary']};font-size:0.75rem;margin:5px 0 0;">
        BSM Live Calculator</p>
</div>
""", unsafe_allow_html=True)

sidebar_label("📊 Option Parameters")
S     = st.sidebar.number_input("Spot Price (S) ₹",     50.0, 100000.0, 22500.0, 50.0)
K     = st.sidebar.number_input("Strike Price (K) ₹",   50.0, 100000.0, 23000.0, 50.0)
T_days= st.sidebar.number_input("Time to Expiry (days)",  1,     730,      30,      1)
sigma = st.sidebar.slider("Volatility σ (%)", 5.0, 80.0, 18.0, 0.5) / 100
r     = st.sidebar.slider("Risk-Free Rate r (%)", 0.0, 20.0, 6.5, 0.1) / 100
opt_type = st.sidebar.radio("Option Type", ["call", "put"])

sidebar_label("📈 3D Surface")
surface_greek = st.sidebar.selectbox("Greek for Surface", ["Delta","Gamma","Vega","Theta","Rho","Price"])

sidebar_label("💰 P&L Settings")
lot_size      = st.sidebar.number_input("Lot Size (contracts)", 1, 10000, 50)
purchase_price= st.sidebar.number_input("Purchase Price ₹ (0=current)", 0.0, 10000.0, 0.0, 0.5)

T = T_days / 365.0
res = bsm(S, K, T, r, sigma, opt_type)
price = res['price']
purch = purchase_price if purchase_price > 0 else price
moneyness = "ITM" if (opt_type=='call' and S>K) or (opt_type=='put' and S<K) else ("ATM" if S==K else "OTM")

# ============================================================================
# LIVE METRICS ROW
# ============================================================================
section_title("⚡ Live BSM Metrics")
cols = st.columns(8)
metrics_data = [
    ("Option Price", f"₹{price:.2f}",    moneyness),
    ("Delta Δ",      f"{res['delta']:.4f}", "Δ per ₹1"),
    ("Gamma Γ",      f"{res['gamma']:.6f}", "ΔΔ/ΔS"),
    ("Vega ν",       f"{res['vega']:.4f}",  "per 1% vol"),
    ("Theta Θ",      f"{res['theta']:.4f}", "per day"),
    ("Rho ρ",        f"{res['rho']:.4f}",   "per 1% r"),
    ("d₁",           f"{res['d1']:.4f}",    ""),
    ("d₂",           f"{res['d2']:.4f}",    ""),
]
for col, (lbl, val, sub) in zip(cols, metrics_data):
    with col:
        metric_card(lbl, val, sub)

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Greek Profiles",
    "🌐 3D Surfaces",
    "💰 P&L Simulation",
    "📊 Sensitivity Table",
    "📚 Theory & Formulae"
])

# ──────────────────────── TAB 1: GREEK PROFILES ───────────────────────────
with tab1:
    section_title("📈 Greek Sensitivity Profiles")

    info_box(
        f"Vertical gold dashed line = current spot (₹{S:,.0f}). "
        f"Grey dashed line = strike (₹{K:,.0f}). "
        f"All Greeks computed live from BSM with T={T_days}d, σ={sigma*100:.1f}%, r={r*100:.1f}%.",
        title="Reading the Charts"
    )

    S_range = np.linspace(S * 0.5, S * 1.5, 200)
    T_fixed = np.full_like(S_range, T)
    greek_fns = {
        'Delta': greeks_vec(S_range, K, T_fixed, r, sigma, 'delta'),
        'Gamma': greeks_vec(S_range, K, T_fixed, r, sigma, 'gamma'),
        'Vega':  greeks_vec(S_range, K, T_fixed, r, sigma, 'vega'),
        'Theta': greeks_vec(S_range, K, T_fixed, r, sigma, 'theta'),
        'Rho':   greeks_vec(S_range, K, T_fixed, r, sigma, 'rho'),
        'Price': bsm_vec(S_range, K, T_fixed, r, sigma, opt_type),
    }
    line_colors = [COLORS['accent_gold'], COLORS['light_blue'], COLORS['success'],
                   COLORS['danger'], '#9b59b6', COLORS['medium_blue']]

    fig = make_subplots(rows=2, cols=3,
        subplot_titles=[f"{g} vs Spot" for g in greek_fns],
        vertical_spacing=0.14, horizontal_spacing=0.08)

    for idx, (greek, vals) in enumerate(greek_fns.items()):
        r_i, c_i = divmod(idx, 3)
        color = line_colors[idx]
        fig.add_trace(go.Scatter(x=S_range, y=vals, name=greek,
            line=dict(color=color, width=2.5)), row=r_i+1, col=c_i+1)
        fig.add_vline(x=S, line_dash="dash", line_color=COLORS['accent_gold'], line_width=1.5,
                      row=r_i+1, col=c_i+1)
        fig.add_vline(x=K, line_dash="dash", line_color=COLORS['text_secondary'], line_width=1,
                      row=r_i+1, col=c_i+1)

    fig.update_layout(height=580, showlegend=False,
        title=dict(text=f"BSM Greek Profiles — {opt_type.title()} | K=₹{K:,.0f} | T={T_days}d | σ={sigma*100:.1f}%",
                   font=dict(color=COLORS['accent_gold'], family='Playfair Display', size=15)),
        paper_bgcolor='#0f1824', plot_bgcolor='#0f1824',
        font=dict(color=COLORS['text_primary']))
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.08)', color=COLORS['text_secondary'])
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.08)', color=COLORS['text_secondary'])
    for ann in fig.layout.annotations:
        ann.font.color = COLORS['accent_gold']
        ann.font.family = 'Source Sans Pro'
    st.plotly_chart(fig, use_container_width=True)

    # Vol & Time profiles
    section_title("📊 Greek vs Volatility & Time")
    col1, col2 = st.columns(2)
    vol_range  = np.linspace(0.05, 0.80, 200)
    time_range = np.linspace(1/365, 2.0, 200)

    with col1:
        fig2 = go.Figure()
        for greek, color in [('Delta',COLORS['accent_gold']),('Vega',COLORS['light_blue']),('Gamma',COLORS['success'])]:
            T_fix2 = np.full_like(vol_range, T)
            vals2 = greeks_vec(np.full_like(vol_range, S), K, T_fix2, r, vol_range, greek.lower())
            fig2.add_trace(go.Scatter(x=vol_range*100, y=vals2, name=greek, line=dict(color=color, width=2.5)))
        fig2.add_vline(x=sigma*100, line_dash="dash", line_color=COLORS['accent_gold'], line_width=1.5)
        fig2.update_layout(height=350, title=dict(text="Greeks vs Volatility", font=dict(color=COLORS['accent_gold'], size=14)),
            paper_bgcolor='#0f1824', plot_bgcolor='#0f1824', font=dict(color=COLORS['text_primary']),
            legend=dict(bgcolor='rgba(17,34,64,0.8)', bordercolor=COLORS['accent_gold'], borderwidth=1),
            xaxis=dict(title="Volatility (%)", gridcolor='rgba(255,255,255,0.08)', color=COLORS['text_secondary']),
            yaxis=dict(gridcolor='rgba(255,255,255,0.08)', color=COLORS['text_secondary']))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = go.Figure()
        for greek, color in [('Theta',COLORS['danger']),('Vega',COLORS['light_blue']),('Rho',COLORS['success'])]:
            T_arr3 = time_range
            vals3 = greeks_vec(np.full_like(time_range, S), K, T_arr3, r, sigma, greek.lower())
            fig3.add_trace(go.Scatter(x=time_range*365, y=vals3, name=greek, line=dict(color=color, width=2.5)))
        fig3.add_vline(x=T_days, line_dash="dash", line_color=COLORS['accent_gold'], line_width=1.5)
        fig3.update_layout(height=350, title=dict(text="Greeks vs Time to Expiry", font=dict(color=COLORS['accent_gold'], size=14)),
            paper_bgcolor='#0f1824', plot_bgcolor='#0f1824', font=dict(color=COLORS['text_primary']),
            legend=dict(bgcolor='rgba(17,34,64,0.8)', bordercolor=COLORS['accent_gold'], borderwidth=1),
            xaxis=dict(title="Days to Expiry", gridcolor='rgba(255,255,255,0.08)', color=COLORS['text_secondary']),
            yaxis=dict(gridcolor='rgba(255,255,255,0.08)', color=COLORS['text_secondary']))
        st.plotly_chart(fig3, use_container_width=True)

# ──────────────────────── TAB 2: 3D SURFACES ──────────────────────────────
with tab2:
    section_title("🌐 3D Greek Surface")

    col_ctrl, col_chart = st.columns([1, 3])
    with col_ctrl:
        x_var = st.selectbox("X Axis", ["Spot Price","Time (days)","Volatility (%)"], key="x3d")
        y_var = st.selectbox("Y Axis", ["Volatility (%)","Spot Price","Time (days)"], key="y3d")

        info_box(
            f"<b>Viewing:</b> {surface_greek} surface<br>"
            f"<b>X:</b> {x_var}<br><b>Y:</b> {y_var}<br>"
            f"<b>Fixed:</b> K=₹{K:,.0f}, r={r*100:.1f}%",
            title="Surface Config"
        )

    with col_chart:
        n = 40
        def make_axis(var_name, S, T_days, sigma):
            if var_name == "Spot Price":       return np.linspace(S*0.6, S*1.4, n)
            elif var_name == "Time (days)":    return np.linspace(1, max(T_days*2,60), n)
            else:                              return np.linspace(0.05, 0.70, n)

        x_arr = make_axis(x_var, S, T_days, sigma)
        y_arr = make_axis(y_var, S, T_days, sigma)
        XX, YY = np.meshgrid(x_arr, y_arr)

        def get_val(x_val, y_val, x_var, y_var, S, T_days, sigma):
            s_v = x_val if x_var=="Spot Price" else (y_val if y_var=="Spot Price" else S)
            t_v = x_val/365 if x_var=="Time (days)" else (y_val/365 if y_var=="Time (days)" else T)
            v_v = x_val/100 if x_var=="Volatility (%)" else (y_val/100 if y_var=="Volatility (%)" else sigma)
            t_v = max(t_v, 1/365)
            return greeks_vec(np.array([s_v]), K, np.array([t_v]), r, v_v, surface_greek.lower())[0] \
                   if surface_greek != 'Price' else bsm_vec(np.array([s_v]), K, np.array([t_v]), r, v_v, opt_type)[0]

        ZZ = np.vectorize(get_val)(XX, YY, x_var, y_var, S, T_days, sigma)

        colorscale = [[0,'#003366'],[0.25,'#004d80'],[0.5,'#ADD8E6'],[0.75,'#FFD700'],[1,'#ffffff']]
        fig_s = go.Figure(go.Surface(x=XX, y=YY, z=ZZ, colorscale=colorscale, opacity=0.92,
            contours=dict(z=dict(show=True, usecolormap=True, project_z=True))))
        fig_s.update_layout(height=560,
            title=dict(text=f"{surface_greek} Surface — {opt_type.title()} Option",
                       font=dict(color=COLORS['accent_gold'], family='Playfair Display', size=15)),
            scene=dict(
                xaxis=dict(title=x_var, gridcolor='rgba(255,255,255,0.15)', color=COLORS['text_secondary']),
                yaxis=dict(title=y_var, gridcolor='rgba(255,255,255,0.15)', color=COLORS['text_secondary']),
                zaxis=dict(title=surface_greek, gridcolor='rgba(255,255,255,0.15)', color=COLORS['text_secondary']),
                bgcolor='#0a1628'),
            paper_bgcolor='#0f1824', font=dict(color=COLORS['text_primary']))
        st.plotly_chart(fig_s, use_container_width=True)

# ──────────────────────── TAB 3: P&L SIMULATION ───────────────────────────
with tab3:
    section_title("💰 P&L Simulation")

    breakeven = K + purch if opt_type == 'call' else K - purch
    info_box(
        f"<b>Position:</b> Long {opt_type.title()} · Purchase Price: ₹{purch:.2f} · "
        f"Lot: {lot_size} contracts · <b>Breakeven: ₹{breakeven:,.2f}</b>",
        title="Position Summary"
    )

    S_sim = np.linspace(S * 0.6, S * 1.4, 300)
    horizons = [T, T*0.75, T*0.5, T*0.25, 1/365]
    horizon_labels = ["Expiry", "75% T", "50% T", "25% T", "1 Day"]
    h_colors = [COLORS['accent_gold'], COLORS['light_blue'], COLORS['success'],
                '#9b59b6', COLORS['danger']]

    fig_pnl = go.Figure()
    for h, lbl, hc in zip(horizons, horizon_labels, h_colors):
        T_h = np.full_like(S_sim, max(h, 1/365))
        prices_h = bsm_vec(S_sim, K, T_h, r, sigma, opt_type)
        pnl = (prices_h - purch) * lot_size
        fig_pnl.add_trace(go.Scatter(x=S_sim, y=pnl, name=lbl,
            line=dict(color=hc, width=2.5 if lbl=="Expiry" else 1.8,
                      dash='solid' if lbl=="Expiry" else 'dash')))

    fig_pnl.add_vline(x=S, line_dash="dash", line_color=COLORS['accent_gold'], line_width=1.5,
                      annotation_text=f"Spot ₹{S:,.0f}", annotation_font_color=COLORS['accent_gold'])
    fig_pnl.add_vline(x=K, line_dash="dot", line_color=COLORS['text_secondary'], line_width=1,
                      annotation_text=f"Strike ₹{K:,.0f}", annotation_font_color=COLORS['text_secondary'])
    fig_pnl.add_vline(x=breakeven, line_dash="dash", line_color=COLORS['success'], line_width=1,
                      annotation_text=f"BEP ₹{breakeven:,.0f}", annotation_font_color=COLORS['success'])
    fig_pnl.add_hline(y=0, line_color='rgba(255,255,255,0.3)', line_width=1)

    fig_pnl.update_layout(height=480,
        title=dict(text=f"P&L vs Spot — Long {opt_type.title()} | {lot_size} contracts",
                   font=dict(color=COLORS['accent_gold'], family='Playfair Display', size=15)),
        xaxis=dict(title="Spot Price (₹)", gridcolor='rgba(255,255,255,0.08)', color=COLORS['text_secondary']),
        yaxis=dict(title="P&L (₹)", gridcolor='rgba(255,255,255,0.08)', color=COLORS['text_secondary']),
        paper_bgcolor='#0f1824', plot_bgcolor='#0f1824', font=dict(color=COLORS['text_primary']),
        legend=dict(bgcolor='rgba(17,34,64,0.8)', bordercolor=COLORS['accent_gold'], borderwidth=1))
    st.plotly_chart(fig_pnl, use_container_width=True)

    # Scenario grid
    section_title("📊 P&L Scenario Grid")
    spot_pcts = [-30,-20,-15,-10,-5,0,5,10,15,20,30]
    vol_chgs  = [-8,-4,0,4,8,12,16]
    grid_rows = []
    for vc in vol_chgs:
        row = {"Vol Δ": f"{vc:+d}%"}
        for sp in spot_pcts:
            new_S = S * (1 + sp/100)
            new_sigma = max(sigma + vc/100, 0.01)
            p = bsm(new_S, K, T, r, new_sigma, opt_type)['price']
            pnl_v = (p - purch) * lot_size
            row[f"{sp:+d}%"] = f"₹{pnl_v:,.0f}"
        grid_rows.append(row)
    st.dataframe(pd.DataFrame(grid_rows).set_index("Vol Δ"), use_container_width=True)

# ──────────────────────── TAB 4: SENSITIVITY TABLE ────────────────────────
with tab4:
    section_title("📊 Greeks Sensitivity Table")
    info_box(
        "Bump-and-reprice: each parameter shifted ±1 unit independently. "
        "Shows absolute and % change in option price.",
        title="Methodology"
    )

    bumps = {
        'Spot +₹1':    bsm(S+1, K, T, r, sigma, opt_type)['price'] - price,
        'Spot -₹1':    bsm(S-1, K, T, r, sigma, opt_type)['price'] - price,
        'Vol +1%':     bsm(S, K, T, r, sigma+0.01, opt_type)['price'] - price,
        'Vol -1%':     bsm(S, K, T, r, max(sigma-0.01,0.001), opt_type)['price'] - price,
        'Time +1 day': bsm(S, K, T+1/365, r, sigma, opt_type)['price'] - price,
        'Time -1 day': bsm(S, K, max(T-1/365,0.001), r, sigma, opt_type)['price'] - price,
        'Rate +0.1%':  bsm(S, K, T, r+0.001, sigma, opt_type)['price'] - price,
        'Rate -0.1%':  bsm(S, K, T, r-0.001, sigma, opt_type)['price'] - price,
    }
    rows = [{"Bump": k,
             "Price Change (₹)": f"₹{v:.4f}",
             "% Change": f"{(v/price*100) if price>0 else 0:.3f}%",
             "Annualised": f"₹{v*252:.2f}" if 'day' in k.lower() else "—"}
            for k, v in bumps.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    section_title("📈 Current Greeks Summary")
    greek_table = [
        {"Greek","Symbol","Value","Interpretation","Range"},
    ]
    df_greeks = pd.DataFrame([
        {"Greek":"Delta","Symbol":"Δ","Value":f"{res['delta']:.4f}",
         "Interpretation":f"Price changes ₹{res['delta']:.4f} per ₹1 spot move","Range":"[0,1] calls / [-1,0] puts"},
        {"Greek":"Gamma","Symbol":"Γ","Value":f"{res['gamma']:.6f}",
         "Interpretation":f"Delta changes {res['gamma']:.6f} per ₹1 spot move","Range":"≥ 0 (same for calls & puts)"},
        {"Greek":"Vega", "Symbol":"ν","Value":f"{res['vega']:.4f}",
         "Interpretation":f"Price changes ₹{res['vega']:.4f} per 1% vol change","Range":"≥ 0 (long options)"},
        {"Greek":"Theta","Symbol":"Θ","Value":f"{res['theta']:.4f}",
         "Interpretation":f"Price loses ₹{abs(res['theta']):.4f} per calendar day","Range":"≤ 0 (long options)"},
        {"Greek":"Rho",  "Symbol":"ρ","Value":f"{res['rho']:.4f}",
         "Interpretation":f"Price changes ₹{res['rho']:.4f} per 1% rate change","Range":"≥0 calls / ≤0 puts"},
    ])
    st.dataframe(df_greeks, use_container_width=True, hide_index=True)

# ──────────────────────── TAB 5: THEORY & FORMULAE ────────────────────────
with tab5:
    section_title("📚 BSM Theory & Formulae")

    col1, col2 = st.columns(2)
    with col1:
        info_box("""
        <b>Call Price:</b> C = S·N(d₁) − K·e<sup>−rT</sup>·N(d₂)<br>
        <b>Put Price:</b>  P = K·e<sup>−rT</sup>·N(−d₂) − S·N(−d₁)<br><br>
        <b>d₁</b> = [ln(S/K) + (r + σ²/2)T] / (σ√T)<br>
        <b>d₂</b> = d₁ − σ√T<br><br>
        <b>Put-Call Parity:</b> C − P = S − K·e<sup>−rT</sup>
        """, title="BSM Pricing Formulae")

        info_box("""
        • Continuous trading; no transaction costs or taxes<br>
        • Constant σ and r over the option's life<br>
        • No dividends on the underlying<br>
        • European-style exercise (at expiry only)<br>
        • Log-normal asset returns<br>
        • No arbitrage — law of one price holds
        """, title="Model Assumptions")

    with col2:
        info_box("""
        Δ<sub>call</sub> = N(d₁) &nbsp;|&nbsp; Δ<sub>put</sub> = N(d₁) − 1<br>
        Γ = φ(d₁) / (S·σ·√T)<br>
        ν = S·φ(d₁)·√T / 100<br>
        Θ<sub>call</sub> = [−S·φ(d₁)·σ/(2√T) − rK·e<sup>−rT</sup>·N(d₂)] / 365<br>
        ρ<sub>call</sub> = K·T·e<sup>−rT</sup>·N(d₂) / 100<br><br>
        <b>Daily P&L (delta-neutral):</b><br>
        ≈ ½Γ·(ΔS)² + Θ·Δt + ν·Δσ
        """, title="Greeks Formulae")

        info_box("""
        <b>Theta-Gamma trade-off:</b><br>
        Long gamma / short theta: pays time decay, gains from large moves<br>
        Short gamma / long theta: collects theta, loses on large moves<br><br>
        <b>Breakeven daily move:</b> |ΔS*| = √(−2Θ/Γ)<br><br>
        <b>India VIX 30-day move:</b> VIX / √12 ≈ expected monthly move
        """, title="Key Insights")

    section_title("🐍 Python Implementation")
    st.code("""
import numpy as np
from scipy.stats import norm

def bsm(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    phi = norm.pdf(d1)
    if option_type == 'call':
        price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (-(S*phi*sigma)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)) / 365
        rho   = K*T*np.exp(-r*T)*norm.cdf(d2) / 100
    else:
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        theta = (-(S*phi*sigma)/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2)) / 365
        rho   = -K*T*np.exp(-r*T)*norm.cdf(-d2) / 100
    gamma = phi / (S * sigma * np.sqrt(T))
    vega  = S * phi * np.sqrt(T) / 100
    return dict(price=price, delta=delta, gamma=gamma,
                vega=vega, theta=theta, rho=rho, d1=d1, d2=d2)

# NIFTY 50 Example
res = bsm(S=22500, K=23000, T=30/365, r=0.065, sigma=0.18, option_type='call')
print(f"Price: ₹{res['price']:.2f}  |  Delta: {res['delta']:.4f}  |  Vega: {res['vega']:.4f}")
    """, language='python')

    section_title("📋 Excel Model Architecture")
    excel_df = pd.DataFrame([
        {"Sheet": "Inputs",            "Contents": "S, K, T, r, σ, option type, lot size (named cells)"},
        {"Sheet": "BSM Calculator",    "Contents": "d₁, d₂, N(d₁), N(d₂), Call price, Put price"},
        {"Sheet": "Greeks Dashboard",  "Contents": "All 5 Greeks with NORM.S.DIST formulas, live"},
        {"Sheet": "Payoff Diagram",    "Contents": "XY scatter of payoff at expiry vs spot range"},
        {"Sheet": "Sensitivity Tables","Contents": "Two-way data tables: Price vs (S×σ) and Greeks vs (S×T)"},
        {"Sheet": "P&L Scenarios",     "Contents": "P&L grid: spot moves vs vol changes"},
        {"Sheet": "Theta Decay",       "Contents": "Option price and theta vs days to expiry"},
    ])
    st.dataframe(excel_df, use_container_width=True, hide_index=True)

# ============================================================================
# FOOTER
# ============================================================================
footer()
