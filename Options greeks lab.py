
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

        /* Expander — Streamlit 1.40+ selectors */
        [data-testid="stExpander"] {{
            background: {COLORS['card_bg']} !important;
            border: 1px solid rgba(255,215,0,0.35) !important;
            border-radius: 10px !important;
            margin-bottom: 0.5rem !important;
        }}
        [data-testid="stExpander"] summary {{
            background: {COLORS['card_bg']} !important;
            border-radius: 8px !important;
            padding: 0.7rem 1rem !important;
        }}
        [data-testid="stExpander"] summary p,
        [data-testid="stExpander"] summary span,
        [data-testid="stExpander"] summary div,
        [data-testid="stExpander"] summary {{
            color: {COLORS['accent_gold']} !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
        }}
        [data-testid="stExpander"] summary:hover {{
            background: rgba(0,51,102,0.6) !important;
        }}
        [data-testid="stExpanderDetails"] {{
            background: rgba(10,22,40,0.5) !important;
            border-top: 1px solid rgba(255,215,0,0.2) !important;
            padding: 0.5rem 0 !important;
        }}
        [data-testid="stExpanderDetails"] p,
        [data-testid="stExpanderDetails"] span,
        [data-testid="stExpanderDetails"] div,
        [data-testid="stExpanderDetails"] li {{
            color: {COLORS['text_primary']} !important;
        }}
        /* Legacy fallback */
        .streamlit-expanderHeader {{
            background: {COLORS['card_bg']} !important;
            border: 1px solid rgba(255,215,0,0.35) !important;
            border-radius: 8px !important;
        }}
        .streamlit-expanderHeader p,
        .streamlit-expanderHeader span,
        .streamlit-expanderHeader label,
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
purchase_price= st.sidebar.number_input("Purchase Price ₹", 0.0, 10000.0, 0.0, 0.5)
st.sidebar.markdown(f"""
<div style="background:rgba(255,215,0,0.07);border:1px solid rgba(255,215,0,0.25);
     border-radius:6px;padding:0.6rem 0.8rem;margin-top:-0.3rem;">
    <p style="color:{COLORS['accent_gold']};font-size:0.75rem;font-weight:700;margin:0 0 0.2rem;">
        📌 What is Purchase Price?</p>
    <p style="color:{COLORS['text_primary']};font-size:0.72rem;line-height:1.5;margin:0;">
        The price you <b>actually paid</b> for the option when you entered the trade.<br><br>
        <b>Enter 0</b> → Lab uses today's live BSM price as your cost (theoretical entry).<br><br>
        <b>Enter your actual premium</b> → P&L tab shows real profit/loss from your entry point.<br><br>
        <i>Example: Bought NIFTY call at ₹180 yesterday. Enter 180 here to see your actual P&L today.</i>
    </p>
</div>
""", unsafe_allow_html=True)

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
# ROW 1 TABS — About + Analytical Tools (6 tabs)
# ============================================================================
st.markdown(f"""
<div style="margin-bottom:-1rem;">
    <p style="color:{COLORS['text_secondary']};font-size:0.72rem;letter-spacing:1.5px;
              text-transform:uppercase;margin:0 0 0.3rem;font-weight:600;">
        🗺️ About &nbsp;·&nbsp; 📊 Analytics & Modelling
    </p>
</div>""", unsafe_allow_html=True)

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ About & User Guide",
    "📈 Greek Profiles",
    "🌐 3D Surfaces",
    "💰 P&L Simulation",
    "📊 Sensitivity Table",
    "📚 Theory & Formulae",
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 0: ABOUT & USER GUIDE (first tab in Row 1)
# ══════════════════════════════════════════════════════════════════════════
with tab0:
    # ── Platform Hero ──────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {COLORS['dark_blue']}, {COLORS['medium_blue']}, #1a3a5c);
                border: 2px solid {COLORS['accent_gold']}; border-radius: 14px;
                padding: 2.5rem; margin-bottom: 1.5rem; text-align: center;">
        <div style="font-size: 4rem; margin-bottom: 0.5rem;">🏔️</div>
        <h1 style="color:{COLORS['accent_gold']}; font-family:'Playfair Display',serif;
                   font-size: 2.2rem; margin: 0.3rem 0;">Options Greeks Lab</h1>
        <p style="color:{COLORS['light_blue']}; font-size: 1.1rem; margin: 0.5rem 0;">
            A comprehensive interactive platform for learning and applying the Black-Scholes-Merton framework
        </p>
        <p style="color:{COLORS['text_secondary']}; font-size: 0.85rem; margin-top: 1rem;">
            Designed for MBA · CFA · FRM · Financial Derivatives students and practitioners
        </p>
        <div style="margin-top: 1.5rem; display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
            <span style="background: rgba(255,215,0,0.12); border: 1px solid {COLORS['accent_gold']};
                         border-radius: 20px; padding: 0.4rem 1.2rem; color: {COLORS['accent_gold']}; font-size: 0.85rem;">
                ⚡ Live BSM Calculator
            </span>
            <span style="background: rgba(255,215,0,0.12); border: 1px solid {COLORS['accent_gold']};
                         border-radius: 20px; padding: 0.4rem 1.2rem; color: {COLORS['accent_gold']}; font-size: 0.85rem;">
                🌐 3D Greek Surfaces
            </span>
            <span style="background: rgba(255,215,0,0.12); border: 1px solid {COLORS['accent_gold']};
                         border-radius: 20px; padding: 0.4rem 1.2rem; color: {COLORS['accent_gold']}; font-size: 0.85rem;">
                💰 P&L Simulation
            </span>
            <span style="background: rgba(255,215,0,0.12); border: 1px solid {COLORS['accent_gold']};
                         border-radius: 20px; padding: 0.4rem 1.2rem; color: {COLORS['accent_gold']}; font-size: 0.85rem;">
                🎓 Education Hub
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── About the Platform ──────────────────────────────────────────────
    section_title("🏔️ About The Mountain Path — World of Finance")

    col_about1, col_about2 = st.columns([2, 1])
    with col_about1:
        st.markdown(f"""
        <div class="info-box">
            <h4 style='color:{COLORS['accent_gold']};margin-top:0;'>Our Mission</h4>
            <p style="color:{COLORS['text_primary']};line-height:1.8;margin:0 0 0.8rem;">
            The Mountain Path is an educational finance platform built to bridge the gap between
            textbook theory and practical application. Every tool, chart, and simulation on this
            platform is designed to make complex quantitative finance concepts intuitive and
            immediately actionable.</p>
            <p style="color:{COLORS['text_primary']};line-height:1.8;margin:0 0 0.8rem;">
            This <b>Options Greeks Lab</b> is the definitive interactive environment for understanding
            the Black-Scholes-Merton pricing framework — from basic call/put pricing to advanced
            3D sensitivity surfaces and real-world P&L simulation.</p>
            <p style="color:{COLORS['text_primary']};line-height:1.8;margin:0;">
            <b>Pedagogy:</b> Learn by doing. Every parameter change on the sidebar instantly updates
            all 7 tabs — so you see, in real time, how a change in volatility ripples through Delta,
            reshapes the 3D surface, and shifts your P&L breakeven.</p>
        </div>""", unsafe_allow_html=True)

    with col_about2:
        st.markdown(f"""
        <div style="background: {COLORS['card_bg']}; border: 1px solid {COLORS['accent_gold']};
                    border-radius: 10px; padding: 1.5rem; text-align: center;">
            <div style="font-size: 2.5rem;">👨‍🏫</div>
            <p style="color:{COLORS['accent_gold']}; font-family:'Playfair Display',serif;
                      font-weight:700; font-size:1rem; margin: 0.5rem 0;">
                Prof. V. Ravichandran
            </p>
            <p style="color:{COLORS['text_secondary']}; font-size:0.78rem; line-height:1.6;">
                28+ Years Corporate Finance & Banking Experience<br>
                10+ Years Academic Excellence<br><br>
                Visiting Faculty: BITS Pilani WILP · Christ University ·
                Goa Institute of Management · ICFAI Bangalore
            </p>
            <div style="margin-top:1rem; border-top:1px solid rgba(255,215,0,0.2); padding-top:1rem;">
                <a href="{BRANDING['linkedin']}" target="_blank"
                   style="color:{COLORS['accent_gold']};text-decoration:none;font-size:0.85rem;">
                    🔗 LinkedIn Profile
                </a><br>
                <a href="{BRANDING['github']}" target="_blank"
                   style="color:{COLORS['accent_gold']};text-decoration:none;font-size:0.85rem;">
                    💻 GitHub Projects
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Target Audience ─────────────────────────────────────────────────
    section_title("🎯 Who Is This Platform For?")
    col_a, col_b, col_c, col_d = st.columns(4)
    audiences = [
        ("🎓", "MBA Students", "Financial Derivatives, Investment Banking, Capital Markets modules"),
        ("📊", "CFA Candidates", "Level 1–3 derivatives, options pricing, risk management"),
        ("⚠️", "FRM Students", "Market risk, options risk measures, Greeks hedging"),
        ("💼", "Practitioners", "Traders, risk managers, quant analysts seeking intuition"),
    ]
    for col, (icon, title, desc) in zip([col_a, col_b, col_c, col_d], audiences):
        with col:
            st.markdown(f"""
            <div style="background:{COLORS['card_bg']};border:1px solid rgba(255,215,0,0.25);
                        border-radius:10px;padding:1.2rem;text-align:center;height:170px;">
                <div style="font-size:2rem;">{icon}</div>
                <p style="color:{COLORS['accent_gold']};font-weight:700;font-size:0.9rem;margin:0.4rem 0;">{title}</p>
                <p style="color:{COLORS['text_secondary']};font-size:0.75rem;line-height:1.5;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    # ── Features Overview ───────────────────────────────────────────────
    section_title("✨ Platform Features")
    features = [
        ("📈", "Greek Profiles", "Six interactive charts — Delta, Gamma, Vega, Theta, Rho, Price vs Spot. Plus Greeks vs Volatility and vs Time to Expiry."),
        ("🌐", "3D Surfaces", "Rotate and explore any Greek or price as a 3D surface. Choose any two axes from Spot, Vol, Time. Ideal for intuition building."),
        ("💰", "P&L Simulation", "Multi-horizon P&L curves at expiry, 75%, 50%, 25% and 1-day. Scenario grid: 7 vol changes × 11 spot moves."),
        ("📊", "Sensitivity Table", "Bump-and-reprice for all parameters. See exact ₹ and % impact of each market move on option price."),
        ("📚", "Theory & Code", "BSM formulae, assumptions, Python implementation and Excel model architecture in one place."),
        ("🗺️", "About & User Guide", "Step-by-step instructions, FAQs, and tips for getting the most from the platform."),
        ("🎓", "Education Hub", "Deep-dive educational content on every Greek — concept, formula, behaviour, trading implications, and worked examples."),
    ]
    for i in range(0, len(features), 3):
        cols_f = st.columns(3)
        for j, col in enumerate(cols_f):
            if i+j < len(features):
                icon, title, desc = features[i+j]
                with col:
                    st.markdown(f"""
                    <div style="background:{COLORS['card_bg']};border:1px solid rgba(255,215,0,0.2);
                                border-left:4px solid {COLORS['accent_gold']};border-radius:10px;
                                padding:1.2rem;margin:0.5rem 0;min-height:140px;">
                        <div style="font-size:1.8rem;margin-bottom:0.4rem;">{icon}</div>
                        <p style="color:{COLORS['accent_gold']};font-weight:700;font-size:0.95rem;margin:0 0 0.4rem;">{title}</p>
                        <p style="color:{COLORS['text_secondary']};font-size:0.8rem;line-height:1.6;margin:0;">{desc}</p>
                    </div>
                    """, unsafe_allow_html=True)

    # ── Step-by-Step User Guide ─────────────────────────────────────────
    section_title("📖 Step-by-Step User Guide")

    guide_steps = [
        ("Step 1 — Set Your Parameters (Sidebar)",
         """<p>The <b>left sidebar</b> is the control centre. All tabs update instantly when you change any value.</p>
         <ul>
           <li><b>Spot Price (S):</b> Current market price of the underlying (e.g., NIFTY 50 index level)</li>
           <li><b>Strike Price (K):</b> The exercise price of the option</li>
           <li><b>Time to Expiry (days):</b> Calendar days remaining until expiration</li>
           <li><b>Volatility σ:</b> Implied or historical volatility in % per annum (India VIX ≈ 13–18%)</li>
           <li><b>Risk-Free Rate r:</b> Use RBI repo rate (~6.5%) or 91-day T-bill yield</li>
           <li><b>Option Type:</b> Call (right to buy) or Put (right to sell)</li>
         </ul>"""),
        ("Step 2 — Read the Live Metrics Bar",
         """<p>The <b>8-metric row</b> at the top updates with every parameter change:</p>
         <ul>
           <li><b>Option Price:</b> BSM fair value in ₹ with moneyness tag (ITM/ATM/OTM)</li>
           <li><b>Δ, Γ, ν, Θ, ρ:</b> All five Greeks, live</li>
           <li><b>d₁ and d₂:</b> The BSM standardised distance measures used in all calculations</li>
         </ul>
         <p>💡 <i>Tip: Set S = K (spot equals strike) to see ATM Greeks — most important for exam questions.</i></p>"""),
        ("Step 3 — Explore Greek Profiles (Tab 1)",
         """<p>This tab shows all 6 Greek profiles against Spot Price simultaneously:</p>
         <ul>
           <li>The <b>gold dashed line</b> marks your current spot</li>
           <li>The <b>grey dashed line</b> marks the strike price</li>
           <li>Change T (time) to see how near-expiry collapse sharpens the Delta step function at ATM</li>
           <li>The bottom charts show Greeks vs Volatility and vs Time — critical for vega exposure</li>
         </ul>"""),
        ("Step 4 — Build 3D Intuition (Tab 2)",
         """<p>The 3D surface is the most powerful learning tool on the platform:</p>
         <ul>
           <li>Select a Greek from the sidebar (e.g., <b>Gamma</b>) and choose axes (e.g., Spot × Time)</li>
           <li><b>Rotate</b> the surface by clicking and dragging</li>
           <li>Gamma surface: notice the sharp ATM peak near expiry — this is "gamma risk"</li>
           <li>Theta surface: observe the accelerating decay cliff as expiry approaches</li>
           <li>Vega surface: flat far from ATM, maximum at ATM — shows where vol risk is concentrated</li>
         </ul>"""),
        ("Step 5 — Simulate P&L (Tab 3)",
         """<p>Enter a realistic trading scenario using the <b>💰 P&L Settings</b> in the sidebar:</p>
         <ul>
           <li><b>Lot Size:</b> Number of option contracts in your position. NIFTY lot = 50, BANK NIFTY = 15, FINNIFTY = 40.</li>
           <li><b>Purchase Price ₹</b> — this is the most important field to understand:</li>
         </ul>
         <div style="background:rgba(255,215,0,0.08);border-left:3px solid #FFD700;border-radius:6px;
                     padding:0.8rem 1.2rem;margin:0.5rem 0 0.8rem 1.5rem;">
           <p style="margin:0 0 0.5rem;"><b>What is Purchase Price?</b></p>
           <p style="margin:0 0 0.4rem;">It is the premium you <b>actually paid</b> when you entered your trade — i.e., your cost of buying the option.</p>
           <p style="margin:0 0 0.6rem;"><b>Enter 0 (default):</b> The lab automatically uses today's live BSM-calculated price as your entry cost. This is the "theoretical" scenario — useful for teaching and exploring Greeks without a specific trade in mind.</p>
           <p style="margin:0 0 0.6rem;"><b>Enter your actual premium</b> (e.g., ₹180): The P&L chart and scenario grid will show your real-world profit or loss from your actual entry point. All P&L = (Current Price − ₹180) × Lot Size.</p>
           <p style="margin:0 0 0.4rem;"><b>Example:</b> Yesterday you bought 2 NIFTY ATM call lots at ₹220 each (lot size 50). Enter Purchase Price = 220, Lot Size = 100 (2 × 50). The P&L tab now shows exactly how much you are making or losing today across all spot and vol scenarios.</p>
           <p style="margin:0;"><b>Breakeven point</b> is automatically calculated: For calls, Breakeven = Strike + Purchase Price. For puts, Breakeven = Strike − Purchase Price. The green dashed line on the P&L chart marks this level.</p>
         </div>
         <ul>
           <li>The <b>5 P&L curves</b> show expected profit at Expiry, 75%T, 50%T, 25%T, and 1-day from now</li>
           <li>The <b>Scenario Grid</b> is a stress test: rows = vol change (−8% to +16%), columns = spot move (−30% to +30%)</li>
         </ul>"""
        ),
        ("Step 6 — Run Sensitivity Analysis (Tab 4)",
         """<p>The bump-and-reprice table quantifies exact risk exposures:</p>
         <ul>
           <li>Each row shows the ₹ price change for a standard-size shock to one factor</li>
           <li>Compare <b>Spot +₹1</b> change (≈ Delta) with the Delta in your metrics bar — they match</li>
           <li>Compare <b>Vol +1%</b> change with Vega × 1 — confirms your Vega calculation</li>
           <li>The Greek Summary table interprets each Greek in plain English with current values</li>
         </ul>"""),
        ("Step 7 — Deep Learning (Tabs 5 & 7)",
         """<p>Use these two tabs for structured learning:</p>
         <ul>
           <li><b>Tab 5 (Theory & Formulae):</b> BSM pricing, assumptions, all Greek formulae, Python code, Excel architecture</li>
           <li><b>Tab 7 (Greek Education Hub):</b> Deep concept cards for each Greek — intuition, formulae, real-world examples, strategies, worked numericals</li>
           <li>Study Tab 7 before an exam — covers first principles through advanced second-order effects</li>
         </ul>"""),
    ]

    for step_title, step_content in guide_steps:
        with st.expander(f"📌 {step_title}"):
            st.markdown(f"""
            <div style="color:{COLORS['text_primary']};line-height:1.8;font-size:0.9rem;">
                {step_content}
            </div>
            """, unsafe_allow_html=True)

    # ── Quick Reference ──────────────────────────────────────────────────
    section_title("⚡ Quick Reference — Parameter Cheatsheet")
    ref_col1, ref_col2 = st.columns(2)

    with ref_col1:
        st.markdown(f"""
        <div class="info-box">
            <h4 style='color:{COLORS["accent_gold"]};margin-top:0;'>📐 Typical Parameter Ranges (Indian Markets)</h4>
            <ul style="color:{COLORS['text_primary']};line-height:2;margin:0;padding-left:1.2rem;">
                <li><b>Spot:</b> NIFTY 50 ≈ 22,000–24,000 | BANK NIFTY ≈ 48,000–52,000</li>
                <li><b>Volatility:</b> India VIX ≈ 12–25% (normal); 30–60% (crisis)</li>
                <li><b>Risk-Free Rate:</b> RBI Repo ≈ 6.25–6.5% (2024–25)</li>
                <li><b>Lot Sizes:</b> NIFTY = 50 | BANK NIFTY = 15 | FINNIFTY = 40</li>
                <li><b>Expiry:</b> Weekly (Thursday) and Monthly contracts available</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    with ref_col2:
        st.markdown(f"""
        <div class="info-box">
            <h4 style='color:{COLORS["accent_gold"]};margin-top:0;'>🔖 Moneyness Quick Reference</h4>
            <ul style="color:{COLORS['text_primary']};line-height:2;margin:0;padding-left:1.2rem;">
                <li><b>ITM Call:</b> S &gt; K — has intrinsic value; Delta &gt; 0.5</li>
                <li><b>OTM Call:</b> S &lt; K — only time value; Delta &lt; 0.5</li>
                <li><b>ATM:</b> S = K — maximum Gamma and Vega; Delta ≈ 0.5</li>
                <li><b>Deep ITM:</b> Delta → 1; Gamma → 0; behaves like stock</li>
                <li><b>Deep OTM:</b> Delta → 0; Gamma → 0; all time value only</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    # ── FAQs ─────────────────────────────────────────────────────────────
    section_title("❓ Frequently Asked Questions")
    faqs = [
        ("Why doesn't the BSM price match the market price?",
         "BSM assumes constant volatility and no dividends. Market prices use implied volatility — "
         "the volatility that makes BSM equal to the traded price. For index options (NIFTY), "
         "dividend treatment also differs. Use this lab for learning sensitivity relationships, "
         "not for direct arbitrage pricing."),
        ("Why is Theta always negative for long options?",
         "As time passes, the uncertainty about the final outcome decreases — there is less time "
         "for the option to move into the money. This loss of optionality is captured by Theta. "
         "Theta decay accelerates as expiry approaches, especially for ATM options. "
         "Short option sellers benefit — they receive Theta daily."),
        ("What does a Delta of 0.45 mean practically?",
         "Your call position behaves like holding 0.45 shares of the underlying. "
         "A ₹100 rise in the stock increases your call price by approximately ₹45. "
         "For hedging: to be delta-neutral, sell 0.45 shares for every 1 call you own."),
        ("Why does Gamma peak at ATM and near expiry?",
         "Gamma measures how quickly Delta changes. Near expiry, a small spot move can dramatically "
         "change whether an ATM option finishes in or out of the money — so Delta changes rapidly. "
         "This is why ATM short options near expiry carry extreme gamma risk (the 'gamma trap')."),
        ("How should I use the P&L Scenario Grid in practice?",
         "Think of it as a stress test. Your current position sits at Vol Δ = 0 and Spot = 0%. "
         "Read: what happens if the market falls 15% AND volatility spikes 12%? "
         "This is exactly what happens during a market crash. Use this grid before entering trades "
         "to understand worst-case scenarios and position sizing."),
        ("What is the difference between Vega and Implied Volatility?",
         "Vega measures sensitivity of the option price to a 1% change in volatility. "
         "Implied Volatility (IV) is the volatility input that makes BSM equal to the market price. "
         "India VIX is a measure of 30-day implied volatility for NIFTY 50 options. "
         "When VIX rises, option prices rise — Vega tells you by exactly how much."),
    ]
    for q, a in faqs:
        with st.expander(f"🔹 {q}"):
            st.markdown(f"""
            <div style="color:{COLORS['text_primary']};font-size:0.9rem;line-height:1.8;
                        background:rgba(0,51,102,0.3);border-radius:8px;padding:1rem;">
                {a}
            </div>
            """, unsafe_allow_html=True)


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
# ROW 2 TABS — Education & Guides (rendered AFTER all Row 1 tab content)
# ============================================================================
st.markdown(f"""
<div style="margin-top:1.5rem;margin-bottom:-0.5rem;">
    <p style="color:{COLORS['text_secondary']};font-size:0.72rem;letter-spacing:1.5px;
              text-transform:uppercase;margin:0 0 0.3rem;font-weight:600;">
        📖 Education &amp; Formula Guides
    </p>
</div>""", unsafe_allow_html=True)

st.markdown(f"""
<style>
    div[data-testid="stRadio"] [data-testid="stWidgetLabel"] {{
        display: none !important;
    }}
    div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {{
        display: none !important;
    }}
    div[data-testid="stRadio"] > div[role="radiogroup"] {{
        display: flex !important;
        flex-direction: row !important;
        gap: 10px !important;
        flex-wrap: wrap !important;
        align-items: center !important;
    }}
    div[data-testid="stRadio"] > div[role="radiogroup"] > label {{
        background: {COLORS['card_bg']} !important;
        border: 1px solid rgba(255,215,0,0.35) !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.2rem !important;
        cursor: pointer !important;
        margin: 0 !important;
        min-height: unset !important;
        transition: all 0.15s ease !important;
    }}
    div[data-testid="stRadio"] > div[role="radiogroup"] > label p,
    div[data-testid="stRadio"] > div[role="radiogroup"] > label span,
    div[data-testid="stRadio"] > div[role="radiogroup"] > label div {{
        color: {COLORS['text_primary']} !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }}
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {{
        background: rgba(0,51,102,0.7) !important;
        border-color: {COLORS['accent_gold']} !important;
    }}
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover p,
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover span,
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover div {{
        color: {COLORS['accent_gold']} !important;
    }}
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {{
        background: {COLORS['dark_blue']} !important;
        border: 2px solid {COLORS['accent_gold']} !important;
    }}
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) p,
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) span,
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) div {{
        color: {COLORS['accent_gold']} !important;
        font-weight: 700 !important;
    }}
</style>
""", unsafe_allow_html=True)

row2_tab = st.radio(
    "Education Tabs",
    ["🎓 Greek Education Hub", "📊 Excel Formula Guide"],
    horizontal=True,
    label_visibility="collapsed",
    key="row2_tabs"
)

# ══════════════════════════════════════════════════════════════════════════
# ROW 2 — TAB A: EDUCATIONAL PLATFORM — OPTIONS GREEKS IN DETAIL
# ══════════════════════════════════════════════════════════════════════════
if row2_tab == "🎓 Greek Education Hub":

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {COLORS['dark_blue']}, {COLORS['medium_blue']});
                border: 2px solid {COLORS['accent_gold']}; border-radius: 14px;
                padding: 2rem; margin-bottom: 1.5rem; text-align: center;">
        <h1 style="color:{COLORS['accent_gold']}; font-family:'Playfair Display',serif; margin:0;">
            🎓 Options Greeks — Complete Educational Reference
        </h1>
        <p style="color:{COLORS['text_primary']};margin:0.8rem 0 0;font-size:0.95rem;">
            From first principles to advanced trading applications · MBA · CFA · FRM level
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Conceptual Foundation ────────────────────────────────────────────
    section_title("📐 What Are the Options Greeks?")
    st.markdown(f"""
    <div class="info-box">
        <h4 style='color:{COLORS["accent_gold"]};margin-top:0;font-family:"Playfair Display",serif;'>
            Conceptual Foundation — The Greeks as Partial Derivatives</h4>
        <p style="color:{COLORS['text_primary']};line-height:1.8;margin:0 0 0.8rem;">
        Options Greeks are <b>partial derivatives</b> of the option pricing function with respect to
        its input parameters. They measure the sensitivity of an option's price to changes in
        market conditions — the fundamental tools of options risk management.</p>
        <p style="color:{COLORS['text_primary']};line-height:1.8;margin:0 0 0.5rem;">
        If the option price is <b>V = f(S, K, T, r, σ)</b>, then:</p>
        <ul style="color:{COLORS['text_primary']};line-height:2;margin:0 0 0.8rem;padding-left:1.5rem;">
            <li><b>Delta (Δ)</b> = ∂V/∂S &nbsp;&nbsp; — sensitivity to spot price</li>
            <li><b>Gamma (Γ)</b> = ∂²V/∂S² — rate of change of Delta (curvature)</li>
            <li><b>Vega (ν)</b> = ∂V/∂σ &nbsp;&nbsp; — sensitivity to volatility</li>
            <li><b>Theta (Θ)</b> = ∂V/∂t &nbsp;&nbsp; — sensitivity to time (decay)</li>
            <li><b>Rho (ρ)</b> = ∂V/∂r &nbsp;&nbsp;&nbsp;— sensitivity to interest rate</li>
        </ul>
        <p style="color:{COLORS['text_primary']};line-height:1.8;margin:0;">
        Together they form the <b>Taylor expansion of option P&L</b>:<br>
        <span style="font-family:'Courier New',monospace;color:{COLORS['light_blue']};font-size:1rem;">
        ΔP&L ≈ Δ·ΔS + ½Γ·(ΔS)² + ν·Δσ + Θ·Δt + ρ·Δr
        </span></p>
    </div>""", unsafe_allow_html=True)

    # ── DELTA ────────────────────────────────────────────────────────────
    section_title("Δ  Delta — Direction & Hedge Ratio")
    col_d1, col_d2 = st.columns([3, 2])

    with col_d1:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{COLORS['card_bg']},rgba(0,51,102,0.6));
                    border:1px solid rgba(255,215,0,0.3);border-radius:12px;padding:1.5rem;margin:1rem 0;">
            <span style="font-size:3rem;color:{COLORS['accent_gold']};float:right;opacity:0.35;
                         font-family:'Playfair Display',serif;">Δ</span>
            <h3 style="color:{COLORS['accent_gold']};font-family:'Playfair Display',serif;margin-top:0;">
                Delta (Δ) — Hedge Ratio & Directional Exposure</h3>
            <p style="color:{COLORS['text_primary']};font-size:0.88rem;line-height:1.75;">
            <b>Definition:</b> Rate of change of option price per ₹1 change in the underlying spot price.</p>
            <div style="background:rgba(255,215,0,0.08);border:1px solid rgba(255,215,0,0.3);border-radius:6px;
                        padding:0.6rem 1rem;font-family:'Courier New',monospace;font-size:0.85rem;
                        color:{COLORS['light_blue']};margin:0.5rem 0;">
                Δ_call = N(d₁) &nbsp;&nbsp;&nbsp; Δ_put = N(d₁) − 1
            </div>
            <p style="color:{COLORS['text_primary']};font-size:0.88rem;line-height:1.75;"><b>Range:</b>
            Calls: [0, +1] | Puts: [−1, 0]</p>
            <p style="color:{COLORS['text_primary']};font-size:0.88rem;line-height:1.75;"><b>Key properties:</b></p>
            <ul style="color:{COLORS['text_primary']};font-size:0.88rem;line-height:1.75;">
                <li>ATM option: Δ ≈ ±0.5 (equal probability of expiring ITM or OTM)</li>
                <li>Deep ITM: Δ → ±1 — behaves identically to the underlying asset</li>
                <li>Deep OTM: Δ → 0 — fully decoupled from spot moves</li>
                <li>Δ is also the <b>risk-neutral probability of expiring ITM</b></li>
                <li>Put-Call parity identity: Δ_call − Δ_put = 1 (always)</li>
            </ul>
            <p style="color:{COLORS['text_primary']};font-size:0.88rem;line-height:1.75;">
            <b>Delta Hedging:</b> To hedge 100 ATM call options (Δ=0.5), short 50 shares of the
            underlying. As spot moves, continuously re-hedge — this is dynamic delta hedging.
            The cost of continuous rehedging is related to Gamma.</p>
            <p style="color:{COLORS['text_primary']};font-size:0.88rem;line-height:1.75;">
            <b>Practical insight:</b> A 0.30 Delta OTM call gives 30% participation in the upside
            at a fraction of the cost of owning the stock outright. This is the leverage proposition
            of options.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_d2:
        S_range_d = np.linspace(S*0.6, S*1.4, 150)
        T_range_d = np.full_like(S_range_d, T)
        delta_call = greeks_vec(S_range_d, K, T_range_d, r, sigma, 'delta')
        delta_put  = delta_call - 1
        fig_delta = go.Figure()
        fig_delta.add_trace(go.Scatter(x=S_range_d, y=delta_call, name='Call Δ',
            line=dict(color=COLORS['accent_gold'], width=2.5)))
        fig_delta.add_trace(go.Scatter(x=S_range_d, y=delta_put, name='Put Δ',
            line=dict(color=COLORS['danger'], width=2.5)))
        fig_delta.add_vline(x=S, line_dash="dash", line_color=COLORS['light_blue'], line_width=1.2)
        fig_delta.add_vline(x=K, line_dash="dash", line_color=COLORS['text_secondary'], line_width=1)
        fig_delta.add_hline(y=0, line_color='rgba(255,255,255,0.2)', line_width=1)
        fig_delta.update_layout(height=300,
            title=dict(text="Delta vs Spot (Live)", font=dict(color=COLORS['accent_gold'], size=13)),
            paper_bgcolor='#0f1824', plot_bgcolor='#0f1824', font=dict(color=COLORS['text_primary']),
            legend=dict(bgcolor='rgba(17,34,64,0.8)', bordercolor=COLORS['accent_gold'], borderwidth=1),
            xaxis=dict(title="Spot ₹", gridcolor='rgba(255,255,255,0.07)', color=COLORS['text_secondary']),
            yaxis=dict(title="Delta", gridcolor='rgba(255,255,255,0.07)', color=COLORS['text_secondary']),
            margin=dict(t=40, b=40))
        st.plotly_chart(fig_delta, use_container_width=True)

        st.markdown(f"""
        <div style="background:{COLORS['card_bg']};border:1px solid rgba(255,215,0,0.3);
                    border-radius:8px;padding:1rem;font-size:0.82rem;">
            <p style="color:{COLORS['accent_gold']};font-weight:700;margin:0 0 0.5rem;">📊 Current Position</p>
            <p style="color:{COLORS['text_primary']};margin:0.25rem 0;">
                Call Δ = <b style="color:{COLORS['accent_gold']};">{res['delta']:.4f}</b>
                &nbsp; Put Δ = <b style="color:{COLORS['danger']};">{res['delta']-1:.4f}</b></p>
            <p style="color:{COLORS['text_secondary']};margin:0.25rem 0;">
                d₁ = {res['d1']:.4f} → N(d₁) = {norm.cdf(res['d1']):.4f}</p>
            <p style="color:{COLORS['text_secondary']};margin:0.25rem 0;font-size:0.78rem;">
                Hedge: Short {res['delta']*50:.1f} NIFTY units per 50-lot call position</p>
        </div>
        """, unsafe_allow_html=True)

    # ── GAMMA ────────────────────────────────────────────────────────────
    section_title("Γ  Gamma — Curvature & Delta Sensitivity")
    col_g1, col_g2 = st.columns([3, 2])

    with col_g1:
        bep_move = np.sqrt(-2*res['theta']/res['gamma']) if res['gamma'] > 0 else 0
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{COLORS['card_bg']},rgba(0,51,102,0.6));
                    border:1px solid rgba(255,215,0,0.3);border-radius:12px;padding:1.5rem;margin:1rem 0;">
            <span style="font-size:3rem;color:{COLORS['accent_gold']};float:right;opacity:0.35;
                         font-family:'Playfair Display',serif;">Γ</span>
            <h3 style="color:{COLORS['accent_gold']};font-family:'Playfair Display',serif;margin-top:0;">
                Gamma (Γ) — Rate of Change of Delta</h3>
            <div style="background:rgba(255,215,0,0.08);border:1px solid rgba(255,215,0,0.3);border-radius:6px;
                        padding:0.6rem 1rem;font-family:'Courier New',monospace;font-size:0.85rem;
                        color:{COLORS['light_blue']};margin:0.5rem 0;">
                Γ = φ(d₁) / (S · σ · √T)
            </div>
            <ul style="color:{COLORS['text_primary']};font-size:0.88rem;line-height:1.75;">
                <li>Always <b>positive</b> for long options — calls and puts both benefit from large moves</li>
                <li><b>Maximum at ATM</b> — Delta changes fastest here relative to spot</li>
                <li><b>Spikes near expiry</b> for ATM — the famous "gamma explosion"</li>
                <li>Identical for calls and puts (same S, K, T, r, σ)</li>
            </ul>
            <p style="color:{COLORS['text_primary']};font-size:0.88rem;line-height:1.75;">
            <b>Theta-Gamma trade-off (most important identity in options):</b></p>
            <div style="background:rgba(255,215,0,0.08);border:1px solid rgba(255,215,0,0.3);border-radius:6px;
                        padding:0.6rem 1rem;font-family:'Courier New',monospace;font-size:0.85rem;
                        color:{COLORS['light_blue']};margin:0.5rem 0;">
                Daily P&L ≈ ½Γ·(ΔS)² + Θ·(1/365)
            </div>
            <p style="color:{COLORS['text_primary']};font-size:0.88rem;line-height:1.75;">
            Long gamma → pays theta daily, profits from large moves.<br>
            Short gamma → earns theta daily, loses on large moves.<br><br>
            <b>Breakeven daily move</b> (live): |ΔS*| = √(−2Θ/Γ) =
            <b style="color:{COLORS['accent_gold']};">₹{bep_move:.1f}</b> — if NIFTY moves more than this
            per day on average, long gamma positions profit.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_g2:
        gamma_vals = greeks_vec(S_range_d, K, T_range_d, r, sigma, 'gamma')
        fig_gamma = go.Figure()
        fig_gamma.add_trace(go.Scatter(x=S_range_d, y=gamma_vals,
            line=dict(color=COLORS['light_blue'], width=2.5),
            fill='tozeroy', fillcolor='rgba(173,216,230,0.08)'))
        fig_gamma.add_vline(x=S, line_dash="dash", line_color=COLORS['accent_gold'], line_width=1.2)
        fig_gamma.add_vline(x=K, line_dash="dash", line_color=COLORS['text_secondary'], line_width=1)
        fig_gamma.update_layout(height=300, showlegend=False,
            title=dict(text="Gamma vs Spot (Live)", font=dict(color=COLORS['accent_gold'], size=13)),
            paper_bgcolor='#0f1824', plot_bgcolor='#0f1824', font=dict(color=COLORS['text_primary']),
            xaxis=dict(title="Spot ₹", gridcolor='rgba(255,255,255,0.07)', color=COLORS['text_secondary']),
            yaxis=dict(title="Gamma", gridcolor='rgba(255,255,255,0.07)', color=COLORS['text_secondary']),
            margin=dict(t=40, b=40))
        st.plotly_chart(fig_gamma, use_container_width=True)

        st.markdown(f"""
        <div style="background:{COLORS['card_bg']};border:1px solid rgba(173,216,230,0.3);
                    border-radius:8px;padding:1rem;font-size:0.82rem;">
            <p style="color:{COLORS['accent_gold']};font-weight:700;margin:0 0 0.5rem;">📊 Current Position</p>
            <p style="color:{COLORS['text_primary']};margin:0.25rem 0;">
                Γ = <b style="color:{COLORS['light_blue']};">{res['gamma']:.6f}</b></p>
            <p style="color:{COLORS['text_secondary']};margin:0.25rem 0;">
                Breakeven daily move: ₹{bep_move:.1f}</p>
            <p style="color:{COLORS['text_secondary']};margin:0.25rem 0;font-size:0.78rem;">
                If |NIFTY daily move| &gt; ₹{bep_move:.0f}, long Γ wins</p>
        </div>
        """, unsafe_allow_html=True)

    # ── VEGA ─────────────────────────────────────────────────────────────
    section_title("ν  Vega — Volatility Exposure")
    col_v1, col_v2 = st.columns([3, 2])

    with col_v1:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{COLORS['card_bg']},rgba(0,51,102,0.6));
                    border:1px solid rgba(255,215,0,0.3);border-radius:12px;padding:1.5rem;margin:1rem 0;">
            <span style="font-size:3rem;color:{COLORS['accent_gold']};float:right;opacity:0.35;
                         font-family:'Playfair Display',serif;">ν</span>
            <h3 style="color:{COLORS['accent_gold']};font-family:'Playfair Display',serif;margin-top:0;">
                Vega (ν) — Sensitivity to Implied Volatility</h3>
            <div style="background:rgba(255,215,0,0.08);border:1px solid rgba(255,215,0,0.3);border-radius:6px;
                        padding:0.6rem 1rem;font-family:'Courier New',monospace;font-size:0.85rem;
                        color:{COLORS['light_blue']};margin:0.5rem 0;">
                ν = S · φ(d₁) · √T / 100
            </div>
            <ul style="color:{COLORS['text_primary']};font-size:0.88rem;line-height:1.75;">
                <li>Always <b>positive</b> for long options (both calls and puts)</li>
                <li><b>Maximum at ATM</b> — where outcome uncertainty is greatest</li>
                <li>Increases with time to expiry — longer options have more vol exposure</li>
                <li>Identical for calls and puts (same parameters)</li>
                <li>Deep ITM/OTM options have near-zero Vega</li>
            </ul>
            <p style="color:{COLORS['text_primary']};font-size:0.88rem;line-height:1.75;">
            <b>India VIX context:</b> India VIX measures 30-day implied volatility of NIFTY options.
            A VIX spike from 15% to 18% (+3pp) increases your call by ≈ ν × 3 =
            ₹{res['vega']*3:.2f} per option. Pre-event (Budget, RBI policy, elections), VIX often
            rises sharply — option buyers benefit, sellers suffer.</p>
            <p style="color:{COLORS['text_primary']};font-size:0.88rem;line-height:1.75;">
            <b>Vega-neutral hedging:</b> Unlike delta, you cannot hedge Vega by trading the underlying
            — it requires trading options with different strikes or expiries whose Vegas offset.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_v2:
        vega_vals = greeks_vec(S_range_d, K, T_range_d, r, sigma, 'vega')
        fig_vega = go.Figure()
        fig_vega.add_trace(go.Scatter(x=S_range_d, y=vega_vals,
            line=dict(color=COLORS['success'], width=2.5),
            fill='tozeroy', fillcolor='rgba(40,167,69,0.08)'))
        fig_vega.add_vline(x=S, line_dash="dash", line_color=COLORS['accent_gold'], line_width=1.2)
        fig_vega.add_vline(x=K, line_dash="dash", line_color=COLORS['text_secondary'], line_width=1)
        fig_vega.update_layout(height=300, showlegend=False,
            title=dict(text="Vega vs Spot (Live)", font=dict(color=COLORS['accent_gold'], size=13)),
            paper_bgcolor='#0f1824', plot_bgcolor='#0f1824', font=dict(color=COLORS['text_primary']),
            xaxis=dict(title="Spot ₹", gridcolor='rgba(255,255,255,0.07)', color=COLORS['text_secondary']),
            yaxis=dict(title="Vega (₹/1%vol)", gridcolor='rgba(255,255,255,0.07)', color=COLORS['text_secondary']),
            margin=dict(t=40, b=40))
        st.plotly_chart(fig_vega, use_container_width=True)

        st.markdown(f"""
        <div style="background:{COLORS['card_bg']};border:1px solid rgba(40,167,69,0.3);
                    border-radius:8px;padding:1rem;font-size:0.82rem;">
            <p style="color:{COLORS['accent_gold']};font-weight:700;margin:0 0 0.5rem;">📊 Current Position</p>
            <p style="color:{COLORS['text_primary']};margin:0.25rem 0;">
                ν = <b style="color:{COLORS['success']};">₹{res['vega']:.4f}</b> per 1% vol</p>
            <p style="color:{COLORS['text_secondary']};margin:0.25rem 0;">
                VIX +3pp → option +₹{res['vega']*3:.2f}</p>
            <p style="color:{COLORS['text_secondary']};margin:0.25rem 0;">
                VIX −3pp → option −₹{res['vega']*3:.2f}</p>
            <p style="color:{COLORS['text_secondary']};margin:0.25rem 0;font-size:0.78rem;">
                σ√T = {sigma*np.sqrt(T):.4f} (vol-time factor)</p>
        </div>
        """, unsafe_allow_html=True)

    # ── THETA ────────────────────────────────────────────────────────────
    section_title("Θ  Theta — Time Value Decay")
    col_t1, col_t2 = st.columns([3, 2])

    with col_t1:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{COLORS['card_bg']},rgba(0,51,102,0.6));
                    border:1px solid rgba(255,215,0,0.3);border-radius:12px;padding:1.5rem;margin:1rem 0;">
            <span style="font-size:3rem;color:{COLORS['accent_gold']};float:right;opacity:0.35;
                         font-family:'Playfair Display',serif;">Θ</span>
            <h3 style="color:{COLORS['accent_gold']};font-family:'Playfair Display',serif;margin-top:0;">
                Theta (Θ) — Time Value Erosion</h3>
            <div style="background:rgba(255,215,0,0.08);border:1px solid rgba(255,215,0,0.3);border-radius:6px;
                        padding:0.6rem 1rem;font-family:'Courier New',monospace;font-size:0.85rem;
                        color:{COLORS['light_blue']};margin:0.5rem 0;">
                Θ_call = [−S·φ(d₁)·σ/(2√T) − r·K·e⁻ʳᵀ·N(d₂)] / 365
            </div>
            <ul style="color:{COLORS['text_primary']};font-size:0.88rem;line-height:1.75;">
                <li>Always <b>negative</b> for long options — time is the enemy of option buyers</li>
                <li><b>Most negative at ATM</b> — where time value is the largest component</li>
                <li>Theta decay <b>accelerates</b> as expiry approaches (non-linear, ∝ √T)</li>
                <li>Deep ITM/OTM options have near-zero Theta (mostly intrinsic or no value)</li>
                <li>Short option positions have <b>positive</b> Theta — sellers collect time premium daily</li>
            </ul>
            <p style="color:{COLORS['text_primary']};font-size:0.88rem;line-height:1.75;">
            <b>Weekend effect:</b> BSM Theta is per calendar day. On Fridays, short option holders
            effectively collect 3 days of Theta as the weekend passes with markets closed —
            a key reason why professional option sellers prefer holding into weekends.</p>
            <p style="color:{COLORS['text_primary']};font-size:0.88rem;line-height:1.75;">
            <b>NIFTY example:</b> You buy an ATM call for ₹200 with Θ = −₹8/day.
            After 5 trading days with no spot move, you lose ₹40 purely from time decay.
            This is why option buyers must be right on direction AND timing.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_t2:
        theta_vals = greeks_vec(S_range_d, K, T_range_d, r, sigma, 'theta')
        fig_theta = go.Figure()
        fig_theta.add_trace(go.Scatter(x=S_range_d, y=theta_vals,
            line=dict(color=COLORS['danger'], width=2.5),
            fill='tozeroy', fillcolor='rgba(220,53,69,0.08)'))
        fig_theta.add_vline(x=S, line_dash="dash", line_color=COLORS['accent_gold'], line_width=1.2)
        fig_theta.add_vline(x=K, line_dash="dash", line_color=COLORS['text_secondary'], line_width=1)
        fig_theta.add_hline(y=0, line_color='rgba(255,255,255,0.2)', line_width=1)
        fig_theta.update_layout(height=240, showlegend=False,
            title=dict(text="Theta vs Spot (Live)", font=dict(color=COLORS['accent_gold'], size=13)),
            paper_bgcolor='#0f1824', plot_bgcolor='#0f1824', font=dict(color=COLORS['text_primary']),
            xaxis=dict(title="Spot ₹", gridcolor='rgba(255,255,255,0.07)', color=COLORS['text_secondary']),
            yaxis=dict(title="Theta ₹/day", gridcolor='rgba(255,255,255,0.07)', color=COLORS['text_secondary']),
            margin=dict(t=40, b=30))
        st.plotly_chart(fig_theta, use_container_width=True)

        # Option price decay curve
        days_arr = np.linspace(1, max(T_days, 10), 80)
        price_decay = [bsm(S, K, d/365, r, sigma, opt_type)['price'] for d in days_arr]
        fig_tdecay = go.Figure()
        fig_tdecay.add_trace(go.Scatter(x=days_arr, y=price_decay,
            line=dict(color=COLORS['accent_gold'], width=2.2),
            fill='tozeroy', fillcolor='rgba(255,215,0,0.06)'))
        fig_tdecay.update_layout(height=200, showlegend=False,
            title=dict(text="Option Price Decay (Time)", font=dict(color=COLORS['accent_gold'], size=12)),
            paper_bgcolor='#0f1824', plot_bgcolor='#0f1824', font=dict(color=COLORS['text_primary']),
            xaxis=dict(title="Days", gridcolor='rgba(255,255,255,0.07)', color=COLORS['text_secondary'],
                       autorange='reversed'),
            yaxis=dict(title="Price ₹", gridcolor='rgba(255,255,255,0.07)', color=COLORS['text_secondary']),
            margin=dict(t=35, b=40))
        st.plotly_chart(fig_tdecay, use_container_width=True)

        st.markdown(f"""
        <div style="background:{COLORS['card_bg']};border:1px solid rgba(220,53,69,0.3);
                    border-radius:8px;padding:0.8rem;font-size:0.82rem;">
            <p style="color:{COLORS['accent_gold']};font-weight:700;margin:0 0 0.4rem;">📊 Current Position</p>
            <p style="color:{COLORS['text_primary']};margin:0.2rem 0;">
                Θ = <b style="color:{COLORS['danger']};">₹{res['theta']:.4f}</b>/day</p>
            <p style="color:{COLORS['text_secondary']};margin:0.2rem 0;">
                Weekly: ₹{res['theta']*7:.2f} | Monthly: ₹{res['theta']*30:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── RHO ──────────────────────────────────────────────────────────────
    section_title("ρ  Rho — Interest Rate Sensitivity")
    col_r1, col_r2 = st.columns([3, 2])

    with col_r1:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{COLORS['card_bg']},rgba(0,51,102,0.6));
                    border:1px solid rgba(255,215,0,0.3);border-radius:12px;padding:1.5rem;margin:1rem 0;">
            <span style="font-size:3rem;color:{COLORS['accent_gold']};float:right;opacity:0.35;
                         font-family:'Playfair Display',serif;">ρ</span>
            <h3 style="color:{COLORS['accent_gold']};font-family:'Playfair Display',serif;margin-top:0;">
                Rho (ρ) — Sensitivity to Interest Rates</h3>
            <div style="background:rgba(255,215,0,0.08);border:1px solid rgba(255,215,0,0.3);border-radius:6px;
                        padding:0.6rem 1rem;font-family:'Courier New',monospace;font-size:0.85rem;
                        color:{COLORS['light_blue']};margin:0.5rem 0;">
                ρ_call = K·T·e⁻ʳᵀ·N(d₂) / 100<br>
                ρ_put  = −K·T·e⁻ʳᵀ·N(−d₂) / 100
            </div>
            <ul style="color:{COLORS['text_primary']};font-size:0.88rem;line-height:1.75;">
                <li>Calls have <b>positive</b> Rho — higher rates increase call values</li>
                <li>Puts have <b>negative</b> Rho — higher rates decrease put values</li>
                <li>Increases with time to expiry (more rate-sensitive for long-dated options)</li>
                <li>Near-zero for short-dated options — least important Greek for monthly NIFTY contracts</li>
            </ul>
            <p style="color:{COLORS['text_primary']};font-size:0.88rem;line-height:1.75;">
            <b>Intuition:</b> Higher interest rates increase the cost of carry for the underlying.
            Holding a call is equivalent to holding the stock synthetically at lower upfront cost,
            so higher rates make calls relatively more attractive → call value rises.</p>
            <p style="color:{COLORS['text_primary']};font-size:0.88rem;line-height:1.75;">
            <b>Indian context:</b> RBI surprises matter. A 50bp rate cut (Δr = −0.5%) changes your
            call by ≈ ρ × (−0.5) = ₹{res['rho']*(-0.5):.3f}. For short-dated NIFTY weekly options,
            this is negligible — but for 1-year LEAPS or interest rate derivatives, Rho dominates.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_r2:
        rho_vals = greeks_vec(S_range_d, K, T_range_d, r, sigma, 'rho')
        fig_rho = go.Figure()
        fig_rho.add_trace(go.Scatter(x=S_range_d, y=rho_vals, name='Call ρ',
            line=dict(color='#9b59b6', width=2.5)))
        fig_rho.add_trace(go.Scatter(x=S_range_d, y=-rho_vals, name='Put ρ',
            line=dict(color=COLORS['danger'], width=2, dash='dash')))
        fig_rho.add_vline(x=S, line_dash="dash", line_color=COLORS['accent_gold'], line_width=1.2)
        fig_rho.update_layout(height=300,
            title=dict(text="Rho vs Spot (Live)", font=dict(color=COLORS['accent_gold'], size=13)),
            paper_bgcolor='#0f1824', plot_bgcolor='#0f1824', font=dict(color=COLORS['text_primary']),
            legend=dict(bgcolor='rgba(17,34,64,0.8)', bordercolor=COLORS['accent_gold'], borderwidth=1),
            xaxis=dict(title="Spot ₹", gridcolor='rgba(255,255,255,0.07)', color=COLORS['text_secondary']),
            yaxis=dict(title="Rho (₹/1%r)", gridcolor='rgba(255,255,255,0.07)', color=COLORS['text_secondary']),
            margin=dict(t=40, b=40))
        st.plotly_chart(fig_rho, use_container_width=True)

        st.markdown(f"""
        <div style="background:{COLORS['card_bg']};border:1px solid rgba(155,89,182,0.3);
                    border-radius:8px;padding:1rem;font-size:0.82rem;">
            <p style="color:{COLORS['accent_gold']};font-weight:700;margin:0 0 0.5rem;">📊 Current Position</p>
            <p style="color:{COLORS['text_primary']};margin:0.25rem 0;">
                Call ρ = <b style="color:#9b59b6;">₹{res['rho']:.4f}</b></p>
            <p style="color:{COLORS['text_primary']};margin:0.25rem 0;">
                Put ρ = <b style="color:{COLORS['danger']};">₹{-res['rho']:.4f}</b></p>
            <p style="color:{COLORS['text_secondary']};margin:0.25rem 0;font-size:0.78rem;">
                RBI 50bp cut → Call: ₹{res['rho']*(-0.5):.3f}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Second-Order Greeks ───────────────────────────────────────────────
    section_title("⚡ Second-Order Greeks — Advanced Risk Measures")
    col_so1, col_so2, col_so3 = st.columns(3)

    second_order = [
        ("Vanna", "∂Δ/∂σ = ∂ν/∂S",
         "Cross-sensitivity between Delta and Volatility.",
         "Vanna = −φ(d₁) · d₂ / σ",
         "Critical for volatility skew trading and delta-hedging in changing vol regimes. "
         "When vol spikes, OTM call Deltas rise → Vanna exposure materialises. Key for risk-reversal pricing."),
        ("Charm (Delta Bleed)", "∂Δ/∂t",
         "Rate of change of Delta with respect to time.",
         "Charm = −φ(d₁)[2rT − d₂σ√T] / (2T·σ√T)",
         "How much the delta hedge needs adjusting each day purely due to time passing. "
         "Critical for daily delta-hedging programmes and near-expiry books. Sometimes called 'Delta bleed'."),
        ("Volga (Vomma)", "∂²V/∂σ²",
         "Rate of change of Vega with respect to volatility (vol convexity).",
         "Volga = ν · d₁ · d₂ / σ",
         "Positive for long options — benefits from vol-of-vol. "
         "Key in exotic options (barriers, binaries) and volatility surface construction. Drives the volatility smile curvature."),
    ]

    for col, (name, defn, desc_short, formula, detail) in zip([col_so1, col_so2, col_so3], second_order):
        with col:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,{COLORS['card_bg']},rgba(0,51,102,0.6));
                        border:1px solid rgba(255,215,0,0.25);border-radius:12px;padding:1.3rem;
                        margin:0.5rem 0;min-height:340px;">
                <h3 style="color:{COLORS['accent_gold']};font-family:'Playfair Display',serif;
                           font-size:1.1rem;margin:0 0 0.3rem;">{name}</h3>
                <p style="color:{COLORS['light_blue']};font-size:0.8rem;margin:0 0 0.5rem;">{defn} — {desc_short}</p>
                <div style="background:rgba(255,215,0,0.08);border:1px solid rgba(255,215,0,0.25);
                            border-radius:5px;padding:0.5rem 0.8rem;font-family:'Courier New',monospace;
                            font-size:0.78rem;color:{COLORS['light_blue']};margin:0.5rem 0;">
                    {formula}
                </div>
                <p style="color:{COLORS['text_primary']};font-size:0.82rem;line-height:1.7;margin:0;">{detail}</p>
            </div>
            """, unsafe_allow_html=True)

    # ── BSM PDE Identity ─────────────────────────────────────────────────
    section_title("🔗 The BSM PDE — Greeks Are Not Independent")
    col_id1, col_id2 = st.columns(2)

    with col_id1:
        st.markdown(f"""
        <div class="info-box">
            <h4 style='color:{COLORS["accent_gold"]};margin-top:0;'>The Master PDE Identity</h4>
            <p style="color:{COLORS['text_primary']};line-height:1.8;margin:0 0 0.5rem;">
            <b>Black-Scholes Partial Differential Equation:</b></p>
            <div style="background:rgba(255,215,0,0.08);border:1px solid rgba(255,215,0,0.3);
                        border-radius:6px;padding:0.6rem 1rem;font-family:'Courier New',monospace;
                        font-size:1rem;color:{COLORS['light_blue']};margin:0.5rem 0;text-align:center;">
                Θ + ½σ²S²Γ + rSΔ − rV = 0
            </div>
            <p style="color:{COLORS['text_primary']};line-height:1.8;margin:0.5rem 0;">
            This links ALL Greeks in one no-arbitrage constraint. For a delta-neutral portfolio (Δ=0):</p>
            <div style="background:rgba(255,215,0,0.08);border:1px solid rgba(255,215,0,0.3);
                        border-radius:6px;padding:0.5rem 1rem;font-family:'Courier New',monospace;
                        font-size:0.9rem;color:{COLORS['light_blue']};margin:0.5rem 0;text-align:center;">
                Θ + ½σ²S²Γ = rV
            </div>
            <p style="color:{COLORS['text_primary']};line-height:1.8;margin:0;">
            Time decay is exactly offset by gamma gain in a risk-free delta-neutral portfolio.
            This is the mathematical foundation of delta-neutral options market-making.</p>
        </div>""", unsafe_allow_html=True)

    with col_id2:
        st.markdown(f"""
        <div class="info-box">
            <h4 style='color:{COLORS["accent_gold"]};margin-top:0;'>Key Identities from Put-Call Parity</h4>
            <p style="color:{COLORS['text_primary']};line-height:2;margin:0 0 0.5rem;">
            <b>Put-Call Parity Greek Relationships:</b></p>
            <ul style="color:{COLORS['text_primary']};line-height:2;margin:0 0 0.8rem;padding-left:1.2rem;">
                <li>Δ_call − Δ_put = 1 &nbsp; (always, no exceptions)</li>
                <li>Γ_call = Γ_put &nbsp; (same parameters)</li>
                <li>ν_call = ν_put &nbsp; (same parameters)</li>
                <li>Θ_call − Θ_put = −r·K·e<sup>−rT</sup></li>
                <li>ρ_call − ρ_put = K·T·e<sup>−rT</sup></li>
            </ul>
            <p style="color:{COLORS['text_primary']};line-height:1.8;margin:0;">
            <b>Delta-neutral hedge:</b> # shares to short = Δ × # calls<br>
            <b>Gamma-neutral:</b> requires trading additional options positions</p>
        </div>""", unsafe_allow_html=True)

    # ── Master Greeks Table ───────────────────────────────────────────────
    section_title("📋 Complete Greeks Reference Table")
    greeks_master = pd.DataFrame([
        {"Greek":"Delta Δ","Formula":"N(d₁) / N(d₁)−1","Call":"[0,+1]","Put":"[−1,0]",
         "Max at":"Deep ITM","Min at":"Deep OTM","Increases with":"↑S (calls), ↑T, ATM",
         "Primary Use":"Directional hedge ratio","Now":f"{res['delta']:.4f}"},
        {"Greek":"Gamma Γ","Formula":"φ(d₁)/(Sσ√T)","Call":"≥ 0","Put":"≥ 0",
         "Max at":"ATM + near expiry","Min at":"Deep ITM/OTM","Increases with":"↓T (ATM), ↓σ",
         "Primary Use":"Delta rehedge frequency","Now":f"{res['gamma']:.6f}"},
        {"Greek":"Vega ν","Formula":"Sφ(d₁)√T/100","Call":"≥ 0","Put":"≥ 0",
         "Max at":"ATM","Min at":"Deep ITM/OTM","Increases with":"↑T, ATM",
         "Primary Use":"Volatility / IV exposure","Now":f"₹{res['vega']:.4f}"},
        {"Greek":"Theta Θ","Formula":"[−Sφσ/(2√T)−rKe⁻ʳᵀN(d₂)]/365","Call":"≤ 0","Put":"≤ 0",
         "Max at":"ATM + near expiry","Min at":"Deep ITM/OTM","Increases with":"↓T, ATM",
         "Primary Use":"Time decay cost/income","Now":f"₹{res['theta']:.4f}"},
        {"Greek":"Rho ρ","Formula":"KTe⁻ʳᵀN(d₂)/100","Call":"≥ 0","Put":"≤ 0",
         "Max at":"Deep ITM, long-dated","Min at":"Short-dated OTM","Increases with":"↑T, ITM, ↑r",
         "Primary Use":"Interest rate risk","Now":f"₹{res['rho']:.4f}"},
    ])
    st.dataframe(greeks_master, use_container_width=True, hide_index=True)

    # ── Option Strategies ─────────────────────────────────────────────────
    section_title("📊 Option Strategies — Greek Profiles at a Glance")
    strategies_df = pd.DataFrame([
        {"Strategy":"Long Call","Δ":"+","Γ":"+","ν":"+","Θ":"−","View":"Bullish + Long Vol","Max Loss":"Premium"},
        {"Strategy":"Long Put","Δ":"−","Γ":"+","ν":"+","Θ":"−","View":"Bearish + Long Vol","Max Loss":"Premium"},
        {"Strategy":"Short Call","Δ":"−","Γ":"−","ν":"−","Θ":"+","View":"Neutral-Bearish + Short Vol","Max Loss":"Unlimited"},
        {"Strategy":"Short Put","Δ":"+","Γ":"−","ν":"−","Θ":"+","View":"Neutral-Bullish + Short Vol","Max Loss":"Strike−Premium"},
        {"Strategy":"Covered Call","Δ":"+(reduced)","Γ":"−","ν":"−","Θ":"+","View":"Mildly Bullish, Income","Max Loss":"Stock−Premium"},
        {"Strategy":"Long Straddle","Δ":"≈0","Γ":"+(large)","ν":"+(large)","Θ":"−(large)","View":"Long Vol, Direction-Neutral","Max Loss":"Both premiums"},
        {"Strategy":"Short Straddle","Δ":"≈0","Γ":"−(large)","ν":"−(large)","Θ":"+(large)","View":"Short Vol, Range-Bound","Max Loss":"Unlimited"},
        {"Strategy":"Iron Condor","Δ":"≈0","Γ":"−","ν":"−","Θ":"+","View":"Low Vol, Tight Range","Max Loss":"Spread−Premium"},
        {"Strategy":"Calendar Spread","Δ":"≈0","Γ":"−","ν":"+","Θ":"+","View":"Short Realised, Long IV","Max Loss":"Net premium"},
        {"Strategy":"Bull Call Spread","Δ":"+(reduced)","Γ":"±","ν":"±","Θ":"±","View":"Moderately Bullish","Max Loss":"Net premium"},
    ])
    st.dataframe(strategies_df, use_container_width=True, hide_index=True)

    # ── Exam Prep ─────────────────────────────────────────────────────────
    section_title("🎯 Exam Preparation — CFA / FRM / MBA")
    exam_col1, exam_col2 = st.columns(2)

    with exam_col1:
        st.markdown(f"""
        <div class="info-box">
            <h4 style='color:{COLORS["accent_gold"]};margin-top:0;'>🔟 Must-Know Facts for Exams</h4>
            <ol style="color:{COLORS['text_primary']};line-height:2;margin:0;padding-left:1.3rem;">
                <li>ATM call Δ ≈ 0.5; ATM put Δ ≈ −0.5</li>
                <li>Deep ITM call: Δ → 1 | Deep OTM call: Δ → 0</li>
                <li>Γ is ALWAYS positive for long options</li>
                <li>Θ is ALWAYS negative for long options</li>
                <li>ν is ALWAYS positive for long options</li>
                <li>Γ = 0 at expiry for ITM/OTM; spikes at ATM</li>
                <li>Long Γ pays Θ; Short Γ earns Θ (core trade-off)</li>
                <li>ATM options most sensitive to volatility (max ν)</li>
                <li>ρ least important for short-dated equity options</li>
                <li>Δ_call − Δ_put = 1 (Put-Call Parity, always)</li>
            </ol>
        </div>""", unsafe_allow_html=True)

    with exam_col2:
        st.markdown(f"""
        <div class="info-box">
            <h4 style='color:{COLORS["accent_gold"]};margin-top:0;'>📝 Common Exam Question Approaches</h4>
            <ol style="color:{COLORS['text_primary']};line-height:2;margin:0;padding-left:1.3rem;">
                <li><b>Price an option:</b> Compute d₁, d₂, N(d₁), N(d₂) → BSM formula</li>
                <li><b>Delta hedge:</b> Shares = Δ × contracts × lot size</li>
                <li><b>Vol impact:</b> ΔPrice ≈ ν × Δσ</li>
                <li><b>Time decay:</b> ΔPrice ≈ Θ × Δdays (always negative for buyer)</li>
                <li><b>Higher Gamma?</b> ATM beats OTM; shorter expiry beats longer</li>
                <li><b>Delta-neutral:</b> long options + short/long underlying</li>
                <li><b>Long vol:</b> Long straddle (long Γ + long ν)</li>
                <li><b>Breakeven at expiry:</b> K + C (call) | K − P (put)</li>
            </ol>
        </div>""", unsafe_allow_html=True)

    # ── Worked Numericals ─────────────────────────────────────────────────
    section_title("🔢 Worked Numerical Examples")

    with st.expander("📌 Example 1 — Full BSM Pricing + Greeks (NIFTY Call)"):
        ex_S, ex_K, ex_T, ex_r, ex_sigma = 22500, 23000, 30/365, 0.065, 0.18
        ex = bsm(ex_S, ex_K, ex_T, ex_r, ex_sigma, 'call')
        st.markdown(f"""
        <div style="color:{COLORS['text_primary']};font-size:0.9rem;line-height:2;
                    background:rgba(0,51,102,0.2);border-radius:8px;padding:1.2rem;">
        <b>Given:</b> NIFTY Call — S=₹22,500 · K=₹23,000 · T=30d · r=6.5% · σ=18%<br><br>
        <b>Step 1 — d₁:</b><br>
        d₁ = [ln(22500/23000) + (0.065 + ½×0.18²)×(30/365)] / (0.18×√(30/365))<br>
        d₁ = [{np.log(22500/23000):.5f} + {(0.065+0.5*0.18**2)*(30/365):.5f}] / {0.18*np.sqrt(30/365):.5f}
        = <b style="color:{COLORS['accent_gold']};">{ex['d1']:.4f}</b><br><br>
        <b>Step 2 — d₂:</b> d₂ = d₁ − σ√T = {ex['d1']:.4f} − {0.18*np.sqrt(30/365):.4f}
        = <b style="color:{COLORS['accent_gold']};">{ex['d2']:.4f}</b><br><br>
        <b>Step 3 — CDF:</b> N(d₁) = {norm.cdf(ex['d1']):.4f} · N(d₂) = {norm.cdf(ex['d2']):.4f}<br><br>
        <b>Step 4 — Call Price:</b><br>
        C = 22500×{norm.cdf(ex['d1']):.4f} − 23000×e⁻⁰·⁰⁰⁵³×{norm.cdf(ex['d2']):.4f}
        = <b style="color:{COLORS['accent_gold']};font-size:1.1rem;">₹{ex['price']:.2f}</b><br><br>
        <b>Step 5 — All Greeks:</b><br>
        Δ=<b style="color:{COLORS['accent_gold']};">{ex['delta']:.4f}</b> · Γ=<b style="color:{COLORS['light_blue']};">{ex['gamma']:.6f}</b> ·
        ν=<b style="color:{COLORS['success']};">₹{ex['vega']:.4f}</b> · Θ=<b style="color:{COLORS['danger']};">₹{ex['theta']:.4f}/day</b> ·
        ρ=<b style="color:#9b59b6;">₹{ex['rho']:.4f}</b><br><br>
        <b>Interpretation:</b><br>
        • OTM call (K>S) — Δ={ex['delta']:.4f} → {ex['delta']*100:.1f}% directional exposure<br>
        • ₹100 NIFTY rise → call +₹{ex['delta']*100:.0f}<br>
        • Each day: costs ₹{abs(ex['theta']):.2f} in time decay (₹{abs(ex['theta']*7):.2f}/week)<br>
        • India VIX +2pp → option +₹{ex['vega']*2:.2f}
        </div>
        """, unsafe_allow_html=True)

    with st.expander("📌 Example 2 — Delta Hedging (Short Call Position)"):
        ex = bsm(22500, 23000, 30/365, 0.065, 0.18, 'call')
        shares = ex['delta'] * 500
        st.markdown(f"""
        <div style="color:{COLORS['text_primary']};font-size:0.9rem;line-height:2;
                    background:rgba(0,51,102,0.2);border-radius:8px;padding:1.2rem;">
        <b>Problem:</b> You sold 10 NIFTY call lots (lot size=50 → 500 options). How to delta-hedge?<br><br>
        <b>Position Delta:</b> −500 × {ex['delta']:.4f} = <b>−{500*ex['delta']:.1f}</b> (negative = short calls)<br><br>
        <b>Hedge Required:</b> Buy {shares:.0f} NIFTY units (via futures or ETF)<br><br>
        <b>After ₹500 NIFTY rise:</b><br>
        • Short call loss = 500 × Δ × ΔS = 500 × {ex['delta']:.4f} × 500 = ₹{500*ex['delta']*500:,.0f}<br>
        • Long {shares:.0f} NIFTY gain = {shares:.0f} × 500 = ₹{shares*500:,.0f}<br>
        • <b style="color:{COLORS['success']};">Net ≈ ₹0 ✓ (hedged)</b><br><br>
        <b>Re-hedge trigger:</b> New Δ after +₹500 = {bsm(23000, 23000, 30/365, 0.065, 0.18)['delta']:.4f}
        (was {ex['delta']:.4f}) → buy more NIFTY units. This continuous rebalancing cost = Gamma exposure.
        </div>
        """, unsafe_allow_html=True)

    with st.expander("📌 Example 3 — Theta/Gamma P&L Decomposition (Gamma Scalping)"):
        ex = bsm(22500, 23000, 30/365, 0.065, 0.18, 'call')
        n_opt = 100
        dS = 500
        gamma_pnl = 0.5 * ex['gamma'] * dS**2 * n_opt
        theta_pnl = ex['theta'] * 1 * n_opt
        net_pnl = gamma_pnl + theta_pnl
        bep = np.sqrt(-2*ex['theta']/ex['gamma']) if ex['gamma'] > 0 else 0
        st.markdown(f"""
        <div style="color:{COLORS['text_primary']};font-size:0.9rem;line-height:2;
                    background:rgba(0,51,102,0.2);border-radius:8px;padding:1.2rem;">
        <b>Scenario:</b> Long 100 ATM NIFTY calls (delta-neutral hedged). NIFTY moves ₹500 in 1 day.<br><br>
        <b>Delta P&L:</b> ≈ ₹0 (delta-neutral; first-order exposure hedged away)<br><br>
        <b>Gamma P&L:</b> = ½ × Γ × (ΔS)² × N<br>
        = ½ × {ex['gamma']:.6f} × 500² × 100
        = <b style="color:{COLORS['success']};">+₹{gamma_pnl:,.2f}</b><br><br>
        <b>Theta P&L (1 day):</b> = Θ × 1 × N = {ex['theta']:.4f} × 100
        = <b style="color:{COLORS['danger']};">−₹{abs(theta_pnl):,.2f}</b><br><br>
        <b>Net P&L = ₹{net_pnl:,.2f}</b>
        {'(profit — gamma exceeded theta)' if net_pnl > 0 else '(loss — theta exceeded gamma)'}<br><br>
        <b>Breakeven daily move:</b> |ΔS*| = √(−2Θ/Γ)
        = √(−2 × {ex['theta']:.4f} / {ex['gamma']:.6f}) = <b style="color:{COLORS['accent_gold']};">₹{bep:.1f}</b><br><br>
        <b>Conclusion:</b> If NIFTY moves &gt; ₹{bep:.0f}/day on average, long gamma profits.
        If &lt; ₹{bep:.0f}/day, theta bleeds out the premium. This is the essence of gamma scalping
        and the volatility trading decision (realised vol vs implied vol).
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# ROW 2 — TAB C: EXCEL FORMULA GUIDE
# ══════════════════════════════════════════════════════════════════════════
if row2_tab == "📊 Excel Formula Guide":

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{COLORS['dark_blue']},{COLORS['medium_blue']});
                border:2px solid {COLORS['accent_gold']};border-radius:14px;
                padding:2rem;margin-bottom:1.5rem;text-align:center;">
        <div style="font-size:3.5rem;margin-bottom:0.4rem;">📊</div>
        <h1 style="color:{COLORS['accent_gold']};font-family:'Playfair Display',serif;margin:0.2rem 0;">
            Excel Formula Reference Guide</h1>
        <p style="color:{COLORS['light_blue']};font-size:0.95rem;margin:0.6rem 0 0;">
            Complete BSM Option Pricing · All Five Greeks · Ready-to-paste Excel formulas
        </p>
        <p style="color:{COLORS['text_secondary']};font-size:0.82rem;margin:0.4rem 0 0;">
            For NIFTY / BANK NIFTY · MBA · CFA · FRM exam preparation
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Setup convention ────────────────────────────────────────────────
    section_title("🗂️ Spreadsheet Setup — Named Cell Convention")
    st.markdown(f"""
    <div class="info-box">
        <h4 style="color:{COLORS['accent_gold']};margin-top:0;">Recommended Cell Mapping (Sheet: <i>Inputs</i>)</h4>
        <p style="color:{COLORS['text_primary']};margin:0 0 0.6rem;line-height:1.8;">
        Create an <b>Inputs</b> sheet with these named cells. All formula sheets reference them by name for
        easy what-if analysis. In Excel: select cell → Name Box (top-left) → type the name → Enter.</p>
    </div>
    """, unsafe_allow_html=True)

    setup_df = pd.DataFrame([
        {"Cell":"B2","Named Range":"S_price",  "Description":"Spot / Current Market Price (₹)","Example":"22500"},
        {"Cell":"B3","Named Range":"K_strike", "Description":"Strike Price (₹)","Example":"23000"},
        {"Cell":"B4","Named Range":"T_years",  "Description":"Time to Expiry in Years (= days/365)","Example":"=B4_days/365"},
        {"Cell":"B4a","Named Range":"T_days",  "Description":"Days to Expiry (enter here)","Example":"30"},
        {"Cell":"B5","Named Range":"r_rate",   "Description":"Risk-Free Rate (decimal, e.g. 6.5% = 0.065)","Example":"0.065"},
        {"Cell":"B6","Named Range":"sigma",    "Description":"Volatility / India VIX (decimal, e.g. 18% = 0.18)","Example":"0.18"},
        {"Cell":"B7","Named Range":"lot_size", "Description":"Lot Size (NIFTY=50, BANKNIFTY=15, FINNIFTY=40)","Example":"50"},
    ])
    st.dataframe(setup_df, use_container_width=True, hide_index=True)

    # ── d1 and d2 ────────────────────────────────────────────────────────
    section_title("📐 Step 1 — Compute d₁ and d₂")
    st.markdown(f"""
    <div class="info-box">
        <h4 style="color:{COLORS['accent_gold']};margin-top:0;">d₁ and d₂ — The Core BSM Intermediate Values</h4>
        <p style="color:{COLORS['text_primary']};line-height:1.8;margin:0 0 0.5rem;">
        Every option price and Greek derives from these two values. Calculate them first in dedicated cells
        (e.g., D2 and D3), then reference them in all subsequent formulas.</p>
        <div style="background:rgba(255,215,0,0.08);border:1px solid rgba(255,215,0,0.3);border-radius:6px;
                    padding:0.6rem 1rem;font-family:'Courier New',monospace;font-size:0.85rem;
                    color:{COLORS['light_blue']};margin:0.5rem 0;">
            d₁ = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)<br>
            d₂ = d₁ − σ·√T
        </div>
    </div>
    """, unsafe_allow_html=True)

    d1d2_df = pd.DataFrame([
        {
            "Cell":"D2","Label":"d1",
            "Excel Formula":"=(LN(S_price/K_strike)+(r_rate+0.5*sigma^2)*T_years)/(sigma*SQRT(T_years))",
            "Live Value":f"{res['d1']:.6f}",
            "Notes":"LN() = natural log. SQRT() for square root."
        },
        {
            "Cell":"D3","Label":"d2",
            "Excel Formula":"=D2 - sigma*SQRT(T_years)",
            "Live Value":f"{res['d2']:.6f}",
            "Notes":"References D2 (d1) directly. Or: =D2-sigma*SQRT(T_years)"
        },
        {
            "Cell":"D4","Label":"N(d1)",
            "Excel Formula":"=NORM.S.DIST(D2,TRUE)",
            "Live Value":f"{norm.cdf(res['d1']):.6f}",
            "Notes":"Cumulative standard normal CDF at d1"
        },
        {
            "Cell":"D5","Label":"N(d2)",
            "Excel Formula":"=NORM.S.DIST(D3,TRUE)",
            "Live Value":f"{norm.cdf(res['d2']):.6f}",
            "Notes":"Cumulative standard normal CDF at d2"
        },
        {
            "Cell":"D6","Label":"N(-d1)",
            "Excel Formula":"=NORM.S.DIST(-D2,TRUE)",
            "Live Value":f"{norm.cdf(-res['d1']):.6f}",
            "Notes":"Used in put pricing and Delta"
        },
        {
            "Cell":"D7","Label":"N(-d2)",
            "Excel Formula":"=NORM.S.DIST(-D3,TRUE)",
            "Live Value":f"{norm.cdf(-res['d2']):.6f}",
            "Notes":"Used in put pricing and Rho"
        },
        {
            "Cell":"D8","Label":"φ(d1) — PDF",
            "Excel Formula":"=NORM.S.DIST(D2,FALSE)",
            "Live Value":f"{norm.pdf(res['d1']):.6f}",
            "Notes":"Standard normal PDF (FALSE = density, not cumulative). Used in all Greeks."
        },
        {
            "Cell":"D9","Label":"e^(-rT)",
            "Excel Formula":"=EXP(-r_rate*T_years)",
            "Live Value":f"{np.exp(-r*T):.6f}",
            "Notes":"Discount factor. EXP() in Excel = e^x"
        },
    ])
    st.dataframe(d1d2_df, use_container_width=True, hide_index=True)

    # ── Option Pricing ───────────────────────────────────────────────────
    section_title("💰 Step 2 — Option Pricing Formulas")

    st.markdown(f"""
    <div class="info-box">
        <h4 style="color:{COLORS['accent_gold']};margin-top:0;">BSM Option Price Formulas</h4>
        <p style="color:{COLORS['text_primary']};line-height:1.8;margin:0 0 0.5rem;">
        <b>Call:</b> C = S·N(d₁) − K·e<sup>−rT</sup>·N(d₂)<br>
        <b>Put:</b> P = K·e<sup>−rT</sup>·N(−d₂) − S·N(−d₁)<br><br>
        All cells below assume d1=D2, d2=D3, N(d1)=D4, N(d2)=D5, N(-d1)=D6, N(-d2)=D7, e^(-rT)=D9</p>
    </div>
    """, unsafe_allow_html=True)

    pricing_df = pd.DataFrame([
        {
            "Output":"Call Price (C)","Cell":"E2",
            "Excel Formula":"=S_price*D4 - K_strike*D9*D5",
            "Live Value (₹)":f"₹{bsm(S,K,T,r,sigma,'call')['price']:.4f}",
            "Breakdown":"S·N(d₁) − K·e⁻ʳᵀ·N(d₂)"
        },
        {
            "Output":"Put Price (P)","Cell":"E3",
            "Excel Formula":"=K_strike*D9*D7 - S_price*D6",
            "Live Value (₹)":f"₹{bsm(S,K,T,r,sigma,'put')['price']:.4f}",
            "Breakdown":"K·e⁻ʳᵀ·N(−d₂) − S·N(−d₁)"
        },
        {
            "Output":"Put-Call Parity Check","Cell":"E4",
            "Excel Formula":"=E2 - E3 - S_price + K_strike*D9",
            "Live Value (₹)":"≈ 0.0000",
            "Breakdown":"C − P − S + K·e⁻ʳᵀ must = 0 (arbitrage check)"
        },
        {
            "Output":"Intrinsic Value (Call)","Cell":"E5",
            "Excel Formula":"=MAX(S_price-K_strike,0)",
            "Live Value (₹)":f"₹{max(S-K,0):.2f}",
            "Breakdown":"Payoff if exercised immediately"
        },
        {
            "Output":"Intrinsic Value (Put)","Cell":"E6",
            "Excel Formula":"=MAX(K_strike-S_price,0)",
            "Live Value (₹)":f"₹{max(K-S,0):.2f}",
            "Breakdown":"Payoff if exercised immediately"
        },
        {
            "Output":"Time Value (Call)","Cell":"E7",
            "Excel Formula":"=E2 - E5",
            "Live Value (₹)":f"₹{bsm(S,K,T,r,sigma,'call')['price']-max(S-K,0):.4f}",
            "Breakdown":"Option Price − Intrinsic Value"
        },
        {
            "Output":"Breakeven (Call)","Cell":"E8",
            "Excel Formula":"=K_strike + E2",
            "Live Value (₹)":f"₹{K+bsm(S,K,T,r,sigma,'call')['price']:.2f}",
            "Breakdown":"Strike + Call Premium"
        },
        {
            "Output":"Breakeven (Put)","Cell":"E9",
            "Excel Formula":"=K_strike - E3",
            "Live Value (₹)":f"₹{K-bsm(S,K,T,r,sigma,'put')['price']:.2f}",
            "Breakdown":"Strike − Put Premium"
        },
    ])
    st.dataframe(pricing_df, use_container_width=True, hide_index=True)

    # ── Greeks ────────────────────────────────────────────────────────────
    section_title("🧮 Step 3 — All Five Greeks Formulas")

    col_g1, col_g2 = st.columns([1,1])

    with col_g1:
        st.markdown(f"""
        <div style="background:{COLORS['card_bg']};border:1px solid rgba(255,215,0,0.3);
                    border-left:4px solid {COLORS['accent_gold']};border-radius:10px;padding:1.2rem;margin-bottom:1rem;">
            <h4 style="color:{COLORS['accent_gold']};margin-top:0;font-family:'Playfair Display',serif;">
                Δ Delta</h4>
            <p style="color:{COLORS['text_secondary']};font-size:0.8rem;margin:0 0 0.3rem;">
                Rate of change of option price per ₹1 move in spot</p>
            <div style="background:rgba(0,0,0,0.3);border-radius:6px;padding:0.7rem;
                        font-family:'Courier New',monospace;font-size:0.78rem;color:{COLORS['light_blue']};
                        margin:0.5rem 0;">
                <b style="color:{COLORS['accent_gold']};">Call Δ (cell F2):</b><br>
                =NORM.S.DIST(D2,TRUE)<br><br>
                <b style="color:{COLORS['accent_gold']};">Put Δ (cell F3):</b><br>
                =NORM.S.DIST(D2,TRUE)-1<br><br>
                <b style="color:{COLORS['text_secondary']};font-size:0.72rem;">
                Or: =F2 - 1 &nbsp;(Put-Call Parity)</b>
            </div>
            <p style="color:{COLORS['text_primary']};font-size:0.8rem;margin:0.4rem 0 0;">
                Live: Call Δ = <b style="color:{COLORS['accent_gold']};">{res['delta']:.4f}</b>
                &nbsp;|&nbsp; Put Δ = <b style="color:{COLORS['danger']};">{res['delta']-1:.4f}</b></p>
            <p style="color:{COLORS['text_secondary']};font-size:0.75rem;margin:0.2rem 0 0;">
                Range: Call [0,1] · Put [−1,0] · ATM ≈ ±0.50</p>
        </div>

        <div style="background:{COLORS['card_bg']};border:1px solid rgba(173,216,230,0.3);
                    border-left:4px solid {COLORS['light_blue']};border-radius:10px;padding:1.2rem;margin-bottom:1rem;">
            <h4 style="color:{COLORS['light_blue']};margin-top:0;font-family:'Playfair Display',serif;">
                Γ Gamma</h4>
            <p style="color:{COLORS['text_secondary']};font-size:0.8rem;margin:0 0 0.3rem;">
                Rate of change of Delta (same for calls and puts)</p>
            <div style="background:rgba(0,0,0,0.3);border-radius:6px;padding:0.7rem;
                        font-family:'Courier New',monospace;font-size:0.78rem;color:{COLORS['light_blue']};
                        margin:0.5rem 0;">
                <b style="color:{COLORS['accent_gold']};">Γ (cell F4):</b><br>
                =D8/(S_price*sigma*SQRT(T_years))<br><br>
                <b style="color:{COLORS['text_secondary']};font-size:0.72rem;">
                D8 = φ(d₁) = NORM.S.DIST(D2,FALSE)</b>
            </div>
            <p style="color:{COLORS['text_primary']};font-size:0.8rem;margin:0.4rem 0 0;">
                Live: Γ = <b style="color:{COLORS['light_blue']};">{res['gamma']:.6f}</b></p>
            <p style="color:{COLORS['text_secondary']};font-size:0.75rem;margin:0.2rem 0 0;">
                Always ≥ 0. Maximum at ATM near expiry.</p>
        </div>

        <div style="background:{COLORS['card_bg']};border:1px solid rgba(40,167,69,0.3);
                    border-left:4px solid {COLORS['success']};border-radius:10px;padding:1.2rem;margin-bottom:1rem;">
            <h4 style="color:{COLORS['success']};margin-top:0;font-family:'Playfair Display',serif;">
                ν Vega</h4>
            <p style="color:{COLORS['text_secondary']};font-size:0.8rem;margin:0 0 0.3rem;">
                ₹ change per 1% rise in implied volatility (same for calls and puts)</p>
            <div style="background:rgba(0,0,0,0.3);border-radius:6px;padding:0.7rem;
                        font-family:'Courier New',monospace;font-size:0.78rem;color:{COLORS['light_blue']};
                        margin:0.5rem 0;">
                <b style="color:{COLORS['accent_gold']};">ν (cell F5):</b><br>
                =S_price*D8*SQRT(T_years)/100<br><br>
                <b style="color:{COLORS['text_secondary']};font-size:0.72rem;">
                Divide by 100 → per 1% vol change</b>
            </div>
            <p style="color:{COLORS['text_primary']};font-size:0.8rem;margin:0.4rem 0 0;">
                Live: ν = <b style="color:{COLORS['success']};">₹{res['vega']:.4f}</b> per 1% vol</p>
            <p style="color:{COLORS['text_secondary']};font-size:0.75rem;margin:0.2rem 0 0;">
                VIX +2pp → option {'+' if res['vega']*2>0 else ''}₹{res['vega']*2:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_g2:
        st.markdown(f"""
        <div style="background:{COLORS['card_bg']};border:1px solid rgba(220,53,69,0.3);
                    border-left:4px solid {COLORS['danger']};border-radius:10px;padding:1.2rem;margin-bottom:1rem;">
            <h4 style="color:{COLORS['danger']};margin-top:0;font-family:'Playfair Display',serif;">
                Θ Theta</h4>
            <p style="color:{COLORS['text_secondary']};font-size:0.8rem;margin:0 0 0.3rem;">
                ₹ change per calendar day (always negative for long options)</p>
            <div style="background:rgba(0,0,0,0.3);border-radius:6px;padding:0.7rem;
                        font-family:'Courier New',monospace;font-size:0.78rem;color:{COLORS['light_blue']};
                        margin:0.5rem 0;">
                <b style="color:{COLORS['accent_gold']};">Call Θ (cell F6):</b><br>
                =(-S_price*D8*sigma/(2*SQRT(T_years))<br>
                &nbsp;-r_rate*K_strike*D9*D5)/365<br><br>
                <b style="color:{COLORS['accent_gold']};">Put Θ (cell F7):</b><br>
                =(-S_price*D8*sigma/(2*SQRT(T_years))<br>
                &nbsp;+r_rate*K_strike*D9*D7)/365<br><br>
                <b style="color:{COLORS['text_secondary']};font-size:0.72rem;">
                Divide by 365 → per calendar day</b>
            </div>
            <p style="color:{COLORS['text_primary']};font-size:0.8rem;margin:0.4rem 0 0;">
                Live: Θ = <b style="color:{COLORS['danger']};">₹{res['theta']:.4f}</b>/day
                &nbsp;(₹{res['theta']*7:.2f}/week)</p>
            <p style="color:{COLORS['text_secondary']};font-size:0.75rem;margin:0.2rem 0 0;">
                Always ≤ 0 for long options. Most negative at ATM.</p>
        </div>

        <div style="background:{COLORS['card_bg']};border:1px solid rgba(155,89,182,0.3);
                    border-left:4px solid #9b59b6;border-radius:10px;padding:1.2rem;margin-bottom:1rem;">
            <h4 style="color:#9b59b6;margin-top:0;font-family:'Playfair Display',serif;">
                ρ Rho</h4>
            <p style="color:{COLORS['text_secondary']};font-size:0.8rem;margin:0 0 0.3rem;">
                ₹ change per 1% rise in risk-free interest rate</p>
            <div style="background:rgba(0,0,0,0.3);border-radius:6px;padding:0.7rem;
                        font-family:'Courier New',monospace;font-size:0.78rem;color:{COLORS['light_blue']};
                        margin:0.5rem 0;">
                <b style="color:{COLORS['accent_gold']};">Call ρ (cell F8):</b><br>
                =K_strike*T_years*D9*D5/100<br><br>
                <b style="color:{COLORS['accent_gold']};">Put ρ (cell F9):</b><br>
                =-K_strike*T_years*D9*D7/100<br><br>
                <b style="color:{COLORS['text_secondary']};font-size:0.72rem;">
                Divide by 100 → per 1% rate change</b>
            </div>
            <p style="color:{COLORS['text_primary']};font-size:0.8rem;margin:0.4rem 0 0;">
                Live: Call ρ = <b style="color:#9b59b6;">₹{res['rho']:.4f}</b>
                &nbsp;| Put ρ = <b style="color:{COLORS['danger']};">₹{-res['rho']:.4f}</b></p>
            <p style="color:{COLORS['text_secondary']};font-size:0.75rem;margin:0.2rem 0 0;">
                Least important for short-dated NIFTY options.</p>
        </div>

        <div style="background:{COLORS['card_bg']};border:1px solid rgba(255,215,0,0.2);
                    border-left:4px solid rgba(255,215,0,0.5);border-radius:10px;padding:1.2rem;margin-bottom:1rem;">
            <h4 style="color:{COLORS['accent_gold']};margin-top:0;font-family:'Playfair Display',serif;">
                ⚡ Breakeven Daily Move (Gamma Scalping)</h4>
            <p style="color:{COLORS['text_secondary']};font-size:0.8rem;margin:0 0 0.3rem;">
                Minimum daily spot move for long gamma to profit</p>
            <div style="background:rgba(0,0,0,0.3);border-radius:6px;padding:0.7rem;
                        font-family:'Courier New',monospace;font-size:0.78rem;color:{COLORS['light_blue']};
                        margin:0.5rem 0;">
                <b style="color:{COLORS['accent_gold']};">BEP Move (cell F10):</b><br>
                =SQRT(-2*F6/F4)<br><br>
                <b style="color:{COLORS['text_secondary']};font-size:0.72rem;">
                F6=Theta, F4=Gamma. Result in ₹/day.</b>
            </div>
            <p style="color:{COLORS['text_primary']};font-size:0.8rem;margin:0.4rem 0 0;">
                Live: BEP Move = <b style="color:{COLORS['accent_gold']};">
                ₹{np.sqrt(-2*res['theta']/res['gamma']) if res['gamma']>0 else 0:.1f}</b>/day</p>
            <p style="color:{COLORS['text_secondary']};font-size:0.75rem;margin:0.2rem 0 0;">
                If |daily move| &gt; this → long gamma profits; else theta bleeds it out.</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Complete formula table ────────────────────────────────────────────
    section_title("📋 Complete Excel Formula Sheet — Copy-Paste Ready")

    st.markdown(f"""
    <div class="info-box">
        <h4 style="color:{COLORS['accent_gold']};margin-top:0;">How to use this table</h4>
        <p style="color:{COLORS['text_primary']};line-height:1.8;margin:0;">
        1. Create a new Excel sheet. Set up named cells (B2–B7) as per the Setup table above.<br>
        2. Copy the formulas below exactly into the specified cells.<br>
        3. All formulas use the named ranges — no hardcoded values.<br>
        4. Live Value column shows current output with sidebar parameters for verification.</p>
    </div>
    """, unsafe_allow_html=True)

    bep = np.sqrt(-2*res['theta']/res['gamma']) if res['gamma'] > 0 else 0
    full_formula_df = pd.DataFrame([
        # Intermediates
        {"Cell":"D2","Output":"d₁",
         "Formula":"=(LN(S_price/K_strike)+(r_rate+0.5*sigma^2)*T_years)/(sigma*SQRT(T_years))",
         "Live Value":f"{res['d1']:.6f}","Category":"Intermediate"},
        {"Cell":"D3","Output":"d₂",
         "Formula":"=D2-sigma*SQRT(T_years)",
         "Live Value":f"{res['d2']:.6f}","Category":"Intermediate"},
        {"Cell":"D4","Output":"N(d₁)",
         "Formula":"=NORM.S.DIST(D2,TRUE)",
         "Live Value":f"{norm.cdf(res['d1']):.6f}","Category":"Intermediate"},
        {"Cell":"D5","Output":"N(d₂)",
         "Formula":"=NORM.S.DIST(D3,TRUE)",
         "Live Value":f"{norm.cdf(res['d2']):.6f}","Category":"Intermediate"},
        {"Cell":"D6","Output":"N(−d₁)",
         "Formula":"=NORM.S.DIST(-D2,TRUE)",
         "Live Value":f"{norm.cdf(-res['d1']):.6f}","Category":"Intermediate"},
        {"Cell":"D7","Output":"N(−d₂)",
         "Formula":"=NORM.S.DIST(-D3,TRUE)",
         "Live Value":f"{norm.cdf(-res['d2']):.6f}","Category":"Intermediate"},
        {"Cell":"D8","Output":"φ(d₁) PDF",
         "Formula":"=NORM.S.DIST(D2,FALSE)",
         "Live Value":f"{norm.pdf(res['d1']):.6f}","Category":"Intermediate"},
        {"Cell":"D9","Output":"e^(−rT)",
         "Formula":"=EXP(-r_rate*T_years)",
         "Live Value":f"{np.exp(-r*T):.6f}","Category":"Intermediate"},
        # Prices
        {"Cell":"E2","Output":"Call Price",
         "Formula":"=S_price*D4-K_strike*D9*D5",
         "Live Value":f"₹{bsm(S,K,T,r,sigma,'call')['price']:.4f}","Category":"Option Price"},
        {"Cell":"E3","Output":"Put Price",
         "Formula":"=K_strike*D9*D7-S_price*D6",
         "Live Value":f"₹{bsm(S,K,T,r,sigma,'put')['price']:.4f}","Category":"Option Price"},
        {"Cell":"E4","Output":"PCP Check",
         "Formula":"=E2-E3-S_price+K_strike*D9",
         "Live Value":"≈ 0.0000","Category":"Option Price"},
        {"Cell":"E5","Output":"Intrinsic (Call)",
         "Formula":"=MAX(S_price-K_strike,0)",
         "Live Value":f"₹{max(S-K,0):.2f}","Category":"Option Price"},
        {"Cell":"E6","Output":"Intrinsic (Put)",
         "Formula":"=MAX(K_strike-S_price,0)",
         "Live Value":f"₹{max(K-S,0):.2f}","Category":"Option Price"},
        {"Cell":"E7","Output":"BEP Call",
         "Formula":"=K_strike+E2",
         "Live Value":f"₹{K+bsm(S,K,T,r,sigma,'call')['price']:.2f}","Category":"Option Price"},
        {"Cell":"E8","Output":"BEP Put",
         "Formula":"=K_strike-E3",
         "Live Value":f"₹{K-bsm(S,K,T,r,sigma,'put')['price']:.2f}","Category":"Option Price"},
        # Greeks
        {"Cell":"F2","Output":"Call Delta",
         "Formula":"=NORM.S.DIST(D2,TRUE)",
         "Live Value":f"{res['delta']:.6f}","Category":"Greek"},
        {"Cell":"F3","Output":"Put Delta",
         "Formula":"=NORM.S.DIST(D2,TRUE)-1",
         "Live Value":f"{res['delta']-1:.6f}","Category":"Greek"},
        {"Cell":"F4","Output":"Gamma",
         "Formula":"=D8/(S_price*sigma*SQRT(T_years))",
         "Live Value":f"{res['gamma']:.8f}","Category":"Greek"},
        {"Cell":"F5","Output":"Vega",
         "Formula":"=S_price*D8*SQRT(T_years)/100",
         "Live Value":f"₹{res['vega']:.6f}","Category":"Greek"},
        {"Cell":"F6","Output":"Call Theta",
         "Formula":"=(-S_price*D8*sigma/(2*SQRT(T_years))-r_rate*K_strike*D9*D5)/365",
         "Live Value":f"₹{res['theta']:.6f}","Category":"Greek"},
        {"Cell":"F7","Output":"Put Theta",
         "Formula":"=(-S_price*D8*sigma/(2*SQRT(T_years))+r_rate*K_strike*D9*D7)/365",
         "Live Value":f"₹{bsm(S,K,T,r,sigma,'put')['theta']:.6f}","Category":"Greek"},
        {"Cell":"F8","Output":"Call Rho",
         "Formula":"=K_strike*T_years*D9*D5/100",
         "Live Value":f"₹{res['rho']:.6f}","Category":"Greek"},
        {"Cell":"F9","Output":"Put Rho",
         "Formula":"=-K_strike*T_years*D9*D7/100",
         "Live Value":f"₹{-res['rho']:.6f}","Category":"Greek"},
        {"Cell":"F10","Output":"BEP Daily Move",
         "Formula":"=SQRT(-2*F6/F4)",
         "Live Value":f"₹{bep:.2f}","Category":"Greek"},
        # Position P&L
        {"Cell":"G2","Output":"Position P&L (Call)",
         "Formula":"=(E2-purchase_price)*lot_size",
         "Live Value":"Depends on purchase_price","Category":"P&L"},
        {"Cell":"G3","Output":"Daily Theta Cost",
         "Formula":"=F6*lot_size",
         "Live Value":f"₹{res['theta']*50:.2f} (per lot)","Category":"P&L"},
        {"Cell":"G4","Output":"VIX +1% Impact",
         "Formula":"=F5*lot_size",
         "Live Value":f"₹{res['vega']*50:.2f} (per lot)","Category":"P&L"},
        {"Cell":"G5","Output":"Delta Hedge Shares",
         "Formula":"=F2*lot_size",
         "Live Value":f"{res['delta']*50:.1f} shares (per lot)","Category":"P&L"},
    ])
    st.dataframe(full_formula_df, use_container_width=True, hide_index=True)

    # ── Common errors ─────────────────────────────────────────────────────
    section_title("⚠️ Common Excel Errors & Fixes")
    errors_df = pd.DataFrame([
        {"Error":"#DIV/0!","Cause":"T_years = 0 (option has expired)","Fix":"Ensure T_years > 0. Add IF(T_years<=0,0,...) guard."},
        {"Error":"#NUM!","Cause":"sigma=0 or S_price=0 or K_strike=0","Fix":"All inputs must be positive non-zero values."},
        {"Error":"#NAME?","Cause":"Named range not defined (e.g. S_price not set)","Fix":"Define named ranges via Name Box or Formulas → Name Manager."},
        {"Error":"#VALUE!","Cause":"Text in a numeric cell (e.g. '22,500' with comma)","Fix":"Remove commas/currency symbols — enter raw numbers only."},
        {"Error":"Wrong sign on Theta","Cause":"Missing negative sign or wrong N() function used","Fix":"Call Theta uses N(d₂) with minus; Put Theta uses N(−d₂) with plus before r term."},
        {"Error":"Gamma too large","Cause":"T_years very small (e.g., same-day expiry)","Fix":"Expected — Gamma explodes near expiry for ATM options. This is correct BSM behaviour."},
        {"Error":"Put-Call Parity ≠ 0","Cause":"Using different d1/d2 values for call vs put","Fix":"Reference the SAME D2, D3 cells for both call and put formulas."},
    ])
    st.dataframe(errors_df, use_container_width=True, hide_index=True)

    # ── Quick verification ────────────────────────────────────────────────
    section_title("✅ Quick Verification — Sanity Check Your Model")
    st.markdown(f"""
    <div class="info-box">
        <h4 style="color:{COLORS['accent_gold']};margin-top:0;">5 Checks to Verify Your Excel Model is Correct</h4>
        <ol style="color:{COLORS['text_primary']};line-height:2.2;margin:0;padding-left:1.3rem;">
            <li><b>Put-Call Parity:</b> Cell E4 = C − P − S + K·e<sup>−rT</sup> must be ≈ 0 (within ₹0.01)</li>
            <li><b>Delta bounds:</b> Call Delta in [0,1] · Put Delta in [−1,0] · At-the-money ≈ ±0.50</li>
            <li><b>Gamma positive:</b> Cell F4 must always be &gt; 0. If negative, formula sign error.</li>
            <li><b>Theta negative (call):</b> Cell F6 must be &lt; 0 always for European call.</li>
            <li><b>Deep OTM option:</b> Set S much lower than K (e.g., S=20000, K=23000) → price near ₹0, Delta near 0.</li>
        </ol>
        <p style="color:{COLORS['text_secondary']};font-size:0.82rem;margin:0.8rem 0 0;">
        💡 Cross-check your Excel values against the live metrics bar at the top of this app
        using the same S, K, T, r, σ — they should match to 4 decimal places.</p>
    </div>
    """, unsafe_allow_html=True)



# ============================================================================
# FOOTER
# ============================================================================
footer()
