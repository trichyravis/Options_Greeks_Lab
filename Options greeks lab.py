
"""
Options Greeks Lab — The Mountain Path: World of Finance
Prof. V. Ravichandran | 28+ Years Corporate Finance & Banking | 10+ Years Academic Excellence
Interactive Black-Scholes Greeks Lab with 3D Surfaces and P&L Simulation
"""

import streamlit as st
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Options Greeks Lab | The Mountain Path",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Colour Palette ─────────────────────────────────────────────────────────
DARK_BLUE   = "#003366"
MID_BLUE    = "#004d80"
LIGHT_BLUE  = "#ADD8E6"
GOLD        = "#FFD700"
WHITE       = "#FFFFFF"
LIGHT_GREY  = "#f5f7fa"

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* Global */
  .stApp {{ background: {LIGHT_GREY}; }}

  /* Hero Header */
  .hero {{
    background: linear-gradient(135deg, {DARK_BLUE} 0%, {MID_BLUE} 60%, #0077b6 100%);
    padding: 2rem 2.5rem 1.5rem;
    border-radius: 14px;
    margin-bottom: 1.5rem;
    border-left: 6px solid {GOLD};
  }}
  .hero h1 {{ color: {WHITE}; font-size: 2.1rem; margin: 0; font-weight: 700; }}
  .hero .sub  {{ color: {LIGHT_BLUE}; font-size: 1rem; margin-top: 0.4rem; }}
  .hero .badge {{
    display: inline-block; background: {GOLD}; color: {DARK_BLUE};
    padding: 3px 12px; border-radius: 20px; font-size: 0.8rem;
    font-weight: 700; margin-top: 0.6rem;
  }}

  /* Metric Cards */
  .metric-row {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.2rem; }}
  .metric-card {{
    background: {WHITE}; border-radius: 10px; padding: 1rem 1.4rem;
    flex: 1; min-width: 130px; border-top: 4px solid {DARK_BLUE};
    box-shadow: 0 2px 8px rgba(0,51,102,0.09);
  }}
  .metric-card.gold {{ border-top-color: {GOLD}; }}
  .metric-card.blue {{ border-top-color: {LIGHT_BLUE}; }}
  .metric-card h4 {{ color: #666; font-size: 0.78rem; margin: 0; text-transform: uppercase; letter-spacing: .06em; }}
  .metric-card .val {{ color: {DARK_BLUE}; font-size: 1.55rem; font-weight: 700; margin: 0.2rem 0 0; }}
  .metric-card .sub {{ color: #888; font-size: 0.72rem; }}

  /* Section Headers */
  .section-title {{
    color: {DARK_BLUE}; font-size: 1.15rem; font-weight: 700;
    border-left: 4px solid {GOLD}; padding-left: 0.7rem; margin: 1.5rem 0 0.8rem;
  }}

  /* Info Boxes */
  .info-box {{
    background: #e8f4f8; border-left: 4px solid {MID_BLUE};
    padding: 0.8rem 1rem; border-radius: 0 8px 8px 0; margin-bottom: 1rem;
    font-size: 0.88rem; color: #1a3a5c;
  }}
  .formula-box {{
    background: #fff9e6; border-left: 4px solid {GOLD};
    padding: 0.8rem 1rem; border-radius: 0 8px 8px 0; margin-bottom: 1rem;
    font-size: 0.88rem; color: #4a3800; font-family: 'Courier New', monospace;
  }}

  /* Footer */
  .footer {{
    text-align: center; padding: 1.2rem; color: #666; font-size: 0.78rem;
    border-top: 1px solid #ddd; margin-top: 2rem;
  }}
  .footer span {{ color: {DARK_BLUE}; font-weight: 600; }}

  /* Sidebar */
  section[data-testid="stSidebar"] {{
    background: {DARK_BLUE} !important;
  }}
  section[data-testid="stSidebar"] * {{ color: {WHITE} !important; }}
  section[data-testid="stSidebar"] .stSlider > div {{ color: {WHITE} !important; }}
  section[data-testid="stSidebar"] h3 {{
    color: {GOLD} !important; border-bottom: 1px solid rgba(255,215,0,0.3);
    padding-bottom: 0.4rem;
  }}
  .sidebar-label {{
    color: {LIGHT_BLUE} !important; font-size: 0.78rem;
    text-transform: uppercase; letter-spacing: .05em;
  }}
  div[data-testid="stSelectbox"] label {{ color: {WHITE} !important; }}

  /* Tab Styling */
  .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
  .stTabs [data-baseweb="tab"] {{
    background: {WHITE}; border-radius: 8px 8px 0 0;
    color: {DARK_BLUE}; font-weight: 600; padding: 0.5rem 1.2rem;
    border: 1px solid #ddd;
  }}
  .stTabs [aria-selected="true"] {{
    background: {DARK_BLUE} !important; color: {GOLD} !important;
    border-color: {DARK_BLUE} !important;
  }}

  /* DataFrame */
  .dataframe {{ font-size: 0.85rem !important; }}
</style>
""", unsafe_allow_html=True)


# ─── Black-Scholes Engine ────────────────────────────────────────────────────
def bsm(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes-Merton pricing and Greeks."""
    if T <= 0 or sigma <= 0:
        return {g: 0.0 for g in ["price","delta","gamma","vega","theta","rho","d1","d2"]}
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    phi  = norm.pdf(d1)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (-(S * phi * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        rho   = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        theta = (-(S * phi * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho   = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    gamma = phi / (S * sigma * np.sqrt(T))
    vega  = S * phi * np.sqrt(T) / 100
    return dict(price=price, delta=delta, gamma=gamma,
                vega=vega, theta=theta, rho=rho, d1=d1, d2=d2)

def bsm_vec(S, K, T, r, sigma, option_type="call"):
    """Vectorized BSM for surface generation."""
    S, K, T, sigma = np.array(S), np.array(K), np.array(T), np.array(sigma)
    mask = (T > 0) & (sigma > 0)
    out  = {g: np.zeros_like(S, dtype=float)
            for g in ["price","delta","gamma","vega","theta","rho"]}
    if not mask.any():
        return out
    d1 = np.where(mask,
         (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(np.maximum(T,1e-10))),
         0.0)
    d2 = d1 - sigma * np.sqrt(np.maximum(T, 1e-10))
    phi = norm.pdf(d1)
    if option_type == "call":
        out["price"]  = np.where(mask, S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2), 0)
        out["delta"]  = np.where(mask, norm.cdf(d1), 0)
        out["theta"]  = np.where(mask, (-(S*phi*sigma)/(2*np.sqrt(np.maximum(T,1e-10)))
                                         - r*K*np.exp(-r*T)*norm.cdf(d2))/365, 0)
        out["rho"]    = np.where(mask, K*T*np.exp(-r*T)*norm.cdf(d2)/100, 0)
    else:
        out["price"]  = np.where(mask, K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1), 0)
        out["delta"]  = np.where(mask, norm.cdf(d1)-1, 0)
        out["theta"]  = np.where(mask, (-(S*phi*sigma)/(2*np.sqrt(np.maximum(T,1e-10)))
                                         + r*K*np.exp(-r*T)*norm.cdf(-d2))/365, 0)
        out["rho"]    = np.where(mask, -K*T*np.exp(-r*T)*norm.cdf(-d2)/100, 0)
    out["gamma"] = np.where(mask, phi / (S * sigma * np.sqrt(np.maximum(T,1e-10))), 0)
    out["vega"]  = np.where(mask, S * phi * np.sqrt(np.maximum(T,1e-10)) / 100, 0)
    return out


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ Options Greeks Lab")
    st.markdown("---")
    st.markdown("### 📐 Option Parameters")

    S     = st.slider("Spot Price (S) ₹",      50,   500, 100, 1)
    K     = st.slider("Strike Price (K) ₹",     50,   500, 100, 1)
    T_days= st.slider("Days to Expiry",          1,   365,  90, 1)
    T     = T_days / 365
    sigma = st.slider("Implied Volatility (σ) %", 5,   100,  20, 1) / 100
    r     = st.slider("Risk-Free Rate (r) %",    0.1,  15.0, 6.5, 0.1) / 100

    st.markdown("---")
    st.markdown("### 🎯 Option Type")
    option_type = st.selectbox("", ["call", "put"], format_func=lambda x: x.title())

    st.markdown("---")
    st.markdown("### 📊 Surface Settings")
    surface_greek = st.selectbox("Greek for 3D Surface",
        ["delta","gamma","vega","theta","rho","price"])
    surface_x = st.selectbox("X-axis", ["Spot Price", "Strike Price", "Volatility"])
    surface_y = st.selectbox("Y-axis (≠ X)", ["Time to Expiry (Days)", "Volatility", "Spot Price"])

    st.markdown("---")
    st.markdown("### 💰 P&L Simulation")
    purchase_price = st.number_input("Option Purchase Price ₹", 0.1, 500.0, 5.0, 0.1)
    lot_size = st.number_input("Lot Size (contracts)", 1, 10000, 50, 1)

    st.markdown("---")
    st.markdown('<div style="font-size:0.75rem; color:#ADD8E6;">The Mountain Path<br>World of Finance</div>',
                unsafe_allow_html=True)


# ─── Compute BSM ─────────────────────────────────────────────────────────────
g = bsm(S, K, T, r, sigma, option_type)
moneyness = S / K


# ─── Hero Header ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <h1>⚡ Options Greeks Lab</h1>
  <div class="sub">Black-Scholes-Merton Framework · 3D Greek Surfaces · P&L Simulation · Sensitivity Analysis</div>
  <span class="badge">LIVE LAB</span>&nbsp;
  <span class="badge" style="background:#e8f4f8;color:{DARK_BLUE};">The Mountain Path · World of Finance</span>
</div>
""", unsafe_allow_html=True)


# ─── Live Metric Cards ────────────────────────────────────────────────────────
moneyness_label = "ITM" if (option_type=="call" and S>K) or (option_type=="put" and S<K) \
    else ("ATM" if S==K else "OTM")

st.markdown(f"""
<div class="metric-row">
  <div class="metric-card gold">
    <h4>Option Price</h4>
    <div class="val">₹{g['price']:.4f}</div>
    <div class="sub">{option_type.title()} · {moneyness_label} · Lot ₹{g['price']*lot_size:,.0f}</div>
  </div>
  <div class="metric-card">
    <h4>Delta (Δ)</h4>
    <div class="val">{g['delta']:.4f}</div>
    <div class="sub">Δ Price per ₹1 move</div>
  </div>
  <div class="metric-card">
    <h4>Gamma (Γ)</h4>
    <div class="val">{g['gamma']:.4f}</div>
    <div class="sub">Δ of Delta per ₹1</div>
  </div>
  <div class="metric-card blue">
    <h4>Vega (ν)</h4>
    <div class="val">{g['vega']:.4f}</div>
    <div class="sub">Price per 1% vol move</div>
  </div>
  <div class="metric-card">
    <h4>Theta (Θ)</h4>
    <div class="val">{g['theta']:.4f}</div>
    <div class="sub">Price decay per day</div>
  </div>
  <div class="metric-card">
    <h4>Rho (ρ)</h4>
    <div class="val">{g['rho']:.4f}</div>
    <div class="sub">Price per 1% rate move</div>
  </div>
  <div class="metric-card">
    <h4>d₁ / d₂</h4>
    <div class="val" style="font-size:1.1rem;">{g['d1']:.3f} / {g['d2']:.3f}</div>
    <div class="sub">N(d₁)={norm.cdf(g['d1']):.3f}</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Greek Profiles",
    "🏔 3D Surfaces",
    "💰 P&L Simulation",
    "🔢 Sensitivity Table",
    "📚 Theory & Formulae"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Greek Profiles (vs Spot)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Greek Profiles vs. Spot Price</div>', unsafe_allow_html=True)

    spot_range = np.linspace(max(5, S*0.5), S*1.5, 300)
    profiles = {gk: [] for gk in ["price","delta","gamma","vega","theta","rho"]}
    for s in spot_range:
        res = bsm(s, K, T, r, sigma, option_type)
        for gk in profiles:
            profiles[gk].append(res[gk])

    COLORS = {
        "price":  DARK_BLUE,
        "delta":  "#e63946",
        "gamma":  "#2a9d8f",
        "vega":   "#e9c46a",
        "theta":  "#f4a261",
        "rho":    "#a8dadc",
    }

    fig1 = make_subplots(rows=2, cols=3,
        subplot_titles=["Option Price", "Delta (Δ)", "Gamma (Γ)",
                         "Vega (ν)", "Theta (Θ)", "Rho (ρ)"],
        vertical_spacing=0.14, horizontal_spacing=0.1)
    positions = [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
    for (gk, (row,col)) in zip(["price","delta","gamma","vega","theta","rho"], positions):
        fig1.add_trace(go.Scatter(
            x=spot_range, y=profiles[gk], name=gk.title(),
            line=dict(color=COLORS[gk], width=2.5), fill='tozeroy',
            fillcolor=COLORS[gk].replace("#","rgba(") + "0.08)" if "#" in COLORS[gk] else None
        ), row=row, col=col)
        fig1.add_vline(x=S, line_dash="dot", line_color="grey", line_width=1, row=row, col=col)
        fig1.add_vline(x=K, line_dash="dash", line_color=GOLD, line_width=1.5, row=row, col=col)

    fig1.update_layout(height=600, showlegend=False,
        plot_bgcolor=WHITE, paper_bgcolor=LIGHT_GREY,
        font=dict(family="Arial", size=11),
        margin=dict(t=50, b=20))
    fig1.update_xaxes(showgrid=True, gridcolor="#eee")
    fig1.update_yaxes(showgrid=True, gridcolor="#eee")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown(f"""
    <div class="info-box">
    🏔 <b>Reading the chart:</b> Vertical grey dotted line = current spot (₹{S}).
    Gold dashed line = strike (₹{K}).
    Greeks are computed live using the Black-Scholes-Merton model.
    </div>
    """, unsafe_allow_html=True)

    # Greek vs Vol & Time profiles side by side
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Greeks vs. Implied Volatility</div>', unsafe_allow_html=True)
        vol_range = np.linspace(0.05, 1.0, 200)
        delta_v, gamma_v, vega_v = [], [], []
        for v in vol_range:
            rv = bsm(S, K, T, r, v, option_type)
            delta_v.append(rv["delta"])
            gamma_v.append(rv["gamma"])
            vega_v.append(rv["vega"])
        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(x=vol_range*100, y=delta_v, name="Delta", line=dict(color="#e63946", width=2)))
        fig_v.add_trace(go.Scatter(x=vol_range*100, y=gamma_v, name="Gamma", line=dict(color="#2a9d8f", width=2)))
        fig_v.add_trace(go.Scatter(x=vol_range*100, y=vega_v,  name="Vega",  line=dict(color=GOLD, width=2)))
        fig_v.add_vline(x=sigma*100, line_dash="dot", line_color=DARK_BLUE, line_width=2)
        fig_v.update_layout(height=300, margin=dict(t=10,b=30),
            xaxis_title="Volatility (%)", plot_bgcolor=WHITE, paper_bgcolor=LIGHT_GREY)
        st.plotly_chart(fig_v, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Theta Decay — Time vs. Price</div>', unsafe_allow_html=True)
        t_range = np.linspace(0.003, 1.0, 200)
        price_t, theta_t = [], []
        for t in t_range:
            rt = bsm(S, K, t, r, sigma, option_type)
            price_t.append(rt["price"])
            theta_t.append(rt["theta"])
        fig_t = make_subplots(specs=[[{"secondary_y": True}]])
        fig_t.add_trace(go.Scatter(x=t_range*365, y=price_t, name="Price",
            line=dict(color=DARK_BLUE, width=2.5)), secondary_y=False)
        fig_t.add_trace(go.Scatter(x=t_range*365, y=theta_t, name="Theta",
            line=dict(color="#f4a261", width=2, dash="dash")), secondary_y=True)
        fig_t.add_vline(x=T_days, line_dash="dot", line_color="grey", line_width=2)
        fig_t.update_layout(height=300, margin=dict(t=10,b=30),
            xaxis_title="Days to Expiry", plot_bgcolor=WHITE, paper_bgcolor=LIGHT_GREY)
        st.plotly_chart(fig_t, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — 3D Greek Surfaces
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">3D Greek Surface Visualization</div>', unsafe_allow_html=True)

    # Build axis ranges
    def make_range(name, S, K, sigma):
        if name == "Spot Price":
            return np.linspace(max(5, S*0.5), S*1.5, 60)
        elif name == "Strike Price":
            return np.linspace(max(5, K*0.5), K*1.5, 60)
        elif name == "Volatility":
            return np.linspace(0.05, 1.0, 60)
        elif name == "Time to Expiry (Days)":
            return np.linspace(1, 365, 60)

    xvals = make_range(surface_x, S, K, sigma)
    yvals = make_range(surface_y, S, K, sigma)
    XX, YY = np.meshgrid(xvals, yvals)

    def resolve(name, val, S, K, T, r, sigma):
        if name == "Spot Price":       return val,    K,     T,           r, sigma
        if name == "Strike Price":     return S,      val,   T,           r, sigma
        if name == "Volatility":       return S,      K,     T,           r, val
        if name == "Time to Expiry (Days)": return S, K,     val/365,     r, sigma

    ZZ = np.zeros_like(XX)
    for i in range(XX.shape[0]):
        for j in range(XX.shape[1]):
            sx, kx, tx, rx, vx = resolve(surface_x, XX[i,j], S, K, T, r, sigma)
            sy, ky, ty, ry, vy = resolve(surface_y, YY[i,j], S, K, T, r, sigma)
            # X-axis wins for shared params; Y modifies only its own param
            sp = sx if surface_x=="Spot Price"  else (sy if surface_y=="Spot Price"  else S)
            kp = kx if surface_x=="Strike Price" else (ky if surface_y=="Strike Price" else K)
            tp = tx if surface_x=="Time to Expiry (Days)" else (ty if surface_y=="Time to Expiry (Days)" else T)
            vp = vx if surface_x=="Volatility"  else (vy if surface_y=="Volatility"  else sigma)
            res = bsm(sp, kp, tp, r, vp, option_type)
            ZZ[i,j] = res.get(surface_greek, 0.0)

    fig3d = go.Figure(data=[go.Surface(
        x=XX, y=YY, z=ZZ,
        colorscale=[
            [0.0, DARK_BLUE], [0.25, MID_BLUE], [0.5, "#0099cc"],
            [0.75, LIGHT_BLUE], [1.0, GOLD]
        ],
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor=GOLD, project_z=True)
        ),
        lighting=dict(ambient=0.7, diffuse=0.8, roughness=0.5, specular=0.3)
    )])

    fig3d.update_layout(
        title=dict(text=f"{surface_greek.title()} Surface — {option_type.title()} Option",
                   font=dict(color=DARK_BLUE, size=16)),
        scene=dict(
            xaxis_title=surface_x,
            yaxis_title=surface_y,
            zaxis_title=surface_greek.title(),
            xaxis=dict(backgroundcolor=LIGHT_GREY, gridcolor="#ccc"),
            yaxis=dict(backgroundcolor=LIGHT_GREY, gridcolor="#ccc"),
            zaxis=dict(backgroundcolor=LIGHT_GREY, gridcolor="#ccc"),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.0))
        ),
        height=620, margin=dict(t=50, b=10),
        paper_bgcolor=LIGHT_GREY, font=dict(family="Arial")
    )
    st.plotly_chart(fig3d, use_container_width=True)

    # Quick insight box
    greek_insights = {
        "delta": "Delta is the slope of the option price curve. Call deltas range [0,1]; put deltas range [-1,0]. ATM options have delta ≈ 0.5 (calls) or -0.5 (puts).",
        "gamma": "Gamma peaks at-the-money and near expiry. High gamma = rapid delta changes = gamma risk for market makers.",
        "vega":  "Vega is highest for ATM options with long expiry. Long options are always long vega — they benefit from rising volatility.",
        "theta": "Theta is negative for long options (time decay). Decay accelerates as expiry approaches, especially for ATM options.",
        "rho":   "Rho is positive for call options and negative for puts. Long-dated options have larger rho exposure.",
        "price": "The option price surface shows intrinsic + time value across spot and time dimensions.",
    }
    st.markdown(f"""
    <div class="info-box">
    <b>📐 {surface_greek.title()} Insight:</b> {greek_insights.get(surface_greek, "")}
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — P&L Simulation
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">P&L Simulation at Expiry & Before Expiry</div>', unsafe_allow_html=True)

    spot_sim = np.linspace(max(5, S*0.5), S*1.5, 400)

    # Payoffs at expiry
    if option_type == "call":
        payoff = np.maximum(spot_sim - K, 0)
    else:
        payoff = np.maximum(K - spot_sim, 0)
    pnl_expiry = (payoff - purchase_price) * lot_size

    # Before expiry (multiple time horizons)
    fig_pnl = go.Figure()

    horizons = [T_days, int(T_days*0.75), int(T_days*0.5), int(T_days*0.25), 1]
    horizons = sorted(set([max(1,h) for h in horizons if h > 0]), reverse=True)
    alphas   = [1.0, 0.75, 0.55, 0.38, 0.22]
    blues    = [DARK_BLUE, MID_BLUE, "#0077b6", "#0096c7", "#00b4d8"]

    for idx, (h, al, bl) in enumerate(zip(horizons, alphas, blues)):
        prices_h = []
        for s in spot_sim:
            r_h = bsm(s, K, h/365, r, sigma, option_type)
            prices_h.append(r_h["price"])
        pnl_h = (np.array(prices_h) - purchase_price) * lot_size
        fig_pnl.add_trace(go.Scatter(
            x=spot_sim, y=pnl_h, name=f"{h}d to expiry",
            line=dict(color=bl, width=2, dash="dot" if idx>0 else "solid"),
            opacity=al
        ))

    # At expiry — solid bold
    fig_pnl.add_trace(go.Scatter(
        x=spot_sim, y=pnl_expiry, name="At Expiry",
        line=dict(color="#e63946", width=3), fill='tozeroy',
        fillcolor="rgba(230,57,70,0.07)"
    ))

    fig_pnl.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
    fig_pnl.add_vline(x=S, line_dash="dot", line_color="grey", line_width=1.5,
                      annotation_text="Spot", annotation_position="top")
    fig_pnl.add_vline(x=K, line_dash="dash", line_color=GOLD, line_width=2,
                      annotation_text="Strike", annotation_position="top")
    # Breakeven
    be = K + purchase_price if option_type=="call" else K - purchase_price
    fig_pnl.add_vline(x=be, line_dash="longdash", line_color="#2a9d8f", line_width=1.5,
                      annotation_text=f"BE ₹{be:.1f}", annotation_position="bottom")

    fig_pnl.update_layout(
        height=480, xaxis_title="Spot Price at Expiry (₹)",
        yaxis_title="P&L (₹)", legend=dict(orientation="h", y=1.04),
        plot_bgcolor=WHITE, paper_bgcolor=LIGHT_GREY,
        font=dict(family="Arial", size=11),
        margin=dict(t=20, b=40)
    )
    st.plotly_chart(fig_pnl, use_container_width=True)

    # P&L Summary stats
    max_loss = -purchase_price * lot_size
    if option_type == "call":
        max_gain = "Unlimited"
        breakeven = K + purchase_price
    else:
        max_gain = f"₹{(K - purchase_price) * lot_size:,.0f}"
        breakeven = K - purchase_price

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Max Loss",     f"₹{max_loss:,.0f}",     "Premium paid × lot size")
    col2.metric("Max Gain",     max_gain)
    col3.metric("Breakeven",    f"₹{breakeven:.2f}",     f"{'K+P' if option_type=='call' else 'K-P'}")
    col4.metric("Current P&L",  f"₹{(g['price']-purchase_price)*lot_size:,.0f}",
                f"{'▲' if g['price']>purchase_price else '▼'} vs purchase")

    st.markdown("---")
    st.markdown('<div class="section-title">Scenario Analysis — Spot × Volatility P&L Grid</div>',
                unsafe_allow_html=True)

    spot_scen  = np.linspace(S*0.7, S*1.3, 11)
    vol_scen   = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])
    grid_data  = {}
    for v in vol_scen:
        row_pnl = []
        for s in spot_scen:
            rr = bsm(s, K, T, r, v, option_type)
            row_pnl.append(round((rr["price"] - purchase_price) * lot_size, 0))
        grid_data[f"Vol {v*100:.0f}%"] = row_pnl

    df_grid = pd.DataFrame(grid_data, index=[f"₹{s:.0f}" for s in spot_scen])

    def color_pnl(val):
        if val > 0:   return f"background-color: #d4edda; color: #155724; font-weight:600"
        elif val < 0: return f"background-color: #f8d7da; color: #721c24; font-weight:600"
        return ""

    st.dataframe(df_grid.style.applymap(color_pnl), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Sensitivity Table
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Greeks Sensitivity — Bump & Reprice</div>',
                unsafe_allow_html=True)

    bumps = {
        "Base": (S, K, T, r, sigma),
        "S +5%": (S*1.05, K, T, r, sigma),
        "S -5%": (S*0.95, K, T, r, sigma),
        "Vol +5%": (S, K, T, r, sigma+0.05),
        "Vol -5%": (S, K, T, r, sigma-0.05),
        "T -30d": (S, K, max(0.003, T-30/365), r, sigma),
        "r +1%": (S, K, T, r+0.01, sigma),
        "r -1%": (S, K, T, r-0.01, sigma),
    }

    rows = []
    base_g = bsm(*bumps["Base"], option_type)
    for label, params in bumps.items():
        bg = bsm(*params, option_type)
        rows.append({
            "Scenario": label,
            "Price ₹":    round(bg["price"], 4),
            "ΔPrice":     round(bg["price"]-base_g["price"], 4),
            "Delta":      round(bg["delta"], 4),
            "Gamma":      round(bg["gamma"], 6),
            "Vega":       round(bg["vega"],  4),
            "Theta":      round(bg["theta"], 4),
            "Rho":        round(bg["rho"],   4),
        })
    df_sens = pd.DataFrame(rows)

    def highlight_base(row):
        if row["Scenario"] == "Base":
            return [f"background-color:{DARK_BLUE};color:{WHITE};font-weight:700"]*len(row)
        return [""]*len(row)

    st.dataframe(df_sens.style.apply(highlight_base, axis=1), use_container_width=True, height=340)

    st.markdown("---")
    st.markdown('<div class="section-title">Greeks vs. Moneyness (K range)</div>',
                unsafe_allow_html=True)

    K_range   = np.linspace(max(5, S*0.6), S*1.4, 80)
    mono_data = {"K": K_range, "delta": [], "gamma": [], "vega": [], "theta": []}
    for k in K_range:
        rk = bsm(S, k, T, r, sigma, option_type)
        for gk in ["delta","gamma","vega","theta"]:
            mono_data[gk].append(rk[gk])

    fig_mono = make_subplots(rows=1, cols=4,
        subplot_titles=["Delta","Gamma","Vega","Theta"])
    colors_m = ["#e63946","#2a9d8f","#e9c46a","#f4a261"]
    for i, (gk, clr) in enumerate(zip(["delta","gamma","vega","theta"], colors_m), 1):
        fig_mono.add_trace(go.Scatter(x=K_range, y=mono_data[gk], name=gk,
            line=dict(color=clr, width=2.5)), row=1, col=i)
        fig_mono.add_vline(x=S, line_dash="dot", line_color=DARK_BLUE, line_width=1.5,
                           row=1, col=i)
    fig_mono.update_layout(height=280, showlegend=False,
        plot_bgcolor=WHITE, paper_bgcolor=LIGHT_GREY,
        margin=dict(t=40, b=20), font=dict(family="Arial", size=10))
    st.plotly_chart(fig_mono, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Theory & Formulae
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">Black-Scholes-Merton Framework</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown("""
        <div class="formula-box">
        <b>BSM Call Price:</b><br>
        C = S·N(d₁) − K·e^{−rT}·N(d₂)<br><br>
        <b>BSM Put Price:</b><br>
        P = K·e^{−rT}·N(−d₂) − S·N(−d₁)<br><br>
        <b>d₁</b> = [ln(S/K) + (r + σ²/2)T] / (σ√T)<br>
        <b>d₂</b> = d₁ − σ√T
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="formula-box">
        <b>Greeks Formulae:</b><br>
        Δ_call = N(d₁)    |   Δ_put = N(d₁) − 1<br>
        Γ      = φ(d₁) / (S·σ·√T)<br>
        ν      = S·φ(d₁)·√T / 100<br>
        Θ_call = −[S·φ(d₁)·σ/(2√T) + rK·e^{−rT}·N(d₂)] / 365<br>
        ρ_call = K·T·e^{−rT}·N(d₂) / 100
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-box">
        <b>Model Assumptions:</b><br>
        • Continuous trading; no transaction costs<br>
        • Constant volatility (σ) and risk-free rate (r)<br>
        • No dividends on underlying<br>
        • European-style option (exercise at expiry only)<br>
        • Log-normal distribution of returns<br>
        • No arbitrage conditions hold
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
        <b>Put-Call Parity:</b><br>
        C − P = S − K·e^{−rT}<br><br>
        <b>Delta-Hedging:</b><br>
        A delta-neutral portfolio: −1 option + Δ shares of stock<br>
        Requires continuous rebalancing as S and t change.<br><br>
        <b>Greeks Risk Management:</b><br>
        Delta: directional risk | Gamma: convexity risk<br>
        Vega: vol risk | Theta: time decay | Rho: rate risk
        </div>
        """, unsafe_allow_html=True)

    # Greek quick-reference table
    st.markdown('<div class="section-title">Greeks Quick Reference</div>', unsafe_allow_html=True)
    ref_data = {
        "Greek": ["Delta (Δ)","Gamma (Γ)","Vega (ν)","Theta (Θ)","Rho (ρ)"],
        "Definition": [
            "Rate of change of option price w.r.t. spot",
            "Rate of change of Delta w.r.t. spot",
            "Rate of change of price w.r.t. volatility",
            "Rate of change of price w.r.t. time",
            "Rate of change of price w.r.t. interest rate",
        ],
        "Call Range": ["[0, 1]","Always ≥ 0","Always ≥ 0","Negative","Positive"],
        "Put Range":  ["[−1, 0]","Always ≥ 0","Always ≥ 0","Negative","Negative"],
        "ATM at Expiry": ["0 or 1","Very High","≈ 0","Very Negative","—"],
        "Hedge Use": [
            "Delta hedge: hold Δ shares per short option",
            "Gamma hedge: add options to reduce convexity",
            "Vega hedge: match vol exposure",
            "Cannot hedge time (Theta is a cost)",
            "Rho hedge: interest rate instruments",
        ]
    }
    st.dataframe(pd.DataFrame(ref_data), use_container_width=True)

    st.markdown('<div class="section-title">Volatility Smile & Surface</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>Volatility Smile:</b> In practice, BSM's assumption of constant vol breaks down.
    OTM options (especially puts) trade at higher implied volatility — this creates a
    "volatility smile" (equity markets often show a "smirk" or skew, with higher IV for
    low strikes). The full 3D plot of IV vs. Strike and Maturity is the <b>Volatility Surface</b>.
    </div>
    """, unsafe_allow_html=True)

    # Simulated vol smile
    K_smile = np.linspace(S*0.7, S*1.3, 100)
    moneyness_smile = K_smile / S
    # Parametric smile: higher vol at wings
    iv_smile = sigma + 0.08*(moneyness_smile - 1)**2 - 0.03*(moneyness_smile - 1)
    fig_smile = go.Figure()
    fig_smile.add_trace(go.Scatter(
        x=K_smile, y=iv_smile*100, name="IV Smile",
        line=dict(color=DARK_BLUE, width=2.5), fill='tozeroy',
        fillcolor=f"rgba(0,51,102,0.07)"
    ))
    fig_smile.add_hline(y=sigma*100, line_dash="dash", line_color=GOLD, line_width=1.5,
                        annotation_text="Flat BSM vol", annotation_position="right")
    fig_smile.add_vline(x=K, line_dash="dot", line_color="grey", line_width=1.5)
    fig_smile.update_layout(
        height=280, xaxis_title="Strike (₹)", yaxis_title="Implied Volatility (%)",
        plot_bgcolor=WHITE, paper_bgcolor=LIGHT_GREY,
        margin=dict(t=10, b=30), font=dict(family="Arial")
    )
    st.plotly_chart(fig_smile, use_container_width=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
  <span>Prof. V. Ravichandran</span> · 28+ Years Corporate Finance & Banking ·
  10+ Years Academic Excellence<br>
  <span>The Mountain Path — World of Finance</span> · Options Greeks Lab ·
  Black-Scholes-Merton Framework
</div>
""", unsafe_allow_html=True)
