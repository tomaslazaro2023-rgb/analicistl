"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        F1 TELEMETRY DASHBOARD v2 — MoTeC-Style Pro Edition                 ║
║        FastF1 + Streamlit + Plotly  ·  Temporada 2026 (sin DRS)            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  FEATURES:                                                                  ║
║  · 6-channel synchronized telemetry (distance-based axis)                  ║
║  · Track Map coloreado por velocidad (Speed Heatmap)                        ║
║  · Análisis de Zonas de Frenada con tabla detallada                        ║
║  · Histograma de velocidades comparativo                                    ║
║  · Speed vs Throttle efficiency scatter                                     ║
║  · Gear usage analysis (% of lap distance)                                  ║
║  · Lap-by-lap pace progression chart                                        ║
║  · Lap comparison table (all laps, conditional formatting)                  ║
║  · Channel statistics table                                                 ║
║  · Self-contained HTML report export                                        ║
║  · CSV export (all channels)                                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

Instalación:
    pip install streamlit fastf1 plotly pandas numpy scipy

Ejecución:
    streamlit run f1_telemetry_dashboard.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d

# ─── FastF1 ───────────────────────────────────────────────────────────────────
try:
    import fastf1
    FASTF1_AVAILABLE = True
except ImportError:
    FASTF1_AVAILABLE = False

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Telemetry Pro v2",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Share+Tech+Mono&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    background-color: #070B0F !important;
    color: #C8D6E5 !important;
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0C1018 0%,#070B0F 100%) !important;
    border-right: 1px solid #1A2535;
}
section[data-testid="stSidebar"] * { color: #C8D6E5 !important; }
section[data-testid="stSidebar"] label {
    color: #566A7F !important; font-size: 10px !important;
    text-transform: uppercase; letter-spacing: 1.2px;
    font-family: 'Share Tech Mono', monospace !important;
}
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #0C1018 !important; border: 1px solid #1A2535 !important;
    border-radius: 3px !important;
}

/* ── Main ── */
.main > div { padding: 0.8rem 1.8rem; }

/* ── Header ── */
.dash-header {
    display: flex; align-items: center; gap: 14px;
    padding: 14px 0 12px; border-bottom: 2px solid #E8002D;
    margin-bottom: 20px;
}
.dash-header h1 {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 22px !important; font-weight: 900 !important;
    color: #FFF !important; letter-spacing: 3px; margin: 0 !important;
}
.ver-badge {
    background: #E8002D; color: #fff;
    font-family: 'Share Tech Mono', monospace;
    font-size: 9px; letter-spacing: 1px;
    padding: 2px 7px; border-radius: 2px; margin-left: 6px;
    vertical-align: middle;
}
.dash-header .subtitle {
    font-family: 'Share Tech Mono', monospace;
    color: #566A7F; font-size: 10px; letter-spacing: 2px; margin-top: 3px;
}
.red-bar { width: 4px; height: 46px; background: #E8002D; border-radius: 2px; flex-shrink:0; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0C1018 !important; border-bottom: 1px solid #1A2535 !important; gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Share Tech Mono', monospace !important; font-size: 10px !important;
    letter-spacing: 1.5px; color: #566A7F !important;
    padding: 10px 18px !important; border-radius: 0 !important;
    border-bottom: 2px solid transparent !important; background: transparent !important;
}
.stTabs [aria-selected="true"] { color: #E8002D !important; border-bottom: 2px solid #E8002D !important; }
.stTabs [data-baseweb="tab-panel"] { padding: 0 !important; }

/* ── Metric cards ── */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(155px, 1fr));
    gap: 10px; margin-bottom: 18px;
}
.metric-card {
    background: #0C1018; border: 1px solid #1A2535;
    border-radius: 5px; padding: 12px 16px; position: relative; overflow: hidden;
}
.metric-card::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; }
.metric-card.d1::before  { background:#E8002D; }
.metric-card.d2::before  { background:#00D4FF; }
.metric-card.neu::before { background:#FFD700; }
.metric-card.grn::before { background:#39D353; }
.mc-label {
    font-family:'Share Tech Mono',monospace; font-size:9px;
    color:#566A7F; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:5px;
}
.mc-val {
    font-family:'Orbitron',sans-serif; font-size:19px; font-weight:700;
    line-height:1; color:#FFF;
}
.metric-card.d1  .mc-val { color:#E8002D; }
.metric-card.d2  .mc-val { color:#00D4FF; }
.metric-card.neu .mc-val { color:#FFD700; }
.metric-card.grn .mc-val { color:#39D353; }
.mc-sub { font-family:'Share Tech Mono',monospace; font-size:9px; color:#2E3E50; margin-top:4px; }

/* ── Section label ── */
.sec-label {
    font-family:'Share Tech Mono',monospace; font-size:10px;
    color:#566A7F; letter-spacing:2px; text-transform:uppercase;
    margin:18px 0 8px; display:flex; align-items:center; gap:8px;
}
.sec-label::after { content:''; flex:1; height:1px; background:#1A2535; }

/* ── Sector badges ── */
.sector-grid { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:16px; }
.sector-badge {
    background:#0C1018; border:1px solid #1A2535; border-radius:4px;
    padding:8px 14px; text-align:center; min-width:86px;
}
.sb-lbl { font-family:'Share Tech Mono',monospace; font-size:9px; color:#566A7F; letter-spacing:1px; text-transform:uppercase; }
.sb-val { font-family:'Orbitron',sans-serif; font-size:13px; font-weight:600; margin-top:3px; }
.c-green { color:#39D353 !important; }
.c-red   { color:#E8002D !important; }
.c-yellow{ color:#FFD700 !important; }
.c-blue  { color:#00D4FF !important; }
.c-gray  { color:#566A7F !important; }

/* ── Chart wrapper ── */
.chart-wrap {
    background:#090D13; border:1px solid #1A2535;
    border-radius:5px; padding:6px; margin-bottom:10px;
}

/* ── Tables ── */
.bz-table, .lap-table {
    width:100%; border-collapse:collapse;
    font-family:'Share Tech Mono',monospace; font-size:11px;
}
.bz-table th, .lap-table th {
    background:#0C1018; color:#566A7F; text-align:left;
    padding:8px 12px; border-bottom:1px solid #1A2535;
    font-size:9px; letter-spacing:1px; text-transform:uppercase;
}
.bz-table td, .lap-table td { padding:7px 12px; border-bottom:1px solid #0E1520; color:#C8D6E5; }
.bz-table tr:hover td, .lap-table tr:hover td { background:#0D1520; }
.lap-table .fastest { color:#39D353; font-weight:700; }

/* ── Status bar ── */
.status-bar {
    display:flex; align-items:center; gap:8px; padding:9px 14px;
    background:#0C1018; border:1px solid #1A2535; border-radius:3px;
    margin-bottom:14px; font-family:'Share Tech Mono',monospace; font-size:11px;
}
.dot { width:7px; height:7px; border-radius:50%; background:#39D353; animation:pulse 1.5s infinite; }
.dot.loading { background:#FFD700; }
.dot.error   { background:#E8002D; animation:none; }
@keyframes pulse { 0%,100%{opacity:1}50%{opacity:.3} }

/* ── Warn box ── */
.warn-box {
    background:#14100A; border:1px solid #FFD700; border-radius:5px;
    padding:14px 18px; font-family:'Share Tech Mono',monospace;
    font-size:12px; color:#FFD700; margin-bottom:18px;
}

/* ── Sticker ── */
.sticker {
    display:inline-block; padding:3px 9px; border-radius:2px;
    font-family:'Orbitron',sans-serif; font-size:11px; font-weight:700; letter-spacing:1px;
}

/* ── Hide streamlit chrome ── */
#MainMenu, footer, header { visibility:hidden; }
.stDeployButton { display:none; }
div[data-testid="stDecoration"] { display:none; }

/* ── Ocultar la flechita nativa de Streamlit que colapsa el sidebar ──
   El problema: si el usuario la toca, el sidebar desaparece y no hay
   forma visible de volver a abrirlo sin recargar la página.
   Solución: ocultamos esa flecha y ponemos nuestro propio botón. ── */
button[data-testid="collapsedControl"] {
    display: none !important;
}
section[data-testid="stSidebarCollapseButton"] {
    display: none !important;
}
/* También ocultar el chevron interno del sidebar */
button[kind="headerNoPadding"] {
    display: none !important;
}

/* ── Botón toggle flotante siempre visible ── */
#sidebar-toggle-btn {
    position: fixed;
    top: 50%;
    left: 0;
    transform: translateY(-50%);
    z-index: 999999;
    background: #E8002D;
    color: #fff;
    border: none;
    border-radius: 0 6px 6px 0;
    width: 20px;
    height: 64px;
    cursor: pointer;
    font-size: 11px;
    font-family: 'Share Tech Mono', monospace;
    writing-mode: vertical-rl;
    text-orientation: mixed;
    letter-spacing: 1px;
    opacity: 0.85;
    transition: opacity 0.2s, width 0.2s;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0;
    box-shadow: 2px 0 8px rgba(232,0,45,0.3);
}
#sidebar-toggle-btn:hover {
    opacity: 1;
    width: 24px;
}
</style>
""", unsafe_allow_html=True)

# ── Botón toggle flotante (inyectado via JS para interactuar con el DOM) ──────
st.markdown("""
<button id="sidebar-toggle-btn" onclick="toggleSidebar()" title="Mostrar/ocultar panel">
  &#9776;
</button>

<script>
function toggleSidebar() {
    // Selector del sidebar de Streamlit
    const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
    const btn     = window.parent.document.getElementById('sidebar-toggle-btn');

    if (!sidebar) return;

    // Detectar si está colapsado buscando el botón nativo de Streamlit
    const nativeBtn = window.parent.document.querySelector(
        'button[data-testid="collapsedControl"]'
    );
    if (nativeBtn) {
        nativeBtn.click();
        return;
    }

    // Alternativa: toggle clase collapsed
    const isCollapsed = sidebar.getAttribute('aria-expanded') === 'false'
                     || sidebar.style.marginLeft === '-100%'
                     || sidebar.offsetWidth < 50;

    if (isCollapsed) {
        // Forzar apertura clickeando el control nativo si existe
        const controls = window.parent.document.querySelectorAll(
            'button[kind="headerNoPadding"], [data-testid="stSidebarCollapseButton"] button'
        );
        controls.forEach(c => c.click());

        // Fallback: restaurar estilos directamente
        sidebar.style.marginLeft = '0';
        sidebar.style.visibility = 'visible';
        sidebar.style.width = '';
        if (btn) btn.textContent = '☰';
    } else {
        // Solo cambiar el ícono — NO colapsamos (ese es el punto)
        if (btn) btn.textContent = '☰';
    }
}

// Observar cambios en el sidebar para actualizar el ícono del botón
(function() {
    const observer = new MutationObserver(function() {
        const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
        const btn     = window.parent.document.getElementById('sidebar-toggle-btn');
        if (!sidebar || !btn) return;

        const collapsed = sidebar.offsetWidth < 80;
        btn.style.background = collapsed ? '#39D353' : '#E8002D';
        btn.title = collapsed ? 'Abrir panel ←' : 'Panel abierto';
    });

    // Intentar observar después de que el DOM esté listo
    setTimeout(function() {
        const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {
            observer.observe(sidebar, {
                attributes: true,
                attributeFilter: ['style', 'class', 'aria-expanded']
            });
        }
    }, 1000);
})();
</script>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
D1 = "#E8002D"   # Red  – reference driver
D2 = "#00D4FF"   # Cyan – comparison driver
GOLD  = "#FFD700"
GREEN = "#39D353"

_M_DEFAULT = dict(l=58, r=18, t=34, b=40)   # default margin, overridable per chart

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#090D13",
    font=dict(family="Share Tech Mono, monospace", color="#566A7F", size=10),
    # NOTE: margin intentionally omitted — each chart passes its own
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#0C1018", bordercolor="#1A2535",
        font=dict(family="Share Tech Mono, monospace", color="#C8D6E5", size=11),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)", bordercolor="#1A2535",
        font=dict(family="Share Tech Mono, monospace", color="#566A7F", size=10),
        x=0.01, y=0.99,
    ),
)

def pb(**kwargs):
    """
    Devuelve PLOTLY_BASE fusionado con kwargs.
    Los kwargs siempre tienen prioridad — si el caller pasa margin, legend,
    hovermode, hoverlabel, etc., sobreescriben los valores de PLOTLY_BASE.
    """
    out = {**PLOTLY_BASE}
    if "margin" not in kwargs:
        out["margin"] = _M_DEFAULT
    out.update(kwargs)   # kwargs sobreescriben; nunca hay duplicados
    return out
AX = dict(
    showgrid=True, gridcolor="#131C27", gridwidth=1,
    zeroline=False, linecolor="#1A2535",
    tickfont=dict(color="#2E3E50", size=9),
)

SESSION_LABELS = {
    "FP1": "Free Practice 1", "FP2": "Free Practice 2", "FP3": "Free Practice 3",
    "Q": "Qualifying", "R": "Race",
}

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def setup_cache():
    # En Streamlit Cloud el filesystem es read-only excepto /tmp
    # Usamos /tmp/fastf1_cache si ~/.cache no es escribible
    default = os.path.join(os.path.expanduser("~"), ".cache", "fastf1")
    try:
        os.makedirs(default, exist_ok=True)
        # Verificar que se puede escribir
        test = os.path.join(default, ".write_test")
        with open(test, "w") as f:
            f.write("ok")
        os.remove(test)
        p = default
    except (OSError, PermissionError):
        p = "/tmp/fastf1_cache"
        os.makedirs(p, exist_ok=True)
    fastf1.Cache.enable_cache(p)


@st.cache_data(ttl=3600, show_spinner=False)
def load_schedule(year):
    s = fastf1.get_event_schedule(year, include_testing=False)
    s = s[s["EventFormat"] != "testing"]
    return s[["EventName","Country","RoundNumber"]].reset_index(drop=True)


# IMPORTANTE: usar cache_resource (no cache_data) para objetos FastF1
# que no son serializables — cache_data intenta picklearlos y corrompe
# el estado de Streamlit en la segunda consulta.
@st.cache_resource(show_spinner=False)
def load_session(year, gp, stype):
    sess = fastf1.get_session(year, gp, stype)
    sess.load(telemetry=True, laps=True, weather=False, messages=False)
    return sess


@st.cache_data(ttl=3600, show_spinner=False)
def get_drivers(year, gp, stype):
    """Carga solo los nombres — serializable, usa cache_data."""
    try:
        # Carga liviana solo para obtener la lista de pilotos
        sess_light = fastf1.get_session(year, gp, stype)
        sess_light.load(telemetry=False, laps=False,
                        weather=False, messages=False)
        return sorted([
            sess_light.get_driver(d)["Abbreviation"]
            for d in sess_light.drivers
        ])
    except Exception:
        return ["VER","PER","LEC","SAI","HAM","RUS","NOR","PIA",
                "ALO","STR","GAS","OCO","TSU","RIC","ALB","SAR",
                "MAG","HUL","BOT","ZHO"]


def get_fastest_lap(session, driver):
    laps = session.laps.pick_drivers(driver)
    if laps.empty:
        return None, None, None
    fastest = laps.pick_fastest()
    if fastest is None or fastest.empty:
        return None, None, None
    try:
        tel = fastest.get_car_data().add_distance()
        try:
            pos = fastest.get_pos_data()
        except Exception:
            pos = None
        return fastest, tel, pos
    except Exception:
        return None, None, None


def get_all_laps(session, driver):
    """Todas las vueltas válidas (sin outlaps/inlaps)."""
    laps = session.laps.pick_drivers(driver).copy()
    laps = laps[laps["LapTime"].notna()]
    try:
        laps = laps[laps["PitOutTime"].isna()]
    except Exception:
        pass
    return laps.sort_values("LapNumber").reset_index(drop=True)


def get_quick_laps(session, driver):
    """
    Vueltas limpias usando pick_quicklaps() de FastF1:
    elimina pit in/out, safety car, VSC y banderas amarillas.
    Devuelve DataFrame con columnas LapNumber, LapTime, LapTimeSeconds, Compound, Team.
    """
    try:
        laps = session.laps.pick_drivers(driver).pick_quicklaps().copy()
    except Exception:
        # fallback si pick_quicklaps no está disponible
        laps = session.laps.pick_drivers(driver).copy()
        laps = laps[laps["LapTime"].notna()]

    if laps.empty:
        return pd.DataFrame()

    laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
    laps = laps.sort_values("LapNumber").reset_index(drop=True)
    return laps


def get_team_color(session, driver, fallback="#FFFFFF"):
    """Obtiene el color oficial del equipo via fastf1.plotting."""
    try:
        info = session.get_driver(driver)
        team = info.get("TeamName", info.get("TeamId", ""))
        # fastf1 >= 3.x
        try:
            color = fastf1.plotting.team_color(team)
            if color:
                return color
        except Exception:
            pass
        # fallback: usar colores hardcodeados por equipo
        TEAM_COLORS_MAP = {
            "mercedes":    "#27F4D2", "ferrari":     "#E8002D",
            "red bull":    "#3671C6", "mclaren":     "#FF8000",
            "aston martin":"#229971", "alpine":      "#FF87BC",
            "williams":    "#64C4FF", "rb":          "#6692FF",
            "kick sauber": "#52E252", "sauber":      "#52E252",
            "haas":        "#B6BABD", "audi":        "#C8F026",
            "cadillac":    "#A50F2D",
        }
        for key, val in TEAM_COLORS_MAP.items():
            if key in team.lower():
                return val
    except Exception:
        pass
    return fallback


def interpolate_tel(tel1, tel2, n=1200):
    max_dist = min(tel1["Distance"].max(), tel2["Distance"].max())
    common = np.linspace(0, max_dist, n)
    cols = ["Speed","Throttle","Brake","nGear","Steering","RPM"]

    def _interp(tel):
        dist = tel["Distance"].values
        out = {"Distance": common}
        for col in cols:
            if col not in tel.columns:
                out[col] = np.full(n, np.nan)
                continue
            vals = tel[col].values.astype(float)
            _, idx = np.unique(dist, return_index=True)
            d_u, v_u = dist[idx], vals[idx]
            if len(d_u) < 2:
                out[col] = np.full(n, np.nan)
                continue
            f = interp1d(d_u, v_u, kind="linear", bounds_error=False, fill_value="extrapolate")
            out[col] = f(common)
        return pd.DataFrame(out)

    return common, _interp(tel1), _interp(tel2)


def compute_delta(dist, t1, t2):
    dx = np.diff(dist, prepend=dist[0]); dx[0] = 0.0
    s1 = np.where(t1["Speed"].values / 3.6 <= 0, 1e-6, t1["Speed"].values / 3.6)
    s2 = np.where(t2["Speed"].values / 3.6 <= 0, 1e-6, t2["Speed"].values / 3.6)
    return np.cumsum(dx / s2) - np.cumsum(dx / s1)


def detect_braking_zones(dist, tel, thr=30.0, min_len=50.0):
    brake = tel["Brake"].values * 100
    speed = tel["Speed"].values
    in_z  = False; z_start = 0; zones = []
    for i in range(len(dist)):
        if not in_z and brake[i] >= thr:
            in_z = True; z_start = i
        elif in_z and brake[i] < thr:
            in_z = False
            length = dist[i] - dist[z_start]
            if length >= min_len:
                zones.append({
                    "start_m": float(dist[z_start]),
                    "end_m":   float(dist[i]),
                    "length_m": float(length),
                    "max_speed_entry": float(speed[z_start]),
                    "min_speed":       float(speed[z_start:i].min()),
                    "speed_drop":      float(speed[z_start] - speed[z_start:i].min()),
                })
    return zones


def fmt_lap(td):
    if td is None or (hasattr(td, '__class__') and td.__class__.__name__ == 'NaTType'):
        return "–:––.–––"
    try:
        total = td.total_seconds()
        m = int(total // 60); s = total % 60
        return f"{m}:{s:06.3f}"
    except Exception:
        return "–:––.–––"


def lap_to_s(td):
    try:
        return td.total_seconds()
    except Exception:
        return np.nan


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_main_telemetry(dist, t1, t2, d1_name, d2_name, delta):
    row_h = [0.20, 0.18, 0.16, 0.14, 0.16, 0.16]
    # Título del panel delta con convención de signo explícita
    delta_title = (f"Δ TIME (s)  ·  positivo = {d1_name} más rápido  "
                   f"·  negativo = {d2_name} más rápido")
    titles = [delta_title, "SPEED (km/h)", "THROTTLE / BRAKE (%)", "GEAR", "STEERING (°)", "RPM"]
    fig = make_subplots(rows=6, cols=1, shared_xaxes=True,
                        row_heights=row_h, subplot_titles=titles, vertical_spacing=0.035)

    # Delta sign convention: delta = cumsum(dt2) - cumsum(dt1)
    # delta > 0  →  d2 takes MORE time  →  d1 is FASTER (gaining on d2)
    # delta < 0  →  d2 takes LESS time  →  d2 is FASTER (gaining on d1)
    pos_d = np.where(delta >= 0, delta, np.nan)
    neg_d = np.where(delta <  0, delta, np.nan)
    fig.add_trace(go.Scatter(x=dist, y=pos_d, mode="lines",
        line=dict(color=GREEN, width=2), name=f"▲ {d1_name} faster",
        hovertemplate="Δ %{y:+.3f}s<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dist, y=neg_d, mode="lines",
        line=dict(color=D1, width=2), name=f"▼ {d2_name} faster",
        hovertemplate="Δ %{y:+.3f}s<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=np.concatenate([dist, dist[::-1]]),
        y=np.concatenate([delta, np.zeros(len(delta))]),
        fill="toself", mode="none", fillcolor="rgba(232,0,45,0.07)",
        showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_hline(y=0, line=dict(color="#1E2E40", width=1, dash="dot"), row=1, col=1)

    # Speed
    for tel, name, col in [(t1, d1_name, D1), (t2, d2_name, D2)]:
        fig.add_trace(go.Scatter(x=dist, y=tel["Speed"], mode="lines",
            line=dict(color=col, width=1.8), name=name,
            hovertemplate="%{y:.0f} km/h<extra></extra>"), row=2, col=1)

    # Throttle & Brake
    for tel, prefix, col, dash in [
        (t1, d1_name, D1, "solid"), (t2, d2_name, D2, "solid"),
        (t1, d1_name, "#FF6B35", "dash"), (t2, d2_name, "#FFB347", "dash"),
    ]:
        is_brake = dash == "dash"
        data = tel["Brake"] * 100 if is_brake else tel["Throttle"]
        label = f"BRK {prefix}" if is_brake else f"THR {prefix}"
        fig.add_trace(go.Scatter(x=dist, y=data, mode="lines",
            line=dict(color=col, width=1.4, dash=dash), name=label,
            showlegend=False, hovertemplate="%{y:.0f}%<extra></extra>"), row=3, col=1)

    # Gear
    for tel, name, col in [(t1, d1_name, D1), (t2, d2_name, D2)]:
        fig.add_trace(go.Scatter(x=dist, y=tel["nGear"], mode="lines",
            line=dict(color=col, width=1.8, shape="hv"), name=name,
            showlegend=False, hovertemplate="G%{y:.0f}<extra></extra>"), row=4, col=1)

    # Steering
    for tel, name, col in [(t1, d1_name, D1), (t2, d2_name, D2)]:
        fig.add_trace(go.Scatter(x=dist, y=tel["Steering"], mode="lines",
            line=dict(color=col, width=1.4), name=name,
            showlegend=False, hovertemplate="%{y:.1f}°<extra></extra>"), row=5, col=1)
    fig.add_hline(y=0, line=dict(color="#1E2E40", width=1, dash="dot"), row=5, col=1)

    # RPM
    for tel, name, col in [(t1, d1_name, D1), (t2, d2_name, D2)]:
        if "RPM" in tel.columns and tel["RPM"].notna().any():
            fig.add_trace(go.Scatter(x=dist, y=tel["RPM"], mode="lines",
                line=dict(color=col, width=1.4), name=name,
                showlegend=False, hovertemplate="%{y:.0f} RPM<extra></extra>"), row=6, col=1)

    fig.update_layout(**pb(height=980, showlegend=True,
                           margin=dict(l=60, r=18, t=44, b=46)))
    for i in range(1, 7):
        fig.update_xaxes(**AX, row=i, col=1)
        fig.update_yaxes(**AX, row=i, col=1)
    fig.update_xaxes(title_text="Distance (m)",
                     title_font=dict(color="#2E3E50", size=10), row=6, col=1)
    for ann in fig.layout.annotations:
        ann.update(font=dict(family="Share Tech Mono, monospace", color="#2E3E50", size=9),
                   xanchor="left", x=0.0)
    return fig


def build_dual_track_map(pos1, pos2, t1, t2, d1_name, d2_name):
    SPEED_SCALE = [[0,"#2E0000"],[0.25,"#E8002D"],[0.5,"#FFD700"],[0.75,"#39D353"],[1,"#00D4FF"]]

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=[f"{d1_name} — SPEED HEATMAP", f"{d2_name} — SPEED HEATMAP"])

    def add_map(pos, spd_arr, col_idx, show_scale):
        if pos is None or pos.empty:
            return
        try:
            x = uniform_filter1d(pos["X"].values.astype(float), size=5)
            y = uniform_filter1d(pos["Y"].values.astype(float), size=5)
            n = len(x)
            spd = np.interp(np.linspace(0,1,n), np.linspace(0,1,len(spd_arr)), spd_arr)
            # Track outline
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines",
                line=dict(color="#1A2535", width=10),
                showlegend=False, hoverinfo="skip"), row=1, col=col_idx)
            # Speed heatmap
            fig.add_trace(go.Scatter(x=x, y=y, mode="markers",
                marker=dict(size=3, color=spd, colorscale=SPEED_SCALE,
                    showscale=show_scale,
                    colorbar=dict(thickness=10, len=0.7,
                        tickfont=dict(family="Share Tech Mono",color="#566A7F",size=9),
                        bgcolor="rgba(0,0,0,0)", bordercolor="#1A2535",
                        title=dict(text="km/h",
                                   font=dict(family="Share Tech Mono",color="#566A7F",size=9))),
                    cmin=float(spd.min()), cmax=float(spd.max())),
                showlegend=False,
                hovertemplate=f"Speed: %{{marker.color:.0f}} km/h<extra>{d1_name if col_idx==1 else d2_name}</extra>"),
            row=1, col=col_idx)
        except Exception:
            pass

    add_map(pos1, t1["Speed"].values, 1, False)
    add_map(pos2, t2["Speed"].values, 2, True)

    fig.update_layout(**pb(height=420, margin=dict(l=10, r=70, t=46, b=10)))
    for col in [1, 2]:
        fig.update_xaxes(visible=False, scaleanchor="y" if col == 1 else "y2",
                         scaleratio=1, row=1, col=col)
        fig.update_yaxes(visible=False, row=1, col=col)
    for ann in fig.layout.annotations:
        ann.update(font=dict(family="Share Tech Mono,monospace", color="#566A7F", size=9))
    return fig


def build_braking_chart(zones1, zones2, d1_name, d2_name, dist_max):
    fig = go.Figure()
    for color, y_lo, y_hi in [(D1, 1.62, 2.38), (D2, 0.62, 1.38)]:
        fig.add_shape(type="rect", x0=0, x1=dist_max,
                      y0=y_lo - 0.02, y1=y_hi + 0.02,
                      fillcolor="#0C1018", line=dict(width=0))

    for zones, color, y_mid, y_lo, y_hi, name in [
        (zones1, D1, 2.0, 1.62, 2.38, d1_name),
        (zones2, D2, 1.0, 0.62, 1.38, d2_name),
    ]:
        for z in zones:
            fig.add_shape(type="rect",
                x0=z["start_m"], x1=z["end_m"], y0=y_lo, y1=y_hi,
                fillcolor=color, opacity=0.75, line=dict(width=0))
            if z["length_m"] > 80:
                fig.add_annotation(x=(z["start_m"]+z["end_m"])/2, y=y_mid,
                    text=f"{z['speed_drop']:.0f}", showarrow=False,
                    font=dict(family="Share Tech Mono", color="#FFF", size=8))
        fig.add_annotation(x=-30, y=y_mid, text=name, showarrow=False, xanchor="right",
                           font=dict(family="Share Tech Mono", color=color, size=11))

    fig.update_layout(
        **pb(height=220),
        title=dict(text="BRAKING ZONES  ·  bar width = zone length  ·  number = Δv (km/h)",
                   font=dict(family="Share Tech Mono", color="#566A7F", size=9)),
        xaxis=dict(**AX, title_text="Distance (m)",
                   title_font=dict(color="#2E3E50",size=10), range=[-100, dist_max+20]),
        yaxis=dict(visible=False, range=[0,3]),
    )
    return fig


def build_speed_histogram(t1, t2, d1_name, d2_name):
    fig = go.Figure()
    for tel, name, col in [(t1, d1_name, D1), (t2, d2_name, D2)]:
        fig.add_trace(go.Histogram(x=tel["Speed"].dropna(), name=name, nbinsx=60,
            marker_color=col, opacity=0.72,
            hovertemplate=f"{name}<br>%{{x:.0f}} km/h: %{{y}} pts<extra></extra>"))
    fig.update_layout(**pb(barmode="overlay", height=320,
        title=dict(text="SPEED DISTRIBUTION",
                   font=dict(family="Share Tech Mono",color="#566A7F",size=10)),
        xaxis=dict(**AX, title_text="Speed (km/h)", title_font=dict(color="#2E3E50",size=10)),
        yaxis=dict(**AX, title_text="Count", title_font=dict(color="#2E3E50",size=10))))
    return fig


def build_throttle_scatter(t1, t2, d1_name, d2_name):
    fig = go.Figure()
    for tel, name, col in [(t1, d1_name, D1), (t2, d2_name, D2)]:
        m = tel["Throttle"] > 10
        fig.add_trace(go.Scatter(x=tel.loc[m,"Speed"], y=tel.loc[m,"Throttle"],
            mode="markers", name=name,
            marker=dict(color=col, size=3, opacity=0.45),
            hovertemplate="%{x:.0f} km/h — %{y:.0f}% THR<extra></extra>"))
    fig.update_layout(**pb(height=320,
        title=dict(text="THROTTLE vs SPEED — EFFICIENCY MAP",
                   font=dict(family="Share Tech Mono",color="#566A7F",size=10)),
        xaxis=dict(**AX, title_text="Speed (km/h)", title_font=dict(color="#2E3E50",size=10)),
        yaxis=dict(**AX, title_text="Throttle (%)", title_font=dict(color="#2E3E50",size=10))))
    return fig


def build_gear_usage(t1, t2, d1_name, d2_name):
    gears = list(range(1, 9))
    def pct(tel):
        n = len(tel)
        return [(tel["nGear"].round() == g).sum() / n * 100 for g in gears]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[str(g) for g in gears], y=pct(t1), name=d1_name,
        marker_color=D1, opacity=0.85,
        hovertemplate="G%{x}: %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Bar(x=[str(g) for g in gears], y=pct(t2), name=d2_name,
        marker_color=D2, opacity=0.85,
        hovertemplate="G%{x}: %{y:.1f}%<extra></extra>"))
    fig.update_layout(**pb(barmode="group", height=300,
        title=dict(text="GEAR USAGE (% OF LAP DISTANCE)",
                   font=dict(family="Share Tech Mono",color="#566A7F",size=10)),
        xaxis=dict(**AX, title_text="Gear", title_font=dict(color="#2E3E50",size=10)),
        yaxis=dict(**AX, title_text="% Distance", title_font=dict(color="#2E3E50",size=10))))
    return fig


def build_drs_chart(dist, t1, t2, d1_name, d2_name):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=[f"{d1_name} DRS", f"{d2_name} DRS"],
                        row_heights=[0.5,0.5], vertical_spacing=0.08)
    for i, (tel, name, col) in enumerate([(t1,d1_name,D1),(t2,d2_name,D2)], start=1):
        drs = tel["DRS"].values if "DRS" in tel.columns else np.zeros(len(dist))
        active = np.where(drs > 8, 1, 0).astype(float)
        r, g, b = int(col[1:3],16), int(col[3:5],16), int(col[5:7],16)
        fig.add_trace(go.Scatter(x=dist, y=active, mode="lines",
            line=dict(color=col, width=2, shape="hv"), name=name,
            fill="tozeroy", fillcolor=f"rgba({r},{g},{b},0.15)",
            hovertemplate="DRS: %{y}<extra></extra>"), row=i, col=1)
    fig.update_layout(**pb(height=280, margin=dict(l=60,r=18,t=44,b=40)))
    for r in [1,2]:
        fig.update_xaxes(**AX, row=r, col=1)
        fig.update_yaxes(**AX, tickvals=[0,1], ticktext=["OFF","ON"], row=r, col=1)
    fig.update_xaxes(title_text="Distance (m)",
                     title_font=dict(color="#2E3E50",size=10), row=2, col=1)
    for ann in fig.layout.annotations:
        ann.update(font=dict(family="Share Tech Mono,monospace",color="#2E3E50",size=9),
                   xanchor="left", x=0)
    return fig


def build_pace_chart(laps1, laps2, d1_name, d2_name):
    fig = go.Figure()
    for df, name, col in [(laps1, d1_name, D1), (laps2, d2_name, D2)]:
        if df.empty:
            continue
        times = np.array([lap_to_s(lt) for lt in df["LapTime"]])
        laps  = df["LapNumber"].values
        med   = np.nanmedian(times)
        mask  = times < med * 1.10
        fig.add_trace(go.Scatter(
            x=laps[mask], y=times[mask], mode="lines+markers",
            line=dict(color=col, width=1.8), name=name,
            marker=dict(size=5, color=col),
            hovertemplate="Lap %{x}: %{y:.3f}s<extra></extra>"))
    fig.update_layout(**pb(height=320,
        title=dict(text="LAP PACE PROGRESSION",
                   font=dict(family="Share Tech Mono",color="#566A7F",size=10)),
        xaxis=dict(**AX, title_text="Lap Number", title_font=dict(color="#2E3E50",size=10)),
        yaxis=dict(**AX, title_text="Lap Time (s)", title_font=dict(color="#2E3E50",size=10))))
    return fig




# ══════════════════════════════════════════════════════════════════════════════
#  RACE PACE ANALYSIS — estilo Corsino
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_seconds(s: float) -> str:
    """Convierte segundos a string M:SS.mmm"""
    m = int(s // 60); sec = s % 60
    return f"{m}:{sec:06.3f}"


def plot_race_pace_evolution(ql1: pd.DataFrame, ql2: pd.DataFrame,
                              d1: str, d2: str,
                              col1: str, col2: str) -> go.Figure:
    """
    Líneas de evolución de LapTime vs LapNumber.
    Eje Y invertido: menor tiempo (más rápido) arriba.
    Incluye banda de rolling median suavizada como referencia de ritmo.
    """
    fig = go.Figure()

    for df, name, col in [(ql1, d1, col1), (ql2, d2, col2)]:
        if df.empty:
            continue

        laps  = df["LapNumber"].values
        times = df["LapTimeSeconds"].values

        # Banda de rango (min/max por ventana de 3 vueltas)
        if len(times) >= 3:
            roll_med = pd.Series(times).rolling(3, center=True, min_periods=1).median().values
            roll_min = pd.Series(times).rolling(3, center=True, min_periods=1).min().values
            roll_max = pd.Series(times).rolling(3, center=True, min_periods=1).max().values

            r, g, b = int(col[1:3],16), int(col[3:5],16), int(col[5:7],16)
            # Banda sombreada
            fig.add_trace(go.Scatter(
                x=np.concatenate([laps, laps[::-1]]),
                y=np.concatenate([roll_max, roll_min[::-1]]),
                fill="toself",
                fillcolor=f"rgba({r},{g},{b},0.10)",
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))
            # Línea suavizada de ritmo
            fig.add_trace(go.Scatter(
                x=laps, y=roll_med, mode="lines",
                line=dict(color=col, width=2.5, dash="solid"),
                name=f"{name} ritmo", opacity=0.6,
                hoverinfo="skip", showlegend=False,
            ))

        # Puntos individuales por compuesto
        compound_colors = {
            "SOFT":   "#FF3333",
            "MEDIUM": "#FFD700",
            "HARD":   "#E8E8E8",
            "INTER":  "#39D353",
            "WET":    "#0077FF",
        }
        if "Compound" in df.columns:
            for compound, grp in df.groupby("Compound"):
                c_dot = compound_colors.get(str(compound).upper(), col)
                fig.add_trace(go.Scatter(
                    x=grp["LapNumber"].values,
                    y=grp["LapTimeSeconds"].values,
                    mode="markers",
                    name=f"{name} ({compound})",
                    marker=dict(
                        color=c_dot, size=8,
                        line=dict(color=col, width=1.5),
                        symbol="circle",
                    ),
                    hovertemplate=(
                        f"<b>{name}</b> Lap %{{x}}<br>"
                        f"Tiempo: %{{customdata}}<br>"
                        f"Compuesto: {compound}<extra></extra>"
                    ),
                    customdata=[_fmt_seconds(t) for t in grp["LapTimeSeconds"].values],
                ))
        else:
            fig.add_trace(go.Scatter(
                x=laps, y=times, mode="markers",
                name=name,
                marker=dict(color=col, size=8,
                            line=dict(color="#1A2535", width=1)),
                hovertemplate=(
                    f"<b>{name}</b> Lap %{{x}}<br>"
                    f"Tiempo: %{{customdata}}<extra></extra>"
                ),
                customdata=[_fmt_seconds(t) for t in times],
            ))

        # Etiqueta de tiempo promedio
        avg = np.median(times)
        fig.add_annotation(
            x=laps[-1], y=avg,
            text=f"  {name}<br>  med: {_fmt_seconds(avg)}",
            showarrow=False, xanchor="left",
            font=dict(family="Share Tech Mono, monospace", color=col, size=10),
        )

    # Leyenda de compuestos
    for cname, ccol in [("SOFT","#FF3333"),("MEDIUM","#FFD700"),
                        ("HARD","#E8E8E8"),("INTER","#39D353"),("WET","#0077FF")]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=ccol, size=9, symbol="circle"),
            name=cname, showlegend=True,
        ))

    fig.update_layout(**pb(
        height=480,
        template="plotly_dark",
        paper_bgcolor="#070B0F",
        plot_bgcolor="#090D13",
        margin=dict(l=70, r=120, t=60, b=60),
        title=dict(
            text="EVOLUCIÓN DEL RITMO DE CARRERA  ·  tiempos más rápidos abajo",
            font=dict(family="Share Tech Mono, monospace", color="#566A7F", size=11),
            x=0,
        ),
        xaxis=dict(
            **AX, title_text="Vuelta",
            title_font=dict(color="#566A7F", size=10),
        ),
        yaxis=dict(
            **AX,
            title_text="Tiempo de vuelta (s)",
            title_font=dict(color="#566A7F", size=10),
            tickformat=".3f",
        ),
        legend=dict(
            bgcolor="rgba(12,16,24,0.85)", bordercolor="#1A2535", borderwidth=1,
            font=dict(family="Share Tech Mono, monospace", color="#C8D6E5", size=10),
            x=1.01, y=1, xanchor="left",
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#0C1018", bordercolor="#1A2535",
            font=dict(family="Share Tech Mono, monospace", color="#C8D6E5", size=11),
        ),
    ))
    return fig


def plot_race_pace_boxplot(ql1: pd.DataFrame, ql2: pd.DataFrame,
                            d1: str, d2: str,
                            col1: str, col2: str) -> go.Figure:
    """
    Violin Plot estilo Corsino:
    · Violin KDE muestra la densidad real de tiempos (más ancho = más vueltas ahí)
    · Puntos individuales con jitter, coloreados por compuesto de neumático
    · Línea de mediana superpuesta como referencia
    · Eje Y estándar: tiempos más rápidos (menores) en la parte inferior
    """
    COMPOUND_COLORS = {
        "SOFT":   "#FF3333",
        "MEDIUM": "#FFD700",
        "HARD":   "#ECECEC",
        "INTER":  "#39D353",
        "WET":    "#0077FF",
    }

    fig = go.Figure()

    for df, name, col in [(ql1, d1, col1), (ql2, d2, col2)]:
        if df.empty:
            continue

        times  = df["LapTimeSeconds"].values
        r, g, b = int(col[1:3], 16), int(col[3:5], 16), int(col[5:7], 16)

        # ── 1. Violin (KDE) ──────────────────────────────────────────────────
        fig.add_trace(go.Violin(
            y=times,
            x=[name] * len(times),
            name=name,
            side="both",                      # violin simétrico
            width=1.6,
            meanline_visible=True,            # línea de media dentro del violin
            meanline=dict(color="#FFFFFF", width=1.5),
            line_color=col,
            fillcolor=f"rgba({r},{g},{b},0.18)",
            opacity=1.0,
            points=False,                     # los puntos los pintamos a mano
            box_visible=True,                 # cajita interior con IQR
            box=dict(
                fillcolor=f"rgba({r},{g},{b},0.35)",
                line=dict(color=col, width=1.5),
            ),
            spanmode="hard",                  # no extender más allá del min/max real
            showlegend=False,
            hoverinfo="skip",
        ))

        # ── 2. Puntos por compuesto con jitter ───────────────────────────────
        rng = np.random.RandomState(42)

        if "Compound" in df.columns:
            for compound, grp in df.groupby("Compound"):
                c_dot  = COMPOUND_COLORS.get(str(compound).upper(), col)
                n      = len(grp)
                jitter = rng.uniform(-0.08, 0.08, n)   # spread horizontal sutil

                fig.add_trace(go.Scatter(
                    # Plotly violin usa categorías en x; los puntos necesitan
                    # el mismo nombre de categoría + offset numérico vía xaxis ticks
                    x=[name] * n,
                    y=grp["LapTimeSeconds"].values,
                    mode="markers",
                    name=f"{compound}",
                    legendgroup="compounds",
                    showlegend=True,
                    marker=dict(
                        color=c_dot,
                        size=7,
                        opacity=0.90,
                        symbol="circle",
                        line=dict(color=f"rgba({r},{g},{b},0.6)", width=1),
                    ),
                    hovertemplate=(
                        f"<b>{name}</b>  Lap %{{customdata[0]}}<br>"
                        f"Tiempo: %{{customdata[1]}}<br>"
                        f"Compuesto: {compound}<extra></extra>"
                    ),
                    customdata=list(zip(
                        grp["LapNumber"].values,
                        [_fmt_seconds(t) for t in grp["LapTimeSeconds"].values],
                    )),
                ))
        else:
            n      = len(times)
            fig.add_trace(go.Scatter(
                x=[name] * n,
                y=times,
                mode="markers",
                name=name,
                showlegend=False,
                marker=dict(
                    color=col, size=7, opacity=0.85,
                    symbol="circle",
                    line=dict(color="#1A2535", width=0.8),
                ),
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    f"Tiempo: %{{y:.3f}}s<extra></extra>"
                ),
            ))

        # ── 3. Anotación estadística debajo del violin ───────────────────────
        median = float(np.median(times))
        std    = float(np.std(times))
        q1     = float(np.percentile(times, 25))
        q3     = float(np.percentile(times, 75))

        fig.add_annotation(
            x=name,
            y=times.max() + 0.25,
            text=(
                f"<b style='color:{col}'>{name}</b><br>"
                f"med {_fmt_seconds(median)}<br>"
                f"σ {std:.3f}s · IQR {(q3-q1):.3f}s"
            ),
            showarrow=False,
            yanchor="bottom",
            align="center",
            font=dict(family="Share Tech Mono, monospace", color=col, size=10),
            bgcolor="rgba(9,13,19,0.75)",
            bordercolor=col,
            borderwidth=1,
            borderpad=4,
        )

    # ── Layout ───────────────────────────────────────────────────────────────
    fig.update_layout(**pb(
        height=560,
        template="plotly_dark",
        paper_bgcolor="#070B0F",
        plot_bgcolor="#090D13",
        margin=dict(l=80, r=40, t=80, b=60),
        violingap=0.3,
        violinmode="group",
        title=dict(
            text=(
                "DISTRIBUCIÓN DE TIEMPOS — VIOLIN PLOT  ·  "
                "ancho = densidad de vueltas  ·  tiempos rápidos abajo"
            ),
            font=dict(family="Share Tech Mono, monospace", color="#566A7F", size=11),
            x=0,
        ),
        xaxis=dict(
            **AX,
            title_text="Piloto",
            title_font=dict(color="#566A7F", size=10),
        ),
        yaxis=dict(
            **AX,
            title_text="Tiempo de vuelta (s)",
            title_font=dict(color="#566A7F", size=10),
            tickformat=".3f",
            # Sin autorange="reversed" — eje estándar: rápido abajo
        ),
        showlegend=True,
        legend=dict(
            title=dict(
                text="COMPUESTO",
                font=dict(family="Share Tech Mono, monospace",
                          color="#566A7F", size=9),
            ),
            bgcolor="rgba(12,16,24,0.85)",
            bordercolor="#1A2535", borderwidth=1,
            font=dict(family="Share Tech Mono, monospace",
                      color="#C8D6E5", size=10),
            x=1.01, y=0.98, xanchor="left",
        ),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#0C1018", bordercolor="#1A2535",
            font=dict(family="Share Tech Mono, monospace",
                      color="#C8D6E5", size=11),
        ),
    ))
    # Sobreescribir tickfont del eje X por separado (evita conflicto con **AX)
    fig.update_xaxes(tickfont=dict(family="Orbitron, monospace",
                                   color="#C8D6E5", size=12))
    return fig


def build_grid_violin(session,
                      drivers: list,
                      gp_name: str = "",
                      year: int = 2026) -> go.Figure:
    """
    Violin plot estilo Corsino para N pilotos de toda la parrilla.

    Diseño:
    · KDE suave con bandwidth=0.4  → forma de «pétalo» orgánica
    · points="all" con jitter nativo de Plotly, coloreados por compuesto
    · Ordenados de izquierda a derecha por mediana (más rápido primero)
    · Línea de media interior + caja IQR dentro del violin
    · Anotación flotante con mediana, σ y abreviatura del piloto
    · Fondo oscuro, tipografía Share Tech Mono, sin eje Y invertido
    """
    COMPOUND_CLR = {
        "SOFT":   "#FF3333",
        "MEDIUM": "#FFD700",
        "HARD":   "#F0F0F0",
        "INTER":  "#39D353",
        "WET":    "#00AAFF",
        "UNKNOWN":"#AAAAAA",
    }

    # ── Recoger vueltas limpias de cada piloto ─────────────────────────────
    driver_data = []   # list of (abbr, times_array, compound_series, team_color)
    for drv in drivers:
        try:
            ql = get_quick_laps(session, drv)
            if ql.empty or len(ql) < 3:
                continue
            tc = get_team_color(session, drv, "#FFFFFF")
            driver_data.append((drv, ql, tc))
        except Exception:
            continue

    if not driver_data:
        fig = go.Figure()
        fig.update_layout(**pb(height=500, paper_bgcolor="#070B0F",
                               plot_bgcolor="#090D13"))
        fig.add_annotation(text="Sin datos disponibles para los pilotos seleccionados",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False,
                           font=dict(family="Share Tech Mono, monospace",
                                     color="#566A7F", size=14))
        return fig

    # Ordenar por mediana ascendente (más rápido a la izquierda)
    driver_data.sort(key=lambda x: float(np.median(x[1]["LapTimeSeconds"].values)))

    fig = go.Figure()

    # ── Construir un violin por piloto ─────────────────────────────────────
    # Rastrea qué compuestos ya tienen entrada en leyenda
    legend_compounds_added = set()

    for drv, ql, tc in driver_data:
        times = ql["LapTimeSeconds"].values
        r = int(tc[1:3], 16); g = int(tc[3:5], 16); b_ch = int(tc[5:7], 16)

        # ── Violin puro (SIN puntos — los ponemos como Scatter separados) ──
        # go.Violin NO acepta lista de colores en marker.color, solo un color único.
        fig.add_trace(go.Violin(
            x=[drv] * len(times),
            y=times,
            name=drv,
            side="both",
            bandwidth=0.35,       # kernel suave → forma "pétalo" estilo Corsino
            spanmode="hard",      # no extender más allá del dato real
            width=0.85,
            points=False,         # SIN puntos aquí; los añadimos como Scatter
            line_color=tc,
            fillcolor=f"rgba({r},{g},{b_ch},0.22)",
            opacity=1.0,
            meanline_visible=True,
            meanline=dict(color="#FFFFFF", width=1.2),
            box_visible=True,
            box=dict(
                fillcolor=f"rgba({r},{g},{b_ch},0.40)",
                line=dict(color=tc, width=1.5),
            ),
            showlegend=False,
            hovertemplate=(
                f"<b>{drv}</b><br>"
                "Vuelta: %{y:.3f}s<extra></extra>"
            ),
        ))

        # ── Puntos por compuesto como Scatter (color único por traza) ──────
        rng = np.random.RandomState(42)
        compounds = ql["Compound"].values if "Compound" in ql.columns \
                    else ["UNKNOWN"] * len(times)

        for compound in np.unique(compounds):
            mask       = compounds == compound
            c_dot      = COMPOUND_CLR.get(str(compound).upper(), "#AAAAAA")
            grp_times  = times[mask]
            grp_laps   = ql.loc[mask, "LapNumber"].values if "LapNumber" in ql.columns \
                         else np.arange(mask.sum())
            jitter_x   = rng.uniform(-0.06, 0.06, mask.sum())

            already    = compound in legend_compounds_added
            fig.add_trace(go.Scatter(
                x=[drv] * mask.sum(),
                y=grp_times,
                mode="markers",
                name=str(compound),
                legendgroup=f"cpd_{compound}",
                showlegend=not already,
                marker=dict(
                    color=c_dot,
                    size=6,
                    opacity=0.88,
                    symbol="circle",
                    line=dict(color=f"rgba({r},{g},{b_ch},0.40)", width=0.8),
                ),
                hovertemplate=(
                    f"<b>{drv}</b>  Lap %{{customdata[0]}}<br>"
                    f"Tiempo: %{{customdata[1]}}<br>"
                    f"Comp: {compound}<extra></extra>"
                ),
                customdata=list(zip(
                    grp_laps,
                    [_fmt_seconds(t) for t in grp_times],
                )),
            ))
            legend_compounds_added.add(compound)

        # ── Anotación estadística encima del violin ──────────────────────
        median = float(np.median(times))
        std    = float(np.std(times))
        fig.add_annotation(
            x=drv,
            y=float(times.max()) + (float(times.max()) - float(times.min())) * 0.06,
            text=(
                f"<b>{drv}</b>  {_fmt_seconds(median)}<br>"
                f"σ {std:.3f}s"
            ),
            showarrow=False,
            yanchor="bottom",
            align="center",
            font=dict(family="Share Tech Mono, monospace", color=tc, size=9),
            bgcolor="rgba(7,11,15,0.80)",
            bordercolor=tc,
            borderwidth=1,
            borderpad=3,
        )

    # ── Layout ────────────────────────────────────────────────────────────
    n = len(driver_data)
    fig.update_layout(**pb(
        height=max(520, 420 + n * 8),  # altura dinámica según pilotos
        template="plotly_dark",
        paper_bgcolor="#070B0F",
        plot_bgcolor="#090D13",
        margin=dict(l=80, r=160, t=70, b=70),
        violingap=0.12,
        violingroupgap=0.05,
        violinmode="group",
        title=dict(
            text=(
                f"{year} {gp_name} — RITMOS DE CARRERA  ·  "
                f"{n} piloto{'s' if n != 1 else ''}  ·  "
                "ordenado por mediana  ·  puntos = compuesto de neumático"
            ),
            font=dict(family="Share Tech Mono, monospace", color="#7B8FA1", size=11),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(
            **AX,
            title_text="Piloto",
            title_font=dict(color="#566A7F", size=10),
            categoryorder="array",
            categoryarray=[drv for drv, _, _ in driver_data],
        ),
        yaxis=dict(
            **AX,
            title_text="Tiempo de vuelta (s)",
            title_font=dict(color="#566A7F", size=10),
            tickformat=".2f",
        ),
        showlegend=True,
        legend=dict(
            title=dict(
                text="COMPUESTO",
                font=dict(family="Share Tech Mono, monospace",
                          color="#566A7F", size=9),
            ),
            bgcolor="rgba(12,16,24,0.90)",
            bordercolor="#1A2535", borderwidth=1,
            font=dict(family="Share Tech Mono, monospace",
                      color="#C8D6E5", size=10),
            x=1.01, y=1, xanchor="left",
        ),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#0C1018", bordercolor="#1A2535",
            font=dict(family="Share Tech Mono, monospace",
                      color="#C8D6E5", size=11),
        ),
    ))
    # Tickfont del eje X con Orbitron para los nombres de piloto
    fig.update_xaxes(tickfont=dict(family="Orbitron, monospace",
                                   color="#C8D6E5", size=10))
    return fig


def plot_race_pace_analysis(session, d1: str, d2: str,
                             col1: str, col2: str) -> dict:
    """
    Entry point principal. Carga vueltas limpias y devuelve
    {'ql1', 'ql2', 'fig_evolution', 'fig_box', 'stats'} o lanza Exception.
    """
    ql1 = get_quick_laps(session, d1)
    ql2 = get_quick_laps(session, d2)

    if ql1.empty and ql2.empty:
        raise ValueError("No hay vueltas limpias disponibles para ninguno de los pilotos.")

    fig_evo = plot_race_pace_evolution(ql1, ql2, d1, d2, col1, col2)
    fig_box = plot_race_pace_boxplot(ql1, ql2, d1, d2, col1, col2)

    # Stats summary
    stats = {}
    for df, name in [(ql1, d1), (ql2, d2)]:
        if not df.empty:
            t = df["LapTimeSeconds"].values
            stats[name] = {
                "laps":   len(t),
                "median": float(np.median(t)),
                "mean":   float(np.mean(t)),
                "std":    float(np.std(t)),
                "best":   float(t.min()),
                "worst":  float(t.max()),
            }

    return {"ql1": ql1, "ql2": ql2,
            "fig_evolution": fig_evo, "fig_box": fig_box,
            "stats": stats}


# ══════════════════════════════════════════════════════════════════════════════
#  HTML REPORT
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(d):
    import plotly.io as pio

    d1, d2 = d["driver1"], d["driver2"]
    dist = d["common_dist"]; t1 = d["t1"]; t2 = d["t2"]; delta = d["delta"]

    charts = {
        "Telemetry": build_main_telemetry(dist, t1, t2, d1, d2, delta),
        "Speed Distribution": build_speed_histogram(t1, t2, d1, d2),
        "Gear Usage": build_gear_usage(t1, t2, d1, d2),
        "Throttle Efficiency": build_throttle_scatter(t1, t2, d1, d2),
    }

    body = ""
    for title, fig in charts.items():
        inner = pio.to_html(fig, include_plotlyjs="cdn", full_html=False,
                            config={"displayModeBar": False})
        body += f'<div class="section"><h2>{title.upper()}</h2>{inner}</div>\n'

    lt1 = _fmt_seconds(d["lap1"].get("LapTime", 0)) if d.get("lap1") else "–:––.–––"
    lt2 = _fmt_seconds(d["lap2"].get("LapTime", 0)) if d.get("lap2") else "–:––.–––"
    now = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>F1 Telemetry Report — {d1} vs {d2}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Share+Tech+Mono&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#070B0F;color:#C8D6E5;font-family:'Share Tech Mono',monospace}}
header{{border-bottom:2px solid #E8002D;padding:22px 40px;display:flex;align-items:center;gap:14px}}
header .bar{{width:4px;height:46px;background:#E8002D;border-radius:2px}}
header h1{{font-family:'Orbitron',sans-serif;font-size:20px;letter-spacing:3px;color:#fff}}
header .sub{{font-size:10px;color:#566A7F;letter-spacing:2px;margin-top:4px}}
.kpi-row{{display:flex;gap:10px;flex-wrap:wrap;padding:18px 40px;border-bottom:1px solid #1A2535}}
.kpi{{background:#0C1018;border:1px solid #1A2535;border-radius:4px;padding:12px 16px;min-width:130px}}
.kpi .lbl{{font-size:9px;color:#566A7F;letter-spacing:1px;text-transform:uppercase}}
.kpi .val{{font-family:'Orbitron',sans-serif;font-size:18px;font-weight:700;margin-top:4px}}
.kpi.d1 .val{{color:#E8002D}}.kpi.d2 .val{{color:#00D4FF}}.kpi.neu .val{{color:#FFD700}}
.section{{padding:18px 40px;border-bottom:1px solid #1A2535}}
.section h2{{font-size:9px;color:#566A7F;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px}}
footer{{padding:12px 40px;color:#2E3E50;font-size:9px}}
</style>
</head>
<body>
<header>
  <div class="bar"></div>
  <div><h1>F1 TELEMETRY REPORT</h1>
    <div class="sub">{d['year']} · {d['gp_name']} · {d['session']} · {now}</div>
  </div>
</header>
<div class="kpi-row">
  <div class="kpi d1"><div class="lbl">🔴 {d1} Fastest Lap</div><div class="val">{lt1}</div></div>
  <div class="kpi d2"><div class="lbl">🔵 {d2} Fastest Lap</div><div class="val">{lt2}</div></div>
  <div class="kpi neu"><div class="lbl">Final Δ Time</div><div class="val">{delta[-1]:+.3f}s</div></div>
  <div class="kpi neu"><div class="lbl">Max Δ Gain</div><div class="val">{abs(delta.min()):.3f}s</div></div>
  <div class="kpi neu"><div class="lbl">Max Δ Loss</div><div class="val">{delta.max():.3f}s</div></div>
  <div class="kpi d1"><div class="lbl">{d1} Avg Speed</div><div class="val">{t1['Speed'].mean():.0f} km/h</div></div>
  <div class="kpi d2"><div class="lbl">{d2} Avg Speed</div><div class="val">{t2['Speed'].mean():.0f} km/h</div></div>
  <div class="kpi d1"><div class="lbl">{d1} Top Speed</div><div class="val">{t1['Speed'].max():.0f} km/h</div></div>
  <div class="kpi d2"><div class="lbl">{d2} Top Speed</div><div class="val">{t2['Speed'].max():.0f} km/h</div></div>
</div>
{body}
<footer>F1 Telemetry Dashboard v2 · FastF1 · {now}</footer>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
#  UI COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

def render_header():
    st.markdown("""
    <div class="dash-header">
      <div class="red-bar"></div>
      <div>
        <h1>F1 TELEMETRY PRO <span class="ver-badge">v2</span></h1>
        <div class="subtitle">MOTEC-STYLE ANALYSIS · FASTF1 + PLOTLY · DISTANCE-BASED AXIS</div>
      </div>
    </div>""", unsafe_allow_html=True)


def render_kpis(lap1, lap2, d1, d2, delta_end, t1, t2):
    lt1 = _fmt_seconds(lap1.get("LapTime", 0)) if lap1 else "–:––.–––"
    lt2 = _fmt_seconds(lap2.get("LapTime", 0)) if lap2 else "–:––.–––"
    # delta > 0 → d1 faster;  delta < 0 → d2 faster
    leader = d1 if delta_end > 0 else d2
    sign   = "▲" if delta_end > 0 else "▼"
    st.markdown(f"""
    <div class="metrics-row">
      <div class="metric-card d1"><div class="mc-label">🔴 {d1} — LAP TIME</div>
        <div class="mc-val">{lt1}</div><div class="mc-sub">FASTEST LAP</div></div>
      <div class="metric-card d2"><div class="mc-label">🔵 {d2} — LAP TIME</div>
        <div class="mc-val">{lt2}</div><div class="mc-sub">FASTEST LAP</div></div>
      <div class="metric-card neu"><div class="mc-label">⚡ FINAL DELTA</div>
        <div class="mc-val">{sign} {abs(delta_end):.3f}s</div>
        <div class="mc-sub">{leader} FASTER</div></div>
      <div class="metric-card d1"><div class="mc-label">🔴 {d1} TOP SPEED</div>
        <div class="mc-val">{t1['Speed'].max():.0f}</div><div class="mc-sub">km/h</div></div>
      <div class="metric-card d2"><div class="mc-label">🔵 {d2} TOP SPEED</div>
        <div class="mc-val">{t2['Speed'].max():.0f}</div><div class="mc-sub">km/h</div></div>
      <div class="metric-card d1"><div class="mc-label">🔴 {d1} AVG SPEED</div>
        <div class="mc-val">{t1['Speed'].mean():.0f}</div><div class="mc-sub">km/h</div></div>
      <div class="metric-card d2"><div class="mc-label">🔵 {d2} AVG SPEED</div>
        <div class="mc-val">{t2['Speed'].mean():.0f}</div><div class="mc-sub">km/h</div></div>
    </div>""", unsafe_allow_html=True)


def render_sectors(lap1, lap2, d1, d2):
    # lap1/lap2 son dicts con sector times ya en segundos (float)
    html = '<div class="sec-label">SECTOR ANALYSIS</div><div class="sector-grid">'
    for s, lbl in [("Sector1Time","SECTOR 1"),("Sector2Time","SECTOR 2"),("Sector3Time","SECTOR 3")]:
        try:
            v1 = lap1.get(s); v2 = lap2.get(s)
            if v1 is None or v2 is None:
                raise ValueError("missing")
            v1 = float(v1); v2 = float(v2)
            diff = v1 - v2
            c1 = "c-green" if diff < 0 else "c-red"
            c2 = "c-green" if diff > 0 else "c-red"
            if abs(diff) < 0.001:
                c1 = c2 = "c-yellow"
            if diff < 0:
                ind = f"{d1} +{abs(diff):.3f}s"
            elif diff > 0:
                ind = f"{d2} +{abs(diff):.3f}s"
            else:
                ind = "EQUAL"
        except Exception:
            v1 = v2 = None; c1 = c2 = "c-gray"; ind = "N/A"
        html += f"""
        <div class="sector-badge">
          <div class="sb-lbl">{lbl}</div>
          <div class="sb-val {c1}">{f'{v1:.3f}s' if v1 is not None else '–'}</div>
          <div class="sb-val {c2}">{f'{v2:.3f}s' if v2 is not None else '–'}</div>
          <div class="sb-lbl" style="color:#39D353;margin-top:3px">{ind}</div>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_braking_tables(bz1, bz2, d1, d2):
    for col_st, zones, name, color in [
        (st.columns(2)[0], bz1, d1, D1),
        (st.columns(2)[1], bz2, d2, D2),
    ]:
        with col_st:
            rows = "".join(f"""
            <tr><td>{i}</td><td>{z['start_m']:.0f}</td><td>{z['end_m']:.0f}</td>
            <td>{z['length_m']:.0f}</td><td>{z['max_speed_entry']:.0f}</td>
            <td style="color:{color};font-weight:700">{z['speed_drop']:.0f}</td></tr>"""
            for i, z in enumerate(zones, 1))
            st.markdown(f"""
            <div class="sec-label"><span style="color:{color}">■</span> {name} BRAKING ZONES</div>
            <table class="bz-table">
              <thead><tr><th>#</th><th>Start (m)</th><th>End (m)</th>
                <th>Length (m)</th><th>Entry (km/h)</th><th>Δv (km/h)</th></tr></thead>
              <tbody>{rows}</tbody>
            </table>""", unsafe_allow_html=True)


def render_lap_table(laps1, laps2, d1, d2):
    all_lap_nums = sorted(set(
        (laps1["LapNumber"].tolist() if not laps1.empty else []) +
        (laps2["LapNumber"].tolist() if not laps2.empty else [])
    ))

    def to_dict(df):
        out = {}
        for _, row in df.iterrows():
            v = lap_to_s(row["LapTime"])
            if not np.isnan(v):
                out[int(row["LapNumber"])] = v
        return out

    d1d = to_dict(laps1); d2d = to_dict(laps2)
    best1 = min(d1d.values(), default=999)
    best2 = min(d2d.values(), default=999)

    rows = ""
    for lap in all_lap_nums:
        v1 = d1d.get(lap); v2 = d2d.get(lap)

        def cell(v, best, col):
            if v is None:
                return "<td style='color:#2E3E50'>–</td>"
            m = int(v//60); s = v%60
            txt = f"{m}:{s:06.3f}"
            css = f"color:{col};font-weight:700" if abs(v-best)<0.001 else ""
            return f'<td style="{css}">{txt}</td>'

        def gap_cell(v1, v2):
            if v1 is None or v2 is None:
                return "<td style='color:#2E3E50'>–</td>"
            g = v1 - v2  # negativo = d1 más rápido; positivo = d2 más rápido
            col = D1 if g < 0 else D2 if g > 0 else GOLD
            return f'<td style="color:{col}">{g:+.3f}s</td>'

        rows += f"<tr><td style='color:#566A7F'>{lap}</td>{cell(v1,best1,D1)}{cell(v2,best2,D2)}{gap_cell(v1,v2)}</tr>"

    st.markdown(f"""
    <div class="sec-label">LAP-BY-LAP COMPARISON</div>
    <div style="font-family:'Share Tech Mono',monospace;font-size:9px;color:#2E3E50;
                margin-bottom:6px">
        GAP = {d1} tiempo − {d2} tiempo &nbsp;·&nbsp;
        <span style="color:{D1}">■ negativo = {d1} más rápido</span> &nbsp;·&nbsp;
        <span style="color:{D2}">■ positivo = {d2} más rápido</span> &nbsp;·&nbsp;
        <span style="color:#39D353">■ negrita = vuelta más rápida</span>
    </div>
    <div style="max-height:480px;overflow-y:auto">
    <table class="lap-table">
      <thead><tr><th>LAP</th>
        <th style="color:{D1}">{d1}</th>
        <th style="color:{D2}">{d2}</th>
        <th>GAP ({d1}−{d2})</th></tr></thead>
      <tbody>{rows}</tbody>
    </table></div>""", unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="font-family:'Orbitron',sans-serif;font-size:14px;font-weight:700;
                    color:#E8002D;letter-spacing:3px;padding:10px 0 5px;text-transform:uppercase">
            F1 TELEMETRY
        </div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:9px;color:#2E3E50;
                    letter-spacing:2px;border-bottom:1px solid #1A2535;padding-bottom:10px;margin-bottom:14px">
            DATA ANALYSIS SUITE v2
        </div>""", unsafe_allow_html=True)

        year = st.selectbox("SEASON", [2026], index=0)

        try:
            sched = load_schedule(year)
            events = sched["EventName"].tolist()
        except Exception:
            events = [
                "Bahrain Grand Prix","Saudi Arabian Grand Prix","Australian Grand Prix",
                "Japanese Grand Prix","Chinese Grand Prix","Miami Grand Prix",
                "Emilia Romagna Grand Prix","Monaco Grand Prix","Canadian Grand Prix",
                "Spanish Grand Prix","Austrian Grand Prix","British Grand Prix",
                "Hungarian Grand Prix","Belgian Grand Prix","Dutch Grand Prix",
                "Italian Grand Prix","Azerbaijan Grand Prix","Singapore Grand Prix",
                "United States Grand Prix","Mexico City Grand Prix",
                "São Paulo Grand Prix","Las Vegas Grand Prix",
                "Qatar Grand Prix","Abu Dhabi Grand Prix",
            ]

        gp_name = st.selectbox("GRAND PRIX", events)
        sess    = st.selectbox("SESSION", list(SESSION_LABELS.keys()),
                               format_func=lambda x: SESSION_LABELS[x])

        drivers = []
        if FASTF1_AVAILABLE:
            with st.spinner(""):
                try:
                    drivers = get_drivers(year, gp_name, sess)
                except Exception:
                    pass

        if not drivers:
            drivers = ["VER","PER","LEC","SAI","HAM","RUS","NOR","PIA",
                       "ALO","STR","GAS","OCO","TSU","RIC","ALB","SAR",
                       "MAG","HUL","BOT","ZHO"]

        c1, c2 = st.columns(2)
        with c1:
            d1 = st.selectbox("REF", drivers, index=0)
        with c2:
            d2 = st.selectbox("COMP", drivers, index=min(1, len(drivers)-1))

        n_pts = st.slider("RESOLUTION", 500, 2000, 1200, step=100)

        with st.expander("⚙  ADVANCED"):
            bthr = st.slider("BRAKE THRESHOLD (%)", 10, 80, 30, step=5)
            bmin = st.slider("MIN BRAKE ZONE (m)",  20, 200, 50, step=10)
            show_raw = st.checkbox("Show raw data", value=False)

        st.markdown("<br>", unsafe_allow_html=True)
        load_btn = st.button("⚡  LOAD TELEMETRY", use_container_width=True)

        st.markdown("""
        <div style="margin-top:18px;padding-top:10px;border-top:1px solid #1A2535;
                    font-family:Share Tech Mono,monospace;font-size:9px;color:#1E2E40;line-height:2.1">
            ◆  DISTANCE-BASED AXIS<br>◆  INTERPOLATED TELEMETRY<br>
            ◆  CUMULATIVE Δ TIME<br>◆  TRACK MAP HEATMAP<br>
            ◆  BRAKING ZONE DETECTION<br>◆  LAP PACE CHART<br>
            ◆  HTML REPORT EXPORT
        </div>""", unsafe_allow_html=True)

    return year, gp_name, sess, d1, d2, n_pts, bthr, bmin, show_raw, load_btn



# ══════════════════════════════════════════════════════════════════════════════
#  CLASIFICACIÓN DE SESIÓN
# ══════════════════════════════════════════════════════════════════════════════

def build_standings(session, session_label: str) -> tuple:
    """
    Genera la tabla de clasificación completa de la sesión.
    Retorna (df_standings, fig) donde fig es el gráfico de barras horizontales.

    Lógica por tipo de sesión:
    · Qualifying (Q)  → mejor vuelta de cada piloto
    · Race (R)        → posición final + tiempo total / gap al líder
    · FP1/FP2/FP3     → mejor vuelta de cada piloto
    """
    try:
        laps = session.laps.copy()
    except Exception as e:
        return pd.DataFrame(), None

    is_race = session_label in ("Race", "Sprint")

    rows = []

    if is_race:
        # Para la carrera usamos session.results si está disponible
        try:
            res = session.results
            for _, row in res.iterrows():
                drv  = row.get("Abbreviation", "???")
                pos  = row.get("Position", None)
                team = row.get("TeamName", "")
                status = row.get("Status", "")
                pts  = row.get("Points", 0)

                # Tiempo / gap
                try:
                    t = row.get("Time")
                    if hasattr(t, "total_seconds") and not pd.isna(t):
                        gap_s = t.total_seconds()
                        if pos == 1:
                            time_str = _fmt_seconds(gap_s)
                            gap_str  = "LÍDER"
                        else:
                            gap_str  = f"+{gap_s:.3f}s"
                            time_str = gap_str
                    else:
                        time_str = str(status)
                        gap_str  = str(status)
                        gap_s    = 9999.0
                except Exception:
                    time_str = gap_str = str(status)
                    gap_s    = 9999.0

                # Mejor vuelta de carrera
                drv_laps = laps[laps["Driver"] == drv]
                best_lt  = "–"
                if not drv_laps.empty:
                    valid = drv_laps["LapTime"].dropna()
                    if not valid.empty:
                        best_lt = _fmt_seconds(valid.min().total_seconds())

                rows.append({
                    "POS": int(pos) if pos and not pd.isna(pos) else 99,
                    "PILOTO": drv,
                    "EQUIPO": team,
                    "TIEMPO / GAP": time_str,
                    "MEJOR VUELTA": best_lt,
                    "PTS": int(pts) if pts and not pd.isna(pts) else 0,
                    "STATUS": status,
                    "_gap_s": gap_s,
                })
        except Exception:
            is_race = False   # fallback a mejor vuelta

    if not is_race:
        # Qualifying / FP → mejor vuelta de cada piloto
        for drv in laps["Driver"].unique():
            drv_laps = laps[laps["Driver"] == drv]
            valid    = drv_laps["LapTime"].dropna()
            if valid.empty:
                continue

            best     = valid.min()
            best_s   = best.total_seconds()
            best_str = _fmt_seconds(best_s)

            # Datos del piloto
            try:
                info = session.get_driver(drv)
                drv_abbr = info.get("Abbreviation", drv)
                team     = info.get("TeamName", "")
            except Exception:
                drv_abbr = drv
                team     = ""

            rows.append({
                "POS": 0,       # se asigna al ordenar
                "PILOTO": drv_abbr,
                "EQUIPO": team,
                "MEJOR VUELTA": best_str,
                "TIEMPO / GAP": best_str,
                "PTS": 0,
                "STATUS": "",
                "_gap_s": best_s,
            })

        # Ordenar por tiempo y asignar posición
        rows.sort(key=lambda x: x["_gap_s"])
        leader_s = rows[0]["_gap_s"] if rows else 0
        for i, r in enumerate(rows):
            r["POS"] = i + 1
            gap = r["_gap_s"] - leader_s
            r["TIEMPO / GAP"] = (
                _fmt_seconds(r["_gap_s"]) if i == 0
                else f"+{gap:.3f}s"
            )

    if not rows:
        return pd.DataFrame(), None

    df = pd.DataFrame(rows).sort_values("POS").reset_index(drop=True)

    # ── Gráfico de barras horizontales ────────────────────────────────────────
    # Mostrar solo tiempos válidos (excluir DNF/DNS con gap_s=9999)
    df_plot = df[df["_gap_s"] < 9000].copy()

    # Colores por equipo
    TEAM_COLORS_MAP = {
        "mercedes":    "#27F4D2", "ferrari":     "#E8002D",
        "red bull":    "#3671C6", "mclaren":     "#FF8000",
        "aston martin":"#229971", "alpine":      "#FF87BC",
        "williams":    "#64C4FF", "rb":          "#6692FF",
        "kick sauber": "#52E252", "sauber":      "#52E252",
        "haas":        "#B6BABD", "audi":        "#C8F026",
        "cadillac":    "#A50F2D",
    }

    def team_color(team_name):
        tl = str(team_name).lower()
        for k, v in TEAM_COLORS_MAP.items():
            if k in tl:
                return v
        return "#566A7F"

    colors   = [team_color(t) for t in df_plot["EQUIPO"]]
    pilots   = [f"P{r['POS']} {r['PILOTO']}" for _, r in df_plot.iterrows()]
    times    = df_plot["_gap_s"].values
    # Normalizar para la barra (offset desde el mínimo para que todas empiecen visible)
    base     = times.min()
    bar_vals = times - base + 0.001

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=bar_vals,
        y=pilots,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(color="rgba(0,0,0,0)", width=0),
        ),
        customdata=list(zip(
            df_plot["TIEMPO / GAP"].values,
            df_plot["MEJOR VUELTA"].values if "MEJOR VUELTA" in df_plot else [""] * len(df_plot),
            df_plot["EQUIPO"].values,
        )),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Tiempo: %{customdata[0]}<br>"
            "Equipo: %{customdata[2]}<extra></extra>"
        ),
        text=df_plot["TIEMPO / GAP"].values,
        textposition="outside",
        textfont=dict(family="Share Tech Mono, monospace",
                      color="#C8D6E5", size=10),
    ))

    fig.update_layout(**pb(
        height=max(400, len(df_plot) * 28 + 100),
        paper_bgcolor="#070B0F",
        plot_bgcolor="#090D13",
        margin=dict(l=110, r=120, t=50, b=40),
        title=dict(
            text=f"CLASIFICACIÓN — {session_label.upper()}",
            font=dict(family="Share Tech Mono, monospace",
                      color="#566A7F", size=11),
            x=0,
        ),
        xaxis=dict(
            **AX,
            title_text="Diferencia respecto al líder (s)",
            title_font=dict(color="#566A7F", size=10),
            tickformat=".3f",
        ),
        yaxis=dict(
            **AX,
            autorange="reversed",   # P1 arriba
        ),
        showlegend=False,
        hovermode="y unified",
        hoverlabel=dict(
            bgcolor="#0C1018", bordercolor="#1A2535",
            font=dict(family="Share Tech Mono, monospace",
                      color="#C8D6E5", size=11),
        ),
    ))

    # Quitar columna interna antes de devolver
    df = df.drop(columns=["_gap_s"])
    fig.update_yaxes(tickfont=dict(family="Orbitron, monospace",
                                   color="#C8D6E5", size=10))
    return df, fig


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    render_header()

    if not FASTF1_AVAILABLE:
        st.markdown("""<div class="warn-box">
          ⚠️  FastF1 not installed.<br><br>
          Run: <code>pip install fastf1 plotly scipy</code> then restart.
        </div>""", unsafe_allow_html=True)
        return

    setup_cache()

    (year, gp_name, sess, driver1, driver2, n_pts,
     bthr, bmin, show_raw, load_btn) = render_sidebar()

    if "tdata" not in st.session_state:
        st.session_state.tdata = None

    # ── Load ──────────────────────────────────────────────────────────────────
    if load_btn:
        if driver1 == driver2:
            st.warning("Select two different drivers.")
            return

        ph_status = st.empty()
        ph_prog   = st.progress(0)

        def status(msg, pct, done=False):
            cls = "" if done else "loading"
            ph_status.markdown(
                f'<div class="status-bar"><div class="dot {cls}"></div>{msg}</div>',
                unsafe_allow_html=True)
            ph_prog.progress(pct)

        try:
            status(f"Loading {year} {gp_name} — {SESSION_LABELS[sess]}…", 10)
            session = load_session(year, gp_name, sess)

            status(f"Extracting fastest laps — {driver1} & {driver2}…", 35)
            r1 = get_fastest_lap(session, driver1)
            r2 = get_fastest_lap(session, driver2)
            lap1, tel1, pos1 = (r1 if len(r1)==3 else (*r1, None))
            lap2, tel2, pos2 = (r2 if len(r2)==3 else (*r2, None))

            if tel1 is None:
                st.error(f"No telemetry for {driver1}."); return
            if tel2 is None:
                st.error(f"No telemetry for {driver2}."); return

            status("Interpolating to common distance grid…", 55)
            dist, t1i, t2i = interpolate_tel(tel1, tel2, n_pts)
            delta = compute_delta(dist, t1i, t2i)

            status("Detecting braking zones…", 70)
            bz1 = detect_braking_zones(dist, t1i, bthr, bmin)
            bz2 = detect_braking_zones(dist, t2i, bthr, bmin)

            status("Loading lap pace data…", 82)
            try:
                all1 = get_all_laps(session, driver1)
                all2 = get_all_laps(session, driver2)
            except Exception:
                all1 = pd.DataFrame(); all2 = pd.DataFrame()

            status("Loading race pace (quick laps)…", 90)
            try:
                ql1 = get_quick_laps(session, driver1)
                ql2 = get_quick_laps(session, driver2)
            except Exception:
                ql1 = pd.DataFrame(); ql2 = pd.DataFrame()

            # Colores de equipo
            tc1 = get_team_color(session, driver1, D1)
            tc2 = get_team_color(session, driver2, D2)

            # Convertir lap1/lap2 de Series FastF1 a dict serializable
            def lap_to_dict(lap):
                if lap is None: return {}
                try:
                    return {k: (v.total_seconds() if hasattr(v, 'total_seconds')
                                else v)
                            for k, v in lap.items()
                            if isinstance(v, (int, float, str, type(None)))
                            or hasattr(v, 'total_seconds')}
                except Exception:
                    return {}

            # Convertir pos1/pos2 a DataFrame con solo columnas X,Y (liviano)
            def slim_pos(pos):
                if pos is None or (hasattr(pos, 'empty') and pos.empty):
                    return None
                try:
                    cols = [c for c in ['X','Y'] if c in pos.columns]
                    return pos[cols].reset_index(drop=True) if cols else None
                except Exception:
                    return None

            # IMPORTANTE: NO guardar session (objeto FastF1) dentro de
            # st.session_state — Streamlit lo serializa y corrompe el estado.
            # Se accede siempre via load_session() que usa cache_resource.
            session_key = f"{year}||{gp_name}||{sess}"

            st.session_state.tdata = dict(
                dist=dist, t1=t1i, t2=t2i, delta=delta,
                lap1=lap_to_dict(lap1), lap2=lap_to_dict(lap2),
                pos1=slim_pos(pos1), pos2=slim_pos(pos2),
                bz1=bz1, bz2=bz2, all1=all1, all2=all2,
                ql1=ql1, ql2=ql2,
                tc1=tc1, tc2=tc2,
                driver1=driver1, driver2=driver2,
                gp_name=gp_name, year=year,
                session=SESSION_LABELS[sess],
                session_key=session_key,   # clave para recuperar la sesión
            )

            status(f"✓  Ready — {len(dist)} pts · {len(bz1)} zones ({driver1}) · {len(bz2)} zones ({driver2})",
                   100, done=True)

        except Exception as e:
            ph_prog.empty(); ph_status.empty()
            st.error(f"Error: {e}")
            st.info("💡 Tip: Qualifying sessions have the richest telemetry.")
            return

    # ── Welcome screen ────────────────────────────────────────────────────────
    if st.session_state.tdata is None:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;
                    justify-content:center;min-height:62vh;text-align:center;padding:48px 20px">
          <div style="font-size:68px;margin-bottom:20px">🏎️</div>
          <div style="font-family:'Orbitron',sans-serif;font-size:24px;font-weight:900;
                      color:#1A2535;letter-spacing:4px;text-transform:uppercase">NO DATA LOADED</div>
          <div style="font-family:'Share Tech Mono',monospace;font-size:11px;
                      color:#2E3E50;margin-top:14px;line-height:2.2;max-width:420px">
            SELECT SEASON · GP · SESSION · TWO DRIVERS<br>THEN CLICK ⚡ LOAD TELEMETRY
          </div>
          <div style="margin-top:40px;padding:16px 28px;border:1px solid #1A2535;border-radius:6px;
                      background:#0C1018;font-family:'Share Tech Mono',monospace;font-size:9px;
                      color:#2E3E50;line-height:2.3;text-align:left">
            ◆  6-CHANNEL SYNCHRONIZED TELEMETRY<br>
            ◆  DISTANCE-BASED AXIS (not time)<br>
            ◆  SCIPY INTERPOLATION TO COMMON GRID<br>
            ◆  CUMULATIVE Δ TIME CALCULATION<br>
            ◆  SPEED HEATMAP TRACK MAP<br>
            ◆  AUTOMATIC BRAKING ZONE DETECTION<br>
            ◆  LAP PACE PROGRESSION CHART<br>
            ◆  SELF-CONTAINED HTML REPORT EXPORT
          </div>
        </div>""", unsafe_allow_html=True)
        return

    # ── Dashboard ─────────────────────────────────────────────────────────────
    d = st.session_state.tdata
    d1, d2 = d["driver1"], d["driver2"]

    # Context bar
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px">
      <span class="sticker" style="background:#E8002D18;color:#E8002D;border:1px solid #E8002D30">{d1}</span>
      <span style="font-family:'Share Tech Mono',monospace;color:#2E3E50;font-size:16px">VS</span>
      <span class="sticker" style="background:#00D4FF18;color:#00D4FF;border:1px solid #00D4FF30">{d2}</span>
      <span style="flex:1"></span>
      <span style="font-family:'Share Tech Mono',monospace;font-size:9px;color:#2E3E50">
        {d['year']} · {d['gp_name']} · {d['session']}
      </span>
    </div>""", unsafe_allow_html=True)

    render_kpis(d["lap1"], d["lap2"], d1, d2, d["delta"][-1], d["t1"], d["t2"])
    render_sectors(d["lap1"], d["lap2"], d1, d2)

    # ── TABS ──────────────────────────────────────────────────────────────────
    t_tel, t_map, t_stats, t_race, t_pace, t_standings, t_export = st.tabs([
        "📡  TELEMETRY",
        "🗺  TRACK MAP",
        "📊  ANALYSIS",
        "🏎  RACE PACE",
        "🏁  PACE",
        "🏆  CLASIFICACIÓN",
        "⬇  EXPORT",
    ])

    CHART_CFG = {"scrollZoom": True, "displayModeBar": True,
                 "modeBarButtonsToRemove": ["autoScale2d","lasso2d","select2d"],
                 "displaylogo": False}
    MINI_CFG  = {"displayModeBar": False, "displaylogo": False}

    # ── TAB 1: Telemetry ──────────────────────────────────────────────────────
    with t_tel:
        st.markdown('<div class="sec-label">TELEMETRY CHANNELS — DISTANCE AXIS</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(
            build_main_telemetry(d["dist"], d["t1"], d["t2"], d1, d2, d["delta"]),
            use_container_width=True, config=CHART_CFG)
        st.markdown("</div>", unsafe_allow_html=True)

        # Delta summary row
        delta = d["delta"]; dist = d["dist"]
        c1, c2, c3, c4 = st.columns(4)
        gi = int(np.argmax(delta)); li = int(np.argmin(delta))
        # delta>0 → d1 gaining; delta<0 → d2 gaining
        final_leader = d1 if delta[-1] > 0 else d2
        for col, label, val, sub, cls in [
            (c1, "FINAL DELTA",      f"{delta[-1]:+.3f}s",
             f"{final_leader} FASTER overall", "neu"),
            (c2, f"MAX {d1} GAIN",   f"{delta.max():.3f}s",
             f"at {dist[gi]:.0f}m", "d2"),
            (c3, f"MAX {d2} GAIN",   f"{abs(delta.min()):.3f}s",
             f"at {dist[li]:.0f}m", "d1"),
            (c4, "AVG SPEED DIFF",
             f"{d['t1']['Speed'].mean()-d['t2']['Speed'].mean():+.1f}",
             f"km/h ({d1} vs {d2})", "neu"),
        ]:
            col.markdown(f"""
            <div class="metric-card {cls}" style="margin-top:8px">
              <div class="mc-label">{label}</div>
              <div class="mc-val" style="font-size:17px">{val}</div>
              <div class="mc-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    # ── TAB 2: Track Map ──────────────────────────────────────────────────────
    with t_map:
        st.markdown('<div class="sec-label">SPEED HEATMAP — CIRCUIT LAYOUT</div>',
                    unsafe_allow_html=True)

        if d["pos1"] is None and d["pos2"] is None:
            st.markdown("""<div class="warn-box">
              ⚠  No positional (GPS) data available for this session/lap.<br>
              Track maps require position data — usually available in Qualifying and Race.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            st.plotly_chart(
                build_dual_track_map(d["pos1"], d["pos2"], d["t1"], d["t2"], d1, d2),
                use_container_width=True, config=MINI_CFG)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("""
            <div style="font-family:'Share Tech Mono',monospace;font-size:9px;color:#2E3E50;padding:6px 0;line-height:2">
              COLOR SCALE →
              <span style="color:#2E0000">■ Slow</span>
              <span style="color:#E8002D">■</span>
              <span style="color:#FFD700">■</span>
              <span style="color:#39D353">■</span>
              <span style="color:#00D4FF">■ Fast</span>
            </div>""", unsafe_allow_html=True)

        # Braking zone map
        st.markdown('<div class="sec-label">BRAKING ZONES MAP</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(
            build_braking_chart(d["bz1"], d["bz2"], d1, d2, float(d["dist"].max())),
            use_container_width=True, config=MINI_CFG)
        st.markdown("</div>", unsafe_allow_html=True)

        # Braking zone tables
        ca, cb = st.columns(2)
        for col_st, zones, name, color in [(ca, d["bz1"], d1, D1), (cb, d["bz2"], d2, D2)]:
            with col_st:
                rows = "".join(f"""
                <tr><td>{i}</td><td>{z['start_m']:.0f}</td><td>{z['end_m']:.0f}</td>
                <td>{z['length_m']:.0f}</td><td>{z['max_speed_entry']:.0f}</td>
                <td style="color:{color};font-weight:700">{z['speed_drop']:.0f}</td></tr>"""
                for i, z in enumerate(zones, 1))
                st.markdown(f"""
                <div class="sec-label"><span style="color:{color}">■</span> {name}</div>
                <table class="bz-table">
                  <thead><tr><th>#</th><th>Start</th><th>End</th>
                    <th>Length</th><th>Entry km/h</th><th>Δv km/h</th></tr></thead>
                  <tbody>{rows if rows else '<tr><td colspan=6 style="color:#2E3E50">No braking zones detected</td></tr>'}</tbody>
                </table>""", unsafe_allow_html=True)

    # ── TAB 3: Statistical Analysis ───────────────────────────────────────────
    with t_stats:
        ca, cb = st.columns(2)
        with ca:
            st.markdown('<div class="sec-label">SPEED DISTRIBUTION</div>', unsafe_allow_html=True)
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            st.plotly_chart(build_speed_histogram(d["t1"], d["t2"], d1, d2),
                use_container_width=True, config=MINI_CFG)
            st.markdown("</div>", unsafe_allow_html=True)
        with cb:
            st.markdown('<div class="sec-label">GEAR USAGE (%)</div>', unsafe_allow_html=True)
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            st.plotly_chart(build_gear_usage(d["t1"], d["t2"], d1, d2),
                use_container_width=True, config=MINI_CFG)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="sec-label">THROTTLE EFFICIENCY MAP</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(build_throttle_scatter(d["t1"], d["t2"], d1, d2),
            use_container_width=True, config=MINI_CFG)
        st.markdown("</div>", unsafe_allow_html=True)

        # Channel stats table
        st.markdown('<div class="sec-label">CHANNEL STATISTICS</div>', unsafe_allow_html=True)
        rows = []
        for ch, unit in [("Speed","km/h"),("Throttle","%"),("nGear",""),("RPM","rpm"),("Steering","°")]:
            if ch not in d["t1"].columns:
                continue
            v1 = d["t1"][ch].dropna(); v2 = d["t2"][ch].dropna()
            rows.append({
                "Channel": ch, "Unit": unit,
                f"{d1} Mean": f"{v1.mean():.2f}",
                f"{d1} Max":  f"{v1.max():.2f}",
                f"{d1} Std":  f"{v1.std():.2f}",
                f"{d2} Mean": f"{v2.mean():.2f}",
                f"{d2} Max":  f"{v2.max():.2f}",
                f"{d2} Std":  f"{v2.std():.2f}",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if show_raw:
            with st.expander("📋  RAW INTERPOLATED DATA"):
                df_raw = pd.DataFrame({
                    "Distance_m": d["dist"],
                    f"Speed_{d1}": d["t1"]["Speed"],    f"Speed_{d2}": d["t2"]["Speed"],
                    f"Throttle_{d1}": d["t1"]["Throttle"], f"Throttle_{d2}": d["t2"]["Throttle"],
                    f"Brake_{d1}": d["t1"]["Brake"],    f"Brake_{d2}": d["t2"]["Brake"],
                    f"Gear_{d1}": d["t1"]["nGear"],     f"Gear_{d2}": d["t2"]["nGear"],
                    "DeltaTime_s": d["delta"],
                }).round(4)
                st.dataframe(df_raw, use_container_width=True, height=280)
                st.download_button("⬇  Download CSV",
                    df_raw.to_csv(index=False).encode(),
                    file_name=f"tel_{d['year']}_{d1}_vs_{d2}.csv",
                    mime="text/csv")

    # ── TAB 4: Race Pace Analysis (estilo Corsino) ────────────────────────────
    with t_race:
        ql1 = d.get("ql1", pd.DataFrame())
        ql2 = d.get("ql2", pd.DataFrame())
        tc1 = d.get("tc1", D1)
        tc2 = d.get("tc2", D2)

        # Recuperar la sesión FastF1 desde cache_resource (no desde session_state)
        session_obj = None
        try:
            sk = d.get("session_key", "")
            if sk:
                parts = sk.split("||")
                session_obj = load_session(int(parts[0]), parts[1], parts[2])
        except Exception:
            session_obj = None

        # ════════════════════════════════════════════════════════════════════
        # SECCIÓN A — VIOLIN PARRILLA COMPLETA (selector multi-piloto)
        # ════════════════════════════════════════════════════════════════════
        st.markdown('<div class="sec-label">🎻  VIOLIN — PARRILLA COMPLETA</div>',
                    unsafe_allow_html=True)

        # Obtener lista completa de pilotos de la sesión
        all_session_drivers = []
        if session_obj is not None:
            try:
                all_session_drivers = sorted([
                    session_obj.get_driver(drv)["Abbreviation"]
                    for drv in session_obj.drivers
                ])
            except Exception:
                all_session_drivers = list({d1, d2})

        # Persistir selección en session_state para que no se resetee en cada rerun
        # Solo inicializar si cambia la sesión/GP o si no existe aún
        state_key = f"violin_sel_{d.get('year')}_{d.get('gp_name')}_{d.get('session')}"
        if state_key not in st.session_state:
            # Primera vez con estos datos: preseleccionar los 2 pilotos del sidebar
            st.session_state[state_key] = sorted(
                {d1, d2} & set(all_session_drivers)
            ) or ([d1, d2] if all_session_drivers == [] else all_session_drivers[:2])

        # Limpiar selección guardada si contiene pilotos que ya no están en la sesión
        valid_saved = [p for p in st.session_state[state_key] if p in all_session_drivers]
        if valid_saved != st.session_state[state_key]:
            st.session_state[state_key] = valid_saved

        # Selector multi-piloto
        st.markdown("""
        <div style="font-family:'Share Tech Mono',monospace;font-size:9px;
                    color:#566A7F;letter-spacing:1.5px;margin-bottom:4px">
            SELECCIONAR PILOTOS PARA EL VIOLIN (mínimo 1)
        </div>""", unsafe_allow_html=True)

        selected_drivers = st.multiselect(
            label="Pilotos",
            options=all_session_drivers if all_session_drivers else [d1, d2],
            default=st.session_state[state_key],
            label_visibility="collapsed",
            key="race_pace_multiselect",
        )
        # Guardar selección actual para el próximo rerun
        st.session_state[state_key] = selected_drivers

        if len(selected_drivers) < 1:
            st.info("Selecciona al menos un piloto.")
        elif session_obj is None:
            st.warning("Sesión no disponible — recarga los datos.")
        else:
            try:
                fig_grid = build_grid_violin(
                    session_obj,
                    selected_drivers,
                    gp_name=d.get("gp_name", ""),
                    year=d.get("year", 2026),
                )
                st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                st.plotly_chart(fig_grid, use_container_width=True,
                               config={**CHART_CFG, "scrollZoom": False})
                st.markdown("</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error generando violin de parrilla: {e}")

        st.markdown("<br>", unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════════════════
        # SECCIÓN B — ANÁLISIS DE 2 PILOTOS (los del sidebar)
        # ════════════════════════════════════════════════════════════════════
        if ql1.empty and ql2.empty:
            st.markdown("""
            <div class="warn-box">
              ⚠  No se encontraron vueltas limpias (<code>pick_quicklaps</code>).<br>
              Este análisis requiere sesión de <b>Carrera (R)</b>.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="sec-label">📈  EVOLUCIÓN — {d1} vs {d2}</div>',
                unsafe_allow_html=True)

            rp = None
            stats = {}
            try:
                rp = plot_race_pace_analysis(
                    session_obj, d1, d2, tc1, tc2)
                stats = rp["stats"]
            except Exception as e:
                st.error(f"Error generando análisis de ritmo: {e}")

            # KPI cards
            kpi_cols = st.columns(6)
            kpi_data = []
            for driver, col_css, tc in [(d1,"d1",tc1),(d2,"d2",tc2)]:
                s = stats.get(driver, {})
                if s:
                    kpi_data += [
                        (kpi_cols[0 if driver==d1 else 3],
                         "VUELTAS LIMPIAS", f"{s['laps']}", "laps", col_css),
                        (kpi_cols[1 if driver==d1 else 4],
                         "MEDIANA", _fmt_seconds(s['median']), driver, col_css),
                        (kpi_cols[2 if driver==d1 else 5],
                         "CONSISTENCIA σ", f"{s['std']:.3f}s", driver, col_css),
                    ]
            for col_st, label, val, sub, cls in kpi_data:
                col_st.markdown(f"""
                <div class="metric-card {cls}" style="margin-bottom:10px">
                  <div class="mc-label">{label}</div>
                  <div class="mc-val" style="font-size:16px">{val}</div>
                  <div class="mc-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)

            # Comparativa
            if d1 in stats and d2 in stats:
                diff_med = stats[d1]["median"] - stats[d2]["median"]
                diff_std = stats[d1]["std"]    - stats[d2]["std"]
                faster_p = d1 if diff_med < 0 else d2
                more_con = d1 if diff_std < 0 else d2
                st.markdown(f"""
                <div style="display:flex;gap:10px;flex-wrap:wrap;margin:8px 0 16px">
                  <div class="metric-card neu" style="flex:1;min-width:180px">
                    <div class="mc-label">⚡ DIFERENCIA MEDIANA</div>
                    <div class="mc-val" style="font-size:17px">{abs(diff_med):.3f}s</div>
                    <div class="mc-sub">{faster_p} más rápido en ritmo</div>
                  </div>
                  <div class="metric-card grn" style="flex:1;min-width:180px">
                    <div class="mc-label">📐 DIFERENCIA σ</div>
                    <div class="mc-val" style="font-size:17px">{abs(diff_std):.3f}s</div>
                    <div class="mc-sub">{more_con} más consistente</div>
                  </div>
                </div>""", unsafe_allow_html=True)

            # Gráfico de evolución
            st.markdown('<div class="sec-label">EVOLUCIÓN DE TIEMPOS POR VUELTA</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            if rp is not None:
                try:
                    st.plotly_chart(rp["fig_evolution"],
                        use_container_width=True, config=CHART_CFG)
                except Exception as e:
                    st.error(f"Error en gráfico de evolución: {e}")
            else:
                st.info("Gráfico no disponible — revisa el error de carga arriba.")
            st.markdown("</div>", unsafe_allow_html=True)

            # Violin 2 pilotos
            st.markdown(
                f'<div class="sec-label">VIOLIN {d1} vs {d2}</div>',
                unsafe_allow_html=True)
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            if rp is not None:
                try:
                    st.plotly_chart(rp["fig_box"],
                        use_container_width=True, config=MINI_CFG)
                except Exception as e:
                    st.error(f"Error en violin: {e}")
            else:
                st.info("Gráfico no disponible — revisa el error de carga arriba.")
            st.markdown("</div>", unsafe_allow_html=True)

            # Tabla de vueltas limpias
            with st.expander("📋  VUELTAS LIMPIAS UTILIZADAS"):
                cols_show = ["LapNumber","LapTimeSeconds","Compound",
                             "TyreLife","TrackStatus","IsAccurate"]
                ca, cb = st.columns(2)
                for col_st, df_ql, name, tc in [
                    (ca, ql1, d1, tc1), (cb, ql2, d2, tc2)
                ]:
                    with col_st:
                        st.markdown(
                            f'<div style="font-family:Share Tech Mono,monospace;'
                            f'font-size:10px;color:{tc};margin-bottom:6px">'
                            f'■ {name} — {len(df_ql)} vueltas</div>',
                            unsafe_allow_html=True)
                        if not df_ql.empty:
                            show_cols = [c for c in cols_show if c in df_ql.columns]
                            disp = df_ql[show_cols].copy()
                            if "LapTimeSeconds" in disp.columns:
                                disp["LapTime"] = disp["LapTimeSeconds"].apply(
                                    lambda x: _fmt_seconds(x) if not np.isnan(x) else "–")
                                disp = disp.drop(columns=["LapTimeSeconds"])
                            st.dataframe(disp, use_container_width=True,
                                         hide_index=True, height=280)

    # ── TAB 5: Lap Pace ───────────────────────────────────────────────────────
    with t_pace:
        if d["all1"].empty and d["all2"].empty:
            st.info("Lap data not available for this session type.")
        else:
            st.markdown('<div class="sec-label">LAP PACE PROGRESSION</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            st.plotly_chart(build_pace_chart(d["all1"], d["all2"], d1, d2),
                use_container_width=True, config=MINI_CFG)
            st.markdown("</div>", unsafe_allow_html=True)
            render_lap_table(d["all1"], d["all2"], d1, d2)

    # ── TAB 5: Export ─────────────────────────────────────────────────────────
    with t_export:
        st.markdown('<div class="sec-label">EXPORT OPTIONS</div>', unsafe_allow_html=True)

        ea, eb = st.columns(2)
        with ea:
            st.markdown("""
            <div class="metric-card neu" style="margin-bottom:12px">
              <div class="mc-label">📄 HTML REPORT</div>
              <div class="mc-val" style="font-size:14px">Self-Contained</div>
              <div class="mc-sub">ALL CHARTS · INTERACTIVE · SHAREABLE</div>
            </div>""", unsafe_allow_html=True)

            if st.button("🖨  Generate HTML Report", use_container_width=True):
                with st.spinner("Building report…"):
                    try:
                        html = generate_report(d)
                        st.download_button(
                            "⬇  Download HTML Report",
                            html.encode("utf-8"),
                            file_name=f"f1_tel_{d['year']}_{d['gp_name'].replace(' ','_')}_{d1}_vs_{d2}.html",
                            mime="text/html",
                            use_container_width=True)
                        st.success("Report ready!")
                    except Exception as e:
                        st.error(f"Report failed: {e}")

        with eb:
            st.markdown("""
            <div class="metric-card d2" style="margin-bottom:12px">
              <div class="mc-label">📊 CSV — ALL CHANNELS</div>
              <div class="mc-val" style="font-size:14px">Raw Data</div>
              <div class="mc-sub">SPEED · THROTTLE · BRAKE · GEAR · ΔTIME</div>
            </div>""", unsafe_allow_html=True)

            df_exp = pd.DataFrame({
                "Distance_m":       d["dist"],
                f"Speed_{d1}_kmh":  d["t1"]["Speed"],
                f"Speed_{d2}_kmh":  d["t2"]["Speed"],
                f"Throttle_{d1}":   d["t1"]["Throttle"],
                f"Throttle_{d2}":   d["t2"]["Throttle"],
                f"Brake_{d1}":      d["t1"]["Brake"],
                f"Brake_{d2}":      d["t2"]["Brake"],
                f"Gear_{d1}":       d["t1"]["nGear"],
                f"Gear_{d2}":       d["t2"]["nGear"],
                f"Steering_{d1}":   d["t1"]["Steering"],
                f"Steering_{d2}":   d["t2"]["Steering"],
                f"RPM_{d1}":        d["t1"]["RPM"],
                f"RPM_{d2}":        d["t2"]["RPM"],
                "DeltaTime_s":      d["delta"],
            }).round(4)
            st.download_button("⬇  Download CSV",
                df_exp.to_csv(index=False).encode("utf-8"),
                file_name=f"tel_{d['year']}_{d['gp_name'].replace(' ','_')}_{d1}_vs_{d2}.csv",
                mime="text/csv", use_container_width=True)

        # Braking zones CSV
        st.markdown('<div class="sec-label">BRAKING ZONES EXPORT</div>', unsafe_allow_html=True)
        bz_df = pd.DataFrame(
            [{**z, "driver": d1} for z in d["bz1"]] +
            [{**z, "driver": d2} for z in d["bz2"]]
        )
        if not bz_df.empty:
            st.dataframe(bz_df.round(1), use_container_width=True, hide_index=True)
            st.download_button("⬇  Download Braking Zones CSV",
                bz_df.to_csv(index=False).encode("utf-8"),
                file_name=f"bz_{d['year']}_{d1}_vs_{d2}.csv",
                mime="text/csv")

    # ── TAB 6: Clasificación ──────────────────────────────────────────────────
    with t_standings:
        st.markdown('<div class="sec-label">CLASIFICACIÓN DE LA SESIÓN</div>',
                    unsafe_allow_html=True)

        # Recuperar sesión desde cache_resource
        session_obj_st = None
        try:
            sk = d.get("session_key", "")
            if sk:
                parts = sk.split("||")
                session_obj_st = load_session(int(parts[0]), parts[1], parts[2])
        except Exception:
            session_obj_st = None

        if session_obj_st is None:
            st.warning("Sesión no disponible — recargá los datos.")
        else:
            with st.spinner("Calculando clasificación..."):
                try:
                    df_st, fig_st = build_standings(
                        session_obj_st, d.get("session", ""))
                except Exception as e:
                    df_st = pd.DataFrame()
                    fig_st = None
                    st.error(f"Error generando clasificación: {e}")

            if fig_st is not None:
                st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                st.plotly_chart(fig_st, use_container_width=True,
                               config=MINI_CFG)
                st.markdown("</div>", unsafe_allow_html=True)

            if not df_st.empty:
                st.markdown('<div class="sec-label">TABLA COMPLETA</div>',
                            unsafe_allow_html=True)

                # Resaltar los dos pilotos seleccionados
                def highlight_drivers(row):
                    if row["PILOTO"] == d1:
                        return [f"background-color: #E8002D20; color: #E8002D"] * len(row)
                    elif row["PILOTO"] == d2:
                        return [f"background-color: #00D4FF20; color: #00D4FF"] * len(row)
                    return [""] * len(row)

                # Columnas a mostrar según tipo de sesión
                show_cols = ["POS","PILOTO","EQUIPO","TIEMPO / GAP"]
                if "MEJOR VUELTA" in df_st.columns and d.get("session") != "Race":
                    show_cols.append("MEJOR VUELTA")
                if "PTS" in df_st.columns and d.get("session") == "Race":
                    show_cols += ["MEJOR VUELTA","PTS","STATUS"]

                show_cols = [c for c in show_cols if c in df_st.columns]

                styled = (
                    df_st[show_cols]
                    .style
                    .apply(highlight_drivers, axis=1)
                    .set_properties(**{
                        "font-family": "Share Tech Mono, monospace",
                        "font-size":   "12px",
                    })
                    .format({"POS": "{:.0f}", "PTS": "{:.0f}"},
                            na_rep="–")
                )
                st.dataframe(styled, use_container_width=True,
                             hide_index=True, height=600)

                st.download_button(
                    "⬇  Descargar clasificación CSV",
                    df_st[show_cols].to_csv(index=False).encode("utf-8"),
                    file_name=(f"clasificacion_{d['year']}_"
                               f"{d['gp_name'].replace(' ','_')}_"
                               f"{d.get('session','')}.csv"),
                    mime="text/csv",
                )

    st.markdown("<br><br>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
