import io, math, re, zipfile
import numpy as np, pandas as pd, streamlit as st
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import engine as E

st.set_page_config(page_title='RC Frame Analyzer', page_icon='🏗️', layout='wide')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.grid': True, 'grid.alpha': .3, 'axes.spines.top': False, 'axes.spines.right': False})
NAVY, ORANGE, GREEN, RED, AMBER = '#1F3B73', '#B5542B', '#2E7D32', '#C62828', '#ED8F00'
SLABC, FOOTC = '#6fa8dc', '#8d8d8d'
st.markdown(f"""<style>
html, body, [class*="css"], .stMarkdown, .stTabs, label {{font-family:'Segoe UI','Helvetica Neue',Arial,sans-serif !important}}
.block-container{{padding-top:3.2rem;padding-bottom:2rem;max-width:1500px}}
.hdr{{background:linear-gradient(90deg,{NAVY},#2c5aa0);border-left:8px solid {ORANGE};color:#fff;padding:14px 22px;border-radius:6px;margin-bottom:14px}}
.hdr-t{{font-size:1.9rem;font-weight:700;letter-spacing:.3px;line-height:1.15}} .hdr-s{{font-size:.9rem;opacity:.9;margin-top:3px}}
.kpis{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:12px}}
.kpi{{border:1px solid rgba(120,140,170,.35);border-top:4px solid {NAVY};border-radius:6px;padding:8px 12px;background:rgba(120,140,170,.07)}}
.kpi .l{{font-size:.72rem;text-transform:uppercase;letter-spacing:.6px;opacity:.75}} .kpi .v{{font-size:1.55rem;font-weight:700;line-height:1.2}} .kpi .s{{font-size:.75rem;opacity:.7}}
.kpi.pass{{border-top-color:{GREEN}}} .kpi.pass .v{{color:{GREEN}}} .kpi.fail{{border-top-color:{RED}}} .kpi.fail .v{{color:{RED}}}
.banner{{padding:10px 16px;border-radius:6px;font-weight:600;margin:6px 0 10px}} .banner.pass{{background:rgba(46,125,50,.15);border-left:6px solid {GREEN};color:{GREEN}}}
.banner.fail{{background:rgba(198,40,40,.13);border-left:6px solid {RED};color:{RED}}} .banner.info{{background:rgba(31,59,115,.09);border-left:6px solid {NAVY};color:{NAVY}}}
.banner.warn{{background:rgba(237,143,0,.14);border-left:6px solid {AMBER};color:#8a5200}}
.step{{font-size:1.02rem;font-weight:700;color:{NAVY};border-bottom:2px solid {ORANGE};padding-bottom:2px;margin:4px 0 6px}}
.need{{font-size:.82rem;color:#444;background:rgba(237,143,0,.12);border-radius:4px;padding:4px 9px;margin:2px 0 6px}}
.guide{{font-size:.85rem;line-height:1.6}} .guide .ok{{color:{GREEN}}} .guide .no{{color:{RED}}} .guide .todo{{color:#777}}
.canvas{{border:2px dashed rgba(120,140,170,.55);border-radius:8px;padding:90px 20px;text-align:center;color:#667;background:rgba(120,140,170,.06)}}
.canvas b{{color:{NAVY};font-size:1.15rem}}
h2, h3 {{color:{NAVY};}} .stTabs [data-baseweb="tab"]{{font-weight:600}}
table.rt{{border-collapse:collapse;width:100%;font-size:.85rem;margin-bottom:8px}} table.rt th{{background:{NAVY};color:#fff;text-align:left;padding:6px 9px}}
table.rt td{{border-bottom:1px solid rgba(120,140,170,.3);padding:6px 9px;vertical-align:top}} table.rt tr:nth-child(even) td{{background:rgba(120,140,170,.08)}}
.badge{{display:inline-block;padding:1px 10px;border-radius:10px;color:#fff;font-weight:700;font-size:.75rem}} .badge.pass{{background:{GREEN}}} .badge.fail{{background:{RED}}} .badge.warn{{background:{AMBER}}}
</style>""", unsafe_allow_html=True)

def tbl(df, **kw):
    """Render a DataFrame as styled HTML (no pyarrow needed – works on PCs where pyarrow DLLs are blocked)."""
    html = df.copy().to_html(index=False, border=0, classes='rt', escape=True, float_format=lambda x: f'{x:.2f}')
    html = html.replace('<td>PASS</td>', '<td><span class="badge pass">PASS</span></td>').replace('<td>FAIL</td>', '<td><span class="badge fail">FAIL</span></td>')
    html = re.sub(r'<td>(WARN[^<]*)</td>', r'<td><span class="badge warn">\1</span></td>', html)
    st.markdown(html, unsafe_allow_html=True)

ss = st.session_state
# ---------------------------------------------------------------- input widgets (all EMPTY by default)
def N(label, key, mn, mx, step, fmt='%.1f', help=None):
    st.number_input(label, min_value=mn, max_value=mx, value=None, step=step, format=fmt, key=key, placeholder='enter value', help=help); return ss.get(key)
def S_(label, key, opts, help=None):
    st.selectbox(label, opts, index=None, placeholder='select', key=key, help=help); return ss.get(key)
def need(txt): st.markdown(f'<div class="need">{txt}</div>', unsafe_allow_html=True)
def step(txt): st.markdown(f'<div class="step">{txt}</div>', unsafe_allow_html=True)
def filled(*v): return all(x is not None for x in v)

KEYS = ['nx', 'ny', 'nst', 'L', 'H', 'fc', 'fy', 'stir', 'qa', 'cem', 'fa', 'sdlf', 'sdlr', 'llf', 'llr', 'wall', 'par', 'zone', 'soil', 'Rf', 'Imp', 'base',
        'c_n', 'c_w', 'c_d', 'c_h', 'c_nb', 'c_bs', 'b_n', 'b_len', 'b_w', 'b_d', 's_n', 's_len', 's_wid', 's_t', 'f_n', 'f_len', 'f_wid', 'f_t']
EXAMPLE = dict(nx=2, ny=2, nst=2, L=15.0, H=10.0, fc=3.0, fy=60.0, stir='#3', qa=3.0, cem=350.0, fa=20.0, sdlf=45.0, sdlr=35.0, llf=40.0, llr=20.0, wall=0.85, par=0.15,
               zone='Zone 2B (Z=0.20)', soil='Sd', Rf=5.5, Imp=1.0, base='fixed', c_n=18, c_w=15.0, c_d=15.0, c_h=10.0, c_nb=8, c_bs='#5',
               b_n=24, b_len=15.0, b_w=9.0, b_d=18.0, s_n=8, s_len=15.0, s_wid=15.0, s_t=5.0, f_n=9, f_len=6.5, f_wid=6.5, f_t=18.0)
def load_example():
    for k, v in EXAMPLE.items(): ss[k] = v
def clear_all():
    for k in KEYS: ss.pop(k, None)
    ss.pop('ran_sig', None)

# ---------------------------------------------------------------- header
st.markdown('<div class="hdr"><div class="hdr-t">RC Frame Analyzer</div><div class="hdr-s">G+1 residential RC frame &nbsp;·&nbsp; build the model component by component &nbsp;·&nbsp; 2-D analysis and design checks &nbsp;·&nbsp; ACI 318 / ASCE 7 / BCP-SP 2007</div></div>', unsafe_allow_html=True)

TABS = st.tabs(['🧱 Build Model', 'Summary', 'Model & Loads', 'Frame Diagrams', 'Beam', 'Columns', 'Slab & Footings', 'Failure Guide', 'Sustainability', 'Export'])
with TABS[0]:
    left, right = st.columns([3, 2], gap='large')
    with right:
        h1, h2 = st.columns([1, 1])
        h1.button('🗑 Clear all inputs', on_click=clear_all, use_container_width=True)
        h2.button('Fill example (optional)', on_click=load_example, use_container_width=True, help='Only for demonstration – fills every box with the project design values.')
        st.markdown('**INPUT TABLE** – nothing is pre-filled. Fill each step; the plot on the left updates immediately.')

        # ---- Step 1: grid
        with st.container(border=True):
            step('Step 1 · Building grid')
            a, b = st.columns(2); nx = a.number_input('Bays in X-direction', 1, 6, value=None, step=1, key='nx', placeholder='enter'); ny = b.number_input('Bays in Y-direction', 1, 6, value=None, step=1, key='ny', placeholder='enter')
            a, b = st.columns(2); nst = a.number_input('Number of storeys (G+1 → 2)', 1, 4, value=None, step=1, key='nst', placeholder='enter'); L = b.number_input('Bay span, centre to centre (ft)', 10.0, 25.0, value=None, step=0.5, format='%.1f', key='L', placeholder='enter')
            H = st.number_input('Storey height (ft)', 8.0, 14.0, value=None, step=0.5, format='%.1f', key='H', placeholder='enter')
        grid_ok = filled(nx, ny, nst, L, H)
        if grid_ok:
            nc, nb_, ns_ = (nx+1)*(ny+1), nx*(ny+1)+ny*(nx+1), nx*ny
            REQ = dict(c=nc*nst, b=nb_*nst, s=ns_*nst, f=nc)
            st.markdown(f'<div class="banner info">Footprint {nx*L:.0f} × {ny*L:.0f} ft = {nx*L*ny*L/9:.0f} yd² &nbsp;·&nbsp; this grid needs: <b>{REQ["c"]}</b> columns, <b>{REQ["b"]}</b> beams, <b>{REQ["s"]}</b> slab panels, <b>{REQ["f"]}</b> footings</div>', unsafe_allow_html=True)

        # ---- Step 2..4: materials, loads, seismic
        with st.container(border=True):
            step('Step 2 · Materials & foundation')
            a, b = st.columns(2); fc = a.number_input('Concrete f′c (ksi)', 2.5, 6.0, value=None, step=0.5, format='%.1f', key='fc', placeholder='enter')
            fy = b.selectbox('Steel fy (ksi)', [40.0, 60.0, 75.0], index=None, placeholder='select', key='fy')
            a, b = st.columns(2); stir = a.selectbox('Stirrup / tie size', ['#3', '#4'], index=None, placeholder='select', key='stir'); qa = b.number_input('Allowable soil bearing qa (ksf)', 1.0, 8.0, value=None, step=0.5, format='%.1f', key='qa', placeholder='enter')
            a, b = st.columns(2); cem = a.number_input('Cement content (kg/m³)', 250.0, 450.0, value=None, step=10.0, format='%.0f', key='cem', placeholder='enter'); fa = b.number_input('Fly-ash replacement (%)', 0.0, 40.0, value=None, step=5.0, format='%.0f', key='fa', placeholder='enter')
        with st.container(border=True):
            step('Step 3 · Loads (area loads in psf, walls in kip/ft)')
            a, b = st.columns(2); sdlf = a.number_input('Floor finishes + partitions (psf)', 0.0, 100.0, value=None, step=5.0, format='%.0f', key='sdlf', placeholder='enter'); sdlr = b.number_input('Roof finish / waterproofing (psf)', 0.0, 100.0, value=None, step=5.0, format='%.0f', key='sdlr', placeholder='enter')
            a, b = st.columns(2); llf = a.number_input('Floor live load (psf)', 20.0, 100.0, value=None, step=5.0, format='%.0f', key='llf', placeholder='enter'); llr = b.number_input('Roof live load (psf)', 10.0, 60.0, value=None, step=5.0, format='%.0f', key='llr', placeholder='enter')
            a, b = st.columns(2); wall = a.number_input('Perimeter wall (kip/ft)', 0.0, 1.5, value=None, step=0.05, format='%.2f', key='wall', placeholder='enter'); par = b.number_input('Parapet (kip/ft)', 0.0, 0.5, value=None, step=0.05, format='%.2f', key='par', placeholder='enter')
        with st.container(border=True):
            step('Step 4 · Seismic (BCP-SP 2007 / UBC-97)')
            a, b = st.columns(2); zone = a.selectbox('Seismic zone', list(E.ZONES), index=None, placeholder='select', key='zone', help='Confirm Sukkur zone from the code map.'); soil = b.selectbox('Soil profile', ['Sa', 'Sb', 'Sc', 'Sd', 'Se'], index=None, placeholder='select', key='soil')
            a, b, c = st.columns(3); Rf = a.number_input('R factor', 2.0, 8.5, value=None, step=0.5, format='%.1f', key='Rf', placeholder='enter'); Imp = b.number_input('Importance I', 1.0, 1.5, value=None, step=0.25, format='%.2f', key='Imp', placeholder='enter'); base = c.selectbox('Column base', ['fixed', 'pinned'], index=None, placeholder='select', key='base')

        # ---- Step 5..8: components (count + dimensions)
        def locked():
            st.markdown('<div class="need">Complete Step 1 (grid) first – the required number of this component depends on the grid.</div>', unsafe_allow_html=True)
        with st.container(border=True):
            step('Step 5 · Component: COLUMNS')
            if not grid_ok: locked()
            else:
                need(f'Required: {REQ["c"]} columns ({nc} per storey × {nst}); height must equal the storey height {H:.1f} ft.')
                c_n = st.number_input('Number of columns on the plot', 1, 500, value=None, step=1, key='c_n', placeholder='enter')
                a, b, c = st.columns(3); c_h = a.number_input('Height (ft)', 8.0, 14.0, value=None, step=0.5, format='%.1f', key='c_h', placeholder='enter'); c_w = b.number_input('Width (in)', 9.0, 24.0, value=None, step=1.0, format='%.0f', key='c_w', placeholder='enter'); c_d = c.number_input('Depth (in)', 9.0, 24.0, value=None, step=1.0, format='%.0f', key='c_d', placeholder='enter')
                a, b = st.columns(2); c_nb = a.selectbox('Main bars (number)', [4, 6, 8, 10, 12], index=None, placeholder='select', key='c_nb'); c_bs = b.selectbox('Main bar size', ['#4', '#5', '#6', '#7', '#8'], index=None, placeholder='select', key='c_bs')
        with st.container(border=True):
            step('Step 6 · Component: BEAMS')
            if not grid_ok: locked()
            else:
                need(f'Required: {REQ["b"]} beams ({nb_} per floor × {nst}); length must equal the bay span {L:.1f} ft; depth must exceed slab thickness.')
                b_n = st.number_input('Number of beams on the plot', 1, 500, value=None, step=1, key='b_n', placeholder='enter')
                a, b, c = st.columns(3); b_len = a.number_input('Length (ft)', 5.0, 30.0, value=None, step=0.5, format='%.1f', key='b_len', placeholder='enter'); b_w = b.number_input('Width (in)', 6.0, 18.0, value=None, step=1.0, format='%.0f', key='b_w', placeholder='enter'); b_d = c.number_input('Depth (in)', 12.0, 30.0, value=None, step=1.0, format='%.0f', key='b_d', placeholder='enter')
        with st.container(border=True):
            step('Step 7 · Component: SLAB PANELS')
            if not grid_ok: locked()
            else:
                need(f'Required: {REQ["s"]} slab panels ({ns_} per floor × {nst}); each panel = bay span {L:.1f} × {L:.1f} ft.')
                s_n = st.number_input('Number of slab panels on the plot', 1, 500, value=None, step=1, key='s_n', placeholder='enter')
                a, b, c = st.columns(3); s_len = a.number_input('Length (ft)', 5.0, 30.0, value=None, step=0.5, format='%.1f', key='s_len', placeholder='enter'); s_wid = b.number_input('Width (ft)', 5.0, 30.0, value=None, step=0.5, format='%.1f', key='s_wid', placeholder='enter'); s_t = c.number_input('Thickness (in)', 3.5, 9.0, value=None, step=0.5, format='%.1f', key='s_t', placeholder='enter')
        with st.container(border=True):
            step('Step 8 · Component: FOOTINGS')
            if not grid_ok: locked()
            else:
                need(f'Required: {REQ["f"]} isolated footings (one under every ground-storey column). Size is checked against bearing and punching shear.')
                f_n = st.number_input('Number of footings on the plot', 1, 500, value=None, step=1, key='f_n', placeholder='enter')
                a, b, c = st.columns(3); f_len = a.number_input('Length (ft)', 2.0, 15.0, value=None, step=0.5, format='%.1f', key='f_len', placeholder='enter'); f_wid = b.number_input('Width (ft)', 2.0, 15.0, value=None, step=0.5, format='%.1f', key='f_wid', placeholder='enter'); f_t = c.number_input('Thickness (in)', 8.0, 48.0, value=None, step=1.0, format='%.0f', key='f_t', placeholder='enter')
    if not grid_ok: c_n = c_w = c_d = c_h = c_nb = c_bs = b_n = b_len = b_w = b_d = s_n = s_len = s_wid = s_t = f_n = f_len = f_wid = f_t = None

    # ---------------------------------------------------------------- requirement validation (decides whether the checklist may appear)
    items = []   # (state, text)  state: ok / no / todo
    def add(ok, done, good, bad, todo):
        items.append(('ok', good) if (done and ok) else (('no', bad) if done else ('todo', todo)))
    add(True, grid_ok, 'Grid complete', '', 'Step 1: enter bays, storeys, span and storey height')
    add(True, filled(fc, fy, stir, qa, cem, fa), 'Materials complete', '', 'Step 2: materials / foundation values missing')
    add(True, filled(sdlf, sdlr, llf, llr, wall, par), 'Loads complete', '', 'Step 3: load values missing')
    add(True, filled(zone, soil, Rf, Imp, base), 'Seismic data complete', '', 'Step 4: seismic data missing')
    if grid_ok:
        close = lambda a, b: abs(a-b) < 0.01
        cd = filled(c_n, c_h, c_w, c_d, c_nb, c_bs)
        add(c_n == REQ['c'] and close(c_h or 0, H), cd, f'Columns: {c_n} of {REQ["c"]}, height {c_h} ft', 'Columns: ' + '; '.join(m for m in ([f'{c_n} entered, grid needs {REQ["c"]}'] if c_n != REQ['c'] else []) + ([f'height {c_h} ft must equal storey height {H:.1f} ft'] if not close(c_h or 0, H) else [])) if cd else '', 'Step 5: columns – count and dimensions missing')
        bd = filled(b_n, b_len, b_w, b_d)
        add(b_n == REQ['b'] and close(b_len or 0, L) and (s_t is None or (b_d or 0) > s_t), bd, f'Beams: {b_n} of {REQ["b"]}, length {b_len} ft', 'Beams: ' + '; '.join(m for m in ([f'{b_n} entered, grid needs {REQ["b"]}'] if b_n != REQ['b'] else []) + ([f'length {b_len} ft must equal span {L:.1f} ft'] if not close(b_len or 0, L) else []) + (['depth must exceed slab thickness'] if (s_t is not None and (b_d or 0) <= s_t) else [])) if bd else '', 'Step 6: beams – count and dimensions missing')
        sd = filled(s_n, s_len, s_wid, s_t)
        add(s_n == REQ['s'] and close(s_len or 0, L) and close(s_wid or 0, L), sd, f'Slab panels: {s_n} of {REQ["s"]}, {s_len}×{s_wid} ft', f'Slabs: {s_n} entered, grid needs {REQ["s"]}; panel {s_len}×{s_wid} ft must equal {L:.1f}×{L:.1f} ft' if sd else '', 'Step 7: slab panels – count and dimensions missing')
        fd = filled(f_n, f_len, f_wid, f_t)
        add(f_n == REQ['f'], fd, f'Footings: {f_n} of {REQ["f"]}', f'Footings: {f_n} entered, grid needs {REQ["f"]}' if fd else '', 'Step 8: footings – count and dimensions missing')
    ready = len(items) == 8 and all(s_ == 'ok' for s_, _ in items)
    supported = grid_ok and nx == 2 and ny == 2 and nst == 2

    # ---------------------------------------------------------------- model plot (left)
    def box(x0, y0, z0, dx, dy, dz):
        x1, y1, z1 = x0+dx, y0+dy, z0+dz; v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        return [[v[i] for i in f] for f in [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4)]]

    def layout():
        """Positions of every REQUIRED slot, in placement order; user counts fill the first n slots."""
        xs = [i*L for i in range(nx+1)]; ys = [j*L for j in range(ny+1)]
        cols = [(xs[i], ys[j], k) for k in range(nst) for i in range(nx+1) for j in range(ny+1)]
        beams = [('x', xs[i]+L/2, ys[j], k+1) for k in range(nst) for j in range(ny+1) for i in range(nx)] + [('y', xs[i], ys[j]+L/2, k+1) for k in range(nst) for i in range(nx+1) for j in range(ny)]
        beams.sort(key=lambda t: t[3])
        slabs = [(xs[i]+L/2, ys[j]+L/2, k+1) for k in range(nst) for i in range(nx) for j in range(ny)]
        foots = [(xs[i], ys[j]) for i in range(nx+1) for j in range(ny+1)]
        return xs, ys, cols, beams, slabs, foots

    def model_figs():
        xs, ys, cols, beams, slabs, foots = layout()
        nC = min(c_n or 0, len(cols)); nB = min(b_n or 0, len(beams)); nS = min(s_n or 0, len(slabs)); nF = min(f_n or 0, len(foots))
        ch = c_h if c_h else H; cw = (c_w or 12)/12; cdp = (c_d or 12)/12
        fig = plt.figure(figsize=(7.6, 5.6)); ax = fig.add_subplot(111, projection='3d'); P3 = {'col': [], 'beam': [], 'slab': [], 'foot': []}
        if c_w and c_d:
            for (x, y, k) in cols[:nC]: P3['col'] += box(x-cw/2, y-cdp/2, k*H, cw, cdp, ch)
        if b_w and b_d and b_len:
            bw, bd_ = b_w/12, b_d/12
            for (d, x, y, k) in beams[:nB]:
                P3['beam'] += box(x-b_len/2, y-bw/2, k*H-bd_, b_len, bw, bd_) if d == 'x' else box(x-bw/2, y-b_len/2, k*H-bd_, bw, b_len, bd_)
        if s_len and s_wid and s_t:
            for (x, y, k) in slabs[:nS]: P3['slab'] += box(x-s_len/2, y-s_wid/2, k*H-s_t/12, s_len, s_wid, s_t/12)
        if f_len and f_wid and f_t:
            for (x, y) in foots[:nF]: P3['foot'] += box(x-f_len/2, y-f_wid/2, -f_t/12, f_len, f_wid, f_t/12)
        for key, colr, al in (('foot', FOOTC, .9), ('col', NAVY, .95), ('beam', ORANGE, .9), ('slab', SLABC, .35)):
            if P3[key]: ax.add_collection3d(Poly3DCollection(P3[key], facecolor=colr, edgecolor='k', linewidths=.3, alpha=al))
        ax.set_xlim(-2, nx*L+2); ax.set_ylim(-2, ny*L+2); ax.set_zlim(-3, nst*H+1); ax.set_box_aspect((nx*L+4, ny*L+4, nst*H+4)); ax.view_init(22, -58)
        ax.set_xlabel('X [ft]'); ax.set_ylabel('Y [ft]'); ax.set_zlabel('Z [ft]'); ax.set_title('3-D model (to scale) – placed components only', fontsize=10)
        ax.legend(handles=[Patch(fc=NAVY, label=f'Columns {nC}/{len(cols)}'), Patch(fc=ORANGE, label=f'Beams {nB}/{len(beams)}'), Patch(fc=SLABC, alpha=.5, label=f'Slabs {nS}/{len(slabs)}'), Patch(fc=FOOTC, label=f'Footings {nF}/{len(foots)}')], loc='upper left', fontsize=8)
        # plan view (first floor)
        fig2, ax2 = plt.subplots(figsize=(7.6, 4.3)); pc, pb, ps = len(xs)*len(ys), len(beams)//nst, len(slabs)//nst
        for j in range(ny+1): ax2.axhline(ys[j], color='#bbb', lw=.6, ls='-.')
        for i in range(nx+1): ax2.axvline(xs[i], color='#bbb', lw=.6, ls='-.')
        if f_len and f_wid:
            for (x, y) in foots[:nF]: ax2.add_patch(Rectangle((x-f_len/2, y-f_wid/2), f_len, f_wid, fc='none', ec=FOOTC, lw=1.2, ls='--'))
        if s_len and s_wid:
            for (x, y, k) in slabs[:min(nS, ps)]: ax2.add_patch(Rectangle((x-s_len/2, y-s_wid/2), s_len, s_wid, fc=SLABC, alpha=.25, ec='none'))
        if b_len and b_w:
            for (d, x, y, k) in beams[:min(nB, pb)]:
                xx, yy = ([x-b_len/2, x+b_len/2], [y, y]) if d == 'x' else ([x, x], [y-b_len/2, y+b_len/2])
                ax2.plot(xx, yy, color=ORANGE, lw=3.2)
        if c_w and c_d:
            for n_, (x, y, k) in enumerate(cols[:min(nC, pc)]): ax2.add_patch(Rectangle((x-cw/2, y-cdp/2), cw, cdp, fc=NAVY, ec='k', zorder=5)); ax2.text(x+.6, y+.6, f'C{n_+1}', fontsize=7, zorder=6)
        ax2.set_xlim(-L*.25, nx*L+L*.25); ax2.set_ylim(-L*.25, ny*L+L*.25); ax2.set_aspect('equal'); ax2.set_xlabel('X [ft]'); ax2.set_ylabel('Y [ft]'); ax2.set_title('Plan view – typical floor (footings dashed)', fontsize=10)
        return fig, fig2

    with left:
        st.markdown('**MODEL PLOT** – updates as you type in the input table')
        if not grid_ok:
            st.markdown('<div class="canvas"><b>Empty canvas</b><br>Enter the building grid (Step 1) in the input table on the right.<br>Then add columns, beams, slabs and footings – each appears here with its number and dimensions.</div>', unsafe_allow_html=True)
        else:
            f3, f2 = model_figs(); st.pyplot(f3); st.pyplot(f2)
            def _png(fg):
                b = io.BytesIO(); fg.savefig(b, format='png', dpi=200, bbox_inches='tight', facecolor='white'); return b.getvalue()
            d1, d2 = st.columns(2)
            d1.download_button('⬇ Download 3-D model (PNG)', _png(f3), 'model_3D.png', 'image/png', use_container_width=True)
            d2.download_button('⬇ Download plan view (PNG)', _png(f2), 'model_plan.png', 'image/png', use_container_width=True)
            plt.close('all')
        st.markdown('**INPUT GUIDE** (what is still needed)')
        st.markdown('<div class="guide">' + '<br>'.join({'ok': '<span class="ok">✔ ', 'no': '<span class="no">✘ ', 'todo': '<span class="todo">○ '}[s_]+t+'</span>' for s_, t in items) + '</div>', unsafe_allow_html=True)

    P = None
    if ready:
        P = dict(fc=float(fc), fy=float(fy), L_ft=float(L), H_ft=float(H), t_slab=float(s_t), beam_b=float(b_w), beam_h=float(b_d), col_h=float(c_d), col_b=float(c_w), col_nbar=int(c_nb), col_bar=c_bs, stirrup=stir,
                 SDL_floor=float(sdlf), SDL_roof=float(sdlr), LL_floor=float(llf), LL_roof=float(llr), wall=float(wall), parapet=float(par), zone=zone, soil=soil, R=float(Rf), I=float(Imp), base=base,
                 qa=float(qa), cement=float(cem), fly_ash=float(fa))
        U = dict(f_len=float(f_len), f_wid=float(f_wid), f_t=float(f_t))

@st.cache_resource(show_spinner='Analysing frame …')
def run(P): return E.analyze(P)

# ---------------------------------------------------------------- results (only after all inputs are complete AND valid, and the user presses Run)

with TABS[0]:
    if not ready:
        st.markdown('<div class="banner warn">🔒 Design checklist and results are locked – they appear only after every input is entered and the plot satisfies all requirements listed in the Input Guide.</div>', unsafe_allow_html=True)
    elif not supported:
        st.markdown(f'<div class="banner warn">The model is complete and consistent ({nx}×{ny} bays, {nst} storeys), but the analysis engine is built for the G+1 project frame: <b>2 × 2 bays, 2 storeys</b>. Change the grid to 2 × 2 × 2 to run the analysis.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="banner pass">✔ All inputs entered and the model satisfies the grid requirements – results in the tabs above (Summary, Frame Diagrams, Beam …) update automatically when you change any input.</div>', unsafe_allow_html=True)

def render_results(P, R, U):
    E.configure(P); FIN, B, S, SEI, J = R['FIN'], R['beam'], R['slab'], R['SEI'], R['joint']; f = FIN['f']; out = FIN['out']
    def fig_png(fig):
        b = io.BytesIO(); fig.savefig(b, format='png', dpi=200, bbox_inches='tight', facecolor='white'); return b.getvalue()
    def show(fig, name):
        st.pyplot(fig); st.download_button('⬇ PNG', fig_png(fig), f'{name}.png', 'image/png', key=f'dl_{name}_{id(fig)}')

    # ---------------------------------------------------------------- helpers: diagrams
    def combo_all(cname): return {e['name']: E.combo_profile(out, e['name'], E.COMBOS[cname], 61) for e in f.elems}
    def draw_diag(ax, key, cname, scale, color, fmt, div=1.0, sg=1):
        Pc = combo_all(cname)
        for e in f.elems:
            xi, yi = np.array(f.nodes[e['i']])/12; xj, yj = np.array(f.nodes[e['j']])/12; ax.plot([xi, xj], [yi, yj], color='k', lw=2.2, zorder=3)
            p = Pc[e['name']]; Le = math.hypot(xj-xi, yj-yi); c, s = (xj-xi)/Le, (yj-yi)/Le; t = p['x']/p['x'][-1]; X = xi+(xj-xi)*t; Y = yi+(yj-yi)*t
            v = sg*p[key]/div; Xo = X-s*v*scale; Yo = Y+c*v*scale
            ax.fill(np.r_[X, Xo[::-1]], np.r_[Y, Yo[::-1]], color=color, alpha=.35, zorder=2); ax.plot(Xo, Yo, color=color, lw=1.2)
            idx = {0, len(t)-1}
            if e['kind'] == 'beam': idx.add(int(np.argmax(abs(v))))
            for i in idx:
                if abs(v[i]) > 0.5: ax.text(Xo[i]+(0.3 if e['kind'] == 'col' else 0), Yo[i]+(0.4 if e['kind'] == 'beam' else 0), fmt % v[i], fontsize=7, ha='center', zorder=6)
        ax.set_aspect('equal'); ax.axis('off')
    def deformed_fig():
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.4)); f_g, out_g, _ = E.run_cases(FIN['col'], Ifac_b=1.0, Ifac_c=1.0, SEI=SEI)
        for ax, (nm, case, scl) in zip(axs, (('Service D+L', None, 60), ('Earthquake E', 'E', 12))):
            for e in f.elems:
                xi, yi = np.array(f.nodes[e['i']])/12; xj, yj = np.array(f.nodes[e['j']])/12; ax.plot([xi, xj], [yi, yj], color='#b0bec5', lw=2, ls='--')
                res = [out_g['D'], out_g['L']] if case is None else [out['E']]
                v = sum(E.profile(r, e['name'], 41)['v'] for r in res); t = np.linspace(0, 1, 41)
                ua = sum(r['el'][e['name']]['ul'][0]+(r['el'][e['name']]['ul'][3]-r['el'][e['name']]['ul'][0])*t for r in res); d0 = res[0]['el'][e['name']]; c, s = d0['c'], d0['s']
                X = xi+(xj-xi)*t+(c*ua-s*v)/12*scl; Y = yi+(yj-yi)*t+(s*ua+c*v)/12*scl; ax.plot(X, Y, color='#c62828' if e['kind'] == 'beam' else '#1f3b73', lw=2.4)
            ax.set_title(f'{nm} (displacements ×{scl})', fontsize=9); ax.set_aspect('equal'); ax.axis('off')
        return fig

    # ---------------------------------------------------------------- status checks
    def chk(ok): return 'PASS' if ok else 'FAIL'
    col = FIN['col']; dm1 = 0.7*P['R']*FIN['drift1']; dm2 = 0.7*P['R']*FIN['drift2']
    rows = []
    if 'top' in B:
        rows += [['Beam flexure (support)', f"Mu = {B['Mneg']:.1f} kip-ft", f"φMn = {B['phiMn_neg']:.1f} kip-ft ({B['top']})", chk(B['phiMn_neg'] >= B['Mneg'])],
                 ['Beam flexure (mid-span)', f"Mu = {B['Mpos']:.1f} kip-ft", f"φMn = {B['phiMn_pos']:.1f} kip-ft ({B['bot']})", chk(B['phiMn_pos'] >= B['Mpos'])],
                 ['Beam shear (end zones)', f"Ve = {B['Ve']:.1f} kip", f"φVn = {B['phiVn_end']:.1f} kip (#3 @ {B['s_end']} in)", chk(B['phiVn_end'] >= B['Ve'] and B['phiVn_end'] >= B['Vu'])],
                 ['Beam deflection', f"λΔD+ΔL = {B['defl']['lt']:.3f} in", f"ℓ/480 = {B['defl']['lim480']:.3f} in; ΔL ≤ ℓ/360 = {B['defl']['lim360']:.3f} in", chk(B['defl']['ok'])]]
    else: rows.append(['Beam design', 'section too small', '; '.join(B['msgs']), 'FAIL'])
    rows += [['Columns: P-M interaction (all combos)', f"max DCR = {FIN['maxDCR']:.2f}", 'DCR ≤ 1.0', chk(FIN['maxDCR'] <= 1.0)],
             ['Frame stability index Q', f"Q = {FIN['Q']:.3f}", 'Q ≤ 0.05 (non-sway)', chk(FIN['Q'] <= 0.05) if FIN['Q'] <= 0.05 else 'WARN – magnified δs = %.2f' % FIN['ds']],
             ['Storey drift (Δm = 0.7·R·Δs)', f"{max(dm1, dm2):.2f} in", f"0.025h = {0.025*E.G['H']:.2f} in", chk(max(dm1, dm2) <= 0.025*E.G['H'])],
             ['Column slenderness / buckling', f"Pu/Pc = {FIN['PuB']/FIN['Pc']:.2f}", f"Pc = {FIN['Pc']:.0f} kip", chk(FIN['PuB']/FIN['Pc'] < 0.5)],
             ['Slab (flexure, shear, min. thickness)', f"Mu⁻ = {S['Mneg']:.2f} kip-ft/ft", f"φMn = {S['phiMn']:.2f}; t ≥ {S['tmin']:.1f} in", chk(S['ok'])]]
    if J: rows += [['Strong column / weak beam', f"ΣMnc/ΣMnb = {J['scwb']:.2f}", '≥ 1.2 recommended', chk(J['scwb'] >= 1.2)], ['Beam-column joint shear', f"Vj = {J['Vj']:.0f} kip", f"φVn = {J['phiVj']:.0f} kip", chk(J['Vj'] <= J['phiVj'])]]
    fr = R['footing']; Breq = max(float(s.split('x')[0]) for s in fr['Footing [ft]']); treq = float(fr['Thickness [in]'].max())
    rows.append(['Footing size provided (all footings)', f"{U['f_len']:.1f} × {U['f_wid']:.1f} ft, t = {U['f_t']:.0f} in", f"required ≥ {Breq:.1f} × {Breq:.1f} ft, t ≥ {treq:.0f} in (bearing + punching)", chk(min(U['f_len'], U['f_wid']) >= Breq and U['f_t'] >= treq)])
    status = pd.DataFrame(rows, columns=['Check', 'Demand', 'Capacity / limit', 'Status'])
    failed = status[status['Status'] == 'FAIL']['Check'].tolist(); all_ok = len(failed) == 0

    # ---------------------------------------------------------------- header
    def kpi(l, v, sub='', cls=''): return f'<div class="kpi {cls}"><div class="l">{l}</div><div class="v">{v}</div><div class="s">{sub}</div></div>'
    st.markdown('<div class="kpis">'+kpi('Base shear V', f"{SEI['V']:.1f} kip", f"Cs = {SEI['Cs']:.3f}")+kpi('Seismic weight W', f"{SEI['W']:.0f} kip", f"T = {SEI['T']:.2f} s")
                +kpi('Max column DCR', f"{FIN['maxDCR']:.2f}", 'limit 1.00', 'pass' if FIN['maxDCR'] <= 1 else 'fail')+kpi('Max column load', f"{FIN['PuB']:.0f} kip", 'factored Pu')
                +kpi('Design status', 'PASS' if all_ok else 'FAIL', f"{len(status)-len(failed)} of {len(status)} checks OK", 'pass' if all_ok else 'fail')+'</div>', unsafe_allow_html=True)

    tabs = TABS[1:]

    with tabs[0]:
        if all_ok: st.markdown(f'<div class="banner pass">ALL {len(status)} CHECKS PASS – this section combination satisfies strength, stability and serviceability for the given inputs.</div>', unsafe_allow_html=True)
        else: st.markdown('<div class="banner fail">' + str(len(failed)) + ' CHECK(S) FAIL: ' + '; '.join(failed) + ' – change the member sizes in the input table.</div>', unsafe_allow_html=True)
        st.subheader('Design checks – demand vs capacity'); tbl(status)
        if 'top' in B:
            st.subheader('Proposed reinforcement (preliminary)')
            tbl(pd.DataFrame([
                ['Slab', f"{P['t_slab']} in", f"#3 @ {S['s']} in both ways, top over beams and bottom"],
                ['Beam', f"{P['beam_b']:.0f}×{P['beam_h']:.0f} in", f"support top {B['top']}, bottom {B['bot']}, mid top 2 #5; stirrups {P['stirrup']} @ {B['s_end']} in (36 in at ends) / {B['s_mid']} in elsewhere"],
                ['Column', f"{col[0]:.0f}×{col[1]:.0f} in", f"{P['col_nbar']} {P['col_bar']} (ρ = {FIN['rho']*100:.2f} %), ties {P['stirrup']} @ {min(5.0, 8*E.DIA[P['col_bar']]):.1f} in near joints / {min(16*E.DIA[P['col_bar']], 48*E.DIA[P['stirrup']], min(col)):.0f} in elsewhere"],
                ['Footings (provided)', f"{U['f_len']:.1f}×{U['f_wid']:.1f} ft, t = {U['f_t']:.0f} in; required " + ' / '.join(R['footing']['Footing [ft]']), '; '.join(R['footing']['Bars each way (#5)'])]], columns=['Member', 'Size', 'Reinforcement']))
        for m in B.get('msgs', []): st.warning(m)
        if FIN['maxDCR'] > 1: st.error('Column P-M demand exceeds capacity – enlarge the column or add steel (see Columns tab).')

    with tabs[1]:
        a, b = st.columns(2)
        with a:
            fig, ax = plt.subplots(figsize=(6, 5)); L_ft, H_ft = P['L_ft'], P['H_ft']
            for e in f.elems:
                xi, yi = np.array(f.nodes[e['i']])/12; xj, yj = np.array(f.nodes[e['j']])/12; ax.plot([xi, xj], [yi, yj], color='#1f3b73' if e['kind'] == 'col' else '#b5542b', lw=4)
            for cn in 'ABC':
                x = {'A': 0, 'B': L_ft, 'C': 2*L_ft}[cn]; ax.add_patch(plt.Polygon([(x-1, -1.2), (x+1, -1.2), (x, 0)], fc='#9e9e9e', ec='k'))
                for lv in range(3): ax.text(x+.5, lv*H_ft+.5, f'{cn}{lv}', fontsize=7)
            for lv, w in ((1, 2*E.G['wD_floor']*L_ft), (2, 2*E.G['wD_roof']*L_ft)):
                for a0 in (0, L_ft): ax.add_patch(plt.Polygon([(a0, lv*H_ft+.3), (a0+L_ft, lv*H_ft+.3), (a0+L_ft/2, lv*H_ft+.3+w/4)], fc='#6a1b9a', alpha=.3)); ax.text(a0+L_ft/2, lv*H_ft+w/4+.8, f'D peak {w:.2f} k/ft', ha='center', fontsize=7)
            for lv, F in ((1, SEI['F1']/9), (2, SEI['F2']/9)):
                for x in (0, L_ft, 2*L_ft): ax.annotate('', (x, lv*H_ft), (x-3, lv*H_ft), arrowprops=dict(arrowstyle='->', color='#00838f', lw=2))
                ax.text(-3.4, lv*H_ft, f'{F:.2f} k/joint', ha='right', va='center', fontsize=7, color='#00838f')
            ax.set_aspect('equal'); ax.axis('off'); ax.set_title('Analysed frame (grid 2): supports, gravity (D) and earthquake joint loads', fontsize=9); show(fig, 'model')
        with b:
            st.markdown('**Area loads**'); G_ = E.G
            tbl(pd.DataFrame([['Slab self-weight', f"{G_['slab_sw']*1000:.1f}", f"{G_['slab_sw']*1000:.1f}"], ['Finishes / partitions / roof finish', f"{P['SDL_floor']:.0f}", f"{P['SDL_roof']:.0f}"],
                                   ['Dead D (psf)', f"{G_['wD_floor']*1000:.1f}", f"{G_['wD_roof']*1000:.1f}"], ['Live L (psf)', f"{P['LL_floor']:.0f}", f"{P['LL_roof']:.0f}"]], columns=['Load', 'Floor', 'Roof']))
            st.markdown(f"Beam stem self-weight: **{G_['beam_sw']:.3f} kip/ft**. Slab → beams as triangular loads (peak = w·S/2); out-of-plane beam reactions applied as joint loads; self-weight counted once.")
            st.markdown('**Seismic (static)**'); tbl(pd.DataFrame([['Seismic weight W1 / W2', f"{SEI['W1']:.0f} / {SEI['W2']:.0f} kip"], ['Period T = Ct·h^¾', f"{SEI['T']:.3f} s"], ['Zone / soil', f"Z = {G_['Z']}, {P['soil']}: Ca = {G_['Ca']}, Cv = {G_['Cv']}"],
                                                      ['V = Cv·I·W/(R·T) (limits applied)', f"{SEI['V']:.1f} kip (Cs = {SEI['Cs']:.3f})"], ['F1 (first floor) / F2 (roof)', f"{SEI['F1']:.1f} / {SEI['F2']:.1f} kip (frame takes 1/3)"]], columns=['Item', 'Value']))
            st.caption('Combinations: C1 = 1.2D+1.6L; C2 = 1.2D+1.0L±E; C3 = 0.9D±E. Stiffness: 0.35Ig beams, 0.70Ig columns.')

    with tabs[2]:
        cname = st.selectbox('Load combination', list(E.COMBOS), 0); kind = st.radio('Diagram', ['Bending moment (BMD)', 'Shear force (SFD)', 'Axial force'], horizontal=True)
        key, col_, fmt, div, sg, scl, unit = {'Bending moment (BMD)': ('M', '#c62828', '%.0f', 12.0, -1, 0.03, 'kip-ft'), 'Shear force (SFD)': ('V', '#1565c0', '%.1f', 1.0, 1, 0.1, 'kip'), 'Axial force': ('N', '#2e7d32', '%.0f', 1.0, 1, 0.02, 'kip')}[kind]
        st.caption('Colour code:  BMD = red (drawn on tension side)  |  SFD = blue  |  Axial = green  |  columns = navy, beams = orange (model view)')
        fig, ax = plt.subplots(figsize=(8.5, 5.2)); draw_diag(ax, key, cname, scl*P['L_ft']/15, col_, fmt, div, sg); ax.set_title(f'{kind} [{unit}] – {cname}' + ('  (tension side)' if key == 'M' else ''), fontsize=10); show(fig, f'{key}_{cname[:2]}')
        st.subheader('Deformed shape'); show(deformed_fig(), 'deformed')
        st.info(f"Storey drift Δs = {FIN['drift1']:.3f} / {FIN['drift2']:.3f} in → Δm = {dm1:.2f} / {dm2:.2f} in (limit 0.025h = {0.025*E.G['H']:.1f} in). Roof sway under E = {abs(out['E']['Unode'][('B', 2)][0]):.2f} in.")
        rr = pd.DataFrame([[n, *[out[k]['R'][(n, 0)][i] for k in ('D', 'L', 'E') for i in ((1,) if k != 'E' else (0, 2))]] for n in 'ABC'], columns=['Support', 'D: Ry [kip]', 'L: Ry [kip]', 'E: Rx [kip]', 'E: M [kip-in]'])
        st.markdown('**Support reactions (unfactored)**'); tbl(rr.round(2))

    with tabs[3]:
        nm = st.selectbox('Beam', ['B1BC', 'B1AB', 'B2BC', 'B2AB'], format_func=lambda x: {'B1BC': 'FB2-BC first floor (critical)', 'B1AB': 'FB2-AB first floor', 'B2BC': 'RB2-BC roof', 'B2AB': 'RB2-AB roof'}[x])
        ENV = B['ENV']; fig, axs = plt.subplots(2, 1, figsize=(8.6, 6.4), sharex=True)
        for cname, p in ENV[nm]['combos'].items():
            x = p['x']/12; lw = 2.4 if cname.startswith('C1') else 1.0; axs[0].plot(x, p['V'], lw=lw, label=cname); axs[1].plot(x, -p['M']/12, lw=lw, label=cname)
        if 'phiVc' in B: axs[0].axhline(B['phiVc'], color='gray', ls=':'); axs[0].axhline(-B['phiVc'], color='gray', ls=':'); axs[0].text(E.G['L_ft']/2, B['phiVc']+.6, f"φVc = {B['phiVc']:.1f} kip", ha='center', fontsize=8)
        if 'Mcr' in B:
            pS = E.combo_profile(out, nm, dict(D=1, L=1, E=0), 361); axs[1].fill_between(pS['x']/12, 0, -pS['M']/12, where=np.abs(pS['M']/12) > B['Mcr'], color='k', alpha=.08, label='M_service > Mcr (cracked)')
        for a_ in axs: a_.axhline(0, color='k', lw=.8)
        axs[0].set_ylabel('Shear V [kip]'); axs[1].set_ylabel('Moment [kip-ft]\n(↑ hogging, ↓ sagging – tension side)'); axs[1].set_xlabel('x along beam [ft]'); axs[0].legend(fontsize=6.5, ncol=2); axs[0].set_title(f'SFD and BMD – beam {nm}', fontsize=10); fig.tight_layout(); show(fig, f'beam_{nm}')
        p1 = ENV[nm]['combos']['C1: 1.2D+1.6L']
        st.markdown(f"**C1 values:** end shear {abs(p1['V'][0]):.1f} / {abs(p1['V'][-1]):.1f} kip; hogging {abs(p1['M'][0])/12:.1f} / {abs(p1['M'][-1])/12:.1f} kip-ft; sagging {p1['M'].max()/12:.1f} kip-ft.  "
                    '• Sagging → bottom tension (bottom bars, vertical cracks at soffit) • Hogging → top tension (top bars over columns) • High shear near supports → diagonal cracks → close stirrups.')
        if 'top' in B:
            d = B['defl']
            tbl(pd.DataFrame([['Flexure – support (face of column)', f"Mu = {B['Mneg']:.1f} kip-ft; As,req = {B['Asn']:.2f} in²", f"{B['top']} = {B['top_As']:.2f} in²; φMn = {B['phiMn_neg']:.1f} kip-ft; εt = {B['et_n']:.4f}"],
                                   ['Flexure – mid-span', f"Mu = {B['Mpos']:.1f} kip-ft; As,req = {B['Asp']:.2f} in²", f"{B['bot']} = {B['bot_As']:.2f} in²; φMn = {B['phiMn_pos']:.1f} kip-ft; εt = {B['et_p']:.4f}"],
                                   ['Cracking', f"Mcr = {B['Mcr']:.1f} kip-ft", 'section cracks under service load (normal for RC); control with bar spacing/cover'],
                                   ['Shear', f"Vu(d) = {B['Vu']:.1f} kip; φVc = {B['phiVc']:.1f}; Ve (IMF) = {B['Ve']:.1f} kip", f"#3 @ {B['s_end']} in (ends), @ {B['s_mid']} in; φVn = {B['phiVn_end']:.0f} / {B['phiVn_mid']:.0f} kip"],
                                   ['Deflection', f"ΔD = {d['dD']:.3f} in, ΔL = {d['dL']:.3f} in; λΔD+ΔL = {d['lt']:.3f} in", f"limits ℓ/360 = {d['lim360']:.2f}, ℓ/480 = {d['lim480']:.3f} in; Ie/Ig = {d['IeIg']:.2f}; λ = {d['lam']:.2f}"]], columns=['Check', 'Demand', 'Provided / limit']))
        for m in B['msgs']: st.warning(m)

    with tabs[4]:
        cv = FIN['curve']; df = FIN['df']; fig, ax = plt.subplots(figsize=(7, 4.8)); ax.plot(cv['phiM']/12, cv['phiP'], color='#1f3b73', lw=2, label='φPn–φMn'); ax.plot(cv['M']/12, cv['P'], color='gray', ls='--', label='nominal')
        ok = df.DCR <= 1; ax.scatter(df.Mu_kipft[ok], df.Pu_kip[ok], c='#2e7d32', s=22, label='demand (all combos, all columns)'); ax.scatter(df.Mu_kipft[~ok], df.Pu_kip[~ok], c='#c62828', marker='x', s=45, label='outside capacity')
        ax.set_xlabel('Mu [kip-ft]'); ax.set_ylabel('Pu [kip]'); ax.set_xlim(left=0); ax.set_ylim(bottom=0); ax.legend(fontsize=8); ax.set_title(f"P-M interaction – {col[0]:.0f}×{col[1]:.0f} in, {P['col_nbar']} {P['col_bar']} (max DCR {FIN['maxDCR']:.2f})", fontsize=10); show(fig, 'pm')
        a, b = st.columns(2)
        with a: st.markdown('**Most critical demands**'); tbl(df.sort_values('DCR', ascending=False).head(8).round(2))
        with b:
            st.markdown('**Slenderness & stability**'); tbl(pd.DataFrame([['k (sway, alignment-chart formula)', f"{FIN['kmax']:.2f}"], ['ℓu/r (k = 1)', f"{FIN['lu_r']:.1f}"], ['kℓu/r', f"{FIN['sl']:.1f}"], ['Stability index Q', f"{FIN['Q']:.3f}" + (' → non-sway' if FIN['Q'] <= 0.05 else f" → magnify δs = {FIN['ds']:.2f}")],
                      ['Elastic buckling Pc', f"{FIN['Pc']:.0f} kip (Pu/Pc = {FIN['PuB']/FIN['Pc']:.2f})"], ['Pinned-base sensitivity', f"DCR = {R['FINp']['maxDCR']:.2f}, Q = {R['FINp']['Q']:.2f}, drift = {R['FINp']['drift1']:.2f} in"]], columns=['Item', 'Value']))

    with tabs[5]:
        st.markdown(f"**Slab ({P['t_slab']} in, two-way, 1-ft strip):** wu = {S['wu']:.0f} psf; Mu⁻ = {S['Mneg']:.2f}, Mu⁺ = {S['Mpos']:.2f} kip-ft/ft; As,min = {S['Asmin']:.3f} in²/ft → **#3 @ {S['s']} in both ways**; φMn = {S['phiMn']:.2f} kip-ft/ft; t,min = {S['tmin']:.1f} in → {chk(S['ok'])}")
        st.markdown('**Punching shear if the slab were a flat plate (no beams)** – why beams are used')
        tbl(pd.DataFrame([[f"{t:.1f} in", f"{p['Vu']:.1f}", f"{p['vu']:.0f}", f"{p['phivc']:.0f}", f"{p['DCR']:.2f}", chk(p['DCR'] <= 1)] for t, p in R['punch'].items()], columns=['Slab t', 'Vu [kip]', 'vu [psi]', 'φvc [psi]', 'DCR', 'Status']))
        st.markdown('**Isolated footings (indicative)**'); tbl(R['footing'].round(2))

    with tabs[6]:
        st.subheader('Failure → likely location → check → remedy'); d = B.get('defl', {})
        fg = pd.DataFrame([
            ['Beam flexural cracking', 'Mid-span soffit; top beside columns', f"Mcr = {B.get('Mcr', 0):.1f} kip-ft ≪ service moment → cracks; strength OK" if 'top' in B else 'section inadequate', 'Correct tension steel, anchorage, bar spacing, curing'],
            ['Beam shear cracks', 'Within ~2d of column faces', f"Vu = {B.get('Vu', 0):.1f} kip vs φVc = {B.get('phiVc', 0):.1f} kip", f"Closer stirrups: #3 @ {B.get('s_end', '–')} in at ends"],
            ['Excessive deflection', 'Mid-span of first-floor beams', f"λΔD+ΔL = {d.get('lt', 0):.3f} in vs ℓ/480 = {d.get('lim480', 0):.3f} in", 'Deeper beam, compression steel, delay partitions'],
            ['Column crushing', 'Ground-storey interior column base', f"Pu = {FIN['PuB']:.0f} kip vs φPn,max = {cv['Pcap']:.0f} kip", 'Larger section, higher f′c, more steel'],
            ['Combined axial + bending', 'Edge columns top/bottom, joints (EQ)', f"max DCR = {FIN['maxDCR']:.2f}", 'Larger column, strong-column/weak-beam, joint ties'],
            ['Buckling / sway', 'Storey-1 columns', f"Q = {FIN['Q']:.3f}; Pu/Pc = {FIN['PuB']/FIN['Pc']:.2f}; drift {max(dm1, dm2):.2f} in", 'Stiffer frame, plinth beams, bracing/shear walls'],
            ['Slab flexure', 'Mid-panel and over beams', f"φMn = {S['phiMn']:.2f} vs Mu = {S['Mneg']:.2f} kip-ft/ft", f"#3 @ {S['s']} in, top steel over beams"],
            ['Slab punching', 'Around column (flat plate only)', f"5 in plate DCR = {R['punch'][P['t_slab']]['DCR']:.2f}", 'Keep beams; else thicker slab, drop panel, shear reinforcement']], columns=['Failure mode', 'Where', 'Evidence (demand vs capacity)', 'Remedy'])
        tbl(fg)

    with tabs[7]:
        s = R['sust']; a, b = st.columns(2)
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.4)); axs[0].bar(['Rule of thumb', 'This design'], [s['V_base'], s['V_opt']], color=['#9e9e9e', '#2e7d32']); axs[0].set_ylabel('Concrete [m³]')
        axs[1].bar(['Rule of thumb', 'Optimised', f"+{P['fly_ash']:.0f}% fly ash"], [s['co2_base'], s['co2_opt'], s['co2_fa']], color=['#9e9e9e', '#81c784', '#2e7d32']); axs[1].set_ylabel('CO₂ from cement [t]')
        for ax in axs:
            for p in ax.patches: ax.text(p.get_x()+p.get_width()/2, p.get_height(), f'{p.get_height():.1f}', ha='center', va='bottom', fontsize=8)
        fig.tight_layout(); show(fig, 'sustain')
        st.markdown(f"- **Efficient sizing:** concrete {s['V_base']:.1f} → {s['V_opt']:.1f} m³ ({(1-s['V_opt']/s['V_base'])*100:.0f} % less than 6 in slab / 12×24 beams / 18×18 columns); cement {s['cem_base']:.1f} → {s['cem_opt']:.1f} t.\n"
                    f"- **Fly ash {P['fly_ash']:.0f} %:** cement CO₂ {s['co2_opt']:.1f} → {s['co2_fa']:.1f} t (slower strength gain – specify 56-day strength, cure longer).\n"
                    f"- **Lightweight infill walls (0.40 kip/ft):** W = {s['W']:.0f} → {s['W_lw']:.0f} kip; base shear {s['V']:.1f} → {s['V_lw']:.1f} kip ({(1-s['V_lw']/s['V'])*100:.0f} % lower).\n- **Durability / maintenance:** adequate cover, low w/c, curing, roof waterproofing, accessible drainage.")
        st.caption(f"Assumptions: {P['cement']:.0f} kg cement/m³, {E.G['co2']} kg CO₂/kg cement, baseline = rule-of-thumb sections.")

    with tabs[8]:
        st.subheader('Download results')
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine='openpyxl') as xw:
            pd.DataFrame(list(P.items()), columns=['input', 'value']).to_excel(xw, sheet_name='Inputs', index=False); status.to_excel(xw, sheet_name='Checks', index=False); FIN['df'].round(3).to_excel(xw, sheet_name='Column demands', index=False)
            R['footing'].round(3).to_excel(xw, sheet_name='Footings', index=False); pd.DataFrame([{k: v for k, v in B.items() if not isinstance(v, (dict, list))}]).T.to_excel(xw, sheet_name='Beam design', header=False)
            pd.DataFrame([{k: v for k, v in SEI.items()}]).T.to_excel(xw, sheet_name='Seismic', header=False); fg.to_excel(xw, sheet_name='Failure guide', index=False)
            rows_ = []
            for cname in E.COMBOS:
                for e in f.elems:
                    p = E.combo_profile(out, e['name'], E.COMBOS[cname], 3); rows_.append([cname, e['name'], p['N'][0], p['V'][0], p['M'][0]/12, p['M'][-1]/12])
            pd.DataFrame(rows_, columns=['combo', 'member', 'N_end [kip]', 'V_start [kip]', 'M_start [kip-ft]', 'M_end [kip-ft]']).round(3).to_excel(xw, sheet_name='Member forces', index=False)
        st.download_button('⬇ Excel workbook (inputs, checks, demands, footings, forces)', bio.getvalue(), 'RC_frame_results.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        zb = io.BytesIO()
        with zipfile.ZipFile(zb, 'w') as z:
            for kname, (key, color, fmt, div, sg, scl) in {'BMD': ('M', '#c62828', '%.0f', 12.0, -1, 0.03), 'SFD': ('V', '#1565c0', '%.1f', 1.0, 1, 0.1), 'Axial': ('N', '#2e7d32', '%.0f', 1.0, 1, 0.02)}.items():
                for cname in E.COMBOS:
                    fig, ax = plt.subplots(figsize=(8, 5)); draw_diag(ax, key, cname, scl*P['L_ft']/15, color, fmt, div, sg); ax.set_title(f'{kname} – {cname}'); z.writestr(f'{kname}_{cname[:3].strip(":")}.png', fig_png(fig)); plt.close(fig)
        st.download_button('⬇ All BMD / SFD / axial diagrams (ZIP of PNG)', zb.getvalue(), 'frame_diagrams.zip', 'application/zip')
        st.caption('Model assumptions: 2-D interior frame, 1/3 lateral share, fixed/pinned bases, static seismic, simplified slab coefficients. Preliminary academic design only.')

if ready and supported:
    render_results(P, run(P), U)
else:
    for t_ in TABS[1:]:
        with t_: st.info('🔒 Locked – go to the **Build Model** tab, enter all inputs and make the model satisfy the Input Guide. These results (design checklist, diagrams, beam, columns, slab, footings, failure guide, sustainability, export) appear here automatically.')
