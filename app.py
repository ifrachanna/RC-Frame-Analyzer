import io, math, re, zipfile
import numpy as np, pandas as pd, streamlit as st
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import engine as E

st.set_page_config(page_title='RC Frame Analyzer', page_icon='🏗️', layout='wide')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.grid': True, 'grid.alpha': .3, 'axes.spines.top': False, 'axes.spines.right': False})
NAVY, ORANGE, GREEN, RED, AMBER = '#1F3B73', '#B5542B', '#2E7D32', '#C62828', '#ED8F00'
st.markdown(f"""<style>
html, body, [class*="css"], .stMarkdown, .stTabs, label {{font-family:'Segoe UI','Helvetica Neue',Arial,sans-serif !important}}
.block-container{{padding-top:3.2rem;padding-bottom:2rem;max-width:1400px}}
.hdr{{background:linear-gradient(90deg,{NAVY},#2c5aa0);border-left:8px solid {ORANGE};color:#fff;padding:14px 22px;border-radius:6px;margin-bottom:14px}}
.hdr-t{{font-size:1.9rem;font-weight:700;letter-spacing:.3px;line-height:1.15}} .hdr-s{{font-size:.9rem;opacity:.9;margin-top:3px}}
.kpis{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:12px}}
.kpi{{border:1px solid rgba(120,140,170,.35);border-top:4px solid {NAVY};border-radius:6px;padding:8px 12px;background:rgba(120,140,170,.07)}}
.kpi .l{{font-size:.72rem;text-transform:uppercase;letter-spacing:.6px;opacity:.75}} .kpi .v{{font-size:1.55rem;font-weight:700;line-height:1.2}} .kpi .s{{font-size:.75rem;opacity:.7}}
.kpi.pass{{border-top-color:{GREEN}}} .kpi.pass .v{{color:{GREEN}}} .kpi.fail{{border-top-color:{RED}}} .kpi.fail .v{{color:{RED}}}
.banner{{padding:10px 16px;border-radius:6px;font-weight:600;margin:6px 0 10px}} .banner.pass{{background:rgba(46,125,50,.15);border-left:6px solid {GREEN};color:{GREEN}}}
.banner.fail{{background:rgba(198,40,40,.13);border-left:6px solid {RED};color:{RED}}}
.sb-t{{font-size:1.25rem;font-weight:700;color:{NAVY};border-bottom:3px solid {ORANGE};padding-bottom:4px;margin-bottom:6px}}
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

# ---------------------------------------------------------------- sidebar inputs
D = E.DEFAULTS
PRESETS = {'Project design (15×15 column)': {}, 'Trial design (12×12 column) – fails': dict(col_h=12.0, col_b=12.0, col_nbar=4, col_bar='#6')}
st.sidebar.markdown('<div class="sb-t">INPUTS</div>', unsafe_allow_html=True)
pre = st.sidebar.selectbox('Preset', list(PRESETS))
base = dict(D); base.update(PRESETS[pre])
def num(label, key, mn, mx, step=None, fmt=None, help=None, ss=None):
    return st.number_input(label, mn, mx, float(base[key]), step or (mx-mn)/100, format=fmt or '%.2f', help=help, key=f'{key}_{pre}')
P = {}
with st.sidebar.expander('Geometry (G+1, 3×3 columns)', True):
    P['L_ft'] = num('Bay span (ft, both directions)', 'L_ft', 10.0, 25.0, 0.5, '%.1f'); P['H_ft'] = num('Storey height (ft)', 'H_ft', 8.0, 14.0, 0.5, '%.1f')
    P['t_slab'] = num('Slab thickness (in)', 't_slab', 3.5, 9.0, 0.5, '%.1f')
    st.caption(f"Footprint = {2*P['L_ft']:.0f} ft × {2*P['L_ft']:.0f} ft = {(2*P['L_ft'])**2/9:.0f} yd²")
with st.sidebar.expander('Member sizes', True):
    P['beam_b'] = num('Beam width b (in)', 'beam_b', 6.0, 18.0, 1.0, '%.0f'); P['beam_h'] = num('Beam depth h (in)', 'beam_h', 12.0, 30.0, 1.0, '%.0f')
    P['col_h'] = num('Column depth in frame plane (in)', 'col_h', 9.0, 24.0, 1.0, '%.0f'); P['col_b'] = num('Column width (in)', 'col_b', 9.0, 24.0, 1.0, '%.0f')
    P['col_nbar'] = st.selectbox('Column bars (number)', [4, 6, 8, 10, 12], [4, 6, 8, 10, 12].index(base['col_nbar']), key=f'nb_{pre}')
    P['col_bar'] = st.selectbox('Column bar size', ['#4', '#5', '#6', '#7', '#8'], ['#4', '#5', '#6', '#7', '#8'].index(base['col_bar']), key=f'cb_{pre}')
    P['stirrup'] = st.selectbox('Stirrup / tie size', ['#3', '#4'], 0)
with st.sidebar.expander('Materials'):
    P['fc'] = num('f′c (ksi)', 'fc', 2.5, 6.0, 0.5, '%.1f'); P['fy'] = st.selectbox('fy (ksi)', [40.0, 60.0, 75.0], 1)
with st.sidebar.expander('Loads (area loads in psf, walls in kip/ft)'):
    P['SDL_floor'] = num('Floor finishes + partitions (psf)', 'SDL_floor', 0.0, 100.0, 5.0, '%.0f'); P['SDL_roof'] = num('Roof finish/waterproofing (psf)', 'SDL_roof', 0.0, 100.0, 5.0, '%.0f')
    P['LL_floor'] = num('Floor live load (psf)', 'LL_floor', 20.0, 100.0, 5.0, '%.0f'); P['LL_roof'] = num('Roof live load (psf)', 'LL_roof', 10.0, 60.0, 5.0, '%.0f')
    P['wall'] = num('Perimeter wall (kip/ft)', 'wall', 0.0, 1.5, 0.05, '%.2f'); P['parapet'] = num('Parapet (kip/ft)', 'parapet', 0.0, 0.5, 0.05, '%.2f')
with st.sidebar.expander('Seismic (BCP-SP 2007 / UBC-97)'):
    P['zone'] = st.selectbox('Seismic zone', list(E.ZONES), 2, help='Sukkur zone must be confirmed from the code map; Zone 2B is a conservative assumption.')
    P['soil'] = st.selectbox('Soil profile', ['Sa', 'Sb', 'Sc', 'Sd', 'Se'], 3); P['R'] = num('R factor', 'R', 2.0, 8.5, 0.5, '%.1f'); P['I'] = num('Importance I', 'I', 1.0, 1.5, 0.25, '%.2f')
    P['base'] = st.selectbox('Column base', ['fixed', 'pinned'], 0)
with st.sidebar.expander('Foundation & sustainability'):
    P['qa'] = num('Allowable bearing qa (ksf)', 'qa', 1.0, 8.0, 0.5, '%.1f'); P['cement'] = num('Cement content (kg/m³)', 'cement', 250.0, 450.0, 10.0, '%.0f')
    P['fly_ash'] = num('Fly-ash replacement (%)', 'fly_ash', 0.0, 40.0, 5.0, '%.0f')

@st.cache_resource(show_spinner='Analysing frame …')
def run(P): return E.analyze(P)
R = run(P); E.configure(P); FIN, B, S, SEI, J = R['FIN'], R['beam'], R['slab'], R['SEI'], R['joint']; f = FIN['f']; out = FIN['out']

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
status = pd.DataFrame(rows, columns=['Check', 'Demand', 'Capacity / limit', 'Status'])
failed = status[status['Status'] == 'FAIL']['Check'].tolist(); all_ok = len(failed) == 0

# ---------------------------------------------------------------- header
st.markdown('<div class="hdr"><div class="hdr-t">RC Frame Analyzer</div><div class="hdr-s">G+1 residential RC frame &nbsp;·&nbsp; 2-D analysis and design checks &nbsp;·&nbsp; ACI 318 / ASCE 7 / BCP-SP 2007 &nbsp;·&nbsp; preliminary academic design</div></div>', unsafe_allow_html=True)
def kpi(l, v, sub='', cls=''): return f'<div class="kpi {cls}"><div class="l">{l}</div><div class="v">{v}</div><div class="s">{sub}</div></div>'
st.markdown('<div class="kpis">'+kpi('Base shear V', f"{SEI['V']:.1f} kip", f"Cs = {SEI['Cs']:.3f}")+kpi('Seismic weight W', f"{SEI['W']:.0f} kip", f"T = {SEI['T']:.2f} s")
            +kpi('Max column DCR', f"{FIN['maxDCR']:.2f}", 'limit 1.00', 'pass' if FIN['maxDCR'] <= 1 else 'fail')+kpi('Max column load', f"{FIN['PuB']:.0f} kip", 'factored Pu')
            +kpi('Design status', 'PASS' if all_ok else 'FAIL', f"{len(status)-len(failed)} of {len(status)} checks OK", 'pass' if all_ok else 'fail')+'</div>', unsafe_allow_html=True)

tabs = st.tabs(['Summary', 'Model & Loads', 'Frame Diagrams', 'Beam', 'Columns', 'Slab & Footings', 'Failure Guide', 'Sustainability', 'Export'])

with tabs[0]:
    if all_ok: st.markdown(f'<div class="banner pass">ALL {len(status)} CHECKS PASS – this section combination satisfies strength, stability and serviceability for the given inputs.</div>', unsafe_allow_html=True)
    else: st.markdown('<div class="banner fail">' + str(len(failed)) + ' CHECK(S) FAIL: ' + '; '.join(failed) + ' – change the member sizes in the sidebar.</div>', unsafe_allow_html=True)
    st.subheader('Design checks – demand vs capacity'); tbl(status)
    if 'top' in B:
        st.subheader('Proposed reinforcement (preliminary)')
        tbl(pd.DataFrame([
            ['Slab', f"{P['t_slab']} in", f"#3 @ {S['s']} in both ways, top over beams and bottom"],
            ['Beam', f"{P['beam_b']:.0f}×{P['beam_h']:.0f} in", f"support top {B['top']}, bottom {B['bot']}, mid top 2 #5; stirrups {P['stirrup']} @ {B['s_end']} in (36 in at ends) / {B['s_mid']} in elsewhere"],
            ['Column', f"{col[0]:.0f}×{col[1]:.0f} in", f"{P['col_nbar']} {P['col_bar']} (ρ = {FIN['rho']*100:.2f} %), ties {P['stirrup']} @ {min(5.0, 8*E.DIA[P['col_bar']]):.1f} in near joints / {min(16*E.DIA[P['col_bar']], 48*E.DIA[P['stirrup']], min(col)):.0f} in elsewhere"],
            ['Footings', ' / '.join(R['footing']['Footing [ft]']), '; '.join(R['footing']['Bars each way (#5)'])]], columns=['Member', 'Size', 'Reinforcement']))
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
