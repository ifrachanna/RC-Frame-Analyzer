"""RC frame analysis + design engine (2-D direct stiffness). Units: kip, inch, ksi (ft for geometry/loads where stated)."""
import math, threading
import numpy as np, pandas as pd

LOCK = threading.Lock()
G = {}          # configured parameters (module globals set by configure)

SOIL_CA = {'Sa': [.06, .12, .16, .24, .32], 'Sb': [.08, .15, .20, .30, .40], 'Sc': [.09, .18, .24, .33, .40], 'Sd': [.12, .22, .28, .36, .44], 'Se': [.19, .30, .34, .36, .36]}
SOIL_CV = {'Sa': [.06, .12, .16, .24, .32], 'Sb': [.08, .15, .20, .30, .40], 'Sc': [.13, .25, .32, .45, .56], 'Sd': [.18, .32, .40, .54, .64], 'Se': [.26, .50, .64, .84, .96]}
ZONES = {'Zone 1 (Z=0.075)': 0, 'Zone 2A (Z=0.15)': 1, 'Zone 2B (Z=0.20)': 2, 'Zone 3 (Z=0.30)': 3, 'Zone 4 (Z=0.40)': 4}
ZVAL = [0.075, 0.15, 0.20, 0.30, 0.40]
BAR = {'#3': 0.11, '#4': 0.20, '#5': 0.31, '#6': 0.44, '#7': 0.60, '#8': 0.79, '#9': 1.00}
DIA = {'#3': .375, '#4': .5, '#5': .625, '#6': .75, '#7': .875, '#8': 1.0, '#9': 1.128}

DEFAULTS = dict(
    fc=3.0, fy=60.0, L_ft=15.0, H_ft=10.0, t_slab=5.0, beam_b=9.0, beam_h=18.0, col_h=15.0, col_b=15.0,
    col_nbar=8, col_bar='#5', stirrup='#3', cover_beam=1.5, cover_col=1.5, cover_slab=0.75,
    SDL_floor=45.0, SDL_roof=35.0, LL_floor=40.0, LL_roof=20.0, wall=0.85, parapet=0.15,
    zone='Zone 2B (Z=0.20)', soil='Sd', R=5.5, I=1.0, Ct=0.030, base='fixed', qa=3.0, cement=350.0, co2=0.85, fly_ash=20.0)

def configure(P):
    p = dict(DEFAULTS); p.update(P); g = G; g.update(p)
    g['Ec'] = 57.0*math.sqrt(p['fc']*1000); g['Es'] = 29000.0; g['gc'] = 0.150/1728.0
    g['L'] = p['L_ft']*12; g['H'] = p['H_ft']*12; g['col'] = (p['col_h'], p['col_b'])
    g['slab_sw'] = g['gc']*1728*p['t_slab']/12
    g['wD_floor'] = g['slab_sw']+p['SDL_floor']/1000; g['wD_roof'] = g['slab_sw']+p['SDL_roof']/1000
    g['wL_floor'] = p['LL_floor']/1000; g['wL_roof'] = p['LL_roof']/1000
    g['beam_sw'] = g['gc']*1728*p['beam_b']*(p['beam_h']-p['t_slab'])/144
    zi = ZONES[p['zone']]; g['Z'] = ZVAL[zi]; g['Ca'] = SOIL_CA[p['soil']][zi]; g['Cv'] = SOIL_CV[p['soil']][zi]
    return g

# ------------------------------------------------------------------ solver
class Frame:
    def __init__(self): self.nodes = {}; self.elems = []; self.sup = {}
    def node(self, nid, x, y): self.nodes[nid] = (x, y)
    def elem(self, name, i, j, E, A, I, kind): self.elems.append(dict(name=name, i=i, j=j, E=E, A=A, I=I, kind=kind, q=None))
    def support(self, nid, fx=True, fy=True, fr=True): self.sup[nid] = (fx, fy, fr)

_gp, _gw = np.polynomial.legendre.leggauss(24)
def element_geom(f, e):
    xi, yi = f.nodes[e['i']]; xj, yj = f.nodes[e['j']]; Le = math.hypot(xj-xi, yj-yi); return Le, (xj-xi)/Le, (yj-yi)/Le
def k_local(E, A, I, Le):
    EA, EI = E*A/Le, E*I; k = np.zeros((6, 6))
    k[0, 0] = k[3, 3] = EA; k[0, 3] = k[3, 0] = -EA
    k[1, 1] = k[4, 4] = 12*EI/Le**3; k[1, 4] = k[4, 1] = -12*EI/Le**3
    k[1, 2] = k[2, 1] = k[1, 5] = k[5, 1] = 6*EI/Le**2; k[2, 4] = k[4, 2] = k[4, 5] = k[5, 4] = -6*EI/Le**2
    k[2, 2] = k[5, 5] = 4*EI/Le; k[2, 5] = k[5, 2] = 2*EI/Le; return k
def T_matrix(c, s):
    R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]); T = np.zeros((6, 6)); T[:3, :3] = R; T[3:, 3:] = R; return T
def equiv_load_local(q, Le):
    x = (_gp+1)*Le/2; w = _gw*Le/2; xi = x/Le; qq = q(x)
    N1 = 1-3*xi**2+2*xi**3; N2 = Le*(xi-2*xi**2+xi**3); N3 = 3*xi**2-2*xi**3; N4 = Le*(-xi**2+xi**3)
    f = [np.sum(w*qq*N) for N in (N1, N2, N3, N4)]; return np.array([0, f[0], f[1], 0, f[2], f[3]])
def solve(f, Pnodal):
    ids = list(f.nodes); idx = {n: i for i, n in enumerate(ids)}; nd = 3*len(ids); K = np.zeros((nd, nd)); P = np.zeros(nd)
    for n, v in Pnodal.items(): P[3*idx[n]:3*idx[n]+3] += v
    cache = []
    for e in f.elems:
        Le, c, s = element_geom(f, e); T = T_matrix(c, s); kl = k_local(e['E'], e['A'], e['I'], Le); fl = np.zeros(6)
        if e['q'] is not None: fl = equiv_load_local(lambda x: -e['q'](x), Le)
        dof = np.r_[3*idx[e['i']]:3*idx[e['i']]+3, 3*idx[e['j']]:3*idx[e['j']]+3]
        K[np.ix_(dof, dof)] += T.T@kl@T; P[dof] += T.T@fl; cache.append((Le, c, s, T, kl, fl, dof))
    fixed = np.zeros(nd, bool)
    for n, r in f.sup.items(): fixed[3*idx[n]:3*idx[n]+3] = r
    free = ~fixed; U = np.zeros(nd); U[free] = np.linalg.solve(K[np.ix_(free, free)], P[free]); R = K@U-P; R[free] = 0
    out = dict(R={n: R[3*idx[n]:3*idx[n]+3] for n in f.sup}, el={}, Unode={n: U[3*idx[n]:3*idx[n]+3] for n in ids})
    for e, (Le, c, s, T, kl, fl, dof) in zip(f.elems, cache):
        ul = T@U[dof]; out['el'][e['name']] = dict(F=kl@ul-fl, ul=ul, Le=Le, c=c, s=s, e=dict(e))
    return out
def profile(res, name, n=121):
    d = res['el'][name]; Le = d['Le']; F = d['F']; e = d['e']; x = np.linspace(0, Le, n)
    q = np.zeros(n) if e['q'] is None else -e['q'](x)
    cq = np.r_[0, np.cumsum((q[1:]+q[:-1])/2*np.diff(x))]; V = F[1]+cq
    M = -F[2]+np.r_[0, np.cumsum((V[1:]+V[:-1])/2*np.diff(x))]; N = -F[0]*np.ones(n)
    kap = M/(e['E']*e['I']); th = d['ul'][2]+np.r_[0, np.cumsum((kap[1:]+kap[:-1])/2*np.diff(x))]
    v = d['ul'][1]+np.r_[0, np.cumsum((th[1:]+th[:-1])/2*np.diff(x))]
    return dict(x=x, V=V, M=M, N=N, v=v)

# ------------------------------------------------------------------ loads / model
def Ig_rect(b, h): return b*h**3/12
def col_weight_each(c): return G['gc']*c[0]*c[1]*G['H']
def seismic(col):
    g = G; L_ft = g['L_ft']; A_fl = (2*L_ft)**2; nbeam = 6*(2*L_ft)
    perim = 4*(2*L_ft)
    W1 = g['wD_floor']*A_fl+g['beam_sw']*nbeam+9*col_weight_each(col)+g['wall']*perim
    W2 = g['wD_roof']*A_fl+g['beam_sw']*nbeam+9*col_weight_each(col)/2+g['wall']*perim/2+g['parapet']*perim
    W = W1+W2; h = 2*g['H_ft']; T = g['Ct']*h**0.75; Ca, Cv, I, R = g['Ca'], g['Cv'], g['I'], g['R']
    V = min(Cv*I*W/(R*T), 2.5*Ca*I*W/R); V = max(V, 0.11*Ca*I*W)
    hh = g['H_ft']; F1 = V*W1*hh/(W1*hh+W2*2*hh); F2 = V*W2*2*hh/(W1*hh+W2*2*hh)
    return dict(W1=W1, W2=W2, W=W, T=T, V=V, Cs=V/W, F1=F1, F2=F2)
def tri_load(peak): L = G['L']; return lambda x: (peak/12.0)*(1-np.abs(2*x/L-1))
def udl(w): return lambda x: (w/12.0)+0*x
def out_of_plane_reactions(peak_slab, w_line):
    L = G['L']; f = Frame(); [f.node(i, x, 0) for i, x in zip((1, 2, 3), (0, L, 2*L))]; Ib = Ig_rect(G['beam_b'], G['beam_h'])
    f.elem('s1', 1, 2, G['Ec'], 200., Ib, 'beam'); f.elem('s2', 2, 3, G['Ec'], 200., Ib, 'beam')
    for e in f.elems: e['q'] = (lambda x, p=peak_slab, w=w_line: tri_load(p)(x)+udl(w)(x))
    f.support(1, True, True, False); f.support(2, False, True, False); f.support(3, False, True, False)
    return solve(f, {})['R'][2][1]
def build_frame(col, base='fixed', Ifac_b=0.35, Ifac_c=0.70):
    g = G; L, H = g['L'], g['H']; f = Frame()
    for j, cn in enumerate('ABC'):
        for lv in range(3): f.node((cn, lv), j*L, lv*H)
        f.support((cn, 0), True, True, base == 'fixed')
    Ib = Ig_rect(g['beam_b'], g['beam_h'])*Ifac_b; Ab = g['beam_b']*g['beam_h']; Ic = col[1]*col[0]**3/12*Ifac_c; Ac = col[0]*col[1]
    for cn in 'ABC':
        for lv in range(2): f.elem(f'C{cn}{lv+1}', (cn, lv), (cn, lv+1), g['Ec'], Ac, Ic, 'col')
    for lv in (1, 2):
        for a, b in (('A', 'B'), ('B', 'C')): f.elem(f'B{lv}{a}{b}', (a, lv), (b, lv), g['Ec'], Ab, Ib, 'beam')
    return f
def load_cases(col, SEI):
    g = G; L_ft = g['L_ft']; ctot = col_weight_each(col); cs = {}; oop = {}
    for lv, (wD, wL, wallw) in {1: (g['wD_floor'], g['wL_floor'], g['wall']), 2: (g['wD_roof'], g['wL_roof'], g['parapet'])}.items():
        pD, pL = wD*L_ft/2, wL*L_ft/2
        oop[lv] = dict(D=dict(A=out_of_plane_reactions(pD, g['beam_sw']+wallw), B=out_of_plane_reactions(2*pD, g['beam_sw'])),
                       L=dict(A=out_of_plane_reactions(pL, 0.0), B=out_of_plane_reactions(2*pL, 0.0)))
        oop[lv]['D']['C'] = oop[lv]['D']['A']; oop[lv]['L']['C'] = oop[lv]['L']['A']
    Dn, Ln = {}, {}
    for cn in 'ABC':
        for lv in range(3):
            d = 0.0; l = 0.0
            if lv < 2: d += ctot/2
            if lv > 0: d += ctot/2
            if lv == 0: d = ctot/2
            if lv >= 1: d += oop[lv]['D'][cn]; l += oop[lv]['L'][cn]
            Dn[(cn, lv)] = (0, -d, 0); Ln[(cn, lv)] = (0, -l, 0)
    cs['D_nodal'] = Dn; cs['L_nodal'] = Ln
    cs['qD'] = {1: lambda x: tri_load(2*g['wD_floor']*L_ft)(x)+udl(g['beam_sw'])(x), 2: lambda x: tri_load(2*g['wD_roof']*L_ft)(x)+udl(g['beam_sw'])(x)}
    cs['qL'] = {1: tri_load(2*g['wL_floor']*L_ft), 2: tri_load(2*g['wL_roof']*L_ft)}
    cs['E_nodal'] = {(cn, lv): ((SEI['F1'] if lv == 1 else SEI['F2'])/9, 0, 0) for cn in 'ABC' for lv in (1, 2)}
    return cs
def run_cases(col, base='fixed', Ifac_b=0.35, Ifac_c=0.70, SEI=None):
    f = build_frame(col, base, Ifac_b, Ifac_c); cs = load_cases(col, SEI or seismic(col)); out = {}
    for name in ('D', 'L'):
        for e in f.elems: e['q'] = cs['q'+name][int(e['name'][1])] if e['kind'] == 'beam' else None
        out[name] = solve(f, cs[name+'_nodal'])
    for e in f.elems: e['q'] = None
    out['E'] = solve(f, cs['E_nodal']); return f, out, cs
COMBOS = {'C1: 1.2D+1.6L': dict(D=1.2, L=1.6, E=0), 'C2a: 1.2D+1.0L+1.0E': dict(D=1.2, L=1.0, E=1.0), 'C2b: 1.2D+1.0L-1.0E': dict(D=1.2, L=1.0, E=-1.0),
          'C3a: 0.9D+1.0E': dict(D=0.9, L=0.0, E=1.0), 'C3b: 0.9D-1.0E': dict(D=0.9, L=0.0, E=-1.0)}
def combo_profile(out, name, comb, n=121):
    p = {k: profile(out[k], name, n) for k in ('D', 'L', 'E')}
    r = {key: sum(comb[k]*p[k][key] for k in ('D', 'L', 'E')) for key in ('V', 'M', 'N', 'v')}; r['x'] = p['D']['x']; return r

# ------------------------------------------------------------------ columns
def col_bars(h, nbar, bar):
    As1 = BAR[bar]; db = DIA[bar]; dp = G['cover_col']+DIA[G['stirrup']]+db/2
    top_n, mid_n = {4: (2, 0), 6: (3, 0), 8: (3, 2), 10: (4, 2), 12: (4, 4)}[nbar]
    bars = [(dp, top_n*As1)]
    if mid_n: bars.append((h/2, mid_n*As1))
    bars.append((h-dp, top_n*As1)); return bars
def pm_curve(h, b, bars):
    fc, fy, Es = G['fc'], G['fy'], G['Es']; Ast = sum(a for _, a in bars); Ag = h*b; P0 = 0.85*fc*(Ag-Ast)+fy*Ast; Pcap = 0.8*P0; dt = max(y for y, _ in bars); pts = []
    for c in np.concatenate([np.linspace(4*h, h, 40), np.linspace(h, 0.12*dt, 120)]):
        a = min(0.85*c, h); Cc = 0.85*fc*a*b; P = Cc; M = Cc*(h/2-a/2)
        for y, As in bars:
            fs = float(np.clip(Es*0.003*(c-y)/c, -fy, fy)); F = As*(fs-(0.85*fc if y < a else 0.0)); P += F; M += F*(h/2-y)
        et = 0.003*(dt-c)/c; ph = float(np.clip(0.65+(et-0.002)*0.25/0.003, 0.65, 0.90)); pts.append((P, M, ph*P, ph*M, et))
    pts = np.array(pts); pts = pts[pts[:, 0] <= Pcap]
    capM = np.interp(Pcap, pts[::-1, 0], pts[::-1, 1]) if len(pts) else 0
    pts = np.vstack([[Pcap, capM, 0.65*Pcap, 0.65*capM, 0.0], pts])
    return dict(P=pts[:, 0], M=pts[:, 1], phiP=pts[:, 2], phiM=pts[:, 3], Pcap=0.65*Pcap, P0=P0)
def phiM_at(cv, Pu): o = np.argsort(cv['phiP']); return float(np.interp(Pu, cv['phiP'][o], cv['phiM'][o]))
def k_sway(a, b): return math.sqrt((1.6*a*b+4*(a+b)+7.5)/(a+b+7.5))
def column_checks(col, bars, base='fixed'):
    g = G; H, L = g['H'], g['L']; SEI = seismic(col); f, out, cs = run_cases(col, base=base, SEI=SEI); h, b = col; cv = pm_curve(h, b, bars)
    EcIc = g['Ec']*b*h**3/12*0.70; EcIb = g['Ec']*Ig_rect(g['beam_b'], g['beam_h'])*0.35
    psi_b = 1.0 if base == 'fixed' else 10.0; lu = H-g['beam_h']; r = 0.3*h
    kk = {cn: k_sway(psi_b, (2*EcIc/H)/((2 if cn == 'B' else 1)*EcIb/L)) for cn in 'ABC'}
    dE = out['E']['Unode']; d1 = abs(dE[('B', 1)][0]); d2 = abs(dE[('B', 2)][0]-dE[('B', 1)][0])
    comb = COMBOS['C2a: 1.2D+1.0L+1.0E']; SumPu1 = sum(-combo_profile(out, f'C{cn}1', comb, 3)['N'][0] for cn in 'ABC')
    Vus = sum(cs['E_nodal'][(cn, lv)][0] for cn in 'ABC' for lv in (1, 2)); Q = SumPu1*d1/(Vus*H); ds = 1/(1-Q) if 0.05 < Q < 0.95 else 1.0
    rows = []
    for cn in 'ABC':
        for lv in (1, 2):
            nm = f'C{cn}{lv}'
            for cname, cb in COMBOS.items():
                p = combo_profile(out, nm, cb, 3); Pu = max(-p['N'][0], 0.0)
                Mu = max(abs(p['M'][0]), abs(p['M'][-1]))*(ds if cb['E'] != 0 else 1.0); Mmin = Pu*(0.6+0.03*h)
                pm = phiM_at(cv, min(Pu, cv['Pcap']*0.9999)) if Pu <= cv['Pcap'] else 0.0
                rows.append(dict(column=nm, combo=cname, Pu_kip=Pu, Mu_kipft=max(Mu, Mmin)/12, phiMn_kipft=pm/12, phiPn_max=cv['Pcap']))
    df = pd.DataFrame(rows); df['DCR'] = np.where(df.Pu_kip > df.phiPn_max, 9.99, (df.Mu_kipft/df.phiMn_kipft.replace(0, np.nan)).fillna(9.99))
    sl = max(kk.values())*lu/r; bdns = 0.6; Pc = math.pi**2*0.4*g['Ec']*b*h**3/12/(1+bdns)/(max(kk.values())*lu)**2
    return dict(col=col, bars=bars, df=df, curve=cv, SEI=SEI, out=out, f=f, cs=cs, drift1=d1, drift2=d2, Q=Q, ds=ds, sl=sl, kmax=max(kk.values()),
                Pc=Pc, PuB=float(df[df.column == 'CB1'].Pu_kip.max()), maxDCR=float(df.DCR.max()), rho=sum(a for _, a in bars)/(b*h), lu_r=lu/r)

# ------------------------------------------------------------------ beam / slab / footing design
def As_req(Mu, b, d, minsteel=True):
    fc, fy = G['fc'], G['fy']; Rn = Mu/(0.9*b*d**2)
    if Rn >= 0.85*fc/2*0.99: return None
    rho = 0.85*fc/fy*(1-math.sqrt(1-2*Rn/(0.85*fc))); return max(rho*b*d, max(3*math.sqrt(fc*1000), 200)/(fy*1000)*b*d if minsteel else 0.0)
def phiMn(As, b, d):
    fc, fy = G['fc'], G['fy']; a = As*fy/(0.85*fc*b); c = a/0.85; et = 0.003*(d-c)/c
    ph = float(np.clip(0.65+(et-0.002)*0.25/0.003, 0.65, 0.9)); return ph*As*fy*(d-a/2), et, ph
def pick_bars(need, b):
    st = G['cover_beam']; opts = []
    for n in (2, 3):
        for s in ('#5', '#6', '#7', '#8', '#9'):
            if n*DIA[s]+(n-1)*max(1.0, DIA[s]) <= b-2*st-2*DIA[G['stirrup']]: opts.append((n*BAR[s], f'{n}{s}'))
    for a, s in [(0.88+0.31, '2#6+1#5'), (0.88+0.60, '2#6+1#7'), (1.2+0.6, '2#7+1#7'), (0.79*2+0.6, '2#8+1#7')]: opts.append((a, s))
    opts = sorted(o for o in opts if o[0] >= need); return opts[0] if opts else (None, 'does not fit in one layer – enlarge section')

def beam_design(FIN):
    g = G; out = FIN['out']; f = FIN['f']; L, col = g['L'], FIN['col']; cc = col[0]; bw, hb = g['beam_b'], g['beam_h']
    d_b = hb-g['cover_beam']-2*DIA[g['stirrup']]/2*0-DIA[g['stirrup']]-0.375  # assumes #6 main bar for d
    d_b = hb-g['cover_beam']-DIA[g['stirrup']]-0.375
    ENV = {}
    for nm in ('B1AB', 'B1BC', 'B2AB', 'B2BC'):
        e = dict(neg_L=0, neg_R=0, pos=0, V=0, combos={})
        for cname, comb in COMBOS.items():
            p = combo_profile(out, nm, comb, 361); x = p['x']; iL = np.argmin(abs(x-cc/2)); iR = np.argmin(abs(x-(L-cc/2)))
            e['neg_L'] = max(e['neg_L'], -p['M'][iL]); e['neg_R'] = max(e['neg_R'], -p['M'][iR]); e['pos'] = max(e['pos'], p['M'].max())
            dL = np.argmin(abs(x-(cc/2+d_b))); dR = np.argmin(abs(x-(L-cc/2-d_b))); e['V'] = max(e['V'], abs(p['V'][dL]), abs(p['V'][dR])); e['combos'][cname] = p
        ENV[nm] = e
    Mneg = max(max(e['neg_L'], e['neg_R']) for e in ENV.values()); Mpos = max(e['pos'] for e in ENV.values()); Vu = max(e['V'] for e in ENV.values())
    Asn, Asp = As_req(Mneg, bw, d_b), As_req(Mpos, bw, d_b); R = dict(ENV=ENV, d=d_b, Mneg=Mneg/12, Mpos=Mpos/12, Vu=Vu, ok=True, msgs=[])
    if Asn is None or Asp is None:
        R['ok'] = False; R['msgs'].append('Beam section too small for the flexural demand (Rn exceeds limit) – increase depth/width.'); return R
    top, bot = pick_bars(Asn, bw), pick_bars(Asp, bw); R.update(Asn=Asn, Asp=Asp, top=top[1], top_As=top[0], bot=bot[1], bot_As=bot[0])
    if top[0] is None or bot[0] is None: R['ok'] = False; R['msgs'].append('Required steel does not fit in one layer – use two layers or a bigger section.'); return R
    pn, etn, phn = phiMn(top[0], bw, d_b); pp, etp, php = phiMn(bot[0], bw, d_b); R.update(phiMn_neg=pn/12, phiMn_pos=pp/12, et_n=etn, et_p=etp)
    Mn_neg, Mn_pos = pn/phn, pp/php; fr = 7.5*math.sqrt(g['fc']*1000)/1000; Ig_b = Ig_rect(bw, hb); Mcr = fr*Ig_b/(hb/2); R['Mcr'] = Mcr/12
    phiVc = 0.75*2*math.sqrt(g['fc']*1000)*bw*d_b/1000; Av = 2*BAR[g['stirrup']]; Vs_req = max(Vu/0.75-phiVc/0.75, 0)
    s_str = Av*g['fy']*d_b/Vs_req if Vs_req > 0 else 99; ln_ft = (L-cc)/12
    wu = 1.2*(2*g['wD_floor']*g['L_ft']/2+g['beam_sw'])+1.0*(2*g['wL_floor']*g['L_ft']/2)
    Ve = (Mn_neg+Mn_pos)/12/ln_ft+wu*ln_ft/2; s_e = Av*g['fy']*d_b/max(Ve/0.75-phiVc/0.75, 1e-6)
    s_imf = min(d_b/4, 8*DIA[bot[1][-2:]] if bot[1][-2:] in DIA else 6, 24*DIA[g['stirrup']], 12); s_end = math.floor(min(s_imf, s_e, d_b/2)*2)/2; s_mid = math.floor(min(d_b/2, s_str if s_str < 99 else 99, 24)*2)/2
    s_mid = min(s_mid, s_end*2.5, 12); Vs_max = 4*math.sqrt(g['fc']*1000)*bw*d_b/1000
    phiVn_end = 0.75*(phiVc/0.75+Av*g['fy']*d_b/s_end); phiVn_mid = 0.75*(phiVc/0.75+Av*g['fy']*d_b/s_mid)
    R.update(phiVc=phiVc, s_req=s_str, Ve=Ve, s_e=s_e, s_end=s_end, s_mid=s_mid, Vs_max=Vs_max, phiVn_end=phiVn_end, phiVn_mid=phiVn_mid, Mn_neg=Mn_neg, Mn_pos=Mn_pos)
    if Vu/0.75-phiVc/0.75 > Vs_max: R['ok'] = False; R['msgs'].append('Vs exceeds 4√f\'c·b·d – enlarge beam section.')
    if phiVn_end < Ve: R['ok'] = False; R['msgs'].append('Shear capacity at ends below capacity-based shear Ve.')
    # deflection
    f_g, out_g, _ = run_cases(col, Ifac_b=1.0, Ifac_c=1.0, SEI=FIN['SEI'])
    def mid(res):
        p = profile(res, 'B1BC', 181); k = len(p['x'])//2; return -(p['v'][k]-(p['v'][0]+p['v'][-1])/2), p
    dD, pD = mid(out_g['D']); dL, pL = mid(out_g['L']); n = g['Es']/g['Ec']
    def Icr_of(As): x = (-n*As+math.sqrt((n*As)**2+2*bw*n*As*d_b))/bw; return bw*x**3/3+n*As*(d_b-x)**2
    def Ie_of(Ma, Icr): return Ig_b if abs(Ma) <= Mcr else min(Ig_b, (Mcr/abs(Ma))**3*Ig_b+(1-(Mcr/abs(Ma))**3)*Icr)
    km = len(pD['x'])//2; iF = int(cc/2/(L/180))
    def sv(p): return p['M'][km], p['M'][iF], p['M'][-1-iF]
    def Ie_avg(Ms): Mm, Ml, Mr = Ms; return 0.7*Ie_of(Mm, Icr_of(bot[0]))+0.15*(Ie_of(Ml, Icr_of(top[0]))+Ie_of(Mr, Icr_of(top[0])))
    Ie_D = Ie_avg(sv(pD)); Ie_DL = Ie_avg(tuple(a+b for a, b in zip(sv(pD), sv(pL))))
    dD_e = dD*Ig_b/Ie_D; dDL_e = (dD+dL)*Ig_b/Ie_DL; dL_e = dDL_e-dD_e; rho_c = BAR['#5']*2/(bw*d_b); lam = 2.0/(1+50*rho_c); lt = lam*dD_e+dL_e
    R['defl'] = dict(dD=dD_e, dL=dL_e, lt=lt, lim360=L/360, lim480=L/480, lam=lam, IeIg=Ie_DL/Ig_b, ok=(lt <= L/480 and dL_e <= L/360))
    if not R['defl']['ok']: R['msgs'].append('Deflection limit exceeded – increase beam depth or add compression steel.')
    return R

def slab_design():
    g = G; t = g['t_slab']; wu = 1.2*g['wD_floor']+1.6*g['wL_floor']; w = wu/2; ln = g['L_ft']-g['beam_b']/12
    Mn_, Mp_ = w*ln**2/12, w*ln**2/16; d = t-g['cover_slab']-0.375/2; Asmin = 0.0018*12*t
    need = max(Asmin, As_req(Mn_*12, 12, d, False) or 9.9); s = min(2*t, 18.0); s_opt = 3
    for sp in (9, 8, 7, 6, 5, 4.5):
        if BAR['#3']*12/sp >= need and sp <= min(2*t, 18): s = sp; break
    As = BAR['#3']*12/s; a = As*g['fy']/(0.85*g['fc']*12); pM = 0.9*As*g['fy']*(d-a/2)/12; Vu = wu*ln/3; pVc = 0.75*2*math.sqrt(g['fc']*1000)*12*d/1000
    tmin = max(ln*12*(0.8+g['fy']/200)/(36+9), 3.5)
    return dict(wu=wu*1000, Mneg=Mn_, Mpos=Mp_, d=d, Asmin=Asmin, s=s, As=As, phiMn=pM, Vu=Vu, phiVc=pVc, tmin=tmin, ok=(pM >= Mn_ and Vu <= pVc and t >= tmin))

def punching(t, c, Mub=30.0):
    g = G; d = t-g['cover_slab']-0.375; c1 = c+d; bo = 4*c1; wu = 1.2*g['wD_floor']+1.6*g['wL_floor']
    Vu = wu*(g['L_ft']**2-(c1/12)**2); gv = 1-1/(1+2/3); Jc = c1*d**3/6+d*c1**3/6+d*c1*c1**2/2
    vu = Vu*1000/(bo*d)+gv*Mub*12*1000*(c1/2)/Jc; vc = min(4, 6, 40*d/bo+2)*math.sqrt(g['fc']*1000)
    return dict(t=t, d=d, Vu=Vu, vu=vu, phivc=0.75*vc, DCR=vu/(0.75*vc))

def footings(FIN):
    g = G; out = FIN['out']; col = FIN['col']; rows = []
    for cn in 'ABC':
        D_ = out['D']['R'][(cn, 0)][1]; L_ = out['L']['R'][(cn, 0)][1]; Ps = 1.1*(D_+L_); B = math.ceil(math.sqrt(Ps/g['qa'])*2)/2
        Pu = 1.2*D_+1.6*L_; qu = Pu/B**2; h_f = 12.0
        while h_f < 60:
            d_f = h_f-3-0.625; c1 = col[0]+d_f; Vp = Pu-qu*(c1/12)**2; pVp = 0.75*4*math.sqrt(g['fc']*1000)*4*c1*d_f/1000
            lc = (B*12-col[0])/2-d_f; V1 = qu*B*lc/12; pV1 = 0.75*2*math.sqrt(g['fc']*1000)*B*12*d_f/1000
            if pVp >= Vp and pV1 >= V1: break
            h_f += 3
        Mu = qu*B*((B*12-col[0])/2/12)**2/2; Asf = max(As_req(Mu*12, B*12, d_f) or 0, 0.0018*B*12*h_f)
        rows.append({'column': f'{cn}2', 'D [kip]': D_, 'L [kip]': L_, 'Service 1.1(D+L) [kip]': Ps, 'Footing [ft]': f'{B:.1f} x {B:.1f}', 'Thickness [in]': h_f, 'Pu [kip]': Pu,
                     'Punching DCR': Vp/pVp, 'Mu [kip-ft]': Mu, 'As [in2]': Asf, 'Bars each way (#5)': f'{math.ceil(Asf/BAR["#5"])} #5'})
    return pd.DataFrame(rows)

def sustainability(FIN, beam):
    g = G; k = 0.0283168; Lft = g['L_ft']; A = (2*Lft)**2; nb = 6*(2*Lft)
    def vol(t, bb, bh, cc_, ff=1.0): return A*t/12*2+nb*bb*(bh-t)/144*2+9*cc_*cc_/144*2*g['H_ft']+(4*25+4*30.25+42.25*1.25)*ff
    Vo = vol(g['t_slab'], g['beam_b'], g['beam_h'], FIN['col'][0]); Vb = vol(6, 12, 24, 18, 1.2)
    cem_o, cem_b = Vo*k*g['cement'], Vb*k*g['cement']; fa = g['fly_ash']/100
    _w = G['wall']; G['wall'] = 0.40; SEIlw = seismic(FIN['col']); G['wall'] = _w
    return dict(V_base=Vb*k, V_opt=Vo*k, cem_base=cem_b/1000, cem_opt=cem_o/1000, co2_base=cem_b*g['co2']/1000, co2_opt=cem_o*g['co2']/1000, co2_fa=cem_o*(1-fa)*g['co2']/1000,
                W=FIN['SEI']['W'], W_lw=SEIlw['W'], V=FIN['SEI']['V'], V_lw=SEIlw['V'])

def analyze(P):
    with LOCK:
        configure(P); g = G; col = g['col']; bars = col_bars(col[0], g['col_nbar'], g['col_bar'])
        FIN = column_checks(col, bars, base=g['base']); B = beam_design(FIN)
        FINp = column_checks(col, bars, base='pinned' if g['base'] == 'fixed' else 'fixed')
        S = slab_design(); pf = {t: punching(t, col[0]) for t in (g['t_slab'], 7.0, 8.0)}
        ft = footings(FIN); su = sustainability(FIN, B)
        # strong column / weak beam and joint shear
        scwb = None; joint = None
        if B.get('ok') or 'top' in B:
            cv = FIN['curve']; o = np.argsort(cv['P']); df = FIN['df']
            Pc = df[(df.column.isin(['CB1', 'CB2'])) & (df.combo.str.startswith('C2a'))].Pu_kip.values
            Mnc = [np.interp(P_, cv['P'][o], cv['M'][o])/12 for P_ in Pc]; Mnb = (B['Mn_neg']+B['Mn_pos'])/12; scwb = sum(Mnc)/Mnb
            Vcol = np.mean([abs(combo_profile(FIN['out'], nm, COMBOS['C2a: 1.2D+1.0L+1.0E'], 3)['V'][0]) for nm in ('CB1', 'CB2')])
            Vj = B['top_As']*g['fy']+B['bot_As']*g['fy']-Vcol; pVj = 0.85*20*math.sqrt(g['fc']*1000)*col[0]*col[1]/1000; joint = dict(Vj=Vj, phiVj=pVj, scwb=scwb)
        cfg = dict(g)
        return dict(P=cfg, FIN=FIN, FINp=FINp, beam=B, slab=S, punch=pf, footing=ft, sust=su, joint=joint, SEI=FIN['SEI'], bars=bars)
