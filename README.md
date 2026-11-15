# RC Frame Analyzer – Structural Analysis II CEP

Working web application (Streamlit) for a G+1 reinforced-concrete frame (3×3 columns).

**What it does:** takes geometry, member sizes, materials, loads and seismic data → builds the 2-D frame (grid 2) →
direct-stiffness analysis (D, L, E) → ASCE 7 load combinations → SFD, BMD, axial force, deformed shape →
ACI 318 checks (beam flexure/shear/cracking/deflection, column P-M interaction, slenderness/Q, strong-column/weak-beam,
joint shear, slab, punching, isolated footings) → failure guide → sustainability numbers → Excel/PNG export.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (free) on Streamlit Community Cloud
1. Create a GitHub repository and upload `app.py`, `engine.py`, `requirements.txt`, `README.md`.
2. Go to https://share.streamlit.io, sign in with GitHub, click **New app**, pick the repo, main file `app.py`, **Deploy**.

## Notes
* Units: kip, inch, ksi (ft / psf for geometry and area loads).
* Zone/soil, bearing capacity and loads are assumptions – confirm for the real site.
* Preliminary academic design only; not a substitute for a licensed structural engineer.
