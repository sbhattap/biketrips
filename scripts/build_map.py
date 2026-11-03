"""Offline interactive map: actual trips, each model's prediction and error, and weeks of service.

Every cell's prediction comes from a model that never saw that cell's res-6 block (5-fold spatial
CV with the tuned spatial-split settings), so the map covers the whole city without leaking. The
Plotly library is embedded and the map is plain SVG with no tiles or WebGL, so it opens offline.
"""
import argparse

import h3
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import get_colorscale
import yaml

from biketrips import config, data, features, splits, tune

LABELS = {"baseline_lr": "Baseline LR", "linear": "Ridge on log1p", "knn": "Spatial kNN", "poisson": "Poisson GLM",
          "xgb": "XGBoost", "hgb": "HistGradientBoosting", "hurdle": "Hurdle", "gcn": "GCN"}
SUPPLY = ["POIs, no trips (gap)", "No POIs, no trips", "Trips some weeks", "Trips every week"]
SUPPLY_COLORS = ["#d62728", "#d9d9d9", "#fdae6b", "#3182bd"]
TOP = 5.6  # log10 of the busiest cell (about 390k trips), so every layer shares one scale
INK, PAPER = "#1d2a2e", "#fbfcfc"
HEADER = """<style>
html, body {{ height: 100%; margin: 0 }}
body {{ display: flex; flex-direction: column; background: {paper}; color: {ink}; font-family: system-ui, sans-serif;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)) }}
header {{ padding-block: 14px 10px; border-bottom: 1px solid #dde4e6; display: grid; gap: 4px }}
h1 {{ font-size: 1.25rem; font-weight: 600; margin: 0 }}
p {{ margin: 0; color: #5b6b70; font-size: 0.875rem; line-height: 1.45; max-width: 70ch }}
.num {{ font-family: ui-monospace, monospace; color: #1d2a2e }}
main {{ flex: 1; min-height: 420px; min-width: 0 }}
main > div {{ height: 100% }}
</style>
<header><h1>Shenzhen bike trip ends, Jul 31 to Aug 29, 2021</h1>
<p><span class="num">{trips:,}</span> trips on <span class="num">{cells:,}</span> H3 cells. Use the menu on the map to
switch between actual trips, each model's prediction, each model's error (red = over, blue = under) and weeks of service
(<b style="color:#d62728">red</b> = POIs but no trips, <b style="color:#3182bd">blue</b> = trips every week). Hover a cell
for every model's number.</p>
<p>Each cell is predicted by a model that never saw its city block. The GCN still sees all cells' features
(transductive). Scroll or pinch to zoom.</p></header>
<main>"""
# Plotly clips the map to the frame its lon/lat ranges define, so the ranges follow the plot area's shape;
# otherwise zooming in cuts the map off at an invisible edge. Mercator near 22.6N: 1 deg lat ~ 1/cos(lat) deg lon.
FIT = """var gd = document.getElementById('{plot_id}'), k = Math.cos(22.63 * Math.PI / 180), t;
function fit() {
  var s = gd._fullLayout._size, lon = Math.max(0.5, 0.25 / k * s.w / s.h), lat = lon * k * s.h / s.w;
  Plotly.relayout(gd, {'geo.lonaxis.range': [114.19 - lon, 114.19 + lon], 'geo.lataxis.range': [22.63 - lat, 22.63 + lat]});
}
fit(); window.addEventListener('resize', function () { clearTimeout(t); t = setTimeout(fit, 250); });"""


def oof_predictions(cfg, cells, refresh):
    path = cfg["data"]["results_dir"] / "oof_predictions.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col="h3").reindex(cells["h3"])
    best = yaml.safe_load((cfg["data"]["results_dir"] / "best_params.yaml").read_text())
    folds = splits.inner_folds(cells, "spatial", cfg["cv"]["n_splits"], cfg["seed"])
    ctx = {"features": features.feature_columns(cells), "seed": cfg["seed"], "cells": cells}
    out = pd.DataFrame({m: tune.out_of_fold(m, best[m]["spatial"], cells, folds, ctx) for m in LABELS},
                       index=cells["h3"])
    out.round(1).to_csv(path)
    return out


def slog(x):
    """Signed log10(1 + |x|): keeps 10-trip and 100k-trip errors readable on one colour scale."""
    return np.sign(x) * np.log10(1 + np.abs(x))


def ticks(signed):
    vals = list(range(-5, 6)) if signed else list(range(6))
    names = ["0", "10", "100", "1k", "10k", "100k"]
    return vals, [("-" if v < 0 else "+" if signed and v > 0 else "") + names[abs(v)] for v in vals]


def layers(cells, pred):
    supply = np.select([(cells["weeks_served"] == 0) & (cells["poi_total"] > 0), cells["weeks_served"] == 0,
                        ~cells["always_served"].astype(bool)], [0, 1, 2], 3)
    step = [[i / 4 + d, c] for i, c in enumerate(SUPPLY_COLORS) for d in (0, 0.25)]
    out = [("Actual trips (30 days)", np.log10(1 + cells["trips"]), "Viridis", 0, TOP, *ticks(False))]
    out += [(f"Predicted: {LABELS[m]}", np.log10(1 + pred[m].clip(lower=0)), "Viridis", 0, TOP, *ticks(False))
            for m in LABELS]
    out += [(f"Error: {LABELS[m]} (red = over)", slog(pred[m].clip(lower=0) - cells["trips"]), get_colorscale("RdBu_r"), -TOP, TOP,
             *ticks(True)) for m in LABELS]
    out.append(("Service: weeks with any trip", supply, step, -0.5, 3.5, [0, 1, 2, 3], SUPPLY))
    return out


def figure(cfg, cells, pred):
    gj = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "id": c, "geometry": {"type": "Polygon", "coordinates": [
            [[round(x, 5), round(y, 5)] for x, y in h3.h3_to_geo_boundary(c, geo_json=True)]]}}
        for c in cells["h3"]]}
    err = pred.clip(lower=0).sub(cells["trips"].to_numpy(), axis=0)
    custom = np.column_stack([cells[["trips", "weeks_served", "poi_total"]].to_numpy(), pred.to_numpy(), err.to_numpy()])
    hover = ("<b>%{location}</b><br>Actual trips: %{customdata[0]:,}<br>Weeks with trips: %{customdata[1]} of 4"
             "<br>POIs: %{customdata[2]:,}<br><br>Predicted (error)<br>" + "<br>".join(
                 f"{LABELS[m]}: %{{customdata[{3 + i}]:,.0f}} (%{{customdata[{11 + i}]:+,.0f}})"
                 for i, m in enumerate(LABELS)) + "<extra></extra>")
    name, z, cs, lo, hi, tv, tt = layers(cells, pred)[0]
    fig = go.Figure(go.Choropleth(
        geojson=gj, locations=cells["h3"], featureidkey="id", z=z, colorscale=cs, zmin=lo, zmax=hi,
        marker_line_width=0, marker_opacity=0.85, customdata=custom, hovertemplate=hover,
        colorbar=dict(tickvals=tv, ticktext=tt, title=dict(text="trips"))))
    geom = data.load_boundary(cfg).geometry.union_all().simplify(0.001)
    lon, lat = [], []
    for poly in getattr(geom, "geoms", [geom]):
        x, y = poly.exterior.xy
        lon, lat = lon + list(x) + [None], lat + list(y) + [None]
    fig.add_trace(go.Scattergeo(lon=lon, lat=lat, mode="lines", line=dict(color="#555", width=1),
                                hoverinfo="skip", showlegend=False))
    buttons = [dict(label=n, method="restyle", args=[{
        "z": [z], "colorscale": [cs], "zmin": [lo], "zmax": [hi], "colorbar.tickvals": [tv],
        "colorbar.ticktext": [tt], "colorbar.title.text": ["" if n.startswith("Service") else "trips"]}, [0]])
        for n, z, cs, lo, hi, tv, tt in layers(cells, pred)]
    fig.update_layout(
        geo=dict(visible=False, projection_type="mercator", lonaxis_range=[113.7, 114.7], lataxis_range=[22.4, 22.9]),
        margin=dict(l=0, r=0, t=8, b=0), font=dict(family="system-ui, sans-serif", color=INK),
        paper_bgcolor=PAPER, geo_bgcolor=PAPER,
        updatemenus=[dict(buttons=buttons, x=0.01, y=0.99, xanchor="left", yanchor="top")])
    return fig


def page(cfg, cells, pred, config=None):
    """Header plus map, as body content. The page is a flex column, so the map fills what the header leaves."""
    plot = figure(cfg, cells, pred).to_html(
        full_html=False, include_plotlyjs=True, default_width="100%", default_height="100%", post_script=FIT,
        config=config or {"responsive": True, "displaylogo": False, "scrollZoom": True})
    # Plotly's geo map downloads a world outline even when hidden; an empty preloaded copy stops that.
    stub = '<script>window.PlotlyGeoAssets={topojson:{world_110m:{type:"Topology",objects:{},arcs:[]}}}</script>'
    return HEADER.format(trips=int(cells["trips"].sum()), cells=len(cells), ink=INK, paper=PAPER) + stub + plot + "</main>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="recompute out-of-fold predictions")
    cfg = config.load_config()
    cells = pd.read_parquet(cfg["data"]["processed_dir"] / "cells.parquet")
    pred = oof_predictions(cfg, cells, ap.parse_args().refresh).reset_index(drop=True)
    path = cfg["data"]["results_dir"] / "figures" / "demand_map.html"
    path.write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, '
                    f'initial-scale=1"><title>Shenzhen Bike Demand Map</title></head><body>{page(cfg, cells, pred)}</body></html>')
    print("wrote", path, f"{path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
