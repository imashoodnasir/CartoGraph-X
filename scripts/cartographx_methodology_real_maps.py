#!/usr/bin/env python3
"""CartoGraph-X publication methodology diagram with REAL GIS map layers.

Reads existing Script 124 scene GeoPackages (never generates fictitious maps).
No model training or inference. Outputs SVG, PDF, and high-resolution PNG.

From D:\Research\CartoGen:
 python scripts\cartographx_methodology_real_maps.py
 python scripts\cartographx_methodology_real_maps.py --city vilnius_central --dpi 600
 python scripts\cartographx_methodology_real_maps.py --scene "path\to\scene.gpkg"

Requires: matplotlib, geopandas, numpy, shapely (pyogrio or fiona).
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import warnings

import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib import patheffects

INK="#1A2B40"; MUTED="#506278"; BLUE="#1763B0"; GREEN="#078257"
ORANGE="#D77B19"; PURPLE="#7546B3"; TEAL="#067F98"; RED="#CE453C"
LIGHT="#F2F5F8"; ROAD="#AAB4BD"; REF="#12548B"; BUILD="#C7C8C2"
BG="#FFFFFF"; SOFT_BLUE="#EAF2FD"; SOFT_GREEN="#E8F5EF"
SOFT_ORANGE="#FFF3E4"; SOFT_PURPLE="#F2EBFC"; SOFT_TEAL="#EAF7FA"
CITY_NAMES={"vilnius_central":"Vilnius","kaunas_central":"Kaunas",
            "klaipeda_central":"Klaipėda","warsaw":"Warsaw",
            "krakow":"Kraków","gdansk":"Gdańsk"}

def root_path():
    here=Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate/"data"/"processed"/"cross_country").is_dir():
            return candidate
    return here.parent.parent

def get_layers(path):
    try:
        import pyogrio
        return set(pyogrio.list_layers(str(path))[:,0])
    except ImportError:
        import fiona
        return set(fiona.listlayers(str(path)))

def choose_scene(args, root):
    if args.scene:
        path=Path(args.scene).resolve()
        if not path.is_file(): raise FileNotFoundError(path)
        return path
    scene_root=root/"data"/"processed"/"cross_country"/"cartographx_v2_publication_map_scenes_v1"/"scenes"
    if not scene_root.is_dir():
        raise FileNotFoundError(
            f"Real GIS scenes not found: {scene_root}\n"
            "First run Script 124 to prepare the georeferenced source/reference/"
            "prediction scene GeoPackages, or specify --scene PATH.")
    city=args.city
    preferred=f"{city}__{'joint_to_lt' if city in CITY_NAMES and city in ['vilnius_central','kaunas_central','klaipeda_central'] else 'joint_to_pl'}__seed42.gpkg"
    candidates=sorted(scene_root.glob(f"{city}__*__seed{args.seed}.gpkg"))
    if not candidates:
        raise FileNotFoundError(f"No real scene for {city}, seed {args.seed} in {scene_root}")
    for c in candidates:
        if c.name==preferred:return c
    for c in candidates:
        if "lt_to_lt" in c.name or "pl_to_pl" in c.name:return c
    return candidates[0]

def read_real(path):
    layers=get_layers(path)
    mandatory={"source_roads","reference_target_roads","predicted_cartographx_roads"}
    missing=mandatory-layers
    if missing:
        raise ValueError(f"Scene {path.name} is missing real GIS layers: {sorted(missing)}")
    names=["source_roads","reference_target_roads","predicted_cartographx_roads",
           "test_road_predictions","buildings"]
    d={name:(gpd.read_file(path,layer=name) if name in layers else None) for name in names}
    source=d["source_roads"]
    if source.empty:raise ValueError("The real source road layer is empty")
    if source.crs is None:raise ValueError("Source layer has no CRS; cannot create a metric scale bar")
    for k,v in d.items():
        if v is not None and v.crs is not None and v.crs!=source.crs:
            d[k]=v.to_crs(source.crs)
    if source.crs.is_geographic:
        raise ValueError("Scene must use a projected metric CRS (Script 124 scenes do).")
    return d

def roundrect(ax,x,y,w,h,fc="white",ec="#D6DFE8",r=.014,lw=1,z=1):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0.003,rounding_size={r}",
                     transform=ax.transAxes,fc=fc,ec=ec,lw=lw,zorder=z,clip_on=False)
    ax.add_patch(p)
    return p

def text(ax,x,y,s,size=10,weight="normal",color=INK,ha="left",va="center",**kwargs):
    return ax.text(x,y,s,transform=ax.transAxes,fontsize=size,fontweight=weight,
                   color=color,ha=ha,va=va,zorder=9,**kwargs)

def arrow(ax,x1,y1,x2,y2,color=INK,lw=1.6,style="-|>",scale=13):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),transform=ax.transAxes,
                 arrowstyle=style,mutation_scale=scale,color=color,lw=lw,zorder=12,
                 connectionstyle="arc3,rad=0"))

def panel(fig,box,title_str,color,number,fill):
    ax=fig.add_axes(box);ax.set_axis_off();ax.set_xlim(0,1);ax.set_ylim(0,1)
    roundrect(ax,.006,.012,.988,.977,fc=fill,ec=color,r=.019,lw=1.2)
    roundrect(ax,.006,.858,.988,.13,fc=color,ec=color,r=.012,lw=0)
    text(ax,.028,.922,f"{number}. {title_str}",size=12.8,weight="bold",color="white")
    return ax

def map_inset(fig, parent_box, rect, data, layers, title_str, road_color,
              pred_only=False, label=None):
    px,py,pw,ph=parent_box
    x,y,w,h=rect
    ax=fig.add_axes([px+x*pw,py+y*ph,w*pw,h*ph],zorder=4)
    ax.set_facecolor("#FAFAF8")
    src=data["source_roads"]
    bbox=src.total_bounds
    minx,miny,maxx,maxy=map(float,bbox)
    dx=max(maxx-minx,1);dy=max(maxy-miny,1)
    padx=dx*.055;pady=dy*.055
    ax.set_xlim(minx-padx,maxx+padx);ax.set_ylim(miny-pady,maxy+pady)
    ax.set_aspect("equal",adjustable="box")
    if data.get("buildings") is not None and not data["buildings"].empty:
        try:
            data["buildings"].plot(ax=ax,color=BUILD,edgecolor="none",alpha=.40,zorder=1)
        except Exception as e:
            warnings.warn(f"Building rendering skipped: {e}")
    if pred_only:
        src.plot(ax=ax,color="#D3D9DE",linewidth=.75,alpha=.55,zorder=2)
    for layer,color,lw,alpha in layers:
        g=data.get(layer)
        if g is not None and not g.empty:
            g.plot(ax=ax,color=color,linewidth=lw,alpha=alpha,zorder=3)
    for spine in ax.spines.values():
        spine.set_color("#CAD3DC");spine.set_linewidth(.8)
    ax.set_xticks([]);ax.set_yticks([])
    # Metric scale bar: CRS is projected, expected EPSG:3346 or EPSG:2180 in meters.
    crs=src.crs
    units=str(crs.axis_info[0].unit_name).lower() if crs.axis_info else ""
    if "metre" in units or "meter" in units:
        span=maxx-minx
        choices=[10,20,50,100,200,500,1000,2000]
        length=max([v for v in choices if v<=span*.29],default=10)
        sx=minx+dx*.065;sy=miny+dy*.085
        ax.plot([sx,sx+length],[sy,sy],color=INK,lw=2.0,zorder=15,solid_capstyle="butt")
        ax.plot([sx,sx],[sy-dy*.011,sy+dy*.011],color=INK,lw=1,zorder=15)
        ax.plot([sx+length,sx+length],[sy-dy*.011,sy+dy*.011],color=INK,lw=1,zorder=15)
        ax.text(sx+length/2,sy+dy*.025,f"{length} m",ha="center",va="bottom",
                fontsize=6.7,color=INK,zorder=15,
                path_effects=[patheffects.withStroke(linewidth=2.5,foreground="white")])
    ax.text(.025,.965,"N ↑",transform=ax.transAxes,fontsize=7,weight="bold",
            va="top",color=INK,bbox=dict(facecolor="white",edgecolor="none",alpha=.8,pad=1))
    # Title outside the map's data area, via a figure label.
    fig.text(px+(x+w/2)*pw,py+(y+h+.027)*ph,title_str,fontsize=8.5,
             ha="center",va="bottom",fontweight="bold",color=INK)
    return ax

def graph_art(ax):
    # Schematic proximity graph: NOT road topology, deliberately labeled as such.
    coords=np.array([[.12,.29],[.23,.57],[.29,.22],[.41,.68],[.46,.39],
                     [.59,.59],[.67,.26],[.75,.69],[.85,.40],[.88,.17]])
    edges=[(0,1),(0,2),(1,3),(1,4),(2,4),(3,4),(3,5),(4,5),
           (4,6),(5,7),(5,8),(6,8),(6,9),(7,8),(8,9),(2,6)]
    for i,j in edges:
        a,b=coords[i],coords[j]
        ax.plot([a[0],b[0]],[.19+a[1]*.58,.19+b[1]*.58],color="#778A9A",lw=1.2,
                transform=ax.transAxes,zorder=3)
    for i,(x,y) in enumerate(coords):
        ax.add_patch(Circle((x,.19+y*.58),.024,transform=ax.transAxes,
                    fc=ORANGE if i in (1,4,5,8) else "#70A9E2",
                    ec=INK,lw=.8,zorder=5))
    text(ax,.50,.14,"kNN proximity, k = 6; distance ≤ 250 m",
         size=8.7,ha="center",weight="medium")
    text(ax,.50,.065,"Edges do not assert physical junctions",size=7.4,ha="center",color=MUTED)

def feature_column(ax,x,y,w,h,heading,items,accent):
    roundrect(ax,x,y,w,h,fc="white",ec="#D8E0E8",r=.012)
    text(ax,x+.035,y+h-.10,heading,size=10,weight="bold",color=accent)
    n=len(items)
    top=y+h-.21
    for i,item in enumerate(items):
        yy=top-i*(h-.28)/max(n-1,1)
        roundrect(ax,x+.03,yy-.039,w-.06,.077,fc="#F5F7FA",ec="#E5EAF0",r=.009,lw=.6)
        text(ax,x+.054,yy,item,size=8.5)

def boxlabel(ax,x,y,w,h,main,sub=None,fill="white",edge="#B6C8D8",font=10):
    roundrect(ax,x,y,w,h,fc=fill,ec=edge,r=.012,lw=1.0)
    text(ax,x+w/2,y+h*(.60 if sub else .5),main,size=font,weight="bold",ha="center")
    if sub:text(ax,x+w/2,y+h*.29,sub,size=8,ha="center",color=MUTED)

def draw_model(ax):
    # Deliberately high-detail modular architecture.
    boxlabel(ax,.02,.27,.115,.50,"Road features","4 geometric",SOFT_ORANGE,ORANGE,9.2)
    boxlabel(ax,.02,.06,.115,.16,"Graph context","19 features",SOFT_PURPLE,PURPLE,8.7)
    boxlabel(ax,.185,.42,.145,.34,"Road feature encoder","MLP",SOFT_ORANGE,ORANGE,9.1)
    boxlabel(ax,.185,.06,.145,.28,"Residual GraphSAGE","2 message-passing layers",SOFT_BLUE,BLUE,9)
    boxlabel(ax,.380,.19,.155,.49,"Gated feature fusion",
             "z = g ⊙ h_graph + (1−g) ⊙ h_road",SOFT_GREEN,GREEN,9.0)
    boxlabel(ax,.585,.28,.145,.31,"Shared classifier","MLP + sigmoid",SOFT_BLUE,BLUE,9.5)
    boxlabel(ax,.785,.49,.183,.20,"Lithuania head","joint training",SOFT_PURPLE,PURPLE,9.4)
    boxlabel(ax,.785,.20,.183,.20,"Poland head","joint training",SOFT_PURPLE,PURPLE,9.4)
    arrow(ax,.135,.56,.183,.56,color=ORANGE)
    arrow(ax,.135,.14,.183,.18,color=PURPLE)
    arrow(ax,.33,.57,.378,.57,color=ORANGE)
    arrow(ax,.33,.20,.378,.30,color=BLUE)
    arrow(ax,.535,.435,.583,.435,color=GREEN)
    arrow(ax,.73,.44,.782,.59,color=BLUE)
    arrow(ax,.73,.41,.782,.30,color=BLUE)
    text(ax,.50,.82,"Two complementary feature branches",ha="center",size=9.5,weight="bold")
    text(ax,.50,.05,"Binary retention probability  pᵢ = P(yᵢ = 1 | road i, spatial context)",
         ha="center",size=9.3,color=INK)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene",help="Explicit Script 124 real scene GeoPackage")
    ap.add_argument("--city",choices=list(CITY_NAMES),default="vilnius_central")
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--root",type=Path,default=None)
    ap.add_argument("--out-dir",type=Path,default=None)
    ap.add_argument("--dpi",type=int,default=600)
    args=ap.parse_args()
    root=args.root.resolve() if args.root else root_path()
    path=choose_scene(args,root)
    print(f"REAL MAP INPUT: {path}",flush=True)
    data=read_real(path)
    print("LAYERS:",{k:(None if v is None else len(v)) for k,v in data.items()},flush=True)
    name=path.name.split("__")[0]
    city=CITY_NAMES.get(name,name.replace("_"," ").title())
    out=args.out_dir or root/"data"/"processed"/"cross_country"/"cartographx_v2_methodology_figure_v1"
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,
        "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none",
        "savefig.facecolor":"white"})
    fig=plt.figure(figsize=(23.5,14.0),facecolor="white")
    fig.text(.027,.976,"CartoGraph-X  |  Context-Aware Road Selection for Cartographic Generalization",
             fontsize=20,fontweight="bold",color=INK,va="top")
    fig.text(.027,.948,f"Real georeferenced road scenes: {city}  ·  Proximity graph  ·  Dual-feature encoding  ·  Country-conditioned prediction",
             fontsize=10.5,color=MUTED,va="top")
    # top: input + graph + features
    A=(.027,.633,.462,.280);B=(.500,.633,.211,.280);C=(.722,.633,.251,.280)
    a=panel(fig,A,"Input data and road correspondence",BLUE,1,SOFT_BLUE)
    b=panel(fig,B,"Road graph construction",GREEN,2,SOFT_GREEN)
    c=panel(fig,C,"Feature extraction",ORANGE,3,SOFT_ORANGE)
    map_inset(fig,A,(.027,.17,.298,.58),data,[("source_roads",ROAD,1.05,.95)],
              "Detailed source roads",ROAD)
    map_inset(fig,A,(.351,.17,.298,.58),data,[("reference_target_roads",REF,1.9,1.0)],
              "Generalized target",REF)
    # Show actual provisional source-road correspondence from test_road_predictions
    labels=data.get("test_road_predictions")
    if labels is not None and not labels.empty and "label" in labels.columns:
        l1=labels.loc[labels["label"]==1].copy()
        l0=labels.loc[labels["label"]==0].copy()
        data2={**data,"label_yes":l1,"label_no":l0}
        lyrs=[("label_no","#C65D56",1.15,.90),("label_yes","#119368",1.25,.95)]
        label_caption="Matched (green) / unmatched (red)"
    else:
        data2=data
        lyrs=[("source_roads",ROAD,.8,.7),
              ("predicted_cartographx_roads",RED,1.5,1)]
        label_caption="Predicted retention (red)"
    map_inset(fig,A,(.676,.17,.298,.58),data2,lyrs,
              "Source segment labels" if labels is not None else "Source segment selection",RED)
    text(a,.50,.065,"Geometric matching: search 15 m  ·  buffer 5 m  ·  source ≥ 0.20  ·  target ≥ 0.03",
         size=8.1,ha="center",color=MUTED)
    graph_art(b)
    feature_column(c,.035,.14,.445,.66,"Road-level geometry",
                   ["Length (m)","Intersections","Road count (150 m)","Mean neighbor distance"],ORANGE)
    feature_column(c,.518,.14,.445,.66,"Graph context",
                   ["Spatial neighborhood","Graph descriptors","Local road structure","Contextual statistics"],PURPLE)
    # center architecture
    D=(.027,.310,.946,.298)
    d=panel(fig,D,"CartoGraph-X model architecture",PURPLE,4,SOFT_PURPLE)
    draw_model(d)
    # bottom output and evaluation
    E=(.027,.065,.468,.217);F=(.506,.065,.467,.217)
    e=panel(fig,E,"Real road-selection output",BLUE,5,SOFT_BLUE)
    f=panel(fig,F,"Experimental evaluation and efficiency",TEAL,6,SOFT_TEAL)
    map_inset(fig,E,(.025,.15,.36,.62),data,[("reference_target_roads",REF,1.65,.95)],
              "Generalized reference",REF)
    map_inset(fig,E,(.410,.15,.36,.62),data,[("predicted_cartographx_roads",RED,1.65,.98)],
              "CartoGraph-X retained roads",RED,pred_only=True)
    text(e,.798,.70,"Source geometries",size=9.1,weight="bold")
    text(e,.798,.55,"selected by the model",size=8.5)
    text(e,.798,.37,"Not geometric",size=8.5,color=MUTED)
    text(e,.798,.25,"simplification",size=8.5,color=MUTED)
    boxlabel(f,.025,.17,.285,.61,"Six evaluation protocols",
             "LT, PL, zero-shot, joint",SOFT_BLUE,BLUE,9.7)
    boxlabel(f,.354,.17,.285,.61,"Predictive performance",
             "MCC · BA · ROC/PR-AUC",SOFT_GREEN,GREEN,9.5)
    boxlabel(f,.683,.17,.285,.61,"Computational efficiency",
             "10 seeds · time · GPU · FLOPs",SOFT_ORANGE,ORANGE,9.2)
    fig.text(.028,.027,"GIS maps are rendered from Script 124 georeferenced scene layers; no synthetic roads or invented predictions.",
             fontsize=9,color=MUTED)
    fig.text(.973,.027,"C2 matching labels provisional  ·  Geographic test findings exploratory",
             fontsize=9,color=MUTED,ha="right")
    stem=out/f"CartoGraphX_Methodology_RealMaps_{name}"
    for fmt in ("pdf","svg","png"):
        target=stem.with_suffix("."+fmt)
        fig.savefig(target,dpi=args.dpi if fmt=="png" else None,
                    bbox_inches="tight",pad_inches=.12,facecolor="white")
        print("SAVED:",target,flush=True)
    plt.close(fig)
    print("METHODOLOGY FIGURE PASS",flush=True)

if __name__=="__main__":
    main()
