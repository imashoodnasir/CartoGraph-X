#!/usr/bin/env python3
"""CartoGraph-X Figure 2: template-matched technical plate using real GIS geometry.

Run from D:\Research\CartoGen:
  python scripts\cartographx_fig2_template_matched.py --city warsaw --dpi 600
  python scripts\cartographx_fig2_template_matched.py --city vilnius_central --dpi 600

Input: Script 124 scene GeoPackages, source_roads and optional buildings.
Output: editable SVG, PDF, 600-dpi PNG. No fabricated basemap or geometry.
"""
import argparse
from pathlib import Path
import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib import patheffects as pe

NAVY='#17324F'; BLUE='#1460B5'; ORANGE='#F08B17'; RED='#E23D36'; GREEN='#07854F'; PURPLE='#7542BA'; GRAY='#AAB3BD'; PALE='#F7F9FC'
CITIES=['vilnius_central','kaunas_central','klaipeda_central','warsaw','krakow','gdansk']

def root_find():
    here=Path(__file__).resolve()
    for d in [here.parent,*here.parents]:
        if (d/'data/processed/cross_country').exists():return d
    return Path.cwd()

def get_scene(root,city,explicit):
    if explicit:
        p=Path(explicit).resolve()
        if not p.is_file():raise FileNotFoundError(p)
        return p
    directory=root/'data/processed/cross_country/cartographx_v2_publication_map_scenes_v1/scenes'
    candidates=sorted(directory.glob(f'{city}__*.gpkg'))
    if not candidates:raise FileNotFoundError(f'No Script 124 scene for {city}: {directory}')
    token='lt_to_lt' if city.endswith('_central') else 'pl_to_pl'
    return next((p for p in candidates if token in p.name),candidates[0])

def read_scene(path):
    try:
        import pyogrio
        layers=set(pyogrio.list_layers(str(path))[:,0])
    except ImportError:
        import fiona
        layers=set(fiona.listlayers(path))
    if 'source_roads' not in layers:raise ValueError('source_roads layer missing')
    roads=gpd.read_file(path,layer='source_roads')
    roads=roads.loc[roads.geometry.notna() & ~roads.geometry.is_empty].copy()
    roads=roads.loc[roads.geom_type.isin(['LineString','MultiLineString'])].reset_index(drop=True)
    if roads.crs is None or roads.crs.is_geographic:raise ValueError('Projected metric source CRS required')
    buildings=gpd.read_file(path,layer='buildings') if 'buildings' in layers else None
    if buildings is not None and buildings.crs!=roads.crs:buildings=buildings.to_crs(roads.crs)
    return roads,buildings

def xy_of(g):
    c=g.geometry.centroid
    return np.column_stack([c.x.to_numpy(),c.y.to_numpy()])

def pick_region(roads,window=700,limit=175):
    xy=xy_of(roads);tree=cKDTree(xy)
    center=np.median(xy,axis=0)
    nearest=np.argsort(np.linalg.norm(xy-center,axis=1))[:min(100,len(xy))]
    anchor=xy[max(nearest,key=lambda i:len(tree.query_ball_point(xy[i],200)))]
    ds=np.linalg.norm(xy-anchor,axis=1)
    ids=np.flatnonzero(ds<window/2)
    if len(ids)<15:ids=np.argsort(ds)[:min(65,len(ds))]
    if len(ids)>limit:ids=ids[np.argsort(ds[ids])[:limit]]
    return roads.iloc[np.sort(ids)].reset_index(drop=True)

def graph(xy,k=6,r=250):
    tree=cKDTree(xy)
    ds,inds=tree.query(xy,k=min(k+1,len(xy)))
    if ds.ndim==1:ds=ds[:,None];inds=inds[:,None]
    edges=set()
    for i in range(len(xy)):
        for d,j in zip(ds[i],inds[i]):
            if i!=j and d<=r:edges.add(tuple(sorted((i,int(j)))))
    return sorted(edges),tree

def focus_node(xy,tree,k,r):
    center=np.median(xy,axis=0)
    order=np.argsort(np.linalg.norm(xy-center,axis=1))[:min(65,len(xy))]
    scored=[]
    for i in order:
        ds,js=tree.query(xy[i],k=min(k+1,len(xy)))
        neighbors=[(int(j),float(d)) for d,j in zip(np.atleast_1d(ds),np.atleast_1d(js)) if j!=i and d<=r][:k]
        if len(neighbors)<2:continue
        # Prefer a visually legible neighborhood with separated centroids, not a dense cluster.
        radii=[d for _,d in neighbors]
        vectors=np.array([xy[j]-xy[i] for j,_ in neighbors])
        angles=np.arctan2(vectors[:,1],vectors[:,0]);coverage=np.ptp(np.sort(angles))
        score=min(len(neighbors),k)*30+min(np.median(radii),140)+min(np.min(radii),70)*.8+coverage*12
        scored.append((score,int(i),neighbors))
    if not scored:raise ValueError('No eligible real proximity neighborhood')
    _,i,neighbors=max(scored,key=lambda t:t[0])
    return i,neighbors

def background(ax,roads,buildings=None,width=1.45):
    ax.set_facecolor('#F6F7F7')
    if buildings is not None and not buildings.empty:
        buildings.plot(ax=ax,color='#CBD0D0',edgecolor='none',alpha=.63,zorder=1)
    roads.plot(ax=ax,color=BLUE,linewidth=width,zorder=2)
    ax.set_xticks([]);ax.set_yticks([])
    for s in ax.spines.values():s.set_visible(False)
    ax.set_aspect('equal')

def limits(ax,bounds,pad=.06):
    x0,y0,x1,y1=bounds;dx=max(x1-x0,1);dy=max(y1-y0,1)
    ax.set_xlim(x0-pad*dx,x1+pad*dx);ax.set_ylim(y0-pad*dy,y1+pad*dy)

def panel_frame(fig,box,title,color):
    x,y,w,h=box
    fig.add_artist(FancyBboxPatch((x,y),w,h,transform=fig.transFigure,boxstyle='round,pad=0.002,rounding_size=.004',fc='none',ec=color,lw=1.35,ls='--',zorder=15))
    fig.add_artist(Rectangle((x,y+h-.044),w,.044,transform=fig.transFigure,facecolor=color,edgecolor='none',zorder=16))
    fig.text(x+.009,y+h-.022,title,ha='left',va='center',color='white',fontsize=12.0,fontweight='bold',zorder=18)

def inset(fig,box,margins=(.014,.018,.014,.065)):
    x,y,w,h=box;l,b,r,t=margins
    return fig.add_axes([x+l,y+b,w-l-r,h-b-t])

def scalebar(ax):
    x0,x1=ax.get_xlim();y0,y1=ax.get_ylim();span=x1-x0
    values=[10,20,50,100,200,500,1000]
    v=max([z for z in values if z<=span*.31],default=10)
    sx=x0+.075*span;sy=y0+.07*(y1-y0)
    ax.plot([sx,sx+v],[sy,sy],lw=2.7,color='black',zorder=30)
    ax.text(sx+v/2,sy+.026*(y1-y0),f'{v} m',ha='center',va='bottom',fontsize=8.5,fontweight='bold',color='black',zorder=31,bbox=dict(fc='white',ec='none',alpha=.85,pad=1))

def text_outline():return [pe.withStroke(linewidth=3,foreground='white')]

def plot_connections(ax,xy,edges,focus=None,neighbors=None,show_labels=False,base_color=ORANGE):
    for i,j in edges:
        ax.plot([xy[i,0],xy[j,0]],[xy[i,1],xy[j,1]],color=base_color,lw=1.1,alpha=.62,zorder=5)
    ax.scatter(xy[:,0],xy[:,1],s=20,c='#90B5D9',edgecolors=NAVY,linewidths=.45,zorder=6)
    if focus is not None:
        ax.scatter(*xy[focus],s=125,c=RED,edgecolors='white',lw=1.5,zorder=11)
        if neighbors:
            for j,d in neighbors:
                ax.plot([xy[focus,0],xy[j,0]],[xy[focus,1],xy[j,1]],color=ORANGE,lw=2.0,zorder=9)
                ax.scatter(*xy[j],s=75,c=ORANGE,edgecolors='white',lw=1,zorder=10)

def label_neighbors(ax,xy,focus,neighbors,distances=True):
    cx,cy=xy[focus]
    ax.scatter(cx,cy,s=165,c=RED,edgecolors='white',linewidths=1.7,zorder=21)
    ax.annotate('s*',xy=(cx,cy),xytext=(11,-18),textcoords='offset points',color=RED,fontsize=11,fontweight='bold',zorder=22,path_effects=text_outline())
    for rank,(j,d) in enumerate(neighbors,1):
        xx,yy=xy[j]
        ax.plot([cx,xx],[cy,yy],color=ORANGE,lw=2.0,zorder=17)
        ax.scatter(xx,yy,s=95,c='#FFB22B',edgecolors='white',linewidths=1.2,zorder=18)
        v=np.array([xx-cx,yy-cy]);norm=np.linalg.norm(v)
        direction=v/max(norm,1e-9)
        ax.annotate(f's{rank}',xy=(xx,yy),xytext=tuple(13*direction+np.array([3,3])),textcoords='offset points',fontsize=9.5,color=NAVY,fontweight='bold',zorder=22,path_effects=text_outline())
        if distances:
            pos=np.array([cx,cy])+.63*v
            normal=np.array([-direction[1],direction[0]])
            pos+=normal*16*((rank%2)*2-1)
            ax.text(*pos,f'{d:.0f} m',ha='center',fontsize=8.6,fontweight='bold',color=NAVY,zorder=23,bbox=dict(fc='white',ec='none',alpha=.9,pad=.8))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--city',choices=CITIES,default='warsaw');ap.add_argument('--gpkg')
    ap.add_argument('--root',type=Path);ap.add_argument('--out-dir',type=Path)
    ap.add_argument('--dpi',type=int,default=600);ap.add_argument('--k',type=int,default=6)
    ap.add_argument('--radius',type=float,default=250);ap.add_argument('--window',type=float,default=700)
    args=ap.parse_args();root=args.root.resolve() if args.root else root_find()
    scene=get_scene(root,args.city,args.gpkg);roads,buildings=read_scene(scene)
    local=pick_region(roads,args.window);xy=xy_of(local);edges,tree=graph(xy,args.k,args.radius)
    focus,neighbors=focus_node(xy,tree,args.k,args.radius)
    print('REAL SCENE:',scene,'\nSOURCE ROADS:',len(roads),'LOCAL ROADS:',len(local),'EDGES:',len(edges),'FOCUS:',focus,'NEIGHBORS:',len(neighbors),flush=True)
    lb=None
    if buildings is not None:
        x0,y0,x1,y1=local.total_bounds
        lb=buildings.cx[x0-args.radius:x1+args.radius,y0-args.radius:y1+args.radius]
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(23.0,11.5),facecolor='white')
    # Faithful template geometry: 4 panels on top, 2 below, no notes.
    top_y=.543;top_h=.413;gap=.008;left=.021;top_w=(.958-3*gap)/4
    boxes=[(left+i*(top_w+gap),top_y,top_w,top_h) for i in range(4)]
    lower1=(.021,.080,.260,.440);lower2=(.291,.080,.688,.440)
    titles=['(a) Input road network (source geometry)','(b) Zoomed road segments and centroids',f'(c) Graph construction (kNN ≤ {args.radius:g} m)','(d) Spatial neighborhood example']
    for box,title,color in zip(boxes,titles,[BLUE,BLUE,GREEN,ORANGE]):panel_frame(fig,box,title,color)
    panel_frame(fig,lower1,'(e) Edge construction procedure',PURPLE)
    panel_frame(fig,lower2,'(f) Distance-based neighborhood illustration',PURPLE)
    A,B,C,D=[inset(fig,b) for b in boxes]
    E=inset(fig,lower1,(.010,.018,.010,.063))
    F=inset(fig,lower2,(.012,.022,.018,.064))
    # (a) full real source scene and local window.
    background(A,roads,buildings,1.15);limits(A,roads.total_bounds)
    x0,y0,x1,y1=local.total_bounds
    A.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fc='none',ec=RED,ls='--',lw=1.9,zorder=20))
    A.text(.03,.95,'N ↑',transform=A.transAxes,fontsize=12,fontweight='bold',va='top',color=NAVY,zorder=21)
    scalebar(A)
    # (b) real zoomed roads with true segment centroids.
    background(B,local,lb,2.0);limits(B,local.total_bounds,.07)
    B.scatter(xy[:,0],xy[:,1],s=19,c='white',edgecolors=NAVY,lw=.7,zorder=7)
    for rank,(j,d) in enumerate(neighbors[:5],1):
        B.annotate(f's{rank}',xy[j],xytext=(5,6),textcoords='offset points',fontsize=8.5,fontweight='bold',color=NAVY,zorder=9,path_effects=text_outline())
    B.scatter(*xy[focus],s=80,c=RED,edgecolors='white',zorder=10)
    scalebar(B)
    # (c) graph only, with real centroid positions and true kNN edges.
    C.set_facecolor('#FAFDFC');plot_connections(C,xy,edges,focus,neighbors)
    limits(C,local.total_bounds,.07);C.set_aspect('equal');C.set_xticks([]);C.set_yticks([])
    for s in C.spines.values():s.set_visible(False)
    scalebar(C)
    # (d) local road view + highlighted 1-hop neighbors; no cluttered distance labels.
    background(D,local,lb,2.0)
    cx,cy=xy[focus];radius=args.radius
    D.set_xlim(cx-radius*.95,cx+radius*.95);D.set_ylim(cy-radius*.95,cy+radius*.95)
    for j,d in neighbors:
        D.plot([cx,xy[j,0]],[cy,xy[j,1]],c=ORANGE,lw=2.3,zorder=10)
        D.scatter(*xy[j],s=75,c='#FFB22B',edgecolors='white',lw=1,zorder=11)
    D.scatter(cx,cy,s=130,c=RED,edgecolors='white',lw=1.6,zorder=12)
    for rank,(j,d) in enumerate(neighbors,1):
        D.annotate(f'{d:.0f} m',xy=(cx+(xy[j,0]-cx)*.58,cy+(xy[j,1]-cy)*.58),xytext=(0,0),textcoords='offset points',fontsize=7.8,ha='center',fontweight='bold',color=NAVY,zorder=15,bbox=dict(fc='white',ec='none',alpha=.9,pad=.7))
    scalebar(D)
    # (e) compact 4-step vertical flowchart.
    E.set_xlim(0,1);E.set_ylim(0,1);E.axis('off')
    steps=[('1. Compute segment centroids','For each road segment sᵢ, compute cᵢ.'),('2. Find spatial neighbors',f'Select k={args.k} nearest centroids within {args.radius:g} m.'),('3. Create undirected edges','Add (i,j) if either centroid selects the other.'),('4. Attach node attributes','Combine road geometry and graph-context features.')]
    for index,(head,body) in enumerate(steps):
        y=.765-index*.245
        E.add_patch(FancyBboxPatch((.025,y),.95,.192,transform=E.transAxes,boxstyle='round,pad=.008,rounding_size=.016',fc='#EAF2FD',ec='#8BAFDE',lw=1.2))
        E.text(.060,y+.132,head,transform=E.transAxes,fontsize=11,fontweight='bold',color=NAVY)
        E.text(.060,y+.060,body,transform=E.transAxes,fontsize=9,color='#405A73',wrap=True)
        if index<3:E.annotate('',xy=(.5,y-.038),xytext=(.5,y-.004),xycoords=E.transAxes,textcoords=E.transAxes,arrowprops=dict(arrowstyle='-|>',lw=1.6,color=BLUE))
    # (f) larger real map with a 250 m circle, separate right-side legend.
    F.set_facecolor('#F7F9FA');F.set_xticks([]);F.set_yticks([])
    Fmap=F.inset_axes([.008,.015,.715,.97]);background(Fmap,local,lb,1.8)
    Fmap.set_xlim(cx-radius*1.28,cx+radius*1.28)
    Fmap.set_ylim(cy-radius*1.12,cy+radius*1.12)
    Fmap.add_patch(Circle((cx,cy),radius,facecolor='#FADDDD',edgecolor='none',alpha=.30,zorder=4))
    Fmap.add_patch(Circle((cx,cy),radius,fill=False,ec=RED,ls='--',lw=1.6,zorder=7))
    label_neighbors(Fmap,xy,focus,neighbors,True);scalebar(Fmap)
    # Right legend kept within (f), as in the template.
    F.add_patch(FancyBboxPatch((.74,.08),.245,.84,transform=F.transAxes,boxstyle='round,pad=.01,rounding_size=.014',fc='white',ec='#CDD6E1',lw=1.1))
    legend=[(RED,'Target segment centroid'),(ORANGE,'One-hop neighbor'),(BLUE,'Real road geometry'),('#6F7E8C','Other source segments')]
    for idx,(color,label) in enumerate(legend):
        yy=.83-idx*.14
        F.scatter([.775],[yy],s=90,c=color,edgecolors='white',transform=F.transAxes,zorder=22)
        F.text(.814,yy,label,fontsize=9.2,va='center',color=NAVY,transform=F.transAxes)
    F.plot([.76,.80],[.245,.245],color=ORANGE,lw=2.5,transform=F.transAxes)
    F.text(.814,.245,'Proximity edge',fontsize=9.2,va='center',transform=F.transAxes,color=NAVY)
    F.plot([.76,.80],[.14,.14],color=RED,lw=1.7,ls='--',transform=F.transAxes)
    F.text(.814,.14,f'{radius:g} m search radius',fontsize=9.2,va='center',transform=F.transAxes,color=NAVY)
    fig.text(.5,.042,'All roads, buildings, centroid positions, and distances derive from the real GIS scene. Proximity edges do not imply physical road connectivity.',ha='center',fontsize=10,color='#52667B')
    output=args.out_dir or root/'data/processed/cross_country/cartographx_v2_methodology_figure_v1'
    output.mkdir(parents=True,exist_ok=True)
    stem=output/f'Fig02_TemplateMatched_RealData_{args.city}'
    for ext in ['pdf','svg','png']:
        path=stem.with_suffix('.'+ext)
        fig.savefig(path,bbox_inches='tight',pad_inches=.09,facecolor='white',dpi=args.dpi if ext=='png' else None)
        print('SAVED:',path,flush=True)
    plt.close(fig);print('FIGURE 2 TEMPLATE-MATCHED PASS',flush=True)
if __name__=='__main__':main()
