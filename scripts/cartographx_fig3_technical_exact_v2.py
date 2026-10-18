#!/usr/bin/env python3
"""High-detail Figure 3: implementation-faithful CartoGraph-X architecture.
Based on frozen scripts 111 and 112, NOT the earlier Script 71 architecture.
No GIS needed. Optional --verify imports actual frozen code and checks layer dimensions.
Exports vector PDF/SVG and 600-dpi PNG.

python scripts/cartographx_fig3_technical_exact_v2.py --verify --dpi 600
"""
import argparse, importlib.util, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

NAVY="#1E3249"; BLUE="#1767B3"; PURPLE="#7651B2"; GREEN="#11845B"
ORANGE="#D17A1C"; RED="#C8464C"; GRAY="#52667B"
PB="#EAF3FD"; PP="#F1EDFA"; PG="#E9F6EE"; PO="#FFF2E4"; PR="#FCEFF0"
plt.rcParams.update({"font.family":"DejaVu Sans","mathtext.fontset":"dejavusans",
                     "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})

def card(ax,xywh,title,body="",fc="white",edge="#C6D2DD",title_fs=10.4,body_fs=8.5):
    x,y,w,h=xywh
    patch=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.007,rounding_size=0.017",
                         fc=fc,ec=edge,lw=1.3,zorder=3)
    ax.add_patch(patch)
    ax.text(x+w/2,y+h*(.67 if body else .5),title,ha="center",va="center",
            fontsize=title_fs,color=NAVY,fontweight="bold",zorder=5)
    if body:
        ax.text(x+w/2,y+h*.33,body,ha="center",va="center",fontsize=body_fs,
                color=GRAY,linespacing=1.25,zorder=5)
    return patch

def arr(ax,a,b,c=NAVY,lw=1.65):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=12,
                 lw=lw,color=c,zorder=6,connectionstyle="arc3,rad=0"))

def section(fig,rect,title,accent,sub=None):
    ax=fig.add_axes(rect)
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis("off")
    ax.add_patch(FancyBboxPatch((.005,.005),.99,.99,boxstyle="round,pad=.003,rounding_size=.018",
                 fc="white",ec=accent,lw=1.5))
    ax.add_patch(FancyBboxPatch((.006,.889),.988,.102,
                 boxstyle="round,pad=.001,rounding_size=.014",fc=accent,ec=accent,lw=0))
    ax.text(.026,.94,title,color="white",fontsize=12.5,fontweight="bold",va="center")
    if sub:ax.text(.027,.857,sub,color=GRAY,fontsize=9.2,va="center")
    return ax

def mini_graph(ax,cx,cy,scale=1,highlight=True):
    pts=[(-.07,.02),(-.04,.11),(.05,.105),(.09,.02),(.02,-.07),(-.08,-.07)]
    edges=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(0,4),(0,2),(2,4)]
    for i,j in edges:
        x1,y1=pts[i];x2,y2=pts[j]
        ax.plot([cx+x1*scale,cx+x2*scale],[cy+y1*scale,cy+y2*scale],
                color="#9CAFC1",lw=1.6,zorder=4)
    for i,(x,y) in enumerate(pts):
        ax.add_patch(Circle((cx+x*scale,cy+y*scale),.012*scale,
                            fc=ORANGE if i==2 and highlight else BLUE,
                            ec="white",lw=.8,zorder=5))

def verify(root):
    path=root/"scripts"/"112_train_cartographx_v2_architecture.py"
    if not path.exists():raise FileNotFoundError(path)
    spec=importlib.util.spec_from_file_location("fig3_exact112",path)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    import torch
    import pandas as pd
    n=mod.base.FEATURES.__len__()
    single=mod.ResidualSAGE(n,64,.2,False)
    joint=mod.ResidualSAGE(n,64,.2,True)
    assert single.project[0].in_features==n
    assert single.project[0].out_features==64
    assert single.gate.in_features==128 and single.gate.out_features==64
    assert single.sage1.kind=="GraphSAGE" and single.sage2.kind=="GraphSAGE"
    assert joint.country_delta.num_embeddings==2
    assert joint.country_delta.embedding_dim==64
    assert joint.country_bias.embedding_dim==1
    print("VERIFIED: frozen Script 112 architecture")
    print("FEATURES:",n,"source params:",sum(x.numel() for x in single.parameters()),
          "joint params:",sum(x.numel() for x in joint.parameters()))

def render(out,dpi):
    fig=plt.figure(figsize=(25,17),facecolor="white")
    fig.text(.5,.982,"CartoGraph-X | Detailed Residual GraphSAGE and Country-Conditioned Road Selection",
             ha="center",va="top",fontsize=21,fontweight="bold",color=NAVY)
    fig.text(.5,.956,"Exact computational operators, feature dimensions, message passing, residual pathways, gated fusion, and logit formulation",
             ha="center",fontsize=11,color=GRAY)
    # TOP ROW: data + graph message aggregation + residual stack
    a=section(fig,[.025,.655,.282,.273],"(a) Road-node input and projection",BLUE,
              "Source road segments; induced split-specific proximity graph")
    card(a,(.055,.60,.39,.18),"Node features",
         r"$X\in\mathbb{R}^{N\times F}$"+"\n23 input features",PB,BLUE)
    card(a,(.545,.60,.40,.18),"Proximity edges",
         r"$E\subseteq V\times V$"+"\nk = 6; radius = 250 m",PB,BLUE)
    arr(a,(.445,.68),(.54,.68),BLUE)
    card(a,(.16,.33,.68,.18),"Train-fitted preprocessing",
         "Median imputation → StandardScaler\nNo test-fit preprocessing",PO,ORANGE)
    arr(a,(.25,.598),(.34,.516),BLUE)
    card(a,(.16,.085,.68,.18),"Input projection",
         r"$z=\mathrm{ReLU}(\mathrm{LN}(W_pX+b_p))$"+"\nLinear(F, 64) → LayerNorm(64) → ReLU",PB,BLUE)
    arr(a,(.50,.326),(.50,.269),BLUE)

    b=section(fig,[.32,.655,.327,.273],"(b) Exact GraphSAGE message operator",PURPLE,
              "Both layers call Script 111's Layer(..., 'GraphSAGE')")
    mini_graph(b,.18,.52,1.1)
    card(b,(.35,.62,.60,.15),"Neighbor mean",
         r"$m_i=\sum_{j:(j,i)\in E}h_j/\max(1,d_i)$",PP,PURPLE,10.5,10)
    card(b,(.35,.39,.60,.15),"Self + neighbor concatenation",
         r"$q_i=[h_i\,\Vert\,m_i]\in\mathbb{R}^{128}$",PP,PURPLE,10.5,10)
    card(b,(.35,.16,.60,.15),"Learned combine",
         r"$\mathrm{SAGE}(h,E)_i=W_cq_i+b_c$",PP,PURPLE,10.5,10)
    arr(b,(.65,.618),(.65,.544),PURPLE)
    arr(b,(.65,.386),(.65,.314),PURPLE)
    b.text(.055,.13,"Directed edge list\n(src, dst)\nindex_add + bincount",
           fontsize=8.7,color=GRAY,ha="left",linespacing=1.35)

    c=section(fig,[.66,.655,.315,.273],"(c) Two residual message-passing layers",GREEN,
              "64 hidden channels; residual additions and LayerNorm")
    card(c,(.04,.66,.24,.16),"Input",r"$z\in\mathbb{R}^{N\times64}$",PB,BLUE)
    card(c,(.36,.66,.60,.16),"Layer 1",
         r"$h_1=\mathrm{LN}_1(z+D(\mathrm{ReLU}(\mathrm{SAGE}_1(z,E))))$",PG,GREEN,9.6,8.5)
    arr(c,(.28,.74),(.352,.74),GREEN)
    card(c,(.36,.38,.60,.16),"Layer 2",
         r"$h_2=\mathrm{LN}_2(h_1+D(\mathrm{ReLU}(\mathrm{SAGE}_2(h_1,E))))$",PG,GREEN,9.5,8.4)
    arr(c,(.65,.655),(.65,.548),GREEN)
    card(c,(.18,.105,.64,.15),"Graph representation",
         r"$h=h_2\in\mathbb{R}^{N\times64}$",PG,GREEN,10.7,10)
    arr(c,(.65,.375),(.54,.264),GREEN)
    c.text(.055,.315,"D = dropout(p = 0.20)\n(training only)",fontsize=8.5,color=GRAY)

    # MIDDLE ROW: feature fusion
    d=section(fig,[.025,.365,.95,.264],"(d) Dimension-wise gated fusion of projected and contextual representations",GREEN,
              "One gate per hidden channel; graph encoder and original-feature projection are fused, not separate input encoders")
    card(d,(.026,.51,.17,.22),"Projected features",
         r"$z\in\mathbb{R}^{N\times64}$"+"\nfrom projection (a)",PB,BLUE)
    card(d,(.026,.17,.17,.22),"Graph context",
         r"$h\in\mathbb{R}^{N\times64}$"+"\nfrom GraphSAGE (c)",PG,GREEN)
    card(d,(.255,.33,.185,.25),"Concatenation",
         r"$u=[z\Vert h]$"+"\n"+r"$u\in\mathbb{R}^{N\times128}$",PP,PURPLE,11.3,10)
    card(d,(.495,.33,.205,.25),"Sigmoid gate",
         r"$g=\sigma(W_gu+b_g)$"+"\n"+r"$g\in(0,1)^{N\times64}$",PG,GREEN,11.3,10)
    card(d,(.755,.33,.215,.25),"Adaptive fusion",
         r"$f=g\odot h+(1-g)\odot z$"+"\n"+r"$f\in\mathbb{R}^{N\times64}$",PG,GREEN,11.1,10)
    arr(d,(.197,.63),(.248,.53),BLUE)
    arr(d,(.197,.28),(.248,.38),GREEN)
    arr(d,(.443,.46),(.488,.46),PURPLE)
    arr(d,(.703,.46),(.748,.46),GREEN)
    d.text(.505,.155,r"$\odot$: element-wise product    |    $\sigma$: logistic sigmoid    |    $g$: dimension-wise mixing coefficient",
           fontsize=10,color=GRAY,ha="center")
    # BOTTOM ROW: country + optimization + shapes
    e=section(fig,[.025,.075,.47,.263],"(e) Shared and country-conditioned prediction",BLUE,
              "Joint-country model adds residual logits; source-only/zero-shot does not")
    card(e,(.03,.59,.28,.18),"Shared logit",
         r"$s_i=W_sf_i+b_s$"+"\nLinear(64 → 1)",PB,BLUE)
    card(e,(.38,.59,.57,.18),"Country embeddings (joint only)",
         r"$\delta_{c_i}\in\mathbb{R}^{64},\ \beta_{c_i}\in\mathbb{R}$"+"\n2 countries; embeddings initialized to zero",PP,PURPLE,10.5,9.2)
    card(e,(.03,.23,.41,.22),"Source-only / zero-shot",
         r"$\ell_i=s_i$"+"\nNo target-country head is trained",PO,ORANGE,10.7,9.3)
    card(e,(.52,.23,.43,.22),"Joint-country",
         r"$\ell_i=s_i+f_i^\top\delta_{c_i}/\sqrt{64}+\beta_{c_i}$",PG,GREEN,10.6,9.6)
    arr(e,(.16,.583),(.21,.458),ORANGE)
    arr(e,(.65,.585),(.71,.459),PURPLE)
    e.text(.50,.12,"Country index: Lithuania = 0  |  Poland = 1",ha="center",fontsize=9.2,color=GRAY)
    f=section(fig,[.507,.075,.468,.263],"(f) Optimization, output and evaluation control",RED,
              "The figure depicts the frozen implementation, not a hypothetical network")
    card(f,(.028,.59,.30,.18),"Probabilities",
         r"$p_i=\sigma(\ell_i)$"+"\nSigmoid after logits",PR,RED)
    card(f,(.375,.59,.30,.18),"Binary selection",
         r"$\hat y_i=\mathbf{1}[p_i\geq\tau]$"+"\nValidation-selected threshold",PR,RED)
    card(f,(.71,.59,.26,.18),"Training",
         "AdamW · 60 epochs\n10 seeds (42–51)",PO,ORANGE,10.5,9.2)
    card(f,(.04,.205,.91,.23),"Country-balanced weighted BCE",
         r"$\mathcal{L}=\frac{1}{N}\sum_i w_i\,\mathrm{BCEWithLogits}(\ell_i,y_i)$"+"\n"+
         r"$w_i=\frac{n_c}{2n_{c,1}}$ if $y_i=1$;  "+
         r"$w_i=\frac{n_c}{2n_{c,0}}$ if $y_i=0$  (TRAIN labels only)",PO,ORANGE,11.3,9.5)
    arr(f,(.332,.677),(.369,.677),RED)
    fig.text(.5,.038,
             "All node states are 64-D after projection. Geometric kNN edges are proximity relations, not road-junction topology.",
             fontsize=9.7,ha="center",color=GRAY)
    out.mkdir(parents=True,exist_ok=True)
    stem=out/"Fig03_CartoGraphX_Technical_Exact_V2"
    for ext in ["pdf","svg","png"]:
        path=stem.with_suffix("."+ext)
        fig.savefig(path,dpi=dpi if ext=="png" else None,
                    facecolor="white",bbox_inches="tight",pad_inches=.13)
        print("SAVED",path,flush=True)
    plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--verify",action="store_true")
    p.add_argument("--root",type=Path,default=Path.cwd())
    p.add_argument("--out-dir",type=Path)
    p.add_argument("--dpi",type=int,default=600)
    a=p.parse_args()
    if a.verify:verify(a.root.resolve())
    out=a.out_dir or a.root/"data/processed/cross_country/cartographx_v2_methodology_figure_v1"
    render(out,a.dpi)
    print("FIGURE 3 PASS")

if __name__=="__main__":main()
