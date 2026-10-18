#!/usr/bin/env python3
"""SCRIPT 124: Prepare publication-grade *real vector* cartographic scenes.

No plots are rendered here. Script 125 is the LAST script and the only final
publication-figure renderer. Script 124 builds audited, geographically aligned
vector layers from original source/target road GPKGs and frozen road predictions.

For each of six cities and two protocols, export scene GPKGs containing:
 source_roads, reference_target_roads, predicted_cartographx_roads,
 predicted_gnn_roads, tp_roads, fp_roads, fn_roads, tn_roads,
 contextual_source_roads, and optional buildings if independently available.

IMPORTANT: CartoGraph-X V2 predicts road RETENTION (classification), not new
road geometry, displacement, merging, or building generalization. The predicted
map layer is therefore a geometry-preserving selection of real source roads.
Never claim its geometry is learned, displaced, or topologically reconstructed.
C2 target correspondences remain provisional; historical test is exploratory.

Requirements: geopandas, shapely, pyproj, fiona/pyogrio, pandas, numpy.
Run: python scripts/124_prepare_cartographx_v2_publication_map_scenes.py
Run quick: python scripts/124_prepare_cartographx_v2_publication_map_scenes.py --cities kaunas_central
"""
from __future__ import annotations
import argparse, hashlib, json, math, re
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

ROOT=Path(__file__).resolve().parents[1]
NODES=ROOT/'data/processed/cross_country/benchmark_v2_unified_labels_v1/road_graph_nodes_v2.csv'
CX=ROOT/'data/processed/cross_country/cartographx_v2_ten_seed_summary_v1/all_seed_selected_test_predictions.csv'
GNN=ROOT/'data/processed/cross_country/benchmark_v2_ten_seed_gnn_baselines_v1'
ERROR=ROOT/'data/processed/cross_country/cartographx_v2_error_failure_analysis_v1/road_failure_example_candidates.csv'
OUT=ROOT/'data/processed/cross_country/cartographx_v2_publication_map_scenes_v1'
CITY_FILES={
 'vilnius_central':('Lithuania','lithuania/validated_pairs/vilnius_central_validated.gpkg'),
 'kaunas_central':('Lithuania','lithuania/validated_pairs/kaunas_central_validated.gpkg'),
 'klaipeda_central':('Lithuania','lithuania/validated_pairs/klaipeda_central_validated.gpkg'),
 'warsaw':('Poland','poland/road_target_validation/warsaw_validated_road_pair.gpkg'),
 'krakow':('Poland','poland/road_target_validation/krakow_validated_road_pair.gpkg'),
 'gdansk':('Poland','poland/road_target_validation/gdansk_validated_road_pair.gpkg')}
CITY_ALIAS={'vilnius':'vilnius_central','kaunas':'kaunas_central','klaipeda':'klaipeda_central',
            'warsaw':'warsaw','krakow':'krakow','gdansk':'gdansk'}
PROTOCOLS={'Lithuania':('lt_to_lt','joint_to_lt','pl_to_lt_zero_shot'),
           'Poland':('pl_to_pl','joint_to_pl','lt_to_pl_zero_shot')}
# Two panels per city: same-country and joint. Zero-shot prediction retained as
# an optional *third prediction overlay* in every scene for failure comparisons.
SELECTED={'Lithuania':('lt_to_lt','joint_to_lt'), 'Poland':('pl_to_pl','joint_to_pl')}
STYLES={
 'source_roads':{'color':'#a3acb6','linewidth_pt':0.65,'alpha':0.70,'zorder':2},
 'reference_target_roads':{'color':'#0b5b8a','linewidth_pt':2.15,'alpha':0.98,'zorder':4},
 'predicted_cartographx_roads':{'color':'#d94d31','linewidth_pt':2.0,'alpha':0.98,'zorder':5},
 'predicted_gnn_roads':{'color':'#8d57b6','linewidth_pt':1.8,'alpha':0.95,'zorder':5},
 'tp_roads':{'color':'#147d64','linewidth_pt':2.0,'alpha':0.97,'zorder':5},
 'fp_roads':{'color':'#df6b23','linewidth_pt':2.1,'alpha':0.98,'zorder':6},
 'fn_roads':{'color':'#be2d52','linewidth_pt':2.1,'alpha':0.98,'zorder':6},
 'tn_roads':{'color':'#a4abb5','linewidth_pt':0.65,'alpha':0.55,'zorder':2},
 'buildings':{'facecolor':'#e0dcd2','edgecolor':'#bcb6a8','linewidth_pt':0.25,'alpha':0.85,'zorder':1},
 'background':{'color':'#f7f8f7'},
 'layout':{'scale_bar':True,'north_arrow':True,'coordinate_grid':True,
           'map_crs_labels':True,'inset_locator':True,'vector_output':True,
           'raster_dpi':600,'legend':True,'aligned_extent':True,
           'panel_design':['A: detailed source','B: reference generalized target',
                           'C: CartoGraph-X predicted selection','D: GNN baseline selection',
                           'E: TP/FP/FN overlay'],
           'annotation':'Road-selection only; target geometry is independently sourced'} }

def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()
def dump(obj,path):
 path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
def safe_slug(s):return re.sub(r'[^a-z0-9_]+','_',str(s).lower()).strip('_')
def load_gnn(seed,protocol):
 selected=pd.read_csv(GNN/'all_seed_selected_protocol_results.csv')
 pick=selected.loc[selected.seed.eq(seed)&selected.protocol.eq(protocol)]
 if len(pick)!=1:raise RuntimeError(f'GNN selection missing: {seed}/{protocol}')
 p=pd.read_csv(GNN/f'seed_{seed}'/'all_candidate_test_predictions.csv',low_memory=False)
 for col in ('model','variant'):
  p=p.loc[p[col].astype(str).eq(str(pick.iloc[0][col]))]
 p=p.loc[p.protocol.eq(protocol)]
 if p.sample_uid.duplicated().any():raise RuntimeError('Duplicate GNN UID')
 return p[['sample_uid','prediction','probability']].rename(columns={'prediction':'gnn_prediction','probability':'gnn_probability'})
def resolve_city(v):
 k=safe_slug(v)
 if k in CITY_FILES:return k
 for alias,canon in CITY_ALIAS.items():
  if k==alias or k.startswith(alias+'_'):return canon
 raise ValueError(f'Unrecognized city {v!r}; known {list(CITY_FILES)}')
def metric_crs(gdf,city):
 if gdf.crs is None:raise RuntimeError(f'Original {city} GPKG missing CRS')
 if gdf.crs.is_geographic:raise RuntimeError(f'{city} has angular CRS; need projected metres')
 units=[axis.unit_name.lower() for axis in gdf.crs.axis_info]
 if not any('metre' in u or 'meter' in u for u in units):
  raise RuntimeError(f'{city} CRS units not metres: {units}')
def layer_data(gpkg,layer,city):
 g=gpd.read_file(gpkg,layer=layer).reset_index(drop=True)
 metric_crs(g,city)
 if g.empty:raise RuntimeError(f'Empty {layer}: {city}')
 if g.geometry.isna().any() or g.geometry.is_empty.any():
  raise RuntimeError(f'Invalid empty geometry in {city}/{layer}')
 return g

def clipped(g,footprint):
 # shapely intersection guarantees each exported feature stays inside the SAME
 # rectangular geographic footprint across every map panel.
 hits=g.geometry.intersects(footprint)
 sub=g.loc[hits].copy()
 if sub.empty:return sub
 sub['geometry']=sub.geometry.intersection(footprint)
 return sub.loc[sub.geometry.notna()&~sub.geometry.is_empty].copy()
def write_layer(g,path,layer,first):
 if g.empty:return False
 if g.crs is None:raise RuntimeError('Missing CRS on output layer')
 g.to_file(path,layer=layer,driver='GPKG',mode='w' if first else 'a',index=False)
 return True

def optional_buildings(city,footprint,crs,explicit):
 # No downloads or synthetic footprints: only independently present vector data.
 paths=[]
 if explicit:paths.append(Path(explicit))
 city_root=ROOT/'data/processed'
 patterns=(f'**/{city}*building*.gpkg',f'**/{city}*building*.geojson',
           f'**/{city}*building*.parquet')
 if not explicit:
  for pat in patterns:paths.extend(sorted(city_root.glob(pat))[:5])
 for p in paths:
  if not p.is_file():continue
  try:
   if p.suffix=='.parquet':g=gpd.read_parquet(p)
   else:g=gpd.read_file(p)
   if g.crs is None:continue
   g=g.to_crs(crs)
   g=g.loc[g.geom_type.isin(['Polygon','MultiPolygon'])]
   sub=clipped(g,footprint)
   if not sub.empty:
    return sub[['geometry']].copy(),str(p)
  except Exception as e:
   print(f'  Optional buildings skipped: {p.name}: {e}',flush=True)
 return None,None

def select_anchor(city_nodes,city_pred,candidates,city,protocol,seed,selection):
 # Reproducible fixed candidate ordering, no evaluation-driven model tuning.
 # Priority: high-confidence FP/FN from Script123, otherwise deterministic
 # central test road. Anchor is a real graph road linked to original source row.
 g=city_pred.loc[city_pred.protocol.eq(protocol)].copy()
 g=g.merge(city_nodes[['sample_uid','centroid_x','centroid_y','source_idx_harmonized','tile_id']],
           on='sample_uid',how='inner',validate='one_to_one')
 if g.empty:raise RuntimeError(f'No predictions for {city}/{protocol}')
 if selection=='errors' and candidates is not None:
  c=candidates.loc[(candidates.model_family=='CartoGraph-X')&
                   (candidates.seed==seed)&(candidates.protocol==protocol)].copy()
  c=c.loc[c.city.map(resolve_city).eq(city)] if len(c) else c
  c=c.loc[c.error_type.isin(['FP','FN'])]
  if not c.empty:
   c=c.sort_values(['decision_margin','sample_uid'],ascending=[False,True])
   match=g.loc[g.sample_uid.isin(c.sample_uid)]
   if not match.empty:
    rank={v:i for i,v in enumerate(c.sample_uid)}
    return match.assign(_rank=match.sample_uid.map(rank)).sort_values('_rank').iloc[0]
 if selection=='balanced':
  # Favor a geographically central road, not a chosen performance outcome.
  xy=g[['centroid_x','centroid_y']].to_numpy(dtype=float)
  middle=np.nanmedian(xy,axis=0)
  g['_d2']=np.sum((xy-middle)**2,axis=1)
  return g.sort_values(['_d2','sample_uid']).iloc[0]
 return g.sort_values('sample_uid').iloc[0]

def make_scene(city,protocol,seed,halfwidth,selection,nodes,preds,candidates,args):
 country,rel=CITY_FILES[city]
 gpkg=ROOT/'data/processed'/rel
 if not gpkg.is_file():raise FileNotFoundError(gpkg)
 source=layer_data(gpkg,'source_roads',city)
 target=layer_data(gpkg,'target_roads',city)
 if source.crs!=target.crs:target=target.to_crs(source.crs)
 source['original_source_row']=np.arange(len(source),dtype=int)
 cn=nodes.loc[(nodes.country==country)&nodes.city.map(resolve_city).eq(city)&nodes.split.eq('test')].copy()
 if cn.empty:raise RuntimeError(f'No test graph nodes in {city}')
 if cn.sample_uid.duplicated().any():raise RuntimeError('Duplicate graph node UIDs')
 cp=preds.loc[preds.protocol.eq(protocol)&preds.country.eq(country)].copy()
 cp=cp.loc[cp.city.map(resolve_city).eq(city)]
 if len(cp)!=len(cn):raise RuntimeError(f'Prediction/node count mismatch {city}/{protocol}: {len(cp)} != {len(cn)}')
 if not set(cp.sample_uid)==set(cn.sample_uid):raise RuntimeError('Prediction/node UID mismatch')
 anchor=select_anchor(cn,cp,candidates,city,protocol,seed,selection)
 row=int(anchor.source_idx_harmonized)
 if row<0 or row>=len(source):raise RuntimeError('Source row out of bounds')
 x,y=float(anchor.centroid_x),float(anchor.centroid_y)
 if not source.iloc[row].geometry.distance(__import__('shapely').geometry.Point(x,y))<max(25,halfwidth):
  raise RuntimeError(f'Graph centroid not aligned with original source geometry: {city}/{row}')
 footprint=box(x-halfwidth,y-halfwidth,x+halfwidth,y+halfwidth)
 # Prefer authoritative V2 node metadata; prediction exports can already
 # contain tile_id, which otherwise becomes tile_id_x/tile_id_y.
 pred=cp.drop(columns=['source_idx_harmonized','retained','tile_id'],errors='ignore').merge(
     cn[['sample_uid','source_idx_harmonized','retained','tile_id']],
     on='sample_uid',how='inner',validate='one_to_one')
 if 'tile_id' not in pred.columns:
  raise RuntimeError('Missing canonical tile_id after prediction/node join')
 if not np.array_equal(pred.label.to_numpy(dtype=int),pred.retained.to_numpy(dtype=int)):
  raise RuntimeError('Prediction label does not match frozen C2 graph label')
 if pred.source_idx_harmonized.duplicated().any():raise RuntimeError('Duplicated original source row')
 gnn=load_gnn(seed,protocol)
 pred=pred.merge(gnn,on='sample_uid',how='left',validate='one_to_one')
 if pred.gnn_prediction.isna().any():raise RuntimeError('Missing selected GNN predictions')
 pred['original_source_row']=pd.to_numeric(pred.source_idx_harmonized,errors='raise').astype(int)
 # Scene rows are those whose ORIGINAL source geometry intersects the footprint.
 # No arbitrary selection of nearest centroids: preserves long crossing roads.
 candidate_rows=set(source.loc[source.geometry.intersects(footprint),'original_source_row'].tolist())
 pred=pred.loc[pred.original_source_row.isin(candidate_rows)].copy()
 if pred.empty:raise RuntimeError('No graph test roads intersect scene')
 geo=source.merge(pred[['original_source_row','sample_uid','tile_id','label','prediction',
                        'probability','threshold','gnn_prediction','gnn_probability']],
                  on='original_source_row',how='inner',validate='one_to_one')
 geo=clipped(geo,footprint)
 if geo.empty:raise RuntimeError('Scene clip removed all predictions')
 geo['error_type']=np.select([(geo.label.eq(0)&geo.prediction.eq(1)),
                             (geo.label.eq(1)&geo.prediction.eq(0)),
                             (geo.label.eq(1)&geo.prediction.eq(1))],['FP','FN','TP'],default='TN')
 # Drop unpredictable GPKG source fields that might include unsupported objects.
 keep=['original_source_row','sample_uid','tile_id','label','prediction','probability',
       'threshold','gnn_prediction','gnn_probability','error_type','geometry']
 geo=geo[keep]
 full_source=clipped(source[['original_source_row','geometry']],footprint)
 ref=clipped(target[['geometry']],footprint)
 if ref.empty:raise RuntimeError('Reference target has no geometry in scene; choose another anchor')
 building,building_origin=optional_buildings(city,footprint,source.crs,args.buildings)
 folder=OUT/'scenes';folder.mkdir(parents=True,exist_ok=True)
 name=f'{city}__{protocol}__seed{seed}'
 dest=folder/f'{name}.gpkg'
 if dest.exists():dest.unlink()
 layers={
  'source_roads':full_source,
  'reference_target_roads':ref,
  'predicted_cartographx_roads':geo.loc[geo.prediction.eq(1)],
  'predicted_gnn_roads':geo.loc[geo.gnn_prediction.eq(1)],
  'tp_roads':geo.loc[geo.error_type.eq('TP')],
  'fp_roads':geo.loc[geo.error_type.eq('FP')],
  'fn_roads':geo.loc[geo.error_type.eq('FN')],
  'tn_roads':geo.loc[geo.error_type.eq('TN')],
  'test_road_predictions':geo}
 if building is not None:layers['buildings']=building
 written=[]
 for layer,g in layers.items():
  if write_layer(g,dest,layer,first=(len(written)==0)):written.append(layer)
 if not written:raise RuntimeError('No GPKG layers exported')
 counts={k:int(len(v)) for k,v in layers.items()}
 properties={
  'scene_id':name,'city':city,'country':country,'protocol':protocol,'seed':seed,
  'source_gpkg':str(gpkg),'source_sha256':digest(gpkg),'output_gpkg':str(dest),
  'output_sha256':digest(dest),'crs':str(source.crs),
  'bbox_projected_m':[x-halfwidth,y-halfwidth,x+halfwidth,y+halfwidth],
  'anchor_sample_uid':str(anchor.sample_uid),'anchor_original_source_row':row,
  'anchor_selection_policy':selection,'extent_width_m':2*halfwidth,
  'layer_counts':counts,'written_layers':written,
  'source_total_city':len(source),'reference_total_city':len(target),
  'test_prediction_roads_in_scene':len(geo),
  'mean_predicted_retention':float(geo.prediction.mean()),
  'mean_reference_C2_retention':float(geo.label.mean()),
  'error_counts':{k:int((geo.error_type==k).sum()) for k in ('TP','FP','FN','TN')},
  'buildings_source':building_origin,
  'important_geometry_semantics':'CartoGraph-X is road-retention selection on original source geometries; reference is independently sourced target geometry; not a generated displacement/simplification map',
  'label_status':'C2 PROVISIONAL_NOT_MANUALLY_VALIDATED',
  'evaluation_status':'HISTORICAL_TEST_EXPLORATORY',
  'figure_rendering':'DEFERRED_TO_SCRIPT_125'}
 dump(properties,folder/f'{name}.json')
 print(f'  {name}: source={len(full_source)} reference={len(ref)} predicted={counts["predicted_cartographx_roads"]} errors FP={properties["error_counts"]["FP"]} FN={properties["error_counts"]["FN"]} buildings={counts.get("buildings",0)}',flush=True)
 return properties

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--seed',type=int,default=42)
 ap.add_argument('--halfwidth-m',type=float,default=350.0,
                 help='Scene radius in projected metres (default 350, square 700x700m)')
 ap.add_argument('--cities',nargs='*',default=list(CITY_FILES))
 ap.add_argument('--selection',choices=['errors','balanced','uid'],default='errors')
 ap.add_argument('--buildings',type=str,default=None,
                 help='Optional real building polygon vector file, spatially filtered per scene')
 args=ap.parse_args()
 if args.seed not in range(42,52):ap.error('Seed must be in frozen range 42..51')
 if not 100<=args.halfwidth_m<=1500:ap.error('Halfwidth must be 100..1500 metres')
 if not args.cities:ap.error('Specify at least one city')
 cities=list(dict.fromkeys(resolve_city(c) for c in args.cities))
 for path in (NODES,CX):
  if not path.is_file():raise FileNotFoundError(path)
 nodes=pd.read_csv(NODES,low_memory=False)
 required={'sample_uid','country','city','split','source_idx_harmonized','centroid_x','centroid_y','retained','tile_id'}
 if not required.issubset(nodes.columns):raise RuntimeError(f'Missing graph columns: {required-set(nodes.columns)}')
 if nodes.sample_uid.duplicated().any():raise RuntimeError('Duplicate graph UIDs')
 preds=pd.read_csv(CX,low_memory=False)
 preds=preds.loc[preds.seed.eq(args.seed)].copy()
 if not preds.split.eq('test').all():raise RuntimeError('Unexpected non-test predictions')
 if preds.duplicated(['protocol','sample_uid']).any():raise RuntimeError('Duplicate prediction UID')
 if not np.array_equal(preds.prediction.to_numpy(dtype=int),
                       (preds.probability.to_numpy()>=preds.threshold.to_numpy()).astype(int)):
  raise RuntimeError('Frozen threshold/prediction mismatch')
 candidates=pd.read_csv(ERROR,low_memory=False) if ERROR.is_file() else None
 OUT.mkdir(parents=True,exist_ok=True)
 dump(STYLES,OUT/'publication_cartographic_style_spec.json')
 scenes=[]
 for city in cities:
  country=CITY_FILES[city][0]
  print(f'\nCITY: {city} [{country}]',flush=True)
  for protocol in SELECTED[country]:
   scenes.append(make_scene(city,protocol,args.seed,args.halfwidth_m,args.selection,
                            nodes,preds,candidates,args))
 manifest={'script':'124','created_utc':datetime.now(timezone.utc).isoformat(),
  'scene_count':len(scenes),'seed':args.seed,'cities':cities,
  'selection_policy':args.selection,'halfwidth_m':args.halfwidth_m,
  'source_predictions_sha256':digest(CX),'source_nodes_sha256':digest(NODES),
  'error_candidates_sha256':digest(ERROR) if ERROR.is_file() else None,
  'publication_cartographic_style_spec':'publication_cartographic_style_spec.json',
  'final_figure_renderer':'Script 125 ONLY',
  'notes':['Genuine georeferenced source and target vector roads; no synthetic map content',
           'Predicted roads are selected source geometries, not learned geometry changes',
           'Building polygons included only if an existing source can be found',
           'The historical test has been explored; C2 labels are provisional'],
  'scenes':scenes}
 dump(manifest,OUT/'scene_catalog_manifest.json')
 flat=[]
 for s in scenes:
  flat.append({k:v for k,v in s.items() if not isinstance(v,(dict,list))})
 pd.DataFrame(flat).to_csv(OUT/'scene_catalog.csv',index=False)
 print(f'\nSCRIPT 124 PASS: {len(scenes)} georeferenced publication-map scenes prepared.')
 print('Saved:',OUT)
 print('NO FIGURES RENDERED. SCRIPT 125 IS THE FINAL CODE AND GENERATES ALL FIGURES.')

if __name__=='__main__':main()
