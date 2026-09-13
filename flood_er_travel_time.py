"""
Flood impact on emergency-room travel time
Benton, Washington and Madison Counties, Arkansas

One straight-line script, pandas style, top to bottom.  Steps:

  1. County boundaries ............ Overture Maps "divisions" theme (OSM-derived)
  2. Drivable road network ........ Overture Maps "transportation" theme (OSM-derived, monthly)
  3. Emergency rooms .............. Overture Maps "places" theme, curated to 24/7 EDs
  4. Grid of locations ............ regular 1 km lattice clipped to the three counties
  5. Baseline travel time ......... NetworkX Dijkstra from every ER over the reversed graph
  6. 100-year floodplain .......... FEMA National Flood Hazard Layer: SFHA zones (SFHA_TF = 'T'),
                                    base-flood-elevation lines (S_BFE) and cross-sections (S_XS)
  7. Water depth on the road ...... road surface from the USGS 3DEP DEM sampled every 10 m along each
                                    edge that touches the floodplain; 1 % water-surface elevation from
                                    the BFE lines / static BFEs where FEMA mapped them, otherwise
                                    inferred from the ground elevation along the floodplain edge
  8. Closures ..................... an edge is closed where water on the pavement >= --passable-depth
                                    (or, with --closure-rule 2d, wherever it touches the floodplain)
  9. Flood travel time ............ Dijkstra again, compare, map, summarise
 10. Method flags ................. output/method_flags.csv lists every inference and gap, with counts

Data are cached in ./data so the script can be re-run offline.  Outputs go to ./output.

Run:
    python flood_er_travel_time.py                      # download NFHL from FEMA's REST service
    python flood_er_travel_time.py --nfhl NFHL_05.zip   # or use a state/county NFHL you downloaded
    python flood_er_travel_time.py --dem my_1m_dem.vrt  # better DEM (metres, NAVD88) than 3DEP 1/3"
    python flood_er_travel_time.py --closure-rule 2d    # old planimetric rule, for comparison
    python flood_er_travel_time.py --bridges-passable   # never close bridge decks

FEMA's NFHL is served from hazards.fema.gov; if that host is not reachable from where you run
this, download the Arkansas state NFHL from https://msc.fema.gov (Search All Products ->
Effective Products -> NFHL Data-State) and pass it with --nfhl.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import geopandas as gpd
import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pyogrio
import rasterio
import requests
import shapely
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from pyproj import Transformer
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from shapely.ops import substring

matplotlib.use("Agg")

# --------------------------------------------------------------------------------------
# 0. Settings
# --------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--nfhl", default=os.environ.get("NFHL_PATH"),
                    help="Local NFHL file (.zip/.gdb/.gpkg with S_FLD_HAZ_AR, S_BFE, S_XS). Default: download from FEMA REST.")
parser.add_argument("--dem", nargs="*", default=None,
                    help="DEM GeoTIFF/VRT file(s), metres NAVD88. Default: USGS 3DEP 1/3 arc-second tiles into data/dem/")
parser.add_argument("--release", default="2026-08-19.0", help="Overture Maps release")
parser.add_argument("--spacing", type=float, default=1000.0, help="grid spacing in metres")
parser.add_argument("--sample-m", type=float, default=10.0, help="DEM sampling interval along roads, metres")
parser.add_argument("--passable-depth", type=float, default=0.15,
                    help="metres of water on the pavement at which a road is impassable (0.15 m = 6 in)")
parser.add_argument("--closure-rule", choices=["depth", "2d"], default="depth",
                    help="depth: close where water depth >= --passable-depth; 2d: close every edge touching the floodplain")
parser.add_argument("--bridges-passable", action="store_true", help="never close edges flagged as bridges")
args = parser.parse_args()

RELEASE = args.release
GRID_SPACING_M = args.spacing
SAMPLE_M = args.sample_m
PASSABLE_DEPTH_M = args.passable_depth
CLOSURE_RULE = args.closure_rule
BRIDGES_PASSABLE = args.bridges_passable
NFHL_PATH = args.nfhl
DEM_FILES = args.dem

COUNTIES = ["Benton County", "Washington County", "Madison County"]
CRS_PROJ = "EPSG:26915"            # NAD83 / UTM zone 15N - metres, covers NW Arkansas
ROAD_BUFFER_M = 35_000             # pull roads/ERs this far beyond the county lines
BFE_MAX_DIST_M = 1500              # a BFE line / cross-section further than this is another reach
EDGE_K = 8                         # floodplain-edge vertices averaged for an inferred water surface
DATA_DIR = Path("data")
OUT_DIR = Path("output")
DEM_DIR = DATA_DIR / "dem"
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)
DEM_DIR.mkdir(exist_ok=True)
NFHL_URL = "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer"
DEM_URL = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/{tile}/USGS_13_{tile}.tif"

# Overture segment classes that a car can drive on (mirrors OSMnx's "drive_service" filter)
DRIVE_CLASSES = ["motorway", "trunk", "primary", "secondary", "tertiary",
                 "residential", "unclassified", "service", "living_street"]

# Default free-flow speeds (mph) when Overture carries no posted speed limit
DEFAULT_MPH = {"motorway": 65, "trunk": 55, "primary": 45, "secondary": 40, "tertiary": 35,
               "residential": 25, "unclassified": 30, "service": 15, "living_street": 15}

# 24/7 emergency departments, keyed by Overture place id.  Facilities in and around the
# three counties, so "nearest ER" is right near county lines.  include=False rows are kept
# in the ER table for reference but are not routed to (restricted populations).
ER_TABLE = pd.DataFrame([
    # Overture id,                             short name,                                      county,        include, note
    ("c7a2a177-989c-467a-afac-8a75f0720920", "Northwest Medical Center - Bentonville",          "Benton",      True,  ""),
    ("8ac2cf85-e5ae-4414-99f7-6a9fa4d83a85", "Northwest Health ED - West Bentonville",          "Benton",      True,  "free-standing ED"),
    ("27cefd97-e9db-4f5b-9ecb-9de0e05d7b1e", "Mercy Hospital Northwest Arkansas (Rogers)",      "Benton",      True,  ""),
    ("e73102e3-b8d1-4f24-b7e1-37ea98e0b3c9", "Mercy ED - Bella Vista",                          "Benton",      True,  "free-standing ED"),
    ("64a15d23-91f0-4f8f-961f-59a25828fb49", "Siloam Springs Regional Hospital",                "Benton",      True,  ""),
    ("c721ee27-4ef3-4676-afed-8a262dc8f167", "Ozarks Community Hospital of Gravette",           "Benton",      True,  "critical-access, 24 h ER"),
    ("23cf9a21-62e2-41b9-a3c6-53471cb482f5", "Northwest Medical Center - Springdale",           "Washington",  True,  ""),
    ("bff02822-7e85-4b01-8559-6fdc650a3826", "Willow Creek Women's Hospital ED (Johnson)",      "Washington",  True,  ""),
    ("3c85e9d7-0722-4f27-9c0b-302dfd58d964", "Washington Regional Medical Center",              "Washington",  True,  ""),
    ("f22c72a1-32ba-4190-84aa-7c74c7ef864e", "Northwest Health Physicians' Specialty Hospital", "Washington",  True,  ""),
    ("92461ced-5059-4310-a0cd-ee1b9b3fe12c", "Northwest Health ED - Fayetteville",              "Washington",  True,  "free-standing ED"),
    ("db8904cf-13ff-46a6-bf5c-fa9b820368cf", "Arkansas Children's Northwest",                   "Washington",  False, "pediatric ER only"),
    ("7c5b4fc8-1a6b-4d33-be20-fa22aac194d6", "VA - Veterans Health Care System of the Ozarks",  "Washington",  False, "veterans only"),
    # Madison County has no hospital; neighbouring-county ERs below keep 'nearest' honest
    ("06575049-5c8e-444c-b504-374e59d97c7a", "Mercy ED - Berryville",                           "Carroll",     True,  ""),
    ("1ea5aa1b-22c4-44bd-afb7-396915bf89c3", "Eureka Springs Hospital",                         "Carroll",     True,  ""),
    ("2ea98bc7-4348-45a1-95af-7c08734be893", "North Arkansas Regional Medical Center (Harrison)","Boone",       True,  ""),
    ("200aa882-b7b9-4db1-9e97-c5d1b3fa3cd9", "Johnson Regional Medical Center (Clarksville)",   "Johnson",     True,  ""),
    ("82ac7aac-d89d-4d78-970c-63e0c329000d", "Mercy ED - Ozark",                                "Franklin",    True,  ""),
    ("3318561a-41ff-4976-ae8a-46b2fe1a846d", "Baptist Health - Van Buren",                      "Crawford",    True,  ""),
    ("08dd2f6f-d0c6-49a9-916d-0c41cde919cc", "Stilwell Memorial Hospital (OK)",                 "Adair OK",    True,  ""),
    ("ae374fce-86f7-4bfb-9afc-c492c4a41b6e", "INTEGRIS Health Grove Hospital (OK)",             "Delaware OK", True,  ""),
    ("9a384a4e-6b99-4c1c-b9b9-49530a1ce734", "Mercy Hospital Cassville (MO)",                   "Barry MO",    True,  ""),
], columns=["overture_id", "er_name", "county", "include", "note"])

# pyarrow talks to the public Overture bucket anonymously; honour an HTTPS proxy if one is set
S3_OPTS = dict(anonymous=True, region="us-west-2")
if os.environ.get("HTTPS_PROXY"):
    S3_OPTS["proxy_options"] = os.environ["HTTPS_PROXY"]
S3 = pafs.S3FileSystem(**S3_OPTS)
OVERTURE = f"overturemaps-us-west-2/release/{RELEASE}/"

t0 = time.time()
print(f"Overture release {RELEASE}; grid {GRID_SPACING_M:.0f} m; closure rule = {CLOSURE_RULE}; "
      f"passable depth {PASSABLE_DEPTH_M} m; bridges passable = {BRIDGES_PASSABLE}")
flags = []      # (flag, count, unit, description) rows for output/method_flags.csv

# --------------------------------------------------------------------------------------
# 1. County boundaries
# --------------------------------------------------------------------------------------
county_file = DATA_DIR / "counties.gpkg"
if county_file.exists():
    counties = gpd.read_file(county_file)
else:
    # rough Arkansas-corner box just to prune the parquet row groups; exact names filter below
    flt = ((pc.field("subtype") == "county") & (pc.field("region") == "US-AR")
           & (pc.field("bbox", "xmin") > -95.0) & (pc.field("bbox", "xmax") < -92.5)
           & (pc.field("bbox", "ymin") > 35.3) & (pc.field("bbox", "ymax") < 36.6))
    tbl = ds.dataset(OVERTURE + "theme=divisions/type=division_area/", filesystem=S3, format="parquet") \
            .to_table(filter=flt, columns=["id", "names", "geometry"])
    counties = tbl.to_pandas()
    counties["name"] = counties["names"].str["primary"]
    counties = counties[counties["name"].isin(COUNTIES)]
    counties = gpd.GeoDataFrame(counties[["id", "name"]],
                                geometry=shapely.from_wkb(counties["geometry"].values), crs="EPSG:4326")
    counties.to_file(county_file, driver="GPKG")
assert sorted(counties["name"]) == sorted(COUNTIES), counties["name"].tolist()
counties_proj = counties.to_crs(CRS_PROJ)
study_area = counties_proj.union_all()
road_box = gpd.GeoSeries([study_area.buffer(ROAD_BUFFER_M)], crs=CRS_PROJ).to_crs("EPSG:4326").total_bounds
W, S, E, N = road_box
print(f"counties ok; road bbox lon {W:.3f}..{E:.3f} lat {S:.3f}..{N:.3f}  [{time.time()-t0:.0f}s]")

# --------------------------------------------------------------------------------------
# 2. Roads from Overture transportation segments
# --------------------------------------------------------------------------------------
roads_file = DATA_DIR / "overture_road_segments.parquet"
if not roads_file.exists():
    flt = ((pc.field("subtype") == "road") & pc.field("class").isin(DRIVE_CLASSES)
           & (pc.field("bbox", "xmin") > W) & (pc.field("bbox", "xmax") < E)
           & (pc.field("bbox", "ymin") > S) & (pc.field("bbox", "ymax") < N))
    tbl = ds.dataset(OVERTURE + "theme=transportation/type=segment/", filesystem=S3, format="parquet") \
            .to_table(filter=flt, columns=["id", "class", "connectors", "speed_limits",
                                           "access_restrictions", "road_flags", "geometry"])
    pq.write_table(tbl, roads_file)
roads = pq.read_table(roads_file).to_pandas()
print(f"{len(roads):,} road segments  [{time.time()-t0:.0f}s]")

# -- speed: posted limit if it is unconditional, else class default; normalise to mph
roads["posted"] = roads["speed_limits"].str[0].str["max_speed"].str["value"]
roads["posted_unit"] = roads["speed_limits"].str[0].str["max_speed"].str["unit"]
roads["posted_when"] = roads["speed_limits"].str[0].str["when"]
roads.loc[roads["posted_unit"] == "km/h", "posted"] = roads["posted"] * 0.621371
roads.loc[roads["posted_when"].notna(), "posted"] = np.nan
roads["speed_mph"] = roads["posted"].fillna(roads["class"].map(DEFAULT_MPH)).clip(lower=5)

# -- flags
roads["flags"] = roads["road_flags"].astype(str)
roads = roads[~roads["flags"].str.contains("is_under_construction|is_abandoned")]
# bridges: an Overture flag normally applies to a 'between' range (0..1 along the segment), not the whole segment
bridge_rows = []
for sid, rf in zip(roads["id"], roads["road_flags"]):
    if rf is None:
        continue
    for rule in rf:
        if rule.get("values") is None or "is_bridge" not in list(rule["values"]):
            continue
        bt = rule.get("between")
        bridge_rows.append((sid, 0.0, 1.0) if bt is None else (sid, float(bt[0]), float(bt[1])))
bridges = pd.DataFrame(bridge_rows, columns=["id", "a", "b"])
bridges.index.name = "bridge_id"

# -- access: private roads are dropped, heading-specific denials make a one-way
fwd_ok, bwd_ok, private = [], [], []
for rules in roads["access_restrictions"]:
    f, b, p = True, True, False
    if rules is not None:
        for r in rules:
            if r.get("access_type") != "denied":
                continue
            when = r.get("when")
            if when is None:
                p = True
                continue
            modes = when.get("mode")
            if modes is not None and len(modes) and not any(m in ("motor_vehicle", "car", "vehicle") for m in modes):
                continue                                   # e.g. denied only to trucks/bikes
            if when.get("during") or when.get("using") or when.get("recognized") or when.get("vehicle") is not None:
                continue                                   # time-of-day / permit / size rules: ignore
            if when.get("heading") == "forward":
                f = False
            elif when.get("heading") == "backward":
                b = False
            else:
                p = True
    fwd_ok.append(f); bwd_ok.append(b); private.append(p)
roads["fwd_ok"], roads["bwd_ok"], roads["private"] = fwd_ok, bwd_ok, private
roads = roads[~roads["private"] & (roads["fwd_ok"] | roads["bwd_ok"])].reset_index(drop=True)
bridges = bridges[bridges["id"].isin(roads["id"])]
print(f"{len(roads):,} public drivable segments  [{time.time()-t0:.0f}s]")

# -- split every segment at its connectors -> one edge per connector pair
roads["geom"] = gpd.GeoSeries(shapely.from_wkb(roads["geometry"].values), crs="EPSG:4326").to_crs(CRS_PROJ).values
con = roads[["id", "connectors"]].explode("connectors", ignore_index=True)
con["node"] = con["connectors"].str["connector_id"]
con["at"] = con["connectors"].str["at"].astype(float)
con = con.sort_values(["id", "at"], kind="stable")
con["node_next"] = con.groupby("id")["node"].shift(-1)
con["at_next"] = con.groupby("id")["at"].shift(-1)
pairs = con.dropna(subset=["node_next"])
pairs = pairs[pairs["at_next"] > pairs["at"]]
pairs = pairs.merge(roads[["id", "class", "speed_mph", "fwd_ok", "bwd_ok", "geom"]], on="id")

edge_geoms = []
for g, a, b in zip(pairs["geom"], pairs["at"], pairs["at_next"]):
    edge_geoms.append(substring(g, a, b, normalized=True))
edges = gpd.GeoDataFrame(
    {"u": pairs["node"].values, "v": pairs["node_next"].values, "segment_id": pairs["id"].values,
     "at": pairs["at"].values, "at_next": pairs["at_next"].values,
     "class": pairs["class"].values, "speed_mph": pairs["speed_mph"].values,
     "fwd_ok": pairs["fwd_ok"].values, "bwd_ok": pairs["bwd_ok"].values},
    geometry=edge_geoms, crs=CRS_PROJ)
edges["length_m"] = edges.length
edges["travel_min"] = edges["length_m"] / (edges["speed_mph"] * 1609.344 / 60)
edges = edges[edges["length_m"] > 0].reset_index(drop=True)
edges.index.name = "edge_id"
# an edge "is a bridge" if any part of it lies in a bridge range (used for --bridges-passable and reporting)
ov = edges[["segment_id", "at", "at_next"]].reset_index().merge(bridges.reset_index(), left_on="segment_id", right_on="id")
ov = ov[(ov["a"] < ov["at_next"]) & (ov["b"] > ov["at"])]
edges["is_bridge"] = edges.index.isin(ov["edge_id"])

# -- node coordinates from edge end points
first = shapely.get_point(edges.geometry.values, 0)
last = shapely.get_point(edges.geometry.values, -1)
nodes = pd.concat([
    pd.DataFrame({"node": edges["u"], "x": shapely.get_x(first), "y": shapely.get_y(first)}),
    pd.DataFrame({"node": edges["v"], "x": shapely.get_x(last), "y": shapely.get_y(last)}),
]).drop_duplicates("node").set_index("node")
print(f"{len(edges):,} edges, {len(nodes):,} nodes  [{time.time()-t0:.0f}s]")

# -- directed edge list (both directions unless one-way), keep fastest of any parallel pair
directed = pd.concat([
    edges.loc[edges["fwd_ok"], ["u", "v", "travel_min"]],
    edges.loc[edges["bwd_ok"], ["v", "u", "travel_min"]].rename(columns={"v": "u", "u": "v"}),
]).groupby(["u", "v"], as_index=False)["travel_min"].min()
G = nx.from_pandas_edgelist(directed, "u", "v", ["travel_min"], create_using=nx.DiGraph)
largest = max(nx.strongly_connected_components(G), key=len)
G = G.subgraph(largest).copy()
nodes = nodes.loc[nodes.index.isin(largest)]
edges["in_graph"] = edges["u"].isin(largest) & edges["v"].isin(largest)
print(f"baseline graph: {G.number_of_nodes():,} nodes / {G.number_of_edges():,} directed edges "
      f"in largest strongly-connected component  [{time.time()-t0:.0f}s]")

# --------------------------------------------------------------------------------------
# 3. Emergency rooms from Overture places
# --------------------------------------------------------------------------------------
places_file = DATA_DIR / "overture_er_places.parquet"
if not places_file.exists():
    flt = ((pc.field("bbox", "xmin") > W) & (pc.field("bbox", "xmax") < E)
           & (pc.field("bbox", "ymin") > S) & (pc.field("bbox", "ymax") < N)
           & pc.field("id").isin(ER_TABLE["overture_id"].tolist()))
    tbl = ds.dataset(OVERTURE + "theme=places/type=place/", filesystem=S3, format="parquet") \
            .to_table(filter=flt, columns=["id", "names", "addresses", "geometry"])
    pq.write_table(tbl, places_file)
places = pq.read_table(places_file).to_pandas()
places["overture_name"] = places["names"].str["primary"]
places["address"] = places["addresses"].str[0].str["freeform"] + ", " + places["addresses"].str[0].str["locality"]
pts = shapely.from_wkb(places["geometry"].values)
places["lon"], places["lat"] = shapely.get_x(pts), shapely.get_y(pts)
ers = ER_TABLE.merge(places[["id", "overture_name", "address", "lon", "lat"]],
                     left_on="overture_id", right_on="id", how="left").drop(columns="id")
assert ers["lon"].notna().all(), "Overture ids not found:\n" + ers.loc[ers["lon"].isna(), "er_name"].to_string()
ers = gpd.GeoDataFrame(ers, geometry=gpd.points_from_xy(ers["lon"], ers["lat"]), crs="EPSG:4326").to_crs(CRS_PROJ)

tree = cKDTree(nodes[["x", "y"]].values)
dist, idx = tree.query(np.c_[ers.geometry.x, ers.geometry.y])
ers["node"] = nodes.index.values[idx]
ers["snap_m"] = dist.round(0)
ers.drop(columns="geometry").to_csv(OUT_DIR / "er_locations.csv", index=False)
print(ers[["er_name", "county", "include", "address", "snap_m"]].to_string(index=False))
routed = ers[ers["include"]].reset_index(drop=True)

# --------------------------------------------------------------------------------------
# 4. Grid of locations across the three counties
# --------------------------------------------------------------------------------------
xmin, ymin, xmax, ymax = study_area.bounds
xs = np.arange(xmin + GRID_SPACING_M / 2, xmax, GRID_SPACING_M)
ys = np.arange(ymin + GRID_SPACING_M / 2, ymax, GRID_SPACING_M)
gx, gy = np.meshgrid(xs, ys)
grid = gpd.GeoDataFrame({"x": gx.ravel(), "y": gy.ravel()},
                        geometry=gpd.points_from_xy(gx.ravel(), gy.ravel()), crs=CRS_PROJ)
grid = gpd.sjoin(grid, counties_proj[["name", "geometry"]], predicate="within", how="inner")
grid = grid.rename(columns={"name": "county"}).drop(columns="index_right")
grid = grid[~grid.index.duplicated()].reset_index(drop=True)
grid.index.name = "point_id"
dist, idx = tree.query(grid[["x", "y"]].values)
grid["node"] = nodes.index.values[idx]
grid["snap_m"] = dist.round(0)
ll = grid.to_crs("EPSG:4326")
grid["lon"], grid["lat"] = ll.geometry.x.round(5), ll.geometry.y.round(5)
print(f"{len(grid):,} grid points ({grid['county'].value_counts().to_dict()}); "
      f"median snap {grid['snap_m'].median():.0f} m  [{time.time()-t0:.0f}s]")

# --------------------------------------------------------------------------------------
# 5. Baseline: minutes from every grid point to its nearest ER
# --------------------------------------------------------------------------------------
# Driving *to* an ER = shortest path *from* the ER over the reversed graph, one Dijkstra per ER.
Gr = G.reverse(copy=True)
tt = pd.DataFrame(index=grid.index)
for name, node in zip(routed["er_name"], routed["node"]):
    reach = nx.single_source_dijkstra_path_length(Gr, node, weight="travel_min")
    tt[name] = grid["node"].map(reach)
grid["base_min"] = tt.min(axis=1).round(1)
reachable = tt.notna().any(axis=1)
grid["base_er"] = tt[reachable].idxmin(axis=1).reindex(grid.index)
grid.drop(columns="geometry").to_csv(OUT_DIR / "grid_baseline.csv")
print("baseline minutes by county:\n" + grid.groupby("county")["base_min"].describe().round(1).to_string()
      + f"\n[{time.time()-t0:.0f}s]")

# --------------------------------------------------------------------------------------
# 6. FEMA NFHL: flood zones, base flood elevation lines, cross-sections
# --------------------------------------------------------------------------------------
NFHL_LAYERS = {   # key: (local layer name, REST layer name, where, fields)
    "zones": ("S_FLD_HAZ_AR", "Flood Hazard Zones", "SFHA_TF = 'T'",
              "DFIRM_ID,FLD_AR_ID,FLD_ZONE,ZONE_SUBTY,SFHA_TF,STATIC_BFE,DEPTH,LEN_UNIT,V_DATUM"),
    "bfe":   ("S_BFE", "Base Flood Elevations", "1=1", "DFIRM_ID,ELEV,LEN_UNIT,V_DATUM"),
    "xs":    ("S_XS", "Cross-Sections", "1=1", "DFIRM_ID,WSEL_REG,LEN_UNIT,V_DATUM"),
}
nfhl = {}
rest_layer_ids = None
for key, (local_name, rest_name, where, fields) in NFHL_LAYERS.items():
    cache = DATA_DIR / f"nfhl_{key}.gpkg"
    if NFHL_PATH:
        layers = pyogrio.list_layers(NFHL_PATH)[:, 0].tolist()
        if local_name not in layers:
            layer = pd.Series(layers)[pd.Series(layers).str.upper() == local_name]
            layer = layer.iloc[0] if len(layer) else None
        else:
            layer = local_name
        if layer is None:
            lyr = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            source = f"{NFHL_PATH}: no {local_name} layer"
        else:
            file_crs = pyogrio.read_info(NFHL_PATH, layer=layer)["crs"]   # state NFHL ships in NAD83 lon/lat
            box = gpd.GeoSeries([shapely.box(W, S, E, N)], crs="EPSG:4326").to_crs(file_crs).total_bounds
            lyr = gpd.read_file(NFHL_PATH, layer=layer, bbox=tuple(box))
            source = f"{NFHL_PATH} (layer {layer})"
    elif cache.exists():
        lyr = gpd.read_file(cache)
        source = f"{cache} (cached)"
    else:
        if rest_layer_ids is None:
            svc = requests.get(NFHL_URL, params={"f": "json"}, timeout=120).json()
            rest_layer_ids = {l["name"]: l["id"] for l in svc["layers"]}
        url = f"{NFHL_URL}/{rest_layer_ids[rest_name]}/query"
        params = {"where": where, "geometry": f"{W},{S},{E},{N}", "geometryType": "esriGeometryEnvelope",
                  "inSR": 4326, "outSR": 4326, "spatialRel": "esriSpatialRelIntersects", "outFields": fields,
                  "returnGeometry": "true", "geometryPrecision": 6, "f": "geojson",
                  "resultOffset": 0, "resultRecordCount": 1000}
        features = []
        while True:
            r = requests.get(url, params=params, timeout=600)
            r.raise_for_status()
            js = r.json()
            if "error" in js:
                raise RuntimeError(json.dumps(js["error"]))
            features += js.get("features", [])
            print(f"  NFHL {rest_name}: {len(features):,} features so far")
            if not js.get("features") or not js.get("properties", {}).get("exceededTransferLimit", False):
                break
            params["resultOffset"] += len(js["features"])
        lyr = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326") if features else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        lyr.to_file(cache, driver="GPKG")
        source = url
    lyr.columns = [c.upper() if c != "geometry" else c for c in lyr.columns]
    lyr = lyr[~lyr.geometry.isna()].to_crs(CRS_PROJ) if len(lyr) else lyr.set_crs(CRS_PROJ, allow_override=True)
    lyr["to_m"] = np.where(lyr["LEN_UNIT"].astype(str).str.upper().str.startswith("M"), 1.0, 0.3048) if "LEN_UNIT" in lyr else 0.3048
    nfhl[key] = lyr
    print(f"NFHL {key}: {len(lyr):,} features from {source}")

flood = nfhl["zones"]
if "SFHA_TF" in flood.columns:
    flood = flood[flood["SFHA_TF"].astype(str).str.upper().str[0] == "T"]
flood = flood[flood.intersects(study_area.buffer(ROAD_BUFFER_M))].reset_index(drop=True)
for c in ["FLD_ZONE", "ZONE_SUBTY", "V_DATUM"]:
    if c not in flood.columns:
        flood[c] = None
for c in ["STATIC_BFE", "DEPTH"]:
    flood[c] = pd.to_numeric(flood[c], errors="coerce") if c in flood.columns else np.nan
    flood.loc[flood[c] <= -9000, c] = np.nan            # NFHL null sentinel
flood_source = source if NFHL_PATH else (str(DATA_DIR / "nfhl_zones.gpkg"))
print(f"floodplain: {len(flood):,} SFHA polygons, {flood.area.sum()/1e6:,.0f} km2  [{time.time()-t0:.0f}s]")
print(flood["FLD_ZONE"].value_counts(dropna=False).to_string())

# -- known 1 % water-surface points: BFE lines (ELEV) and cross-sections (WSEL_REG), every 20 m
bfe = nfhl["bfe"]
xsec = nfhl["xs"]
known = []
if len(bfe) and "ELEV" in bfe:
    v = pd.to_numeric(bfe["ELEV"], errors="coerce") * bfe["to_m"]
    xy, i = shapely.get_coordinates(shapely.segmentize(bfe.geometry.values, 20), return_index=True)
    known.append(pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "wse_m": v.values[i], "src": "bfe_line",
                               "datum": bfe["V_DATUM"].astype(str).values[i] if "V_DATUM" in bfe else "unknown"}))
if len(xsec) and "WSEL_REG" in xsec:
    v = pd.to_numeric(xsec["WSEL_REG"], errors="coerce") * xsec["to_m"]
    xy, i = shapely.get_coordinates(shapely.segmentize(xsec.geometry.values, 20), return_index=True)
    known.append(pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "wse_m": v.values[i], "src": "cross_section",
                               "datum": xsec["V_DATUM"].astype(str).values[i] if "V_DATUM" in xsec else "unknown"}))
known = pd.concat(known, ignore_index=True) if known else pd.DataFrame(columns=["x", "y", "wse_m", "src", "datum"])
known = known[known["wse_m"].notna() & (known["wse_m"] > -1000)].reset_index(drop=True)
print(f"known water-surface points: {len(known):,} ({known['src'].value_counts().to_dict()})")

# --------------------------------------------------------------------------------------
# 7. Water depth on the road surface
# --------------------------------------------------------------------------------------
# 7a. every edge that touches the floodplain (this is also the old 2-D rule)
hit = gpd.sjoin(edges[["geometry"]], flood[["geometry"]], predicate="intersects", how="inner")
edges["flooded_2d"] = edges.index.isin(hit.index.unique())
cand = edges[edges["flooded_2d"]]
print(f"{len(cand):,} edges touch the floodplain ({cand['length_m'].sum()/1609.344:,.0f} miles)  [{time.time()-t0:.0f}s]")

# 7b. points to look up in the DEM: road samples every SAMPLE_M along candidate edges, plus the
#     vertices of the dissolved floodplain outline (where the 1 % water surface meets the ground)
xy, i = shapely.get_coordinates(shapely.segmentize(cand.geometry.values, SAMPLE_M), return_index=True)
road_pts = pd.DataFrame({"edge_id": cand.index.values[i], "x": xy[:, 0], "y": xy[:, 1], "kind": "road"})
road_pts["order"] = road_pts.groupby("edge_id").cumcount()
sfha_union = flood.buffer(2).union_all().buffer(-2)              # close hairline gaps between adjacent polygons
bxy = shapely.get_coordinates(shapely.segmentize(sfha_union.boundary, 20))
bnd_pts = pd.DataFrame({"edge_id": -1, "x": bxy[:, 0].round(-1), "y": bxy[:, 1].round(-1), "kind": "boundary", "order": 0}) \
            .drop_duplicates(["x", "y"])
bx0, by0, bx1, by1 = study_area.buffer(ROAD_BUFFER_M + 2000).bounds        # long river polygons run past the road box
bnd_pts = bnd_pts[bnd_pts["x"].between(bx0, bx1) & bnd_pts["y"].between(by0, by1)]
# abutments: the two ends of every bridge range, where the deck meets the approach grade
seg_geom = roads.set_index("id")["geom"]
ab_a = shapely.line_interpolate_point(seg_geom.reindex(bridges["id"]).values, bridges["a"].values, normalized=True)
ab_b = shapely.line_interpolate_point(seg_geom.reindex(bridges["id"]).values, bridges["b"].values, normalized=True)
abut_pts = pd.DataFrame({"edge_id": -1, "x": np.r_[shapely.get_x(ab_a), shapely.get_x(ab_b)],
                         "y": np.r_[shapely.get_y(ab_a), shapely.get_y(ab_b)], "kind": "abutment", "order": 0,
                         "bridge_id": np.r_[bridges.index.values, bridges.index.values]})
pts = pd.concat([road_pts, bnd_pts, abut_pts], ignore_index=True)
to_ll = Transformer.from_crs(CRS_PROJ, "EPSG:4326", always_xy=True)
pts["lon"], pts["lat"] = to_ll.transform(pts["x"].values, pts["y"].values)
print(f"{len(road_pts):,} road samples, {len(bnd_pts):,} floodplain-edge vertices to sample  [{time.time()-t0:.0f}s]")

# 7c. DEM: user-supplied files, or the 3DEP 1/3 arc-second tiles (metres, NAVD88) that cover the points
if DEM_FILES:
    dem_files = [Path(p) for p in DEM_FILES]
    dem_desc = ", ".join(DEM_FILES)
else:
    tiles = pd.DataFrame({"lat": np.floor(pts["lat"]).astype(int), "lon": np.floor(pts["lon"]).astype(int)}).drop_duplicates()
    tiles["name"] = "n" + (tiles["lat"] + 1).astype(str).str.zfill(2) + "w" + tiles["lon"].abs().astype(str).str.zfill(3)
    dem_files = []
    for tile in sorted(tiles["name"]):
        f = DEM_DIR / f"USGS_13_{tile}.tif"
        if not f.exists():
            print(f"  downloading 3DEP tile {tile} ...")
            with requests.get(DEM_URL.format(tile=tile), stream=True, timeout=3600) as r:
                r.raise_for_status()
                with open(f, "wb") as fh:
                    for chunk in r.iter_content(1 << 22):
                        fh.write(chunk)
        dem_files.append(f)
    dem_desc = f"USGS 3DEP 1/3 arc-second (~10 m) tiles {', '.join(sorted(tiles['name']))}"

pts["elev_m"] = np.nan
for f in dem_files:
    with rasterio.open(f) as src:
        to_dem = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        px, py = to_dem.transform(pts["lon"].values, pts["lat"].values)
        col = (px - src.transform.c) / src.transform.a
        row = (py - src.transform.f) / src.transform.e
        inside = (col >= 0.5) & (col < src.width - 0.5) & (row >= 0.5) & (row < src.height - 0.5) & pts["elev_m"].isna().values
        if not inside.any():
            continue
        arr = src.read(1).astype("float32")
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        vals = map_coordinates(arr, [row[inside] - 0.5, col[inside] - 0.5], order=1, mode="nearest")
        pts.loc[inside, "elev_m"] = vals
        del arr
print(f"DEM sampled ({dem_desc}); {pts['elev_m'].isna().sum():,} points with no elevation  [{time.time()-t0:.0f}s]")

samples = pts[pts["kind"] == "road"].copy()
boundary = pts[(pts["kind"] == "boundary") & pts["elev_m"].notna()]
deck_elev = pts[pts["kind"] == "abutment"].groupby("bridge_id")["elev_m"].max()   # deck ~ higher abutment

# 7d. which zone each road sample sits in (samples outside every SFHA polygon are dry)
samples = gpd.GeoDataFrame(samples, geometry=gpd.points_from_xy(samples["x"], samples["y"]), crs=CRS_PROJ)
zone_cols = ["FLD_ZONE", "ZONE_SUBTY", "STATIC_BFE", "DEPTH", "V_DATUM", "to_m"]
samples = gpd.sjoin(samples, flood[zone_cols + ["geometry"]], predicate="within", how="left")
samples = samples[~samples.index.duplicated()].drop(columns="index_right")
samples["in_sfha"] = samples["FLD_ZONE"].notna()
in_sfha = samples["in_sfha"].values

# 7e. water-surface elevation, best source first:
#     ao_depth  : zone AO carries a depth directly
#     static_bfe: polygon-level BFE (ponding / lakes / AH)
#     bfe_line  : inverse-distance between the two nearest distinct BFE lines or cross-sections
#     floodplain_edge: median DEM elevation of the nearest floodplain-outline vertices (inferred)
samples["wse_m"] = np.nan
samples["wse_source"] = np.where(in_sfha, "none", "outside_sfha")
samples["wse_dist_m"] = np.nan
if len(boundary):
    btree = cKDTree(boundary[["x", "y"]].values)
    d, j = btree.query(samples.loc[in_sfha, ["x", "y"]].values, k=min(EDGE_K, len(boundary)))
    d, j = np.atleast_2d(d).reshape(in_sfha.sum(), -1), np.atleast_2d(j).reshape(in_sfha.sum(), -1)
    samples.loc[in_sfha, "wse_m"] = np.median(boundary["elev_m"].values[j], axis=1)
    samples.loc[in_sfha, "wse_dist_m"] = d[:, 0]
    samples.loc[in_sfha, "wse_source"] = "floodplain_edge"
if len(known):
    ktree = cKDTree(known[["x", "y"]].values)
    k = min(12, len(known))
    d, j = ktree.query(samples.loc[in_sfha, ["x", "y"]].values, k=k)
    long = pd.DataFrame({"sample": np.repeat(samples.index[in_sfha], k), "dist": np.ravel(d),
                         "wse": known["wse_m"].values[np.ravel(j)], "src": known["src"].values[np.ravel(j)]})
    long = long[long["dist"] <= BFE_MAX_DIST_M].sort_values("dist").drop_duplicates(["sample", "wse"])
    long = long.groupby("sample").head(2)
    long["w"] = 1 / long["dist"].clip(lower=1)
    idw = long.groupby("sample").apply(lambda g: pd.Series({"wse": np.average(g["wse"], weights=g["w"]),
                                                            "dist": g["dist"].min(), "src": g["src"].iloc[0]}),
                                       include_groups=False)
    samples.loc[idw.index, "wse_m"] = idw["wse"]
    samples.loc[idw.index, "wse_dist_m"] = idw["dist"]
    samples.loc[idw.index, "wse_source"] = idw["src"]
static = samples["STATIC_BFE"].notna() & in_sfha
samples.loc[static, "wse_m"] = samples.loc[static, "STATIC_BFE"] * samples.loc[static, "to_m"]
samples.loc[static, "wse_source"] = "static_bfe"
samples.loc[static, "wse_dist_m"] = 0

# 7f. road surface: DEM at the sample, except on a bridge deck, which bare-earth DEMs remove:
#     samples inside a bridge range take the higher of that bridge's two abutment elevations
samples = samples.sort_values(["edge_id", "order"])
samples["road_elev_m"] = samples["elev_m"]
step = np.hypot(samples.groupby("edge_id")["x"].diff().fillna(0), samples.groupby("edge_id")["y"].diff().fillna(0))
samples["along_m"] = step.groupby(samples["edge_id"]).cumsum()
e_at = edges["at"].reindex(samples["edge_id"]).values
e_span = (edges["at_next"] - edges["at"]).reindex(samples["edge_id"]).values
samples["seg_at"] = e_at + e_span * samples["along_m"].values / edges["length_m"].reindex(samples["edge_id"]).values
onb = samples[["edge_id", "seg_at"]].assign(segment_id=edges["segment_id"].reindex(samples["edge_id"]).values) \
        .reset_index().merge(bridges.reset_index(), left_on="segment_id", right_on="id")
onb = onb[(onb["seg_at"] >= onb["a"]) & (onb["seg_at"] <= onb["b"])].drop_duplicates("index").set_index("index")
samples["on_bridge"] = samples.index.isin(onb.index)
samples.loc[onb.index, "road_elev_m"] = onb["bridge_id"].map(deck_elev)

# 7g. depth on the pavement
samples["depth_m"] = np.where(in_sfha, samples["wse_m"] - samples["road_elev_m"], 0.0)
ao = (samples["FLD_ZONE"].astype(str).str.upper() == "AO") & samples["DEPTH"].notna() & in_sfha
samples.loc[ao, "depth_m"] = samples.loc[ao, "DEPTH"] * samples.loc[ao, "to_m"]
samples.loc[ao, "wse_source"] = "ao_depth"
samples.loc[in_sfha & samples["wse_m"].isna() & ~ao, "depth_m"] = np.nan
samples.loc[in_sfha & samples["road_elev_m"].isna() & ~ao, "depth_m"] = np.nan

# 7h. roll up to the edge: worst sample decides
worst = samples.sort_values("depth_m", ascending=False, na_position="last").drop_duplicates("edge_id").set_index("edge_id")
per_edge = samples.groupby("edge_id").agg(n_samples=("depth_m", "size"),
                                          n_in_sfha=("in_sfha", "sum"),
                                          n_unknown=("depth_m", lambda s: int(s.isna().sum())),
                                          road_elev_min_m=("road_elev_m", "min"))
per_edge["depth_max_m"] = worst["depth_m"]
per_edge["wse_source"] = worst["wse_source"]
per_edge["wse_m"] = worst["wse_m"]
per_edge["fld_zone"] = worst["FLD_ZONE"]
per_edge["depth_unknown"] = per_edge["depth_max_m"].isna() & (per_edge["n_in_sfha"] > 0)
for c in per_edge.columns:
    edges[c] = per_edge[c]
edges["closed_2d"] = edges["flooded_2d"]
edges["closed_depth"] = (edges["depth_max_m"] >= PASSABLE_DEPTH_M) | edges["depth_unknown"].fillna(False)
edges["closed"] = edges["closed_depth"] if CLOSURE_RULE == "depth" else edges["closed_2d"]
edges["closed"] = edges["closed"].fillna(False) & ~(edges["is_bridge"] & BRIDGES_PASSABLE)

closed = edges[edges["closed"] & edges["in_graph"]]
miles = lambda s: s.sum() / 1609.344
print(f"closures ({CLOSURE_RULE} rule): {len(closed):,} of {edges['in_graph'].sum():,} edges, "
      f"{miles(closed['length_m']):,.0f} miles; 2-D rule would close {miles(edges.loc[edges['closed_2d'] & edges['in_graph'], 'length_m']):,.0f} miles  "
      f"[{time.time()-t0:.0f}s]")
print("floodplain edges by water-surface source (miles):\n"
      + (cand.assign(src=edges["wse_source"], closed=edges["closed_depth"])
             .groupby(["src", "closed"])["length_m"].sum().div(1609.344).round(0).unstack(fill_value=0).to_string()))
print("closed miles by class:\n" + (closed.groupby("class")["length_m"].sum() / 1609.344).round(0).to_string())
cand_out = edges.loc[edges["flooded_2d"], ["u", "v", "segment_id", "class", "is_bridge", "length_m", "fld_zone",
                                            "wse_source", "wse_m", "road_elev_min_m", "depth_max_m",
                                            "depth_unknown", "closed_2d", "closed_depth", "closed", "geometry"]]
cand_out.to_crs("EPSG:4326").to_file(OUT_DIR / "floodplain_road_edges.gpkg", driver="GPKG")
samples.drop(columns="geometry").to_parquet(OUT_DIR / "road_depth_samples.parquet")

# --------------------------------------------------------------------------------------
# 8. Travel time with the closures applied
# --------------------------------------------------------------------------------------
keep = pd.concat([
    edges.loc[edges["fwd_ok"] & ~edges["closed"], ["u", "v", "travel_min"]],
    edges.loc[edges["bwd_ok"] & ~edges["closed"], ["v", "u", "travel_min"]].rename(columns={"v": "u", "u": "v"}),
]).groupby(["u", "v"], as_index=False)["travel_min"].min()
Gf = nx.from_pandas_edgelist(keep, "u", "v", ["travel_min"], create_using=nx.DiGraph)
Gf = Gf.subgraph(largest).copy()          # same node universe as the baseline
Gfr = Gf.reverse(copy=True)
ttf = pd.DataFrame(index=grid.index)
for name, node in zip(routed["er_name"], routed["node"]):
    reach = nx.single_source_dijkstra_path_length(Gfr, node, weight="travel_min") if node in Gfr else {}
    ttf[name] = grid["node"].map(reach)
grid["flood_min"] = ttf.min(axis=1).round(1)
reachable = ttf.notna().any(axis=1)
grid["flood_er"] = ttf[reachable].idxmin(axis=1).reindex(grid.index)
grid["delta_min"] = (grid["flood_min"] - grid["base_min"]).round(1)
grid["pct_increase"] = (100 * grid["delta_min"] / grid["base_min"]).round(0)
grid["status"] = np.select(
    [grid["flood_min"].isna(), grid["delta_min"] > 0.05, grid["delta_min"] <= 0.05],
    ["unreachable", "slower", "unchanged"], default="unreachable")
grid["er_changed"] = (grid["flood_er"] != grid["base_er"]) & grid["flood_er"].notna()
grid["flood_source"] = flood_source
grid["closure_rule"] = CLOSURE_RULE
grid["passable_depth_m"] = PASSABLE_DEPTH_M
grid["bridges_passable"] = BRIDGES_PASSABLE
grid.drop(columns="geometry").to_csv(OUT_DIR / "grid_flood_impact.csv")
grid.to_crs("EPSG:4326").to_file(OUT_DIR / "grid_flood_impact.gpkg", driver="GPKG")

# --------------------------------------------------------------------------------------
# 9. Summary tables
# --------------------------------------------------------------------------------------
both = pd.concat([grid, grid.assign(county="All three")])
summary = both.groupby("county").agg(
    points=("status", "size"),
    base_median_min=("base_min", "median"),
    flood_median_min=("flood_min", "median"),
    delta_mean_min=("delta_min", "mean"),
    delta_median_min=("delta_min", "median"),
    delta_p90_min=("delta_min", lambda s: s.quantile(0.9)),
    pct_slower=("status", lambda s: 100 * (s == "slower").mean()),
    pct_unreachable=("status", lambda s: 100 * (s == "unreachable").mean()),
    pct_over_15_min_worse=("delta_min", lambda s: 100 * (s > 15).mean()),
    pct_er_changed=("er_changed", lambda s: 100 * s.mean()),
).round(1)
summary.to_csv(OUT_DIR / "summary_by_county.csv")
print(f"\nSUMMARY (minutes to nearest ER, {CLOSURE_RULE} closure rule)\n" + summary.to_string())

er_use = pd.DataFrame({"baseline": grid["base_er"].value_counts(), "flood": grid["flood_er"].value_counts()}).fillna(0).astype(int)
er_use.to_csv(OUT_DIR / "nearest_er_counts.csv")
print("\nGrid points served by each ER:\n" + er_use.to_string())

# --------------------------------------------------------------------------------------
# 10. Method flags: everything inferred, approximated or not directly possible
# --------------------------------------------------------------------------------------
fp = edges[edges["flooded_2d"]]
src_miles = fp.groupby("wse_source")["length_m"].sum() / 1609.344
datums = pd.concat([flood["V_DATUM"].astype(str), known["datum"].astype(str)]).str.upper()
flags += [
    ("dem_resolution", len(dem_files), "tiles",
     f"Road surface from {dem_desc}. A ~10 m cell averages pavement with roadside ditches and cannot resolve a narrow "
     f"embankment, so road elevations are biased low on raised rural roads (closures over-called). Pass 1 m 3DEP lidar with --dem."),
    ("bridge_deck_inferred", int(samples["on_bridge"].sum()), "road samples",
     f"Bare-earth DEMs remove bridge decks. Samples inside an Overture bridge range ({int(fp['is_bridge'].sum())} floodplain edges) take the "
     "higher of the bridge's two abutment elevations (approach grade). Bridges are therefore judged by their abutments, not the span, "
     "and a deck that sags below its abutments, scour and debris are not modelled."),
    ("wse_inferred_from_floodplain_edge", round(float(src_miles.get("floodplain_edge", 0)), 1), "miles of floodplain road",
     f"No BFE line, cross-section or static BFE within {BFE_MAX_DIST_M} m. Water surface = median DEM elevation of the {EDGE_K} nearest "
     "vertices of the dissolved floodplain outline (where the 1 % water meets the ground). Inherits the FEMA boundary's positional error "
     "(Zone A outlines were often drawn on coarse contours) and ignores the along-valley slope between the road and the outline."),
    ("wse_from_bfe_lines", round(float(src_miles.get("bfe_line", 0) + src_miles.get("cross_section", 0)), 1), "miles of floodplain road",
     "Water surface interpolated (inverse distance) between the two nearest distinct BFE lines / cross-section WSELs. BFE lines are whole-foot "
     "contours of the modelled surface, so +/-0.15 m is inherent."),
    ("wse_static_bfe", round(float(src_miles.get("static_bfe", 0)), 1), "miles of floodplain road",
     "Polygon-level STATIC_BFE used directly (ponding, lakes, Zone AH)."),
    ("wse_none", round(float(src_miles.get("none", 0)), 1), "miles of floodplain road",
     "No water-surface estimate at all (no outline vertex with a DEM value nearby). Treated as closed."),
    ("zone_a_no_bfe", round(float(flood.loc[flood["FLD_ZONE"].astype(str).str.upper() == "A"].area.sum() / 1e6), 1), "km2",
     "Approximate Zone A has no FEMA elevation; every road there relies on the floodplain-edge inference above."),
    ("zone_ao_depth", int(ao.sum()), "road samples",
     "Zone AO (sheet flow) carries a depth attribute which is used as the depth on the pavement directly."),
    ("depth_unknown_closed", int(edges["depth_unknown"].fillna(False).sum()), "edges",
     "Road elevation or water surface missing (DEM nodata / no estimate); closed conservatively."),
    ("ngvd29_records", int((datums == "NGVD29").sum()), "NFHL features",
     "FEMA elevations referenced to NGVD29 are compared to the NAVD88 DEM without conversion (offset in NW Arkansas is roughly "
     "0.0 to 0.2 m). Convert with VERTCON if this count is not zero."),
    ("passable_depth_threshold", PASSABLE_DEPTH_M, "m",
     "A road is impassable when the worst sample along the edge has at least this much water on the pavement. "
     "0.15 m (6 in) stalls most cars; 0.3 m floats them. Sensitivity: --passable-depth."),
    ("sample_interval", SAMPLE_M, "m",
     "Depth is checked at points this far apart along each edge; a dip narrower than this can be missed."),
    ("static_water_surface", 1, "assumption",
     "The NFHL 1 % surface is a regulatory, steady-state elevation. Timing, velocity, duration, drainage failure and "
     "flooding outside the mapped SFHA are not represented."),
    ("outline_seams", 1, "assumption",
     "Where the SFHA outline follows a map-panel or study limit rather than the water's edge, the inferred water surface there is wrong. "
     "Not quantified."),
    ("untagged_low_water_crossings", 1, "assumption",
     "Culverts and low-water crossings are not tagged in Overture and are treated as ground-level road, which is the correct direction."),
    ("closure_rule", CLOSURE_RULE, "setting", "depth = water on the pavement; 2d = every edge touching the floodplain (upper bound)."),
]
pd.DataFrame(flags, columns=["flag", "value", "unit", "description"]).to_csv(OUT_DIR / "method_flags.csv", index=False)
print("\nMETHOD FLAGS\n" + pd.DataFrame(flags, columns=["flag", "value", "unit", "description"])[["flag", "value", "unit"]].to_string(index=False))

# --------------------------------------------------------------------------------------
# 11. Maps (single-hue sequential ramps; unreachable in one reserved colour with a legend)
# --------------------------------------------------------------------------------------
routed_in = routed[routed.geometry.within(study_area.buffer(5000))]
for col, cmap, title, fname in [
    ("base_min", "Blues", "Baseline drive time to nearest ER (minutes)", "map_baseline_minutes.png"),
    ("delta_min", "Oranges", f"Added drive time with 100-year floodplain roads closed ({CLOSURE_RULE} rule, minutes)", "map_flood_increase.png"),
]:
    fig, ax = plt.subplots(figsize=(11, 9))
    counties_proj.boundary.plot(ax=ax, color="#555", linewidth=0.8)
    if col == "delta_min":
        closed.plot(ax=ax, color="#b3b3b3", linewidth=0.3, zorder=1)
    ok = grid[grid["status"] != "unreachable"] if col == "delta_min" else grid[grid[col].notna()]
    vmax = max(np.nanpercentile(ok[col], 98), 1)
    ok.plot(ax=ax, column=col, cmap=cmap, markersize=9, vmin=0, vmax=vmax, legend=True,
            legend_kwds={"shrink": 0.6, "label": "minutes"}, zorder=2)
    bad = grid[grid["status"] == "unreachable"] if col == "delta_min" else grid[grid[col].isna()]
    if len(bad):
        bad.plot(ax=ax, color="#7b1fa2", marker="x", markersize=14, zorder=3)
    routed_in.plot(ax=ax, color="#d32f2f", marker="P", markersize=70, edgecolor="white", linewidth=0.6, zorder=4)
    handles = [Line2D([], [], marker="P", color="#d32f2f", markeredgecolor="white", linestyle="", markersize=10, label="24/7 emergency room")]
    if col == "delta_min":
        handles += [Line2D([], [], color="#b3b3b3", linewidth=2, label="road closed"),
                    Line2D([], [], marker="x", color="#7b1fa2", linestyle="", markersize=8, label="no ER reachable")]
    ax.legend(handles=handles, loc="lower left", frameon=True)
    ax.set_xlim(xmin - 4000, xmax + 4000)
    ax.set_ylim(ymin - 4000, ymax + 4000)
    ax.set_title(title + "\nBenton, Washington and Madison Counties, AR", fontsize=12)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)

print(f"\ndone in {time.time()-t0:.0f}s; outputs in {OUT_DIR}/")
