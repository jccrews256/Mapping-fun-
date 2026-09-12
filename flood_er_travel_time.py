"""
Flood impact on emergency-room travel time
Benton, Washington and Madison Counties, Arkansas

One straight-line script, pandas style, top to bottom.  Steps:

  1. County boundaries ............ Overture Maps "divisions" theme (OSM-derived)
  2. Drivable road network ........ Overture Maps "transportation" theme (OSM-derived, monthly)
  3. Emergency rooms .............. Overture Maps "places" theme, curated to 24/7 EDs
  4. Grid of locations ............ regular 1 km lattice clipped to the three counties
  5. Baseline travel time ......... NetworkX Dijkstra from every ER over the reversed graph
  6. 100-year floodplain .......... FEMA National Flood Hazard Layer, SFHA_TF = 'T'
                                    (Zones A, AE, AH, AO, AR, A99, V, VE = 1 % annual chance)
  7. Closures ..................... every road edge that intersects the floodplain is removed
  8. Flood travel time ............ Dijkstra again, compare, map, summarise

Data are cached in ./data so the script can be re-run offline.  Outputs go to ./output.

Run:
    python flood_er_travel_time.py                      # download NFHL from FEMA's REST service
    python flood_er_travel_time.py --nfhl NFHL_05.zip   # or use a state/county NFHL you downloaded
    python flood_er_travel_time.py --bridges-passable   # optional: keep bridges open

FEMA's NFHL is served from hazards.fema.gov; if that host is not reachable from where you run
this, download the Arkansas state NFHL from https://msc.fema.gov (Search All Products ->
Effective Products -> NFHL Data-State) and pass it with --nfhl.
"""

import argparse
import json
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
import requests
import shapely
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
from shapely.ops import substring

matplotlib.use("Agg")

# --------------------------------------------------------------------------------------
# 0. Settings
# --------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--nfhl", default=os.environ.get("NFHL_PATH"),
                    help="Local NFHL file (.zip/.gdb/.gpkg/.shp with S_FLD_HAZ_AR). Default: download from FEMA REST.")
parser.add_argument("--release", default="2026-08-19.0", help="Overture Maps release")
parser.add_argument("--spacing", type=float, default=1000.0, help="grid spacing in metres")
parser.add_argument("--bridges-passable", action="store_true",
                    help="keep road edges flagged as bridges open even when they cross the floodplain")
args = parser.parse_args()

RELEASE = args.release
GRID_SPACING_M = args.spacing
BRIDGES_PASSABLE = args.bridges_passable
NFHL_PATH = args.nfhl

COUNTIES = ["Benton County", "Washington County", "Madison County"]
CRS_PROJ = "EPSG:26915"            # NAD83 / UTM zone 15N - metres, covers NW Arkansas
ROAD_BUFFER_M = 35_000             # pull roads/ERs this far beyond the county lines
DATA_DIR = Path("data")
OUT_DIR = Path("output")
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

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
print(f"Overture release {RELEASE}; grid {GRID_SPACING_M:.0f} m; bridges passable = {BRIDGES_PASSABLE}")

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
roads["is_bridge"] = roads["flags"].str.contains("is_bridge")
roads = roads[~roads["flags"].str.contains("is_under_construction|is_abandoned")]

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
pairs = pairs.merge(roads[["id", "class", "speed_mph", "is_bridge", "fwd_ok", "bwd_ok", "geom"]], on="id")

edge_geoms = []
for g, a, b in zip(pairs["geom"], pairs["at"], pairs["at_next"]):
    edge_geoms.append(substring(g, a, b, normalized=True))
edges = gpd.GeoDataFrame(
    {"u": pairs["node"].values, "v": pairs["node_next"].values, "segment_id": pairs["id"].values,
     "class": pairs["class"].values, "speed_mph": pairs["speed_mph"].values,
     "is_bridge": pairs["is_bridge"].values, "fwd_ok": pairs["fwd_ok"].values, "bwd_ok": pairs["bwd_ok"].values},
    geometry=edge_geoms, crs=CRS_PROJ)
edges["length_m"] = edges.length
edges["travel_min"] = edges["length_m"] / (edges["speed_mph"] * 1609.344 / 60)
edges = edges[edges["length_m"] > 0].reset_index(drop=True)
edges.index.name = "edge_id"

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
# 6. 100-year floodplain: FEMA NFHL Special Flood Hazard Area
# --------------------------------------------------------------------------------------
flood_file = DATA_DIR / "nfhl_sfha_100yr.gpkg"
if NFHL_PATH:
    import pyogrio
    layers = pyogrio.list_layers(NFHL_PATH)[:, 0].tolist()
    layer = "S_FLD_HAZ_AR" if "S_FLD_HAZ_AR" in layers else layers[0]
    file_crs = pyogrio.read_info(NFHL_PATH, layer=layer)["crs"]      # state NFHL ships in NAD83 lon/lat
    box = gpd.GeoSeries([shapely.box(W, S, E, N)], crs="EPSG:4326").to_crs(file_crs).total_bounds
    flood = gpd.read_file(NFHL_PATH, layer=layer, bbox=tuple(box))
    flood.columns = [c.upper() if c.lower() in ("fld_zone", "zone_subty", "sfha_tf", "dfirm_id") else c for c in flood.columns]
    flood_source = f"{NFHL_PATH} (layer {layer})"
elif flood_file.exists():
    flood = gpd.read_file(flood_file)
    flood_source = str(flood_file) + " (cached)"
else:
    # FEMA NFHL MapServer, layer 28 = S_FLD_HAZ_AR (flood hazard zones); paginate through the bbox
    url = "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28/query"
    params = {"where": "SFHA_TF = 'T'", "geometry": f"{W},{S},{E},{N}", "geometryType": "esriGeometryEnvelope",
              "inSR": 4326, "outSR": 4326, "spatialRel": "esriSpatialRelIntersects",
              "outFields": "DFIRM_ID,FLD_AR_ID,FLD_ZONE,ZONE_SUBTY,SFHA_TF,STATIC_BFE",
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
        print(f"  NFHL features so far: {len(features):,}")
        if not js.get("features") or not js.get("properties", {}).get("exceededTransferLimit", False):
            break
        params["resultOffset"] += len(js["features"])
    flood = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    flood.to_file(flood_file, driver="GPKG")
    flood_source = url
if "SFHA_TF" in flood.columns:
    flood = flood[flood["SFHA_TF"].astype(str).str.upper().str[0] == "T"]
flood = flood[~flood.geometry.isna()].to_crs(CRS_PROJ)
flood = flood[flood.intersects(study_area.buffer(ROAD_BUFFER_M))]
print(f"floodplain: {len(flood):,} SFHA polygons, {flood.area.sum()/1e6:,.0f} km2, from {flood_source}  [{time.time()-t0:.0f}s]")
if "FLD_ZONE" in flood.columns:
    print(flood["FLD_ZONE"].value_counts().to_string())

# --------------------------------------------------------------------------------------
# 7. Road closures: every edge touching the floodplain
# --------------------------------------------------------------------------------------
hit = gpd.sjoin(edges[["geometry"]], flood[["geometry"]], predicate="intersects", how="inner")
edges["flooded"] = edges.index.isin(hit.index.unique())
edges["closed"] = edges["flooded"] & ~(edges["is_bridge"] & BRIDGES_PASSABLE)
edges["in_graph"] = edges["u"].isin(largest) & edges["v"].isin(largest)
closed = edges[edges["closed"] & edges["in_graph"]]
print(f"closures: {len(closed):,} of {edges['in_graph'].sum():,} edges "
      f"({closed['length_m'].sum()/1609.344:,.0f} of {edges.loc[edges['in_graph'], 'length_m'].sum()/1609.344:,.0f} miles)  "
      f"[{time.time()-t0:.0f}s]")
print("closed miles by class:\n" + (closed.groupby("class")["length_m"].sum() / 1609.344).round(0).to_string())
closed[["u", "v", "segment_id", "class", "is_bridge", "length_m", "geometry"]].to_crs("EPSG:4326") \
    .to_file(OUT_DIR / "flooded_road_edges.gpkg", driver="GPKG")

# --------------------------------------------------------------------------------------
# 8. Travel time with the floodplain impassable
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
grid["bridges_passable"] = BRIDGES_PASSABLE
grid.drop(columns="geometry").to_csv(OUT_DIR / "grid_flood_impact.csv")
grid.to_crs("EPSG:4326").to_file(OUT_DIR / "grid_flood_impact.gpkg", driver="GPKG")

# --------------------------------------------------------------------------------------
# 9. Summary tables
# --------------------------------------------------------------------------------------
summary = grid.groupby("county").agg(
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
summary.loc["All three"] = grid.assign(county="All three").groupby("county").agg(
    points=("status", "size"), base_median_min=("base_min", "median"), flood_median_min=("flood_min", "median"),
    delta_mean_min=("delta_min", "mean"), delta_median_min=("delta_min", "median"),
    delta_p90_min=("delta_min", lambda s: s.quantile(0.9)),
    pct_slower=("status", lambda s: 100 * (s == "slower").mean()),
    pct_unreachable=("status", lambda s: 100 * (s == "unreachable").mean()),
    pct_over_15_min_worse=("delta_min", lambda s: 100 * (s > 15).mean()),
    pct_er_changed=("er_changed", lambda s: 100 * s.mean())).round(1).iloc[0]
summary.to_csv(OUT_DIR / "summary_by_county.csv")
print("\nSUMMARY (minutes to nearest ER, floodplain impassable)\n" + summary.to_string())

er_use = pd.DataFrame({"baseline": grid["base_er"].value_counts(), "flood": grid["flood_er"].value_counts()}).fillna(0).astype(int)
er_use.to_csv(OUT_DIR / "nearest_er_counts.csv")
print("\nGrid points served by each ER:\n" + er_use.to_string())

# --------------------------------------------------------------------------------------
# 10. Maps (single-hue sequential ramps; unreachable in one reserved colour with a legend)
# --------------------------------------------------------------------------------------
routed_in = routed[routed.geometry.within(study_area.buffer(5000))]
for col, cmap, title, fname in [
    ("base_min", "Blues", "Baseline drive time to nearest ER (minutes)", "map_baseline_minutes.png"),
    ("delta_min", "Oranges", "Added drive time with the 100-year floodplain impassable (minutes)", "map_flood_increase.png"),
]:
    fig, ax = plt.subplots(figsize=(11, 9))
    counties_proj.boundary.plot(ax=ax, color="#555", linewidth=0.8)
    if col == "delta_min":
        closed.plot(ax=ax, color="#b3b3b3", linewidth=0.3, zorder=1)
    ok = grid[grid["status"] != "unreachable"] if col == "delta_min" else grid[grid[col].notna()]
    vmax = np.nanpercentile(ok[col], 98)
    ok.plot(ax=ax, column=col, cmap=cmap, markersize=9, vmin=0, vmax=vmax, legend=True,
            legend_kwds={"shrink": 0.6, "label": "minutes"}, zorder=2)
    bad = grid[grid["status"] == "unreachable"] if col == "delta_min" else grid[grid[col].isna()]
    if len(bad):
        bad.plot(ax=ax, color="#7b1fa2", marker="x", markersize=14, zorder=3)
    routed_in.plot(ax=ax, color="#d32f2f", marker="P", markersize=70, edgecolor="white", linewidth=0.6, zorder=4)
    handles = [Line2D([], [], marker="P", color="#d32f2f", markeredgecolor="white", linestyle="", markersize=10, label="24/7 emergency room")]
    if col == "delta_min":
        handles += [Line2D([], [], color="#b3b3b3", linewidth=2, label="road closed (in floodplain)"),
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
