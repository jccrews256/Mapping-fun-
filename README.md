# Flood impact on emergency-room travel time, Northwest Arkansas

How much longer does it take to drive to the nearest emergency room from every part of
Benton, Washington and Madison Counties, Arkansas, when the roads that would be under water in
the FEMA 100-year flood are impassable?

Everything lives in one straight-line script, `flood_er_travel_time.py`, written pandas-first
with plain loops (no helper functions). Open-source stack: pandas, GeoPandas/Shapely for the
overlays, rasterio for the DEM, NetworkX for routing, pyarrow to read Overture Maps from S3.

## Data

| Layer | Source | Notes |
|---|---|---|
| County boundaries | Overture Maps `divisions` theme (OSM-derived), release 2026-08-19.0 | Benton, Washington, Madison |
| Road network | Overture Maps `transportation` theme (OSM-derived, refreshed monthly) | drivable classes only; private roads dropped; one-ways honoured; posted speed limits where present, otherwise class defaults; bridge flag kept |
| Emergency rooms | Overture Maps `places` theme, curated in `ER_TABLE` | 24/7 EDs in the three counties plus the ring of neighbouring-county ERs (Berryville, Eureka Springs, Harrison, Clarksville, Ozark, Van Buren, Stilwell OK, Grove OK, Cassville MO) so "nearest" is right near county lines. Arkansas Children's (pediatric) and the VA (veterans) are listed but not routed to |
| 100-year floodplain | FEMA National Flood Hazard Layer: `S_FLD_HAZ_AR` where `SFHA_TF = 'T'`, plus `S_BFE` base-flood-elevation lines and `S_XS` cross-sections | the Special Flood Hazard Area = 1 % annual-chance flood = zones A, AE, AH, AO, AR, A99, V, VE. Zone X (0.2 %) is *not* closed |
| Road surface elevation | USGS 3DEP 1/3 arc-second DEM (about 10 m), NAVD88 metres, pulled from the National Map S3 bucket | swap in 1 m lidar with `--dem` (3DEP has 1 m projects for Benton and Washington Counties) |

Roads and ERs are pulled for a 35 km ring around the counties so routes and nearest-ER
choices are not cut off at the county line. Everything downloaded is cached in `data/`.

## Method

1. Build a directed road graph: each Overture segment is split at its connectors, each piece
   gets `travel_min = length / speed`, both directions are added unless the segment is one-way,
   and the largest strongly-connected component is kept.
2. Lay a 1 km lattice over the three counties (about 6,900 points) and snap each point, and
   each ER, to its nearest graph node.
3. Baseline: one Dijkstra per ER over the *reversed* graph gives minutes from every node to
   that ER; the minimum across ERs is the nearest-ER time.
4. Find every road edge that touches the SFHA and sample the DEM every 10 m along it. That is
   the road surface, except on bridges, where the deck is missing from the bare-earth DEM: samples
   inside an Overture bridge range take the higher of that bridge's two abutment elevations (the
   approach grade).
5. Put a 1 % water-surface elevation on every sample inside the floodplain, best source first:
   - Zone AO: FEMA's depth attribute is used directly.
   - `STATIC_BFE` on the polygon (ponding, lakes, Zone AH).
   - BFE lines / cross-section `WSEL_REG`: inverse-distance between the two nearest distinct
     values within 1.5 km.
   - Otherwise (most of Zone A) the surface is **inferred from the floodplain edge**: the SFHA
     polygons are dissolved, the outline is sampled every 20 m from the DEM, and each road sample
     takes the median elevation of its 8 nearest outline vertices. The outline is, by definition,
     where the 1 % water meets the ground, so its elevation approximates the water surface.
6. Depth on the pavement = water surface minus road surface. An edge is closed when its worst
   sample is at or above `--passable-depth` (default 0.15 m, six inches). Samples with no
   elevation or no water-surface estimate close the edge conservatively.
7. Repeat step 3 on the reduced graph and compare. `--closure-rule 2d` reproduces the old
   planimetric rule (every edge touching the floodplain) as an upper bound.

Every inference and gap is written to `output/method_flags.csv` with a count: miles of road
whose water surface came from each source, bridge edges whose deck was inferred, Zone A area
with no FEMA elevation, NGVD29 records compared without datum conversion, DEM resolution, the
depth threshold, and the assumptions that cannot be quantified (static surface, outline seams,
untagged low-water crossings).

## Running it

**In Colab:** open `colab_run_flood_er.ipynb` (File → Open notebook → GitHub, paste the repo URL,
pick this branch). It clones the branch, installs the stack, checks FEMA's service, runs the default
scenario and three sensitivity scenarios into `output/<scenario>/`, shows the tables and maps, and
zips the results. Output directories are set with `--out`.

**Locally:**

```bash
pip install -r requirements.txt
python flood_er_travel_time.py                      # pulls NFHL from FEMA's REST service, DEM from USGS S3
python flood_er_travel_time.py --nfhl NFHL_05.zip   # or point at a downloaded state/county NFHL
python flood_er_travel_time.py --dem nwa_1m.vrt     # better DEM (metres, NAVD88)
python flood_er_travel_time.py --passable-depth 0.3 # sensitivity on the depth threshold
python flood_er_travel_time.py --closure-rule 2d    # planimetric upper bound
python flood_er_travel_time.py --bridges-passable   # never close bridge decks
```

FEMA serves the NFHL from `hazards.fema.gov` (MapServer layers "Flood Hazard Zones", "Base
Flood Elevations", "Cross-Sections"). If that host is blocked where you run this, download the
Arkansas state NFHL from https://msc.fema.gov (Search All Products → Effective Products → NFHL
Data-State) and pass the zip with `--nfhl`; the script reads `S_FLD_HAZ_AR`, `S_BFE` and `S_XS`
from it. A full run takes about seven minutes after the downloads are cached; the six DEM tiles
are about 2.7 GB.

## Outputs (`output/`)

| File | Contents |
|---|---|
| `er_locations.csv` | the ER table with Overture address, coordinates and snap distance |
| `grid_baseline.csv` | every grid point with county, nearest ER and baseline minutes |
| `grid_flood_impact.csv` / `.gpkg` | baseline vs flood minutes, nearest ER before/after, added minutes, % increase, status (`unchanged` / `slower` / `unreachable`) |
| `floodplain_road_edges.gpkg` | every edge touching the floodplain with zone, water-surface source and value, road elevation, max depth, and the 2-D and depth closure flags |
| `road_depth_samples.parquet` | the 10 m samples behind those edges |
| `summary_by_county.csv` | medians, mean/median/p90 added minutes, share slower, share unreachable, share worse by more than 15 min |
| `nearest_er_counts.csv` | how many grid points each ER serves before and after |
| `method_flags.csv` | every inference, approximation and gap with a count |
| `map_baseline_minutes.png`, `map_flood_increase.png` | maps |

## Assumptions and caveats

- Free-flow speeds: posted limits from Overture where available (about a third of segments),
  otherwise 65/55/45/40/35/30/25/15 mph by class from motorway down to service roads. No
  congestion, no emergency-vehicle privileges.
- The 10 m DEM smears a narrow embankment into the ditch beside it, so raised rural roads read
  low and closures are over-called there. Use 1 m lidar via `--dem` for a tighter answer.
- The floodplain-edge inference inherits whatever positional error FEMA's Zone A outline has
  and ignores the water-surface slope between the road and the outline.
- Travel time is node-to-node on the road graph; the short hop from a grid point to its
  snapped node (median about 245 m on the 1 km grid) is ignored. Points that snap far from any
  road (`snap_m`) are mostly lake surface and forest.
- The ER roster is a judgement call recorded in `ER_TABLE`; edit it there.
