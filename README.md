# Flood impact on emergency-room travel time, Northwest Arkansas

How much longer does it take to drive to the nearest emergency room from every part of
Benton, Washington and Madison Counties, Arkansas, if every road inside the FEMA 100-year
floodplain is impassable?

Everything lives in one straight-line script, `flood_er_travel_time.py`, written pandas-first
with plain loops (no helper functions). Open-source stack: pandas, GeoPandas/Shapely for the
overlay, NetworkX for routing, pyarrow to read Overture Maps straight from S3.

## Data

| Layer | Source | Notes |
|---|---|---|
| County boundaries | Overture Maps `divisions` theme (OSM-derived), release 2026-08-19.0 | Benton, Washington, Madison |
| Road network | Overture Maps `transportation` theme (OSM-derived, refreshed monthly) | drivable classes only; private roads dropped; one-ways honoured; posted speed limits where present, otherwise class defaults |
| Emergency rooms | Overture Maps `places` theme, curated in `ER_TABLE` | 24/7 EDs in the three counties plus the ring of neighbouring-county ERs (Berryville, Eureka Springs, Harrison, Clarksville, Ozark, Van Buren, Stilwell OK, Grove OK, Cassville MO) so "nearest" is right near county lines. Arkansas Children's (pediatric) and the VA (veterans) are listed but not routed to |
| 100-year floodplain | FEMA National Flood Hazard Layer, `S_FLD_HAZ_AR` where `SFHA_TF = 'T'` | the Special Flood Hazard Area = 1 % annual-chance flood = zones A, AE, AH, AO, AR, A99, V, VE. Zone X (0.2 %) is *not* closed |

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
4. Flood: spatially join the road edges to the SFHA polygons; every edge that intersects the
   floodplain is removed (bridges included, unless `--bridges-passable`).
5. Repeat step 3 on the reduced graph and compare.

## Running it

```bash
pip install -r requirements.txt
python flood_er_travel_time.py                      # pulls the floodplain from FEMA's REST service
python flood_er_travel_time.py --nfhl NFHL_05.zip   # or point at a downloaded state/county NFHL
python flood_er_travel_time.py --bridges-passable   # sensitivity: leave bridges open
python flood_er_travel_time.py --spacing 500        # denser grid
```

FEMA serves the NFHL from `hazards.fema.gov` (MapServer layer 28). If that host is blocked
where you run this, download the Arkansas state NFHL from https://msc.fema.gov
(Search All Products → Effective Products → NFHL Data-State) and pass the zip with `--nfhl`.
The whole run takes a few minutes; Overture and NFHL downloads are cached in `data/`.

## Outputs (`output/`)

| File | Contents |
|---|---|
| `er_locations.csv` | the ER table with Overture address, coordinates and snap distance |
| `grid_baseline.csv` | every grid point with county, nearest ER and baseline minutes |
| `grid_flood_impact.csv` / `.gpkg` | baseline vs flood minutes, nearest ER before/after, added minutes, % increase, status (`unchanged` / `slower` / `unreachable`) |
| `flooded_road_edges.gpkg` | the road edges removed as closures |
| `summary_by_county.csv` | medians, mean/median/p90 added minutes, share slower, share unreachable, share worse by more than 15 min |
| `nearest_er_counts.csv` | how many grid points each ER serves before and after |
| `map_baseline_minutes.png`, `map_flood_increase.png` | maps |

## Assumptions and caveats

- Free-flow speeds: posted limits from Overture where available (about a third of segments),
  otherwise 65/55/45/40/35/30/25/15 mph by class from motorway down to service roads. No
  congestion, no emergency-vehicle privileges.
- "Impassable" is applied literally to every edge that touches the SFHA polygon, including
  bridges and elevated approaches. `--bridges-passable` shows how much of the impact is bridges.
- Travel time is node-to-node on the road graph; the short hop from a grid point to its
  snapped node (median about 245 m on the 1 km grid) is ignored. Points that snap far from any
  road (`snap_m`) are mostly lake surface and forest.
- The ER roster is a judgement call recorded in `ER_TABLE`; edit it there.
