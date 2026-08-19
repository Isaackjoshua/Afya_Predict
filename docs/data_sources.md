# Data sources

Six or more independent streams are fused per disease. That is a deliberate
defence: Google Flu Trends depended on search queries alone, and when Google's
ranking changed 86 times in two months in 2012 the model broke silently.

## What is wired up

| Source | Adapter | Variables | Cadence | Resolution | Access |
|---|---|---|---|---|---|
| **CHIRPS** | `chirps` | `rainfall_mm` | pentadal | ~5 km | open HTTP (UCSB) |
| **ERA5** | `era5` | `temperature_c`, `humidity_pct`, `wind_speed_ms` | hourly→weekly | ~28 km | free, CDS API key |
| **MODIS** | `modis` | `ndvi` | 16-day | 250 m–1 km | free, NASA Earthdata login |
| **Sentinel-5P** | `sentinel5p` | `no2_mol_m2`, `pm25_ug_m3`, `aerosol_index` | daily | ~7 km | free, Copernicus Data Space |
| **DHIS2** | `dhis2` | `cases_*`, `reporting_completeness` | weekly | council | ministry credentials |
| **CDR mobility** | `cdr_mobility` | inbound/outbound/internal trips | weekly | council OD pairs | negotiated with telcos |
| **WorldPop** | `population_density` | `population`, `population_density_km2` | annual | 100 m | open HTTP |
| **WASH / JMP** | `wash_indicators` | `wash_access`, `improved_sanitation` | annual | subnational | open |
| **Livestock** | `livestock_disease` | outbreak events, mortality, spillover index | weekly | council | ministry / WOAH |
| **Google Trends** | `search_trends` | `search_*` | daily | region | `pytrends` |

## Why each one is there

**Climate is primary, and that is a choice about equity.** Search and social
signals do not work where disease burden is highest: Tanzanian smartphone
penetration is around 35% against ~87% feature-phone ownership. Satellite climate
has 100% spatial coverage and DHIS2 reaches ~93.9% national completeness, so the
system works where infrastructure is weakest (shortcoming #14). `search_trends`
is flagged `optional = True` and the platform refuses to let it act as a primary
predictor.

**Mobility answers "where next".** Real CDR arrives as pre-aggregated
origin-destination counts — never subscriber records — and requires a negotiated
agreement. The platform never depends on that negotiation completing: without it,
a gravity or radiation model supplies the travel matrix (critical rule #8).

**Livestock is the One Health arm.** Human, animal and environmental
surveillance normally live in separate systems; the 2014 Ebola response was
delayed by exactly that incompatibility. Putting animal signals on the same
`district × week` grid is what makes cross-domain spillover detection possible.

## Configuring credentials

Copy `.env.example` to `.env`. **Every credential is optional.** Without one, the
adapter degrades to cache, then to a deterministic synthetic climatology, and
says so in `GET /data/status`.

```bash
DHIS2_BASE_URL=https://hmis.moh.go.tz
DHIS2_USERNAME=...
CDS_API_KEY=...              # https://cds.climate.copernicus.eu
EARTHDATA_USERNAME=...       # https://urs.earthdata.nasa.gov
CDSE_CLIENT_ID=...           # https://dataspace.copernicus.eu
CDR_DATA_DIR=./data/raw/cdr
```

Verify what is actually live:

```bash
python -m src.data_ingestion.scheduler --status
curl localhost:8000/data/status | jq '.meets_fusion_rule, .warnings'
```

## Supplying data by file

Two sources read local drops rather than APIs:

**CDR mobility** — CSV or Parquet in `CDR_DATA_DIR`, columns
`origin, destination, week, trips`. Weeks are ISO labels (`2026-W07`), district
names must match `config/regions/tanzania.yaml`.

**WASH** — `data/raw/wash/jmp_subnational.csv` with `district, year, water,
sanitation` (coverage as 0–1 shares).

## Data quality

The rule the ingestion layer follows is **flag, never silently drop or
zero-fill**. Checks applied to every source:

| Check | Effect |
|---|---|
| missing values | kept as `NaN`, quality → 0 |
| out of plausible range | quality halved, flagged |
| robust z-score > 6 | quality × 0.6, flagged as a possible reporting artefact |
| < 90% of requested weeks present | temporal-gap flag |
| constant ≥ 8 weeks | stuck-sensor flag |
| < 5 districts present | sparse-coverage flag |

Driver gaps are filled in a recorded order (short interpolation → neighbour
borrowing → own seasonal climatology → national median), and every filled cell is
marked. **Case counts are never imputed.**

## Adding a source

```python
from src.data_ingestion.base_adapter import BaseAdapter
from src.data_ingestion.registry import register_adapter

@register_adapter
class RainfallRadarAdapter(BaseAdapter):
    source_name = "rainfall_radar"
    variables = ("radar_rainfall_mm",)
    update_frequency_days = 1

    def is_configured(self) -> bool:
        return bool(self.settings.radar_api_key)

    def fetch_live(self, weeks):
        ...      # return self.tidy([...])

    def synthesize(self, weeks):
        ...      # deterministic fallback so the pipeline still runs
```

Then reference `source: "rainfall_radar"` from a disease YAML. Implementing
`synthesize()` is not optional — it is what keeps the platform runnable when the
source is unavailable.
