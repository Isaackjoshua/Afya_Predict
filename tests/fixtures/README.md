# Test fixtures

Most test data in this suite is **generated, not stored**: every adapter ships a
deterministic synthetic path (`BaseAdapter.synthesize`), so `conftest.py` can
build a realistic multi-year panel on demand and the repository stays small.

What lives here instead are **captured upstream payloads** — the shapes real
services actually return. These exist because the live parsing paths cannot be
exercised by the synthetic generator: they are only reached when credentials are
present, which is exactly when nobody is watching a test suite.

| File | Source | What it pins |
|---|---|---|
| `dhis2_analytics.json` | DHIS2 `/api/analytics.json` | header/row ordering, `pe` period ids, name-based `outputIdScheme` |
| `dhis2_data_elements.json` | DHIS2 `/api/dataElements.json` | the name-search response used to resolve UIDs |
| `cdr_origin_destination.csv` | telco aggregate drop | column names and the pre-aggregated OD shape |
| `jmp_subnational.csv` | WHO/UNICEF JMP extract | the water/sanitation columns the WASH adapter expects |

These are small, hand-checked samples with no real subscriber, patient or
facility data in them. Field values are illustrative; district names match
`config/regions/tanzania.yaml` so the fixtures flow through the normaliser.
