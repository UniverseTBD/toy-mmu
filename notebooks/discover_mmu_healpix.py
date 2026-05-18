"""Discover which MultimodalUniverse v1 sub-catalogs cover which healpixels.

Crawls https://users.flatironinstitute.org/~polymathic/data/MultimodalUniverse/v1/
top-level -> sub-catalog -> healpix=N/ directories and emits a pandas DataFrame
with columns: catalog, sub_catalog, healpix, size_bytes, link.

Boundary regions are ignored (we only record the healpix number stamped on the
directory name).
"""

from __future__ import annotations

import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin

import pandas as pd

BASE_URL = "https://users.flatironinstitute.org/~polymathic/data/MultimodalUniverse/v1/"
HREF_RE = re.compile(r'<a href="([^"]+)">')
HEALPIX_RE = re.compile(r"^healpix=(\d+)/$")
# A row looks like:
#   <td ...><span class="fas fa-folder-open"></span></td>
#   <td data-order="healpix=1189"><a href="healpix=1189/">healpix=1189/</a></td>
#   <td data-order="75603936">72.10MB</td>
#   <td data-order="1733169470.630">2024-12-02 19:57 UTC</td>
ROW_RE = re.compile(
    r'<a href="(healpix=\d+/)">.*?<td data-order="(\d+)">',
    re.DOTALL,
)


def fetch(url: str) -> str:
    with urllib.request.urlopen(url) as resp:
        return resp.read().decode()


def list_subdirs(url: str) -> list[str]:
    return [
        h
        for h in HREF_RE.findall(fetch(url))
        if h.endswith("/") and h != "../"
    ]


def healpix_for_subcatalog(catalog: str, sub_catalog: str) -> list[dict]:
    sub_url = urljoin(BASE_URL, f"{catalog}/{sub_catalog}/")
    html = fetch(sub_url)
    rows = []
    for entry, size_bytes in ROW_RE.findall(html):
        m = HEALPIX_RE.match(entry)
        if not m:
            continue
        rows.append(
            {
                "catalog": catalog,
                "sub_catalog": sub_catalog.rstrip("/"),
                "healpix": int(m.group(1)),
                "size_bytes": int(size_bytes),
                "link": urljoin(sub_url, entry),
            }
        )
    return rows


def build_dataframe(max_workers: int = 16) -> pd.DataFrame:
    catalogs = list_subdirs(BASE_URL)

    sub_catalog_jobs: list[tuple[str, str]] = []
    for catalog in catalogs:
        for sub in list_subdirs(urljoin(BASE_URL, catalog)):
            sub_catalog_jobs.append((catalog.rstrip("/"), sub))

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for result in pool.map(lambda job: healpix_for_subcatalog(*job), sub_catalog_jobs):
            rows.extend(result)

    return pd.DataFrame(
        rows, columns=["catalog", "sub_catalog", "healpix", "size_bytes", "link"]
    )


if __name__ == "__main__":
    df = build_dataframe()
    print(f"Discovered {len(df)} (sub_catalog, healpix) pairs "
          f"across {df['sub_catalog'].nunique()} sub-catalogs.")
    print(df.head())
    tt = df[df["catalog"].isin(["gaia", "chandra"])].groupby("healpix").agg({"sub_catalog": ["count", list], "size_bytes": "sum"})
    tt.columns = ["count", "list", "sum"]
    # list the healpixels that appear in more than one sub-catalog, sorted by total size
    # We are interested in overlaps with small sizes
    tt[tt["count"] > 1].sort_values("sum", ascending=True).to_csv("mmu_healpix_overlap.csv")


    overlap = (
        df.groupby("healpix")["sub_catalog"]
        .nunique()
        .sort_values(ascending=False)
        .head(20)
    )
    print("\nTop healpixels by sub-catalog overlap:")
    print(overlap)

    df.to_csv("mmu_healpix_index.csv", index=False)
    print("\nWrote mmu_healpix_index.csv")
