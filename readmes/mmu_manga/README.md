
---
configs:
- config_name: default
  data_dir: mmu_manga/dataset
tags:
- astronomy
license: cc-by-4.0
pretty_name: mmu_manga
size_categories:
- 10K<n<100K
---

# mmu_manga HATS Catalog Collection

This is the collection of HATS catalogs representing mmu_manga.

This dataset is part of the [Multimodal Universe](https://github.com/MultimodalUniverse/MultimodalUniverse),
a large-scale collection of multimodal astronomical data. For full details, see the paper:
[The Multimodal Universe: Enabling Large-Scale Machine Learning with 100TBs of Astronomical Scientific Data](https://arxiv.org/abs/2412.02527).


### Access the catalog

We recommend the use of the [LSDB](https://lsdb.io) Python framework to access HATS catalogs.
LSDB can be installed via `pip install lsdb` or `conda install conda-forge::lsdb`,
see more details [in the docs](https://docs.lsdb.io/).
The following code provides a minimal example of opening this catalog:

```python
import lsdb

# Full sky coverage.
catalog = lsdb.open_catalog("https://huggingface.co/datasets/hugging-science/mmu_manga")
# One-degree cone.
catalog = lsdb.open_catalog(
    "https://huggingface.co/datasets/hugging-science/mmu_manga",
    search_filter=lsdb.ConeSearch(ra=123.0, dec=25.0, radius_arcsec=3600.0),
)
```

Each catalog in this collection is represented as a separate [Apache Parquet dataset](https://arrow.apache.org/docs/python/dataset.html) and can be accessed with a variety of tools, including `pandas`, `pyarrow`, `dask`, `Spark`, `DuckDB`.

### File structure

This catalog is represented by the following files and directories:

- [`collection.properties`](https://huggingface.co/datasets/hugging-science/mmu_manga/collection.properties) — textual metadata file describing the HATS collection of catalogs
- [`mmu_manga`](https://huggingface.co/datasets/hugging-science/mmu_manga/mmu_manga) — main HATS catalog directory
  - [`dataset/`](https://huggingface.co/datasets/hugging-science/mmu_manga/mmu_manga/dataset/) — Apache Parquet dataset directory for the main catalog
    - ... parquet metadata and data files in sub directories ...
  - [`hats.properties`](https://huggingface.co/datasets/hugging-science/mmu_manga/mmu_manga/hats.properties) — textual metadata file describing the main HATS catalog
  - [`partition_info.csv`](https://huggingface.co/datasets/hugging-science/mmu_manga/mmu_manga/partition_info.csv) — CSV file with a list of catalog HEALPix tiles (catalog partitions)
  - [`skymap.fits`](https://huggingface.co/datasets/hugging-science/mmu_manga/mmu_manga/skymap.fits) — HEALPix skymap FITS file with row-counts per HEALPix tile of fixed order 12

### Catalog metadata

Metadata of the main HATS catalog, excluding margins and indexes:

| **Number of rows** | **Number of columns** | **Number of partitions** | **Size on disk** | **HATS Builder** |
| --- | --- | --- | --- | --- |
| 10,735 | 9 | 3,705 | 1.1 TiB | hats-import v0.9.0, hats v0.9.0 |


### Catalog columns

The main HATS catalog contains the following columns:

| **Name** |  **`_healpix_29`** | **`z`** | **`spaxel_size`** | **`spaxel_size_units`** | **`spaxels.flux`** | **`spaxels.ivar`** | **`spaxels.mask`** | **`spaxels.lsf`** | **`spaxels.lambda`** | **`spaxels.x`** | **`spaxels.y`** | **`spaxels.spaxel_idx`** | **`spaxels.flux_units`** | **`spaxels.lambda_units`** | **`spaxels.skycoo_x`** | **`spaxels.skycoo_y`** | **`spaxels.ellcoo_r`** | **`spaxels.ellcoo_rre`** | **`spaxels.ellcoo_rkpc`** | **`spaxels.ellcoo_theta`** | **`spaxels.skycoo_units`** | **`spaxels.ellcoo_r_units`** | **`spaxels.ellcoo_rre_units`** | **`spaxels.ellcoo_rkpc_units`** | **`spaxels.ellcoo_theta_units`** | **`images.filter`** | **`images.flux`** | **`images.flux_units`** | **`images.psf`** | **`images.psf_units`** | **`images.scale`** | **`images.scale_units`** | **`ra`** | **`dec`** | **`maps.group`** | **`maps.label`** | **`maps.flux`** | **`maps.ivar`** | **`maps.mask`** | **`maps.array_units`** | **`object_id`** |
| --- |  --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Data Type** |  int64 | double | double | string | list[fixed_size_list<element: fixed_size_list<element: float>[4563]>[1]] | list[fixed_size_list<element: fixed_size_list<element: float>[4563]>[1]] | list[fixed_size_list<element: fixed_size_list<element: int32>[4563]>[1]] | list[fixed_size_list<element: fixed_size_list<element: float>[4563]>[1]] | list[fixed_size_list<element: fixed_size_list<element: float>[4563]>[1]] | list[int64] | list[int64] | list[int64] | list[string] | list[string] | list[float] | list[float] | list[float] | list[float] | list[float] | list[float] | list[string] | list[string] | list[string] | list[string] | list[string] | list[string] | list[fixed_size_list<element: fixed_size_list<element: float>[96]>[96]] | list[string] | list[fixed_size_list<element: fixed_size_list<element: float>[96]>[96]] | list[string] | list[float] | list[string] | double | double | list[string] | list[string] | list[fixed_size_list<element: fixed_size_list<element: float>[96]>[96]] | list[fixed_size_list<element: fixed_size_list<element: float>[96]>[96]] | list[fixed_size_list<element: fixed_size_list<element: float>[96]>[96]] | list[string] | string |
| **Nested?** |  — | — | — | — | spaxels | spaxels | spaxels | spaxels | spaxels | spaxels | spaxels | spaxels | spaxels | spaxels | spaxels | spaxels | spaxels | spaxels | spaxels | spaxels | spaxels | spaxels | spaxels | spaxels | spaxels | images | images | images | images | images | images | images | — | — | maps | maps | maps | maps | maps | maps | — |
| **Value count** |  10,735 | 10,735 | 10,735 | 10,735 | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | 10,735 | 10,735 | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | 10,735 |
| **Example row** |  341185542854091424 | 0.03407 | 0.5 | arcsec | [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, … (4563 total)]], … (9216 total)] | [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, … (4563 total)]], … (9216 total)] | [[[1024, 1024, 1024, 1024, 1024, … (4563 total)]], … (9216 total)] | [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, … (4563 total)]], … (9216 total)] | [[[3622, 3622, 3623, 3624, 3625, … (4563 total)]], … (9216 total)] | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, … (9216 total)] | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, … (9216 total)] | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, … (9216 total)] | [1E-17 erg/s/cm^2/Angstrom/spaxel, … (9216 total)] | [Angstrom, Angstrom, Angstrom, … (9216 total)] | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, … (9216 total)] | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, … (9216 total)] | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, … (9216 total)] | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, … (9216 total)] | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, … (9216 total)] | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, … (9216 total)] | [arcsec, arcsec, arcsec, arcsec, … (9216 total)] | [arcsec, arcsec, arcsec, arcsec, … (9216 total)] | [, , , , , , , , , , , , , , , , , … (9216 total)] | [kpc/h, kpc/h, kpc/h, kpc/h, … (9216 total)] | [degrees, degrees, degrees, … (9216 total)] | [g, r, i, z] | [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, … (96 total)], … (96 total)], … | [nanomaggies/pixel, … (4 total)] | [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, … (96 total)], … (96 total)], … | [nanomaggies/pixel, … (4 total)] | [0.5, 0.5, 0.5, 0.5] | [arcsec, arcsec, arcsec, arcsec] | 123.4 | 25.13 | [spx_skycoo, spx_skycoo, … (914 total)] | [spx_skycoo_on_sky_x, … (914 total)] | [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, … (96 total)], … (96 total)], … | [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, … (96 total)], … (96 total)], … | [[[1.074e+09, 1.074e+09, 1.074e+09, … (96 total)], … (96 total)], … (… | [arcsec, arcsec, arcsec, arcsec, … (914 total)] | 10221-6103 |
| **Minimum value** |  51879464109029 | -9999.0 | 0.5 | arcsec | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | 0.0088083389 | -30.188009 | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | 10001-12701 |
| **Maximum value** |  3458755478246244569 | 0.14971851 | 0.5 | arcsec | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | 359.85858 | 68.453178 | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | 9894-9102 |


"Nested" indicates whether the column is stored as a nested field inside another "struct" column.


"Value count" may be different from the total number of rows for nested columns: each nested element is counted as a single value.




### Crossmatch with another catalog

HATS catalogs can be efficiently crossmatched using [LSDB](https://lsdb.io),
which leverages the HEALPix partitioning to avoid loading the full datasets into memory:

```python
import lsdb

mmu_manga = lsdb.open_catalog("https://huggingface.co/datasets/hugging-science/mmu_manga")
other = lsdb.open_catalog("https://huggingface.co/datasets/<org>/<other_catalog>")

crossmatched = mmu_manga.crossmatch(other, radius_arcsec=1.0)
print(crossmatched)
```

See the [LSDB documentation](https://docs.lsdb.io/) for more details on crossmatching and other operations.
