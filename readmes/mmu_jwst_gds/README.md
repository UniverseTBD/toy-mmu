
---
configs:
- config_name: default
  data_dir: mmu_jwst_gds/dataset
tags:
- astronomy
license: cc-by-4.0
pretty_name: mmu_jwst_gds
size_categories:
- 10K<n<100K
---

<div align="center">
<img src="example_figure.png" width="600">
</div>

# mmu_jwst_gds HATS Catalog Collection

This is the collection of HATS catalogs representing mmu_jwst_gds.

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
catalog = lsdb.open_catalog("https://huggingface.co/datasets/UniverseTBD/mmu_jwst_gds")
# One-degree cone.
catalog = lsdb.open_catalog(
    "https://huggingface.co/datasets/UniverseTBD/mmu_jwst_gds",
    search_filter=lsdb.ConeSearch(ra=53.0, dec=-28.0, radius_arcsec=3600.0),
)
```

Each catalog in this collection is represented as a separate [Apache Parquet dataset](https://arrow.apache.org/docs/python/dataset.html) and can be accessed with a variety of tools, including `pandas`, `pyarrow`, `dask`, `Spark`, `DuckDB`.

### File structure

This catalog is represented by the following files and directories:

- [`collection.properties`](https://huggingface.co/datasets/UniverseTBD/mmu_jwst_gds/collection.properties) — textual metadata file describing the HATS collection of catalogs
- [`mmu_jwst_gds`](https://huggingface.co/datasets/UniverseTBD/mmu_jwst_gds/mmu_jwst_gds) — main HATS catalog directory
  - [`dataset/`](https://huggingface.co/datasets/UniverseTBD/mmu_jwst_gds/mmu_jwst_gds/dataset/) — Apache Parquet dataset directory for the main catalog
    - ... parquet metadata and data files in sub directories ...
  - [`hats.properties`](https://huggingface.co/datasets/UniverseTBD/mmu_jwst_gds/mmu_jwst_gds/hats.properties) — textual metadata file describing the main HATS catalog
  - [`partition_info.csv`](https://huggingface.co/datasets/UniverseTBD/mmu_jwst_gds/mmu_jwst_gds/partition_info.csv) — CSV file with a list of catalog HEALPix tiles (catalog partitions)
  - [`skymap.fits`](https://huggingface.co/datasets/UniverseTBD/mmu_jwst_gds/mmu_jwst_gds/skymap.fits) — HEALPix skymap FITS file with row-counts per HEALPix tile of fixed order 10
- [`mmu_jwst_gds_10arcs/`](https://huggingface.co/datasets/UniverseTBD/mmu_jwst_gds/mmu_jwst_gds_10arcs) — default margin catalog to ensure data completeness in cross-matching, the margin threshold is 10.0 arcseconds
  - ... margin catalog files and directories ...

### Catalog metadata

Metadata of the main HATS catalog, excluding margins and indexes:

| **Number of rows** | **Number of columns** | **Number of partitions** | **Size on disk** | **HATS Builder** |
| --- | --- | --- | --- | --- |
| 17,494 | 11 | 8 | 6.3 GiB | hats-import v0.7.3, hats v0.7.3 |


### Catalog columns

The main HATS catalog contains the following columns:

| **Name** |  **`_healpix_29`** | **`image.band`** | **`image.flux`** | **`image.ivar`** | **`image.mask`** | **`image.psf_fwhm`** | **`image.scale`** | **`mag_auto`** | **`flux_radius`** | **`flux_auto`** | **`fluxerr_auto`** | **`cxx_image`** | **`cyy_image`** | **`cxy_image`** | **`object_id`** | **`ra`** | **`dec`** |
| --- |  --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Data Type** |  int64 | list[string] | list[list<element: list<element: float>>] | list[list<element: list<element: float>>] | list[list<element: list<element: bool>>] | list[float] | list[float] | float | float | float | float | float | float | float | string | double | double |
| **Nested?** |  — | image | image | image | image | image | image | — | — | — | — | — | — | — | — | — | — |
| **Value count** |  17,494 | 122,458 | *N/A* | *N/A* | *N/A* | 122,458 | 122,458 | 17,494 | 17,494 | 17,494 | 17,494 | 17,494 | 17,494 | 17,494 | 17,494 | 17,494 | 17,494 |
| **Example row** |  2528743702220181960 | [f090w, f115w, f150w, f200w, f277w, … (7 total)] | [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, … (96 total)], … (96 total)], … | [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, … (96 total)], … (96 total)], … | [[[False, False, False, False, False, … (96 total)], … (96 total)], …… | [0.033, 0.04, 0.05, 0.066, 0.092, … (7 total)] | [0.02, 0.02, 0.02, 0.02, 0.04, 0.04, … (7 total)] | 25.85 | 4.665 | 0.1431 | 0.0007781 | 0.05192 | 0.07733 | -0.03422 | -7567302259786491648 | 53.17 | -27.87 |
| **Minimum value** |  2528743693306351542 | f090w | *N/A* | *N/A* | *N/A* | 0.032999999821186066 | 0.019999999552965164 | 17.325851440429688 | 0.6315596699714661 | 0.02997756563127041 | 0.00019976860494352877 | 7.402321352856234e-05 | 2.9240540243336e-05 | -1.2243539094924927 | -7567302259786424111 | 53.051927973650635 | -27.87760950265767 |
| **Maximum value** |  2528752620411181000 | f444w | *N/A* | *N/A* | *N/A* | 0.14499999582767487 | 0.03999999910593033 | 27.499950408935547 | 276.15155029296875 | 415.23297119140625 | 0.3415043354034424 | 2.141709089279175 | 1.9386420249938965 | 1.0062775611877441 | -7567302259786494449 | 53.22671287579358 | -27.723125018856216 |


"Nested" indicates whether the column is stored as a nested field inside another "struct" column.


"Value count" may be different from the total number of rows for nested columns: each nested element is counted as a single value.




### Crossmatch with another catalog

HATS catalogs can be efficiently crossmatched using [LSDB](https://lsdb.io),
which leverages the HEALPix partitioning to avoid loading the full datasets into memory:

```python
import lsdb

mmu_jwst_gds = lsdb.open_catalog("https://huggingface.co/datasets/UniverseTBD/mmu_jwst_gds")
other = lsdb.open_catalog("https://huggingface.co/datasets/<org>/<other_catalog>")

crossmatched = mmu_jwst_gds.crossmatch(other, radius_arcsec=1.0)
print(crossmatched)
```

See the [LSDB documentation](https://docs.lsdb.io/) for more details on crossmatching and other operations.

### Dataset-specific context

**Original survey**  
This dataset is based on the James Webb Space Telescope (JWST) NIRCam observations from early deep field surveys.

**Data modality**  
The dataset consists of fixed-size image cutouts (96×96 pixels) centered on sources from photometric catalogs. The images are multi-band, with 6 or 7 filters covering wavelengths from approximately 0.9μm to 4.4μm.

**Typical use cases**  
Images from these JWST deep field surveys have been used in a large number of scientific publications, including machine learning applications.

**Caveats**  
Different surveys have different wavelength coverage, and missing bands are represented as arrays of zeros to simplify data loading.

**Citation**  
The data are in the public domain. The dataset uses data products retrieved from the Dawn JWST Archive (DJA), an initiative of the Cosmic Dawn Center (DAWN).
