"""Generate skymap.fits for an existing HATS catalog by streaming its partitions.

`skymap.fits` is normally written by `hats-import` at the end of the "main"
catalog pipeline (see `hats_import/catalog/run_import.py`). For catalogs that
were uploaded without one (e.g. `hugging-science/mmu_manga`), this script
rebuilds it post-hoc by streaming `(ra, dec)` columns directly from the remote
parquet partitions and accumulating a HEALPix density histogram.

Self-contained — no full download, no writes back to the source catalog. The
resulting `skymap.fits` is written locally and can be uploaded to HF
separately (e.g. `huggingface-cli upload <repo> ./skymap.fits skymap.fits`).

Example::

    python verification/generate_skymap.py \\
        --catalog hf://datasets/hugging-science/mmu_manga/manga
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
from upath import UPath

from hats.catalog.catalog_collection import CatalogCollection
from hats.io.file_io import read_parquet_file
from hats.io.paths import pixel_catalog_file
from hats.io.skymap import write_skymap
from hats.loaders import read_hats
from hats.pixel_math.partition_stats import empty_histogram, generate_histogram


def _resolve_main_catalog(loaded):
    """Unwrap a CatalogCollection into its main Catalog; pass Catalog through."""
    if isinstance(loaded, CatalogCollection):
        return loaded.main_catalog
    return loaded


def _choose_order(catalog, requested: int | None) -> int:
    if requested is not None:
        return requested
    info_order = getattr(catalog.catalog_info, "skymap_order", None)
    if info_order is not None:
        return int(info_order)
    return catalog.partition_info.get_highest_order()


def _parse_alt_orders(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return [int(x) for x in raw.split(",") if x.strip()]


@click.command(help=__doc__)
@click.option(
    "--catalog",
    "catalog_uri",
    required=True,
    help="HATS catalog URI (e.g. hf://datasets/hugging-science/mmu_manga/manga, "
    "or a local path).",
)
@click.option(
    "--output",
    "output_dir",
    default=".",
    show_default=True,
    type=click.Path(file_okay=False, dir_okay=True),
    help="Local directory to write skymap.fits into.",
)
@click.option(
    "--order",
    "order",
    type=int,
    default=None,
    help="Target HEALPix order. Defaults to catalog's hats_skymap_order, "
    "or the highest partition order if unset.",
)
@click.option(
    "--alt-orders",
    "alt_orders_raw",
    default=None,
    help="Comma-separated list of additional (coarser) orders to also write as "
    "skymap.K.fits (e.g. '0,2,4').",
)
@click.option("--ra-col", default="ra", show_default=True)
@click.option("--dec-col", default="dec", show_default=True)
@click.option(
    "--batch-size",
    default=100_000,
    show_default=True,
    type=int,
    help="Row-group batch size when streaming each parquet partition.",
)
def main(
    catalog_uri: str,
    output_dir: str,
    order: int | None,
    alt_orders_raw: str | None,
    ra_col: str,
    dec_col: str,
    batch_size: int,
) -> int:
    catalog_path = UPath(catalog_uri)
    print(f"Loading HATS catalog: {catalog_path}", flush=True)
    catalog = _resolve_main_catalog(read_hats(catalog_path, read_moc=False))

    target_order = _choose_order(catalog, order)
    alt_orders = _parse_alt_orders(alt_orders_raw)
    pixels = catalog.partition_info.get_healpix_pixels()
    print(
        f"  catalog_type={catalog.catalog_info.catalog_type} "
        f"partitions={len(pixels)} target_order={target_order}",
        flush=True,
    )
    if alt_orders:
        print(f"  alt orders: {alt_orders}", flush=True)

    histogram = empty_histogram(target_order)
    base_dir = catalog.catalog_base_dir
    total_rows = 0

    for i, pixel in enumerate(pixels, start=1):
        part_path = pixel_catalog_file(base_dir, pixel)
        pqf = read_parquet_file(part_path)
        part_rows = 0
        for batch in pqf.iter_batches(columns=[ra_col, dec_col], batch_size=batch_size):
            df = batch.to_pandas()
            partial = generate_histogram(df, target_order, ra_col, dec_col)
            histogram += partial
            part_rows += len(df)
        total_rows += part_rows
        print(
            f"  [{i}/{len(pixels)}] Norder={pixel.order} Npix={pixel.pixel}: "
            f"{part_rows} rows (running total: {total_rows})",
            flush=True,
        )

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_skymap(histogram, out_dir, orders=alt_orders)

    fits_sum = int(histogram.sum())
    declared = catalog.catalog_info.total_rows
    print(
        f"\nWrote skymap.fits to {out_dir} "
        f"(npix={len(histogram)}, sum={fits_sum}"
        + (f", catalog total_rows={declared}" if declared is not None else "")
        + ")",
        flush=True,
    )
    if declared is not None and fits_sum != declared:
        print(
            f"WARNING: histogram sum {fits_sum} != declared total_rows {declared}",
            file=sys.stderr,
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(standalone_mode=False) or 0)
