#!/bin/bash

wget --spider -r -np -nH \
  --cut-dirs=4 \
  -P data \
  -A "*.py" \
  -R "*.fits,*.hdf5,*.h5,*.npy,*.npz,*.pkl,*.parquet,*.tar,*.gz,*.zip" \
  --reject-regex ".*healpix.*" \
  https://users.flatironinstitute.org/~polymathic/data/MultimodalUniverse/v1/

wget -r -np -nH --cut-dirs=4 \
  -P data \
  https://users.flatironinstitute.org/~polymathic/data/MultimodalUniverse/v1/desi/desi.py

wget -r -np -nH --cut-dirs=4 \
  -P data \
  https://users.flatironinstitute.org/~polymathic/data/MultimodalUniverse/v1/desi/edr_sv3/healpix=1708/

wget -r -np -nH --cut-dirs=4 \
  -P data \
  https://users.flatironinstitute.org/~polymathic/data/MultimodalUniverse/v1/hsc/hsc.py

wget -r -np -nH --cut-dirs=4 \
  -P data \
  https://users.flatironinstitute.org/~polymathic/data/MultimodalUniverse/v1/hsc/pdr3_dud_22.5/healpix=1708/
