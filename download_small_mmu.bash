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
  https://users.flatironinstitute.org/~polymathic/data/MultimodalUniverse/v1/desi/healpix=1708/

wget -r -np -nH --cut-dirs=4 \
  -P data \
  https://users.flatironinstitute.org/~polymathic/data/MultimodalUniverse/v1/hsc/hsc.py

wget -r -np -nH --cut-dirs=4 \
  -P data \
  https://users.flatironinstitute.org/~polymathic/data/MultimodalUniverse/v1/hsc/healpix=1708/
