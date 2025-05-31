from pathlib import Path
from kraken.geometry import get_binding_geometry_of_ligand

import logging
import sys

from kraken.file_io import write_xyz

logging.basicConfig(
        level=logging.DEBUG,
        format='[%(levelname)-5s - %(asctime)s] [%(module)s] %(message)s',
        datefmt='%m/%d/%Y:%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    coords, elements = get_binding_geometry_of_ligand(smiles='c1ccc(cc1)c1nn(c(c1n1nccc1[P](C12CC3CC(C2)CC(C1)C3)C12CC3CC(C2)CC(C1)C3)c1ccccc1)c1ccccc1',
                                                      coordination_distance=1.8,
                                                      nconfs=25)

    write_xyz(Path('./final_geom.xyz'),
              coords=coords,
              elements=elements,
              comment='testing!')