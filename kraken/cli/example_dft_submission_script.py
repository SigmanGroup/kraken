
import re
import subprocess

from pathlib import Path


def write_job_file(kraken_id: str,
                   directory: Path,
                   destination: Path,
                   template: Path,
                   nprocs: int,
                   mem: int) -> Path:
    '''

    '''
    with open(template, 'r', encoding='utf-8') as infile:
        text = infile.read()

    text = re.sub(r'\$KID', str(kraken_id), text)
    text = re.sub(r'\$NPROCS', str(nprocs), text)
    text = re.sub(r'\$MEM', str(mem), text)
    text = re.sub(r'\$DIR', str(directory.absolute()), text)

    with open(destination, 'w', encoding='utf-8') as o:
        o.write(text)

if __name__ == "__main__":

    dft_template = Path('./dft_slurm_template.sh')

    parent_dir = Path('./data/')

    kraken_ids = ['00000015', '00000030', '00000062', '00000068',
                  '00000069', '00000079', '00000084', '00000089',
                  '00000102', '00000104', '00000116', '00000130',
                  '00000139', '00000148', '00000217', '00000251',
                  '00000280', '00000327', '00000329', '00000338',
                  '00000340', '00000351', '00000401', '00000449',
                  '00000458', '00000487', '00000640', '00000644',
                  '00000648', '00000650']

    for id in kraken_ids:
        dest = Path(f'./{id}_dft.slurm')

        write_job_file(kraken_id=id,
                       directory=parent_dir / id,
                       destination=dest,
                       template=dft_template,
                       nprocs=8,
                       mem=16)

