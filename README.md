# Kraken
In 2024, the code for Kraken was refactored to streamline the addition of new monophosphines and improve portability.
This update uses modern software versions including xTB 6.4.0 and CREST 2.12. In our testing, the new workflow performs
nearly identically to the original with minor differences.

Different versions of xTB lead to small differences in GFN2-level properties. We compared DFT properties for more than
60 monophosphines generated using both the original and updated versions of Kraken. For a small number of properties
(typically octant volumes), the difference between the old and new values for a given monophosphine exceeds 75% of the
original value. These differences likely arise from randomness in the conformer search.

__A detailed report comparing 28 monophosphines from the original Kraken workflow is provided in the validation folder__

## Installation (Linux systems only)

1.  Create a conda environment with the included environment YAML file.

```bash
conda env create --name kraken --file=kraken.yml
```

2.  Activate the new environment

```bash
conda activate kraken
```

3. (Optional) [Download an appropriate version of xTB and CREST](https://github.com/crest-lab/crest). The precompiled versions
   for xTB v6.6.0 and CREST v2.12 worked in our testing. Alternatively, CHPC users can use `module load crest/2.12` to
   load xTB v6.4.0 and CREST v2.12.

4. Place the precompiled binaries for xTB and CREST somewhere on your PATH. Kraken will call these through subprocesses.

5. Install the `kraken` package by navigating to the parent directory containing `setup.py` and running this command:

```bash
pip install .
```

## Example usage (submission to CHPC for the Sigman group)
These instructions are for Sigman group members to submit batches of calculations to the Sigman owner nodes on Notchpeak.
For users outside of the Sigman group, see instructions in the next section.

1. Format a `.csv` file that contains the columns SMILES, KRAKEN_ID, CHARGE, CONVERSION_FLAG:

    | KRAKEN_ID | SMILES           | CHARGE | CONVERSION_FLAG |
    |-----------|------------------|--------|-----------------|
    | 5039      | CP(C)C           |    0   |        4        |
    | 10596     | CP(C1=CC=CC=C1)C |    0   |        4        |
    | ...       | ...              |   ...  |       ...       |

2. Run the example submission script with your requested inputs and configurations:

```bash
submit_conf_search --csv input.csv --nprocs 8 --mem 16 --time 6 --calculation-dir ./data/ --debug
```

3. After completion, inspect SLURM log files (`*.log`, `*.error`) for errors/warnings (`ERROR`, `WARNING`).

4. Use the CLI utilities provided by Kraken if you wish to run all of your calculations from one directory (recommended):

    a. Move DFT files to a common directory:

    ```bash
    extract_dft_files --input ./data/ --destination ./dft_calculation_folder_for_convenience/
    ```

    b. Submit DFT calculations:

    ```bash
    for i in *.com; do subg16 $i -c sigman -m 32 -p 16 -t 12; done
    ```

    c. Return results to their directories:

    ```bash
    return_dft_files --input ./dft_calculation_folder_for_convenience/ --destination ./data/
    ```

5. Evaluate DFT jobs for errors. For help, use [GaussianLogfileAssessor](https://github.com/thejameshoward/GaussianLogfileAssessor.git).

6. Submit the DFT portion of the Kraken workflow:

```bash
submit_dft_calcs --csv input.csv --nprocs 8 --mem 16 --time 6 --calculation-dir ./data/ --debug
```

7. Check SLURM `.log` and `.error` files and raise an issue on this repo if necessary.

8. Convert the resulting YAML files into a convenient spreadsheet.

```bash
yaml_to_csv -d ./data/ --debug
```


## Creating custom submission templates
To submit batches of calculations to different HPC systems, you must create a submission script template similar to those in
the `kraken/slurm_templates` directory to accommodate your job scheduler. Please note that special symbols exist in the SLURM
templates that are substituted with actual values required by the scheduler including `$KID`, `$NPROCS`, `$MEM`, and several others.
Specify the template file using the `--slurm-template` argument for the submission scripts.

1. Format a `.csv` file that contains the columns SMILES, KRAKEN_ID, CHARGE, CONVERSION_FLAG (see above)

2. Run the example submission script with your requested inputs and configurations:

```bash
submit_conf_search --csv input.csv --nprocs 8 --mem 16 --time 6 --calculation-dir ./data/ --debug --slurm-template ~/path/to/template.slurm
```

3. Continue with steps 4-7 from the previous procedure.


## Running directly on a compute node
Kraken can also be executed directly from the command line. This can be useful if you wish to create your own wrapper scripts for submission to
other HPC systems. Please note that running this script will call computationally intensive programs and should not be run on head/login nodes.

1. Format a `.csv` file that contains the columns SMILES, KRAKEN_ID, CHARGE, CONVERSION_FLAG (see above)

2. Run the first Kraken script on the `.csv` file.

```bash
run_kraken_conf_search -i ./data/input.csv --nprocs 4 --calculation-dir ./data/ --debug > kraken_conf_search.log
```

3. After the script terminates, navigate to `./data/` to find the conformer search directories. Each `<KRAKEN_ID>/dft/` folder contains the `.com`
   files for Gaussian16. Run these calculations and place all result files back in the `<KRAKEN_ID>/dft/` for your monophosphine.

4. After confirming the `.log`, `.chk`, and `.wfn` files are present in `<KRAKEN_ID>/dft/`, run the final Kraken DFT processing step.
   This step operates on individual Kraken IDs (CSV input is not supported):<br>

```bash
run_kraken_dft.py --kid 90000001 --dir ./data/ --nprocs 4 --force > kraken_dft_processing_90000001.log
```

5. Final `.yml` output files from both the CREST and DFT steps will be found in `./data/<KRAKEN_ID>/`:

```
./90000001/
├── 90000001_confdata.yml
├── 90000001_data.yml
├── 90000001_Ni_combined.yml
├── 90000001_Ni_confs.yml
├── 90000001_Ni.yml
├── 90000001_noNi_combined.yml
├── 90000001_noNi_confs.yml
├── 90000001_noNi.yml
├── 90000001_relative_energies.csv
├── crest_calculations
│   ├── 90000001_Ni
│   └── 90000001_noNi
├── dft
│   ├── 90000001_errors.txt
│   ├── 90000001_noNi_00000
│   ├── 90000001_noNi_00001
│   ├── ...
│   ├── confselection_minmax_Ni.txt
│   ├── confselection_minmax_noNi.txt
│   ├── rmsdmatrix.csv
│   └── selected_conformers
└── xtb_scr_dir
```

## Citations
Please cite the original Kraken publication if you used this software. The executables for Multiwfn, dftd3, and dftd4 are included
in this repository and are used in the Kraken workflow. Please cite the Multiwfn, dftd3, and dftd4 publications.

__A Comprehensive Discovery Platform for Organophosphorus Ligands for Catalysis__<br>
Tobias Gensch, Gabriel dos Passos Gomes, Pascal Friederich, Ellyn Peters, Théophile Gaudin, Robert Pollice, Kjell Jorner, Akshat Kumar Nigam, Michael Lindner-D’Addario, Matthew S. Sigman, Alán Aspuru-Guzik. <br>
*J*. *Am*. *Chem*. *Soc*. __2022__ *144* (3), 1205-1217. DOI: [10.1021/jacs.1c09718](https://doi.org/10.1021/jacs.1c09718)

__Multiwfn: A Multifunctional Wavefunction Analyzer__<br>
Tian Lu, Feiwu Chen. <br>
*J*. *Comput*. *Chem*., __2012__ *33*, 580-592. DOI: [10.1002/jcc.22885](https://doi.org/10.1002/jcc.22885)

__A consistent and accurate ab initio parametrization of density functional dispersion correction (DFT-D) for the 94 elements H-Pu__<br>
Stefan Grimme, Jens Antony, Stephan Ehrlich, Helge Krieg. <br>
*J*. *Chem*. *Phys*. __2010__, *132*, 154104. DOI: [10.1063/1.3382344](https://doi.org/10.1063/1.3382344)

__Effect of the damping function in dispersion corrected density functional theory__<br>
Stefan Grimme, Stephan Ehrlich, Lars Goerigk. <br>
*J*. *Comput*. *Chem*., __2011__, *32*, 1456-1465. DOI: [10.1002/jcc.21759](https://doi.org/10.1002/jcc.21759)

__Extension of the D3 dispersion coefficient model__<br>
Eike Caldeweyher, Christoph Bannwarth, Stefan Grimme. <br>
*J*. *Chem*. *Phys*., __2017__, *147*, 034112. DOI: [10.1063/1.4993215](https://doi.org/10.1063/1.4993215)

__A generally applicable atomic-charge dependent London dispersion correction__<br>
Eike Caldeweyher, Sebastian Ehlert, Andreas Hansen, Hagen Neugebauer, Sebastian Spicher, Christoph Bannwarth, Stefan Grimme. <br>
*J*. *Chem*. *Phys*., __2019__, *150*, 154122. DOI: [10.1063/1.5090222](https://doi.org/10.1063/1.5090222 )

__Extension and evaluation of the D4 London-dispersion model for periodic systems__<br>
Eike Caldeweyher, Jan-Michael Mewes, Sebastian Ehlert, Stefan Grimme. <br>
*Phys*. *Chem*. *Chem*. *Phys*., __2020__, *22*, 8499-8512. DOI: [10.1039/D0CP00502A](https://doi.org/10.1039/d0cp00502a )


## New features
1.  Kraken rejects SMILES strings with undefined stereochemistry that would lead to inconsistent results (diastereomers)
2.  Included CLI scripts for automatically submitting Kraken calculations
3.  Included CLI scripts for converting Kraken result directories into CSV files or SDF files for viewing
4.  Automatic fchk generation


## Known differences and issues
1.  The conformer search produces slightly different conformers each run, so results vary slightly (around 1%) between runs
2.  The original code is designed to ignore descriptors that are assigned `None` as a result of xTB failure. This behavior is retained.
3.  Despite refactoring, the codebase still contains unused code.
5.  The conda versions of xTB and CREST are incompatible with Kraken. They frequently crashed during the --vipea calculations. The precompiled binaries of each release should be used
    or compiled directly. This workflow was developed with CREST 2.12 and xTB 6.4.0.
6.  CREST 2.12 produces many more conformers than CREST 2.8. Because conformers are selected based on properties, the number of conformers for DFT calculations should remain unchanged.
7.  Several descriptors vary substantially with xTB 6.7.0 or greater (EA/IP descriptors, nucleophilicity) because IPEA-xTB is not used for vertical IP/EA calculations.
    This will likely not affect the DFT level descriptors.


## Comparison between old and new workflows
The code for this updated workflow was adapted from the original Kraken code. Updates to the code should be done carefully so as to not impact the final descriptors.
We have included a comparison between the descriptors from the original Kraken publications and the new workflow for approximately 30 monophosphines in the validation/ folder.

## Including new templates for submission to HPC clusters
If you wish to submit batches of Kraken calculations (either the conformer search or the DFT portion of the workflow) to other systems, you must
create additional `.slurm` templates that are compatible with `/kraken/cli/submit_conf_search.py` and `/kraken/cli/submit_dft_calcs.py`. The
slurm scripts should contain the call to `run_kraken_conf_search` and `run_kraken_dft` along with placeholders for the following variables.

$TIME - Time in hours for the jobs <br>
$NPROCS - Number of processors to request for the job <br>
$MEM - Amount of memory in Gigabytes to request for the job <br>
$KID - 8-digit Kraken ID <br>
$CALCDIR - Calculation directory for the job <br>
$SMILES - Placeholder for the SMILES string of the monophosphines (only required for conf search portion) <br>
$CONVERSION_FLAG - Flag for method for generating coordinates from SMILES (default should be 4, only for conf search portion) <br>

Once you have created the new `.slurm` template, you can use the submission scripts (`submit_conf_search.py`) and specify the `--slurm-template` argument.
