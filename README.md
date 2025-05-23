# Kraken
In 2024, the code for the Kraken workflow was significantly refactored to streamline the addition of new ligands and improve portability.
This update enables the use of modern software versions including xtb v6.4.0 and crest v2.12 in addition to the original versions of these
programs used to build Kraken (xtb v6.2.2 and crest v2.8). The provided environment is capable of running both versions simply by calling
the desired program version through a subprocess. In our testing, the new workflow performs similarly to the original workflow with several
exceptions depending on the program versions used.

There are some significant differences between xTB versions that lead to different properties at the xTB-GFN2 level. However, we have compared
the DFT properties of over 60 monophosphines to the original Kraken results. For a single monophosphine, the differences between the old and
new properties may exceed 75% of the original value (mostly quadrant/octant volumes). However, these differences likely result from either
incomplete conformer searches in the original dataset or the stochastic nature of conformer searches.

__A detailed report comparing 28 monophosphines from the original Kraken workflow is provided in the validation folder__
## Installation (Linux systems only)

1.  Create a conda environment with the included environment yaml file.

```bash
conda env create --name kraken --file=kraken.yml
```

2.  Activate the new environment

```bash
conda activate kraken
```

3. (Optional) [Download an appropriate version of xtb and crest](https://github.com/crest-lab/crest). The precompiled versions
   for xtb v6.6.0 and crest v2.12 worked in our testing. Alternatively, CHPC users can use `module load crest/2.12` to
   load xTB v6.4.0 and CREST v2.12.

4. Place the precompiled binaries for xtb and crest somewhere on your PATH. Kraken will call these through subprocess.

5. Install the `kraken` package by navigating to the parent directory containing `setup.py` and running this command

```bash
pip install .
```
## Example Usage (submission to CHPC for the Sigman group)
These instructions are for Sigman group members to submit batches of calculations to the Sigman owner nodes on Notchpeak. For other users outside of the Sigman group, please
see (and modify) the SLURM templates in the `kraken/slurm_templates` directory to accommodate your job scheduler __before installation__. Please note that special symbols exist in the SLURM templates that are substituted with actual values required by SLURM including `$KID`, `$NPROCS`, `$MEM`, and several others.

1. Format a `.csv` file that contains your monophosphine SMILES string, kraken id, and conversion flag:

    | KRAKEN_ID | SMILES           | CONVERSION_FLAG |
    |-----------|------------------|-----------------|
    | 5039      | CP(C)C           | 4               |
    | 10596     | CP(C1=CC=CC=C1)C | 4               |
    | ...       | ...              | ...             |

2. Run the example submission script with your requested inputs and configurations:

```bash
example_conf_search_submission_script --csv small_molecules.csv --nprocs 8 --mem 16 --time 6 --calculation-dir ./data/ --debug
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
example_dft_submission_script --csv small_molecules.csv --nprocs 8 --mem 16 --time 6 --calculation-dir ./data/ --debug
```

7. Check SLURM `.log` and `.error` files and raise an issue on this repo if necessary.

## Example Usage (directly running on a compute node)
Kraken can also be executed directly from the commandline. This can be useful if you wish to create your own wrapper scripts for submission to other HPC systems.
Please not that running this script will call computationally intensive programs and should not be run on head nodes.

1. Format a `.csv` file that contains your monophosphine SMILES string, KRAKEN_ID, and CONVERSION_FLAG:

    | KRAKEN_ID | SMILES           | CONVERSION_FLAG |
    |-----------|------------------|-----------------|
    | 5039      | CP(C)C           | 4               |
    | 10596     | CP(C1=CC=CC=C1)C | 4               |
    | ...       | ...              | ...             |

2. Run the first Kraken script on a `.csv` file containing the columns `SMILES`, `KRAKEN_ID`, and `CONVERSION_FLAG`:

```bash
run_kraken_conf_search -i ./data/input_file.csv --nprocs 4 --calculation-dir ./data/ --debug > kraken_conf_search.log
```

3. After the script terminates, navigate to `./data/` to find the conformer search directories. Each `<KRAKEN_ID>/dft/` folder contains the `.com` files
   for Gaussian16. You can run them directly in-place or follow the steps below if evaluating many ligands:<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;a. Move DFT input files to a centralized directory:<br>

```bash
extract_dft_files --input ./data/ --destination ./dft_calculation_folder_for_convenience/
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;b. After running the DFT calculations, return the results to their original locations:<br>

```bash
return_dft_files --input ./dft_calculation_folder_for_convenience/ --destination ./data/
```

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
│   ├── 90000001_noNi_00002
│   ├── 90000001_noNi_00003
│   ├── 90000001_noNi_00004
│   ├── 90000001_noNi_00005
│   ├── 90000001_noNi_00006
│   ├── 90000001_noNi_00007
│   ├── 90000001_noNi_00009
│   ├── confselection_minmax_Ni.txt
│   ├── confselection_minmax_noNi.txt
│   ├── fort.7
│   ├── rmsdmatrix.csv
│   └── selected_conformers
└── xtb_scr_dir
```


## Citations
Please cite the original kraken publication if you used this software. The executables for Multiwfn, dftd3, and dftd4 are included
in this repository and are used in the Kraken workflow. Please cite the Multiwfn, dftd3, and dftd4 publications.

__A Comprehensive Discovery Platform for Organophosphorus Ligands for Catalysis__<br>
Tobias Gensch, Gabriel dos Passos Gomes, Pascal Friederich, Ellyn Peters, Théophile Gaudin, Robert Pollice, Kjell Jorner, AkshatKumar Nigam, Michael Lindner-D’Addario, Matthew S. Sigman, Alán Aspuru-Guzik. <br>
*J*. *Am*. *Chem*. *Soc*. __2022__ *144* (3), 1205-1217. DOI: [10.1021/jacs.1c09718](https://doi.org/10.1021/jacs.1c09718)

__Multiwfn: A Multifunctional Wavefunction Analyzer__<br>
Tian Lu, Feiwu Chen. <br>
*J*. *Comput*. *Chem*., __2012__ *33*, 580-592. DOI: [10.1002/jcc.22885](https://doi.org/10.1002/jcc.22885)

__A consistent and accurate ab initio parametrization of density functional dispersion correction (DFT-D) for the 94 elements H-Pu__<br>
Stefan Grimme, Jens Antony, Stephan Ehrlich, Helge Krie. <br>
*J*. *Chem*. *Phys*. __2010__, *132*, 154104. DOI: [10.1063/1.3382344](https://doi.org/10.1063/1.3382344)

__Effect of the damping function in dispersion corrected density functional theory__<br>
Stefan Grimme, Stephan Ehrlich, Lars Goerigk. <br>
*J*. *Comput*. *Chem*., __2011__, *32*: 1456-1465. DOI: [10.1002/jcc.21759](https://doi.org/10.1002/jcc.21759)

__Extension of the D3 dispersion coefficient model__<br>
Eike Caldeweyher, Christoph Bannwarth, Stefan Grimme. <br>
*J*. *Chem*. *Phys*., __2017__, *147*, 034112. DOI: [10.1063/1.4993215](https://doi.org/10.1063/1.4993215)

__A generally applicable atomic-charge dependent London dispersion correction__<br>
Eike Caldeweyher, Sebastian Ehlert, Andreas Hansen, Hagen Neugebauer, Sebastian Spicher, Christoph Bannwarth, Stefan Grimme. <br>
*J*. *Chem*. *Phys*., __2019__, *150*, 154122. DOI: [10.1063/1.5090222](https://doi.org/10.1063/1.5090222 )

__Extension and evaluation of the D4 London-dispersion model for periodic systems__<br>
Eike Caldeweyher, Jan-Michael Mewes, Sebastian Ehlert, Stefan Grimme. <br>
*Phys*. *Chem*. *Chem*. *Phys*., __2020__, *22*, 8499-8512. DOI: [10.1039/D0CP00502A](https://doi.org/10.1039/d0cp00502a )

## Known Issues
1.  The conformers generated by crest will differ between runs, so it can be difficult to compare xTB properties.
2.  The original version of xTB often fails or produces incorrect results when using the --esp and --vipea flag. Newer versions have not failed in our testing.
3.  The original code is designed to ignore descriptors that are assigned `None` as a result of xTB failure. This behavior is retained.
4.  Despite refactoring, the codebase still contains unused code.
5.  The conda versions of xtb and CREST are incompatible with Kraken. They frequently crashed during the --vipea calculations. The precompiled binaries of each release should be used or compiled directly. This workflow was developed with CREST 2.12 and xTB 6.4.0.

```

       ==============================================
       |                                            |
       |                 C R E S T                  |
       |                                            |
       |  Conformer-Rotamer Ensemble Sampling Tool  |
       |          based on the GFN methods          |
       |             P.Pracht, S.Grimme             |
       |          Universitaet Bonn, MCTC           |
       ==============================================
       Version 2.12,   Thu 19. Mai 16:32:32 CEST 2022
  Using the xTB program. Compatible with xTB version 6.4.0

```


# Developers
## Differences when using newer SQM programs
1.  Several descriptors vary substantially with xtb 6.7.0 or greater (EA/IP descriptors, nucleophilicity) because IPEA-xTB is not used for vertical IP/EA calculations. This will likely not affect the DFT level descriptors.
2.  Crest v2.12 produces many more conformers than crest v2.8. Because conformers for DFT calculations are selected based on properties, the number of conformers for DFT calculations should remain unchanged.

## Comparison between old and new workflows
The code for this updated workflow was adapted from the original Kraken code. Some aspects have been altered for ease of use like automating .fchk generation. Updates to the
code should be done carefully so as to not impact the final descriptors produced at the end of the workflow. We have included a comparison between the descriptors from the
original Kraken publications and the new workflow for approximately 30 monophosphines in the validation/ folder.

## Including new templates for submission to HPC clusters
If you wish to submit batches of Kraken calculations (either the conformer search or the DFT portion of the workflow) to other systems that are not the Notchpeak Sigman owner nodes,
you will need to create additional `.slurm` templates that are compatible with `/kraken/cli/example_conf_search_submission_script.py` and `/kraken/cli/example_dft_submission_script.py`
The slurm scripts should contain the call to `run_kraken_conf_search` and `run_kraken_dft` along with placeholders for the following variables.

$TIME - Time in hours for the jobs <br>
$NPROCS - Number of processors to request for the job <br>
$MEM - Amount of memory in Gigabytes to request for the job <br>
$KID - 8-digit Kraken ID <br>
$CALCDIR - Calculation directory for the job <br>
$SMILES - Placeholder for the SMILES string of the monophosphines (only required for conf search portion) <br>
$CONVERSION_FLAG - Flag for method for generating coordinates from SMILES (default should be 4, only for conf search portion) <br>

Once you have created the new `.slurm` template, place it in the `/kraken/slurm_templates/` directory. You can then modify the `SLURM_TEMPLATE` variable in both  `/kraken/cli/example_conf_search_submission_script.py` and `/kraken/cli/example_dft_submission_script.py` submission scripts to point to your new `.slurm` file. Finally, install the Kraken package
using the instructions above. Your new `.slurm` file will be used instead of the one provided in the repository.

## TO-DO
1. Refactor SLURM_TEMPLATE usage in the submission scripts to allow users to change the way Kraken CLI scripts interact with HPC job schedulers.