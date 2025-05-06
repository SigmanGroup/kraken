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
see (and modify) the SLURM templates in the `kraken/slurm_templates` directory to accommodate your job scheduler. Please note that special symbols exist in the SLURM templates
that are substituted with actual values required by SLURM including `$KID`, `$NPROCS`, `$MEM`, and several others.

1. Format a .csv file that contains your monophosphine SMILES string, kraken id, and conversion flag.


        +-----------+------------------+-----------------+
        | KRAKEN_ID | SMILES           | CONVERSION_FLAG |
        +-----------+------------------+-----------------+
        | 5039      | CP(C)C           | 4               |
        | 10596     | CP(C1=CC=CC=C1)C | 4               |
        | ...       | ...              | ...             |
        +-----------+------------------+-----------------+


2. Run the example submission script with your requested inputs and configurations for the SLURM job. This will split your CSV file into individual conformer searches
   and submit them to nodes as their own job. The conformer searches are quick and not too computationally demanding, so use resources sparingly.

    ```bash
    example_conf_search_submission_script.py --csv small_molecules.csv --nprocs 8 --mem 16 --time 6 --calculation-dir ./data/ --debug
    ```

3. Once all jobs are complete, inspect the individual SLURM logfiles to ensure that each one terminated properly. You can search the SLURM logfiles for logging errors (search for "ERROR")
   and warnings (search for "WARNING"). If the jobs did not complete, be sure to check the .error file produced by SLURM and raise an issue on this repository.

4. After all jobs complete successfully, use the included CLI scripts that are installed along with Kraken to move your .com files into a common directory so you can submit them
   all at once instead of navigating to the individual `<KRAKEN_ID>/dft/` directories.

   a. For your convenience, CLI scripts have been included to move the DFT files from the `<KRAKEN_ID>/dft/` directory to somewhere else if you
      wish to run all of these calculations in another directory (or on another system entirely). This can be executed with the following command.

      ```bash
      extract_dft_files.py --input ./data/ --destination ./dft_calculation_folder_for_convenience/
      ```

   b. Use the group submission scripts to run the DFT calculations.

      ```bash
      for i in *.com; do subg16 $i -c sigman -m 32 -p 16 -t 12; done
      ```

   c. After completing all DFT calculations, you can return the results of the calculations to the appropriate `<KRAKEN_ID>/dft/` directories with
      a complementary script like this.

      ```bash
      return_dft_files.py --input ./dft_calculation_folder_for_convenience/ --destination ./data/
      ```
5. The DFT jobs should be evaluated for completeness and errors before returning them to the `<KRAKEN_ID>/dft/` directories. Kraken can accomodate some errors, but error handling
   is not fully tested. We recommend using [this tool to check your Gaussian16 log files](https://github.com/thejameshoward/GaussianLogfileAssessor.git). If your jobs are not converging
   or have imaginary frequencies, try implementing the CalcAll keyword in your optimization job in the `.com` file.

6. The DFT portion of the Kraken workflow can then be submitted to the compute nodes similarly to the conformer search portion.

    ```bash
    example_dft_submission_script.py --csv small_molecules.csv --nprocs 8 --mem 16 --time 6 --calculation-dir ./data/ --debug
    ```

7. Check the resulting SLURM .log and .error files for any indication that the individual SLURM jobs failed. If there is an unhandled error, be sure to raise an issue on this repository.

## Example Usage (directly running on a compute node)

1. Format a .csv file that contains your monophosphine SMILES string, kraken id, and conversion flag.


        +-----------+------------------+-----------------+
        | KRAKEN_ID | SMILES           | CONVERSION_FLAG |
        +-----------+------------------+-----------------+
        | 5039      | CP(C)C           | 4               |
        | 10596     | CP(C1=CC=CC=C1)C | 4               |
        | ...       | ...              | ...             |
        +-----------+------------------+-----------------+


2.  Run the first Kraken script on a .csv file containing the columns 'SMILES', 'KRAKEN_ID', and 'CONVERSION_FLAG'. Here are two examples. <br>

    ```bash
    run_kraken_conf_search.py -i ./data/input_file.csv --nprocs 4 --calculation-dir ./data/ --debug > kraken_conf_search.log
    ```

3. Once that script terminates, you can navigate to the calculation-dir (in this case `./data`) and find the directories for your Kraken conformer
   search calculations. The `<KRAKEN_ID>/dft/` directory contains the `.com` files that should be run with Gaussian16. You can run these directly
   in the existing `<KRAKEN_ID>/dft/` directory, or follow the steps below if you have many different monophosphines to evaluate.

   a. For your convenience, CLI scripts have been included to move the DFT files from the `<KRAKEN_ID>/dft/` directory to somewhere else if you
      wish to run all of these calculations in another directory (or on another system entirely). This can be executed with the following command.

      ```bash
      extract_dft_files.py --input ./data/ --destination ./dft_calculation_folder_for_convenience/
      ```

   b. After completing all DFT calculations, you can return the results of the calculations to the appropriate `<KRAKEN_ID>/dft/` directories with
      a complementary script like this.

      ```bash
      return_dft_files.py --input ./dft_calculation_folder_for_convenience/ --destination ./data/
      ```

4. After completing all DFT calculations and ensuring the requisite .log, .chk, and .wfn files are present in the correct `<KRAKEN_ID>/dft/` directories,
   the final step of the workflow can be executed with the `run_kraken_dft.py` script. This script must be run on each Kraken ID directory directly (i.e.,
   it does not currently support the .csv input). In the example below, one of the entries (KID 90000001) is processed with the script.

   Note that the `--force` flag ensures that if there is a problem with this script and it must be run again, all steps will be performed and no files
   will be read from potentially incomplete runs.

    ```bash
    run_kraken_dft.py --kid 90000001 --dir ./data/ --nprocs 4 --force > kraken_dft_processing_90000001.log
    ```

5. The resulting `.yml` files from both the CREST conformer search portion and the DFT processing portion of the Kraken workflow will exist in the `<KRAKEN_ID>`
   directory. In the present case, this directory's path is `./data/<KRAKEN_ID>` because we specified flags like `--calculation-dir` and `--dir` to be `./data` in
   previous steps. The final results will look something like this.

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
5.  The conda versions of xtb and CREST are incompatible with Kraken. They frequently crashed during the --vipea calculations. The precompiled binaries of each release should be used or compiled directly.

# Developer notes
## Differences when using newer SQM programs
1.  Several descriptors vary substantially with xtb 6.7.0 or greater (EA/IP descriptors, nucleophilicity) because IPEA-xTB is not used for vertical IP/EA calculations. This will likely not affect the DFT level descriptors.
2.  Crest v2.12 produces many more conformers than crest v2.8. Because conformers for DFT calculations are selected based on properties, the number of conformers for DFT calculations should remain unchanged.

## Setting up environment for old version of xtb

1.  Activate kraken

    ```bash
    conda activate kraken
    ```
   <br>
2.  Add the desired version of xtb to your path (this is written for development but will be clean in the final docs)

```export PATH=/home/sigman/kraken-xtb/6.2.2/bin/:/home/sigman/opt/openmpi/bin:/home/sigman/orca:/home/sigman/anaconda3/envs/krakendevclean/bin:/home/sigman/anaconda3/condabin:/home/sigman/.vscode/cli/servers/Stable-abd2f3db4bdb28f9e95536dfa84d8479f1eb312d/server/bin/remote-cli:/home/sigman/opt/openmpi/bin:/home/sigman/orca:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin```

3.  export XTBPATH=/home/sigman/xtb/xtb-6.6.0/share/xtb/

4.  Run Kraken on a single ligand

## TODO
1.  Provide details on the differences between xTB descriptors when using different xTB/CREST versions.
2.  Add instructions for installation and using different xtb/crest versions.
3.  Complete docstrings on functions
