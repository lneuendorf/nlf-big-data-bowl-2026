# NFL Big Data Bowl 2026
## Setup Instructions
Follow these steps to set up the repository and run the notebooks:

1. Prepare the Data
    1. Download the data from the 2026 NFL Big Data Bowl Kaggle competition.
    2. Rename the outer folder of the downloaded data to data and place it in the root of the repository. The structure should look like this:
        ```bash 
        /nfl-big-data-bowl-2026
        ├── data/
        │   ├── supplementary_data.csv
        │   └── train/
        │       └── input_2023_w01.csv
        ├── conda.yml
        ├── notebooks/
        └── ...
        ```
2. Set Up the Environment
    1. Create a conda environment using the provided conda.yml file:
        ```bash 
        conda env create -f conda.yml
        ```
    2. Activate the environment:
        ```bash 
        conda activate bdb26
        ```
    3. Create an IPython kernel for the environment:
        ```bash 
        python -m ipykernel install --user --name bdb26 --display-name "bdb26"
        ```

3. Run the Notebook
    1. Open a notebook, e.g., notebooks/eda.ipynb:
    2. Select the kernel named Python (bdb26) from the kernel dropdown menu.
    3. Run the notebook cells.

You should now be able to run the project successfully!