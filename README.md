# Source Code For OTGCN
## Paper: Population Graph Cross-Network Node Classification for Autism Detection Across Sample Groups

## Setup
Ensure that you have all requirements installed. See `requirements.txt`

## Data Prep
To execute this you will first need to gather and prepare the ABIDE dataset that was used for this project. You can do so by running the `00_fetch_data.py` and `01_prep_data.py` scripts.

```bash
python3 00_fetch_data.py
python3 01_prep_data.py
```

## Running the Code
If you would like to run our cross fold validation script which tunes for the desired hyperparameters then you can run the `tune.py` script as shown below. You can also review the logs from the last time we ran the script at  `log/cv/abide_large`.

```bash
python3 tune.py -s abide_large
```

If you do not want to run the full tune script and simply wish to run our code with the chosen hyperparameters then you can use the `run.py` script as shown below. The logs from the last time we ran this script can be seen at `log/run/abide_large__abie_small`

```bash
python3 run.py -s abide_large -t abide_small
```
