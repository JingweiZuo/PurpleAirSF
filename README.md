## Unleashing Realistic Air Quality Forecasting: Introducing the Ready-to-Use PurpleAirSF Dataset

This is the companion repository for the PurpleAirSF Dataset. 

The preprocessed datasets can be downloaded [here](https://tiiuae-my.sharepoint.com/:u:/g/personal/michele_baldo_tii_ae/EdaRXDgCVNtJuBh6hqSJ3rsBTfm-sD3xf_NHXvXxsWKwfA?e=6SgFbh) 

### Datasets with three temporal granularities are provided:
<p float="left">
  <img src="figures/10M_316.png" width="30%" />
  <img src="figures/1H_232.jpg" width="30%" /> 
  <img src="figures/6H_112.jpg" width="30%" />
  <figcaption> Figure 1 (Left): 10-min granular data with 316 stations; (Middle): 1-hour granular data with 232 stations; (Right): 6-hour granular data with 112 stations</figcaption>
</p>
The statistical summary of the datasets are shown in the below table: 
<p float="left">
  <img src="figures/dataset_summary.png" width="60%" />
</p>



## How to use PurpleAirSF?

The preprocessed data has a shape of (N, L, F)
- N is the number of sensor stations
- L is the entire sequence length
- F is the number of features/measures in each station. 

Users are free to split the dataset with different window size. 

Here are a list of ordered measures that we considered during data collection:
```
    'humidity', 'temperature', 'pressure',
    'pm2.5_alt', 'scattering_coefficient', 'deciviews', 'visual_range',
    '0.3_um_count', '0.5_um_count', '1.0_um_count', '2.5_um_count',
    '5.0_um_count', '10.0_um_count', 'pm1.0_cf_1', 'pm1.0_atm', 'pm2.5_atm',
    'pm2.5_cf_1', 'pm10.0_atm', 'pm10.0_cf_1'
```

The detailed descriptions of the measures are shown in the below table: 
<p float="left">
  <img src="figures/dataset_measures.png" width="60%" />
</p>




## Getting data with the APIs

Users can also use our provided scripts to fetch raw data via PurpleAir API.
For some feature you need the private keys for the APIs.

PurpleAir: write a email to contact@purpleair.com with subject "API keys for PurpleAirAPI". 
They will send you your API private key. Once you have it, just create a file in `keys/PurpleAir_API_key.conf`
with the following structure:
```
[purpleair.com]
API_readKey = YOUR-PRIVATE-READ-KEY
```

For Airly go on the website and ask for your private key. 
Then create the file `keys/Airly_API_key.conf` as follows:
```
[airly.org]
API_key = YOUR-PRIVATE-KEY
```