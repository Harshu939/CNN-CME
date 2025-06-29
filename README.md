# CNN-Based Model for 3D CME Parameter Prediction

This repository contains the code and dataset used in the research work *"CNN-Based Model for 3D CME Parameter Prediction."* 
It includes scripts for generating synthetic data, preprocessing, training the CNN model, and making predictions.

1. Contents

- **IDL Code**: For generating synthetic CME image data and metadata.
- **Python Code**: For processing data, training the model, and evaluating predictions.
- **Sample Dataset**: A subset of the data used in the experiments.

-------------------------------------------------------------------

2. Synthetic Data Generation (IDL)

The synthetic data (images, metadata, and labels) is generated using IDL scripts.

### Files:
- `call_synth_cme.pro`: Main driver script that sets parameters and calls the main procedure 'synth_cme.pro'.
- `synth_cme.pro`: Main code to generate and download synthetic CME images as jpegs and metadata as csv.

### Parameters set in `call_synth_cme.pro`:
- Date range for modeling
- Output directories for FITS, JPEGs, and CSVs
- Mission and instrument selection
- Initial and final heights of the CME

### How to Run:
1. Place all IDL files within a **SolarSoft** environment.
2. Open IDL and run:
   ```idl
   .compile call_synth_cme
   call_synth_cme

