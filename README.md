# Zodiacal Polarized
Zodiacal-polarized is a research module that simulates polarized zodiacal-light (ZL) emission and scattering 
using [Zodipy](https://github.com/Cosmoglobe/zodipy/tree/main),
processes them using polarimetric and noise models, 
adds polarimetric calibration errors, and estimates the errors.

## Project parts
The main module is 'zodipol', which contains the following functionalities:
* `zodipol.zodipol`: Contains the main class 'ZodiPol' that simulates ZL emission and scattering, the 'Observation' class that handles the Stokes vector values of an observation, and data generation code.
* `zodipol.imager`: Contains the 'Imager' class that simulates the imaging process, including the noise model.
* `zodipol.estimation`: Contains the error estimation functionality. The 'Calibration' class handles the calibration process, and the 'SelfCalibration' class handles the self-calibration process.
* `zodipol.background_radiation`: Contains the different background radiation sources that are simulated, 'IntegratedStarlight' which simulates the background stars, and 'PlanetaryLight' which simulates planets.
* `zodipol.utils`: Contains utility functions.
* `zodipol.visualization`: Contains plotting functions.
* `zodipol.zodipy_local`: Contains a local copy of the Zodipy module, integrated with an ad-hock model for polarized zodiacal light.

## Scripts
The scripts folder contains the following scripts:
* `self_calibration.py`: Runs the self-calibration process example.
* `calibration.py`: Runs the calibration process example.
* `anomaly_detection.py`: Runs the anomaly detection example.
* `create_image_example.py`: Runs the image creation example.
* `create_isl_model.py`: Creates the pre-calculated integrated starlight model.
* `create_model_image.py`: Creates a visualization of the sky model.



