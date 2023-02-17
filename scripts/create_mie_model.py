import numpy as np

from zodipol.mie_scattering import MieScatteringModel


if __name__ == '__main__':
    spectrum = np.logspace(np.log10(300), np.log10(700), 20)  # white light wavelength in nm
    mie = MieScatteringModel.train(spectrum)
    mie.save('saved_models/white_light_mie_model')
