import astropy.units as u

# camera constants
exposure_time = 10 * u.s
pixel_area = (3.45 * u.um) ** 2
lens_diameter = 20.0 * u.mm
lens_focal_length = 86.2 * u.mm
optical_loss = 0.96  # percentage
quantum_efficiency = 30

# noise constants
std_read = 2.31  # electrons
beta_t = 3.51 * (u.s ** -1)  # electron per second @ 25 C
full_well = 10500 # electrons
n_bits = 10  # number of bits for quantization
