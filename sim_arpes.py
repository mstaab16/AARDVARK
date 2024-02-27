import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import sawtooth, convolve2d
from skimage.transform import resize
import time

def sph_to_xyz(sph_coords):
    def single_coord(coord):
        theta, phi = coord
        return (
                    np.cos(phi) * np.sin(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(theta)
                )
    return np.array(list(map(lambda x: single_coord(x), sph_coords)))

def xyz_to_sph(xyz_coords):
    def single_coord(coord):
        x, y, z = coord
        # print(x,y,z)
        # print(np.arccos(x/np.sqrt(x**2 + y**2 + 1e-6)))
        return (
                    np.arccos(np.clip(z, -1, 1)),
                    np.sign(y + 1.12941e-10)*np.arccos(x/np.sqrt(x**2 + y**2 + 1e-10))
                )
    return np.array(list(map(lambda x: single_coord(x), xyz_coords)))

def rotate_z(sph_coords, phi):
    xyz_coords = sph_to_xyz(sph_coords)
    rz = np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi), np.cos(phi), 0],
        [0, 0, 1],
    ])
    return xyz_to_sph([np.dot(rz,coord) for coord in xyz_coords])

def rotate_y(sph_coords, theta):
    xyz_coords = sph_to_xyz(sph_coords)
    rz = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)],
    ])
    return xyz_to_sph([np.dot(rz,coord) for coord in xyz_coords])

def rotate_x(sph_coords, theta):
    xyz_coords = sph_to_xyz(sph_coords)
    rz = np.array([
        [1,0,0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])
    return xyz_to_sph([np.dot(rz,coord) for coord in xyz_coords])

def get_normed_xyz_coords_from_orientation(acceptance_angle, num_angles, polar, azimuthal, tilt, angles_in_degrees = True):
    # All angle arguments are in degrees. To use radians use angles_in_degrees = False
    # Zero polar, azimuthal, and tilt corresponds to coordinates along the ky=0 plane (phi = 0) (physics convention)
    # Rotations are applied in this order: 
    #   - Azimuthal (rotate about kz axis)
    #   - Polar_y (rotate about ky axis)
    #   - Polar_x (rotate about kx axis)

    if angles_in_degrees:
        acceptance_angle = np.radians(acceptance_angle)
        polar = np.radians(polar)
        azimuthal = np.radians(azimuthal)
        tilt = np.radians(tilt)
    
    slit_coords = np.array([np.linspace(-acceptance_angle/2, acceptance_angle/2, num_angles), np.zeros(num_angles)]).T

    rotated_coords = sph_to_xyz(rotate_x(rotate_y(rotate_z(slit_coords, azimuthal), tilt), polar))

    kx, ky, kz = rotated_coords.T

    return kx, ky, kz

def fermi_function(temperature, energy):
    """
    This function returns the Fermi function for a given energy and temperature
    """
    beta = 1/(8.617e-5*temperature) #eV^-1
    return 1/(np.exp(beta*(energy)) + 1)

def detector_grid(shape):
    x = np.linspace(-3.14,3.14,shape[1])
    y = np.linspace(-3.14,3.14,shape[0])
    xx, yy = np.meshgrid(x,y)
    # zz = np.sin(15*(np.sin(np.degrees(45))*xx + np.sin(np.degrees(45))*yy))
    # zz += np.sin(15*(np.sin(np.degrees(45))*xx + np.cos(np.degrees(45))*yy))
    freq = 20
    cutoff = 0.5
    zz1 = np.sin((xx + yy)*freq)
    zz2 = np.sin((xx - yy)*freq)
    zz1 = np.where(zz1 > cutoff, 1, .7)
    zz2 = np.where(zz2 > cutoff, 1, .7)
    # zz = np.where(zz1 + zz2 > cutoff, 1,0)
    # zz = np.where(zz1 > 0.5 or zz2 > 0.5, 1,0)
    zz = zz1 + zz2
    zz = convolve2d(zz, np.ones((3,3)), mode='same', boundary='symm')
    return zz

def lorentzian(x, x0, gamma):
    return (1/np.pi)*(gamma/((x - x0)**2 + (gamma/2)**2))


def tight_binding(kx, ky, kz, energies, intrinsic_broadening=0.005, num_bands=None, realistic_cuprate=False):
    if num_bands is None:
        num_bands = np.random.randint(1,10)
    t0 = np.random.uniform(0.001, 0.5, num_bands)
    t1 = np.random.uniform(-0.9, -0.5, num_bands)
    t2 = np.random.uniform(0.01, 0.2, num_bands)
    t3 = np.random.uniform(-0.4, -0.1, num_bands)
    t4 = np.random.uniform(-0.01, -0.1, num_bands)
    t5 = np.random.uniform(0.05, 0.09, num_bands)
    scaling_factors = np.random.uniform(.1, 1, num_bands)

    if realistic_cuprate:
        t0 = [0.090555]
        t1 = [-0.724474 * (1+0.6*np.cos(kz))]# * (np.sin(1*kz))*0.5]
        t2 = [0.0627473]
        t3 = [-0.165944]
        t4 = [-0.0150311]
        t5 = [0.0731361]
        num_bands = 1


    energy_func = lambda t0, t1, t2, t3, t4, t5, kx, ky: t0 + 0.5 * t1 * ( np.cos(kx) + np.cos(ky)) + t2 * np.cos(kx) * np.cos(ky) + 0.5* t3 * (np.cos(2 * kx) + np.cos(2*ky)) + 0.5 * t4 * (np.cos(2 * kx) * np.cos(ky) + np.cos(kx) * np.cos(2*ky)) + t5 * np.cos(2*kx) * np.cos(2*ky)

    # kx_, ky_, energies = np.meshgrid(kx, ky, energies)
    energies_ = np.tile(energies, (len(kx),1))
    intensity = np.zeros((len(energies), len(kx)))
    for t0_, t1_, t2_, t3_, t4_, t5_, a in zip(t0, t1, t2, t3, t4, t5, scaling_factors):
        energy_locations = energy_func(t0_, t1_, t2_, t3_, t4_, t5_, kx, ky)
        # print(energy_locations)
        intensity_ = lorentzian(energies_.T, energy_locations, intrinsic_broadening)
        # print(intensity_.shape)
        intensity += a * intensity_
    # print(energy_locations.shape)
    # print(intensity.shape)
    return intensity.T / intensity.max()

def pure_spectral_weight(kx, ky, kz, temperature = 30.0,
                         k_resolution = 0.011, e_resolution = 0.025,
                         energy_range = (-0.7, 0.1), num_energies = 200,
                         noise_level=0.3, lorentzian_width=0.005, random_bands=False):
    """
    This function computes the APRES intesnity after meshing the kx and ky.
    Returns a 3D array of the spectral weight at each kx, ky and energy.
    The spectral weight has the fermi function applied.
    The size of the gaussian convolution is determined by the e and k resolutions.
    The k and e units are in inverse Angstroms and eV respectively.
    """
    if noise_level > 1 or noise_level < 0:
        raise ValueError(f"Noise level {noise_level} must be in the range [0,1]")

    noise_level = noise_level**2

    low_energy = energy_range[0] - 25*(energy_range[1] - energy_range[0])/num_energies
    high_energy = energy_range[1] + 25*(energy_range[1] - energy_range[0])/num_energies
    energies = np.linspace(low_energy, high_energy, num_energies+50)
    # spectral_weight = np.abs(np.random.normal(0,0.005, size=(len(kx), len(energies))).astype(np.float32))
    spectral_weight = np.zeros((len(kx), len(energies)), dtype=np.float32)

    energy_bands = tight_binding(kx, ky, kz, energies, intrinsic_broadening=lorentzian_width, realistic_cuprate=not random_bands)

    # Set spectral weight to 1 where the energy is for a given kx, ky
    # for energy in energy_bands:
    #     for i in range(len(kx)):
    #         for j in range(len(energies)-1):
    #             intensity = (1/2*np.pi)*(lorentzian_width/((energies[j] - energy[i])**2 + (lorentzian_width/2)**2))
    #             # if intensity > 0 and intensity < 1:
    #             spectral_weight[i][j] += intensity
    #             # #if energy[i] is within the energy range pixel, set the spectral weight to 1
    #             # if energy[i] > energies[j] and energy[i] < energies[j+1]:
    #             #     spectral_weight[i][j] += 
    spectral_weight = energy_bands
    
    # calculate the fermi function in the shape of spectral weight
    fermi = fermi_function(temperature, energies)
    fermi = np.tile(fermi, (len(kx),1))
    # multiply the spectral weight by the fermi function
    # spectral_weight = gaussian_filter(spectral_weight, sigma = 0)
    spectral_weight = spectral_weight
    
    raw_spectrum = spectral_weight.copy()
    spectral_weight *= fermi
    #spectral_weight *= fermi_function(energies[j])

    # if K_RES and E_RES are not 0, use scipy gaussian_filter to apply resolution blur
    k_width = np.sqrt((kx[-1] - kx[0])**2 + (ky[-1] - ky[0])**2)
    k_resolution_for_scipy = (k_resolution/(k_width))*len(kx)/10
    e_resolution_for_scipy = (e_resolution/(np.max(energy_range) - np.min(energy_range)))*num_energies
    if k_resolution != 0 or e_resolution != 0:
        arpes_sim = gaussian_filter(spectral_weight, sigma = (k_resolution_for_scipy, e_resolution_for_scipy))
        high_stats_spectrum = arpes_sim.copy()# * fermi
        fermi_noise = np.random.normal(1, 0.1, size=arpes_sim.shape) * arpes_sim.max()
        scattered_signal = gaussian_filter(raw_spectrum, sigma = (100*k_resolution_for_scipy, 10*e_resolution_for_scipy))
        scattered_signal = (scattered_signal * fermi_noise)
        scattered_signal -= scattered_signal.min()
        scattered_signal /= scattered_signal.max()
        scattered_signal *= noise_level
        arpes_sim *= np.random.uniform(1-noise_level, 1, size=arpes_sim.shape)


    # arpes_sim -= arpes_sim.min()
    # arpes_sim /= arpes_sim.max()
    if k_resolution != 0 or e_resolution != 0:
        arpes_sim += scattered_signal

    arpes_sim += scattered_signal * detector_grid(arpes_sim.shape) * np.sqrt(noise_level)
    arpes_sim *= fermi
    arpes_sim += np.random.exponential(noise_level/10, size=arpes_sim.shape)

    # high_stats_spectrum *= fermi
    raw_spectrum *= fermi

    return arpes_sim.transpose()/arpes_sim.max(), high_stats_spectrum.transpose()/high_stats_spectrum.max(), raw_spectrum.transpose()/raw_spectrum.max()

def simulate_ARPES_measurement(polar=0.0, tilt=0.0, azimuthal=0.0,
                               photon_energy=100.0, noise_level=0.3,
                               acceptance_angle=30.0, num_angles=250,
                               num_energies=200, temperature=30.0,
                               k_resolution=0.011, e_resolution=0.025,
                               energy_range=(-0.7, 0.1), random_bands=False):
    """
    This function simulates an ARPES measurement for a given set of parameters.
    Returns the simulated ARPES spectrum.
    """
    inner_potential = 14
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # for polar in np.linspace(-10,10,25):
    inverse_hbar_times_sqrt_2me = 0.512316722 #eV^-1/2 Angstrom^-1
    kx, ky, kz = get_normed_xyz_coords_from_orientation(acceptance_angle, num_angles+50, polar, azimuthal, tilt)
    r = inverse_hbar_times_sqrt_2me*np.sqrt((photon_energy-4.3)+inner_potential) #(1/hbar) * sqrt(2 m_e KE)
    kx *= r
    ky *= r
    kz *= r
    # a = b = c = 3.7
    a = b = c = 4.258
    c = 17.46
    a_star = 2*np.pi/a
    b_star = 2*np.pi/b
    c_star = 2*np.pi/c
    kx_in_bz = sawtooth(2*np.pi*(kx/a_star-0.5))*a_star/2
    ky_in_bz = sawtooth(2*np.pi*(ky/b_star-0.5))*b_star/2
    kz_in_bz = sawtooth(2*np.pi*(kz/c_star-0.5))*c_star/2
        # ax.scatter(kx_in_bz, ky_in_bz, kz_in_bz, cmap='Spectral_r', c=np.arange(kx_in_bz.shape[0]))

    # ax.set_xlim(-a_star/2, a_star/2)
    # ax.set_ylim(-b_star/2, b_star/2)
    # ax.set_zlim(-c_star/2, c_star/2)
    # ax.set_xlabel('kx')
    # ax.set_ylabel('ky')
    # ax.set_zlabel('kz')
    # plt.show()
    arpes, high_stats, raw_spectrum = pure_spectral_weight(
            kx_in_bz*a, ky_in_bz*b, kz_in_bz*c,
            temperature=temperature, k_resolution=k_resolution,
            e_resolution=e_resolution, energy_range=energy_range,
            num_energies=num_energies, noise_level=noise_level,
            random_bands=random_bands)
    return arpes[25:-25,25:-25], high_stats[25:-25,25:-25],raw_spectrum[25:-25,25:-25], kx_in_bz[25:-25], ky_in_bz[25:-25], kz_in_bz[25:-25]

def generate_batch(n, noise=None, k_resolution=None, e_resolution=None, num_angles=256, num_energies=256, random_bands=True):
    print("Generating data...")
    X = np.zeros((n, num_angles, num_energies), dtype=np.float32)
    Y = np.zeros((n, num_angles, num_energies), dtype=np.float32)
#     polar = np.random.uniform(-15,15, n)
#     tilt = np.random.uniform(-15,15, n)
#     azimuthal = np.random.uniform(0, 360, n)
#     photon_energy = np.random.normal(100, 10, n)
    polar=[5]
    tilt=[5]
    azimuthal=[30]
    photon_energy=[100]

    if noise is None:
        noise_level = np.abs(np.random.normal(0.2, 0.1, n))
    else:
        noise_level = np.abs(np.random.normal(noise, noise*0.1, n))
    if k_resolution is None:
        k_resolution = np.random.normal(0.0011, 0.003, n)
    else:
        k_resolution = np.random.normal(k_resolution, k_resolution*0.1, n)
    if e_resolution is None:
        e_resolution = np.random.normal(0.015, 0.003, n)
    else:
        e_resolution = np.random.normal(e_resolution, e_resolution*0.1, n)


    for i in range(n):
        print(f"Generating spectrum {i+1}/{n}", end='\r')
        X_, _, Y_, _, _, _ = simulate_ARPES_measurement(
                                        polar=polar[i], azimuthal=azimuthal[i], tilt=tilt[i],
                                        k_resolution=k_resolution[i], e_resolution=e_resolution[i],
                                        noise_level=noise_level[i], photon_energy=photon_energy[i],
                                        num_angles=256, num_energies=256,
                                        random_bands=random_bands,
                                        )
        # X[i] = resize(X_, (64,64))
        # Y[i] = resize(Y_, (64,64))
        X[i] = X_
        Y[i] = Y_
    print(" "*100)
    return X, Y


if __name__ == "__main__":
    polar = 4
    tilt = 3
    azimuthal = 16
    energy_min = -0.7
    energy_max = 0.1
    k_resolution = 0.015
    e_resolution = 0.020
    noise_level = .5
    photon_energy = 110
    temperature = 100.0
    acceptance_angle=30.0
    num_angles = 512
    num_energies = 512

    inner_potential = 14
    inverse_hbar_times_sqrt_2me = 0.512316722 #eV^-1/2 Angstrom^-1
    # a = b = c = 3.7
    a = b = c = 4.258
    c = 17.46
    a_star = 2*np.pi/a
    b_star = 2*np.pi/b
    c_star = 2*np.pi/c


    cmap='terrain'
    fig, axes = plt.subplots(2, 5, figsize=(10,4))
    for i, noise_level in enumerate(np.linspace(0.001,.9999,10)):
        k_res = k_resolution*(1+5*noise_level)
        e_res = e_resolution*(1+5*noise_level)
        measured, high_stats, raw, kx_, ky_, kz_ = simulate_ARPES_measurement(
                                                    polar=polar, azimuthal=azimuthal, tilt=tilt,
                                                    k_resolution=k_res, e_resolution=e_res,
                                                    noise_level=noise_level, photon_energy=photon_energy,
                                                    num_angles=num_angles, num_energies=num_energies,
                                                    energy_range=(energy_min, energy_max),
                                                    random_bands=False, temperature=temperature,
                                                    )
        print(i)
        noise_image = np.random.poisson(high_stats*1/(noise_level**3+0.00001), size=high_stats.shape).astype(np.float32)
        noise_image /= noise_image.max()
        detector_grid_noise = np.random.poisson(detector_grid(noise_image.shape), size=high_stats.shape).astype(np.float32)
        detector_grid_noise /= detector_grid_noise.max()
        fermi_image = fermi_function(temperature, np.linspace(energy_min, energy_max, num_energies))
        fermi_image = np.tile(fermi_image, (num_energies,1)).T
        scatter_noise = gaussian_filter(high_stats, np.array(high_stats.shape)/10.0)
        scatter_noise *= fermi_image
        scatter_noise /= scatter_noise.max()
        background = np.tile(np.linspace(1,0,num_energies), (num_energies,1)).T
        random_noise = np.random.poisson(fermi_image, size=noise_image.shape).astype(np.float32)
        random_noise /= random_noise.max()
        final = (noise_image*3 + (random_noise*2 + scatter_noise*5 + background*5) * noise_level * 2) * detector_grid_noise
        final /= final.max()
        axes.flatten()[i].imshow(final, cmap=cmap, origin='lower', extent=[-acceptance_angle/2, acceptance_angle/2, energy_min, energy_max])
        axes.flatten()[i].set_title(f"noise level: {noise_level:.2f}")
        axes.flatten()[i].axhline(0, color='k', linestyle='--')
        # Hide X and Y axes label marks
        axes.flatten()[i].xaxis.set_tick_params(labelbottom=False)
        axes.flatten()[i].yaxis.set_tick_params(labelleft=False)

        # Hide X and Y axes tick marks
        axes.flatten()[i].set_xticks([])
        axes.flatten()[i].set_yticks([])
        axes.flatten()[i].set_box_aspect(1)
        axes.flatten()[i].set_aspect(acceptance_angle/(energy_max-energy_min))
    fig.tight_layout()
    plt.show()


    # for inner_potential in np.arange(30):
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7,3))
    #     for i, photon_energy in enumerate(np.linspace(110,144,34)):
    #         print(i,end='\r')
    #         measured, high_stats, raw, kx_, ky_, kz_ = simulate_ARPES_measurement(
    #                                             polar=polar, azimuthal=azimuthal, tilt=tilt,
    #                                             k_resolution=k_resolution, e_resolution=e_resolution,
    #                                             noise_level=noise_level, photon_energy=photon_energy,
    #                                             num_angles=num_angles, num_energies=num_energies,
    #                                             random_bands=False, temperature=temperature,
    #                                             )
    #         kx, ky, kz = get_normed_xyz_coords_from_orientation(acceptance_angle, num_angles+50, polar, azimuthal, tilt)
    #         r = inverse_hbar_times_sqrt_2me*np.sqrt((photon_energy-4.3)+inner_potential) #(1/hbar) * sqrt(2 m_e KE)
    #         kx *= r
    #         ky *= r
    #         kz *= r
    #         kx_in_bz = sawtooth(2*np.pi*(kx/a_star-0.5))*a_star/2
    #         ky_in_bz = sawtooth(2*np.pi*(ky/b_star-0.5))*b_star/2
    #         kz_in_bz = sawtooth(2*np.pi*(kz/c_star-0.5))*c_star/2
    #         kx_in_bz = kx_in_bz[25:-25]
    #         ky_in_bz = ky_in_bz[25:-25]
    #         kz_in_bz = kz_in_bz[25:-25]
    #         ax1.scatter(kx_in_bz, kz_in_bz, c=measured[160:170,:].mean(axis=0), cmap='Greys',s=1, alpha=1)
    #         ax2.scatter(kx_, kz_, c=measured[160:170,:].mean(axis=0), cmap='Greys',s=1, alpha=1)
    #         #plt.scatter(kx, kz, c=measured[45:50,:].mean(axis=0), cmap='Greys',s=1)


    #     #plt.imshow(im, cmap='terrain', origin='lower')
    #     ax1.set_title(f"Inner Potential: {inner_potential:.2f} eV")
    #     ax2.set_title(f"Inner Potential: 14.00eV")
    #     ax1.set_aspect('equal')
    #     ax2.set_aspect('equal')
    #     ax1.set_xlim(-a_star/2,a_star/2)
    #     ax1.set_ylim(-c_star/2,c_star/2)
    #     ax2.set_xlim(-a_star/2,a_star/2)
    #     ax2.set_ylim(-c_star/2,c_star/2)
    #     fig.tight_layout()
    #     plt.savefig(f"{inner_potential:04}.png")
    #     plt.close()

    

