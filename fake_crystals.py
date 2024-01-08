import zmq
from messages import *
import numpy as np
import matplotlib.pyplot as plt
from sim_ARPES import simulate_ARPES_measurement
from scipy.spatial import Voronoi, voronoi_plot_2d

from time import perf_counter

context = zmq.Context()

class FakeVoronoiCrystal:
    def __init__(self):
        np.random.seed(40)
        num_crystalllites = 4
        vor_points = np.random.uniform(-3,3,(num_crystalllites,2))
        self.vor_azimuths = np.random.uniform(-180,180, num_crystalllites)
        self.vor_tilts = np.random.uniform(-15,15, num_crystalllites)
        self.vor_polars = np.random.uniform(-15,15, num_crystalllites)
        self.vor_intensities = np.random.uniform(1,1, num_crystalllites)
        self.vor = Voronoi(vor_points)
        self.xcoords = np.linspace(-3,3,500)
        self.ycoords = np.linspace(-3,3,500)

    def measure(self, x, y):
        num_angles = 128
        num_energies = 128
        # if np.abs(x) > 2 or np.abs(y) > 2:
        
        if not((np.abs(x-1.5)**2 + np.abs(y+1.5)**2 < 1) or\
                (x>-2.75 and x<-1 and y>-2 and y<2) or\
                (np.abs(x-1.5)**2 + np.abs(y-1.5)**2 < 2)):
            return x, y, np.random.exponential(0.001,(num_energies, num_angles))

        dx_sq = (self.vor.points[:,0] - x)**2
        dy_sq = (self.vor.points[:,1] - y)**2
        idx = np.argmin(dx_sq + dy_sq)
        azimuth = self.vor_azimuths[idx]
        tilt = self.vor_tilts[idx]
        polar = self.vor_polars[idx]
        intensity = self.vor_intensities[idx]

        spectrum = simulate_ARPES_measurement(
                        polar=polar, tilt=tilt, azimuthal=azimuth,
                        num_angles=num_angles, num_energies=num_energies,
                        k_resolution=0.005, e_resolution=0.01)
        return x, y, spectrum*intensity

    def get_boundaries(self):
        return np.min(self.xcoords), np.max(self.xcoords), np.min(self.ycoords), np.max(self.ycoords)

    def plot(self):
        voronoi_plot_2d(self.vor)
        plt.xlim(-3,3)
        plt.ylim(-3,3)
        plt.grid(which='both')
        plt.show()
        fig, axes = plt.subplots(len(self.vor.points)//3, 3, figsize=(8,8))
        for ax, (x,y) in zip(axes.ravel(),self.vor.points):
            ax.imshow(self.measure(x,y)[2], cmap='gray_r', origin='lower')
            ax.set_title(f'x={x:.2f}, y={y:.2f}')
            ax.grid(which='both')
        fig.tight_layout()
        plt.show()


class FakeGrapheneCrystal:
    def __init__(self):
        import h5py
        self.filename =  r"20190915_01325_binned.h5"
        self.file = h5py.File(self.filename, 'r')
        self.data = self.file['2D_Data']['Fixed_Spectra1'][:]
        self.xcoords = self.file['0D_Data']['Scan X'][:]
        self.ycoords = self.file['0D_Data']['Scan Y'][:]


    def measure(self, x, y):
        dx = (self.xcoords - x)**2
        dy = (self.ycoords - y)**2
        d = np.sqrt(dx + dy)
        i = np.argmin(d)
        measured_x = self.xcoords[i]
        measured_y = self.ycoords[i]
        spectrum = self.data[:,:,i]
        return measured_x, measured_y, spectrum

    def get_boundaries(self):
        return np.min(self.xcoords), np.max(self.xcoords), np.min(self.ycoords), np.max(self.ycoords)

        
class FakeWSe2Crystal:
    def __init__(self):
        from astropy.io import fits
        self.filename =  r"20161215_00045_binned.fits"
        self.file = fits.open(self.filename)
        self.data = self.file[1].data['Fixed_Spectra0']
        self.xcoords = self.file[1].data['Scan Z']
        self.ycoords = self.file[1].data['Scan Y']


    def measure(self, x, y):
        dx = (self.xcoords - x)**2
        dy = (self.ycoords - y)**2
        d = np.sqrt(dx + dy)
        i = np.argmin(d)
        measured_x = self.xcoords[i]
        measured_y = self.ycoords[i]
        spectrum = self.data[i,:,:]
        return measured_x, measured_y, spectrum

    def get_boundaries(self):
        return np.min(self.xcoords), np.max(self.xcoords), np.min(self.ycoords), np.max(self.ycoords)


def main():
#     gr = FakeGrapheneCrystal()
#     xmin, xmax, ymin, ymax = gr.get_boundaries()
#     start = perf_counter()
#     for _ in range(10):
#         x_choice, y_choice = np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax)
#         measured_x, measured_y, spectrum = gr.measure(x_choice, y_choice)
#         print(f"Measuring at {x_choice}, {y_choice}")
#         print(f"Measured at {measured_x}, {measured_y}")
# 
#     end = perf_counter()
#     print(f"Time per measurement: {(end-start)/10:.3f} s")
#     
#     plt.imshow(spectrum.T, origin='lower', cmap='Greys')
#     plt.show()
     vor = FakeVoronoiCrystal()
     vor.plot()
#     wse2 = FakeWSe2Crystal()
#     xmin, xmax, ymin, ymax = wse2.get_boundaries()

#     for _ in range(10):
#         x_choice, y_choice = np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax)
#         measured_x, measured_y, spectrum = wse2.measure(x_choice, y_choice)
#         print(f"Measuring at {x_choice}, {y_choice}")
#         print(f"Measured at {measured_x}, {measured_y}")
#         print(f"Spectrum Sum: {np.sum(spectrum)}")


if __name__ == "__main__":
    main()