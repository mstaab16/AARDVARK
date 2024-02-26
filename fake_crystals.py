import numpy as np
import matplotlib.pyplot as plt
from sim_arpes import simulate_ARPES_measurement
from scipy.spatial import Voronoi, voronoi_plot_2d

from time import perf_counter


class FakeVoronoiCrystal:
    def __init__(self, num_angles, num_energies, bounds, min_steps, num_crystallites, plot=True):
        # np.random.seed(40)
        self.num_angles = num_angles
        self.num_energies = num_energies
        self.num_crystallites = num_crystallites
        self.min_steps = min_steps
        self.bounds = bounds
        vor_points = np.array([
            np.random.uniform(bounds[0][0], bounds[0][1], num_crystallites),
            np.random.uniform(bounds[1][0], bounds[1][1], num_crystallites),
        ]).T
        # vor_points = np.array([
        #     [0.2, 0.4],
        #     [0.4, 0.6],
        #     [0.75, 0.5],
        # ])
        self.vor_azimuths = np.random.uniform(-180,180, num_crystallites)
        self.vor_tilts = np.random.uniform(-15,15, num_crystallites)
        self.vor_polars = np.random.uniform(-15,15, num_crystallites)
        self.vor_intensities = np.random.uniform(1,1, num_crystallites)
        self.vor = Voronoi(vor_points)
        self.xcoords = np.arange(bounds[0][0], bounds[0][1], min_steps[0])
        self.ycoords =  np.arange(bounds[1][0], bounds[1][1], min_steps[1])
        self.xmin, self.xmax, self.ymin, self.ymax = self.get_boundaries()
        self.xdelta = min_steps[0]
        self.ydelta = min_steps[1]
        self.plot()

    def get_label(self, x, y):
        # if not(#(np.abs(x-1.5)**2 + np.abs(y+1.5)**2 < 1) or\
        #     (x>0.1 and x<0.3 and y>0.2 and y<0.8) or\
        #     ((x-0.6)**2 + (y-0.6)**2 < 0.15**2)):
        #     return len(self.vor.points)
        label = np.argmin(np.sum((self.vor.points - np.array([x,y]))**2, axis=1))
        # print(f"{x}, {y}: {label}")
        return label

    def measure(self, x, y):
        # if np.abs(x) > 2 or np.abs(y) > 2:
        
        # if not(#(np.abs(x-1.5)**2 + np.abs(y+1.5)**2 < 1) or\
        #         (x>0.1 and x<0.3 and y>0.2 and y<0.8) or\
        #         ((x-0.6)**2 + (y-0.6)**2 < 0.15**2)):
        #     return x, y, np.random.exponential(0.001,(self.num_energies, self.num_angles))

        dx_sq = (self.vor.points[:,0] - x)**2
        dy_sq = (self.vor.points[:,1] - y)**2
        idx = np.argmin(dx_sq + dy_sq)
        azimuth = self.vor_azimuths[idx]
        tilt = self.vor_tilts[idx]
        polar = self.vor_polars[idx]
        intensity = self.vor_intensities[idx]

        measurement = simulate_ARPES_measurement(
                        polar=polar, tilt=tilt, azimuthal=azimuth,
                        num_angles=self.num_angles, num_energies=self.num_energies,
                        k_resolution=0.005, e_resolution=0.01)
        return x, y, measurement[0]*intensity

    def get_boundaries(self):
        return np.min(self.xcoords), np.max(self.xcoords), np.min(self.ycoords), np.max(self.ycoords)
    
    def plot_grid(self, shape):
        print(f"plotting grid: {shape}")
        xs = np.linspace(self.bounds[0][0], self.bounds[0][1], shape[0])
        ys = np.linspace(self.bounds[1][0], self.bounds[1][1], shape[1])
        xs, ys = np.meshgrid(xs, ys)
        labels = np.zeros(xs.shape,dtype=np.int32) * np.nan
        for i, (x, y) in enumerate(zip(xs.flatten(), ys.flatten())):
            i = np.unravel_index(i, xs.shape)
            label = self.get_label(x,y)
            labels[i] = label
            
        # print(labels)
        plt.imshow(labels, extent=np.ravel(self.bounds), origin='lower', cmap='terrain')
        plt.xlim(*self.bounds[0])
        plt.ylim(*self.bounds[1])
        plt.colorbar()
        # plt.grid(which='both')
        plt.scatter(*self.vor.points.T, c='r', marker='x')
        plt.savefig(f"test_{shape[0]}x{shape[1]}.png")
        plt.clf()

    def plot(self):
        print("Plotting crystal")
        xs, ys = np.meshgrid(self.xcoords, self.ycoords)
        labels = np.zeros(xs.shape,dtype=np.int32) * np.nan
        for i, (x, y) in enumerate(zip(xs.flatten(), ys.flatten())):
            i = np.unravel_index(i, xs.shape)
            label = self.get_label(x,y)
            labels[i] = label
            
        # print(labels)
        plt.imshow(labels, extent=np.ravel(self.bounds), origin='lower', cmap='terrain')
        plt.xlim(*self.bounds[0])
        plt.ylim(*self.bounds[1])
        plt.colorbar()
        # plt.grid(which='both')
        plt.scatter(*self.vor.points.T, c='r', marker='x')
        plt.savefig("test.png")
        plt.clf()
        self.plot_grid((10,10))
        self.plot_grid((20,20))
        self.plot_grid((50,50))
        self.plot_grid((100,100))


class FakeGrapheneCrystal:
    def __init__(self):
        import h5py
        self.filename =  r"/mnt/MAESTROdata/nARPES/2019/2019_09/Rotenberg_Eli - 275451/Robinson/Twisted/20190915_01325.h5"
        self.file = h5py.File(self.filename, 'r')
        print("loading data...")
        self.data = self.file['2D_Data']['Fixed_Spectra1']
        print("loading coords...")
        self.xcoords = self.file['0D_Data']['Scan X'][:]
        self.ycoords = self.file['0D_Data']['Scan Y'][:]
        self.xmin, self.xmax, self.ymin, self.ymax = self.get_boundaries()
        print(f"{len(self.xcoords)=}")
        self.xdelta = (self.xmax - self.xmin)/91
        self.ydelta = (self.ymax - self.ymin)/91


    def measure(self, x, y):
        print(f"Measuring at {x}, {y}")
        dx = (self.xcoords - x)**2
        dy = (self.ycoords - y)**2
        d = np.sqrt(dx + dy)
        i = np.argmin(d)
        measured_x = self.xcoords[i]
        measured_y = self.ycoords[i]
        spectrum = self.data[:,:,i]
        return measured_x, measured_y, spectrum.reshape((spectrum.shape[1], spectrum.shape[0]))

    def get_boundaries(self):
        return np.min(self.xcoords), np.max(self.xcoords), np.min(self.ycoords), np.max(self.ycoords)

        
class FakeWSe2Crystal:
    def __init__(self):
        from astropy.io import fits
        self.filename =  r"/mnt/MAESTROdata/nARPES/2016/201612/WS2_hBN_TiO2_sample26/20161215_00045.fits"
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
    #  vor = FakeVoronoiCrystal(num_crystallites=50, num_angles=128, num_energies=128, bounds=[[-10,10], [-10,10]], min_steps=[0.05, 0.05])
    #  vor.plot()
    start = perf_counter()
    wse2 = FakeGrapheneCrystal()
    end = perf_counter()
    print(f"Time to load: {(end-start):.3f} s")
    xmin, xmax, ymin, ymax = wse2.get_boundaries()
    print(wse2.xmin, wse2.xmax, wse2.ymin, wse2.ymax, wse2.xdelta, wse2.ydelta)
    start = perf_counter()
    x = np.random.uniform(xmin, xmax)
    y = np.random.uniform(ymin, ymax)
    _, _, spectrum = wse2.measure(x, y)
    end = perf_counter()
    print(f"Time per measurement: {(end-start):.3f} s")
    print(f"Spectrum shape: {spectrum.shape}")
    plt.imshow(spectrum.reshape(spectrum.shape[::-1]), origin='lower', cmap='Greys')
    plt.savefig("gr.png")

#     for _ in range(10):
#         x_choice, y_choice = np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax)
#         measured_x, measured_y, spectrum = wse2.measure(x_choice, y_choice)
#         print(f"Measuring at {x_choice}, {y_choice}")
#         print(f"Measured at {measured_x}, {measured_y}")
#         print(f"Spectrum Sum: {np.sum(spectrum)}")


if __name__ == "__main__":
    main()