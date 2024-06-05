import numpy as np
# from skimage.transform import resize
from scipy.interpolate import griddata
import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize, basinhopping

from . import tasks
from .celery_app import app
import matplotlib.pyplot as plt

from typing import Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, TypedDict, Union
from numpy.typing import ArrayLike

from abc import ABC, abstractmethod

import time
import pickle

# This is an adaptation of the Agent base class from bluesky-adaptive that works for my use

"""Abstract base class for a single plan agent. These agents should consume data, decide where to measure next,
    and execute a single type of plan (something akin to move and count).
    Alternatively, these agents can be used for soley reporting.

    Base agent sets up a kafka subscription to listen to new stop documents, a catalog to read for experiments,
    a catalog to write agent status to, a kafka publisher to write agent documents to,
    and a manager API for the queue-server. Each time a stop document is read,
    the respective BlueskyRun is unpacked by the ``unpack_run`` method into an independent and dependent variable,
    and told to the agent by the ``tell`` method.

    Children of Agent should implment the following, through direct inheritence or mixin classes:
    Experiment specific:
    - measurement_plan
    - unpack_run
    Agent specific:
    - tell
    - ask
    - report (optional)
    - name (optional)
    """
class Agent(ABC):
    # @abstractmethod
    # def measurement_plan(self, point: ArrayLike) -> Tuple[str, List, dict]:
    #     """Fetch the string name of a registered plan, as well as the positional and keyword
    #     arguments to pass that plan.

    #     Args/Kwargs is a common place to transform relative into absolute motor coords, or
    #     other device specific parameters.

    #     Parameters
    #     ----------
    #     point : ArrayLike
    #         Next point to measure using a given plan

    #     Returns
    #     -------
    #     plan_name : str
    #     plan_args : List
    #         List of arguments to pass to plan from a point to measure.
    #     plan_kwargs : dict
    #         Dictionary of keyword arguments to pass the plan, from a point to measure.
    #     """
    #     ...

    # @staticmethod
    # @abstractmethod
    # def unpack_run(run) -> Tuple[Union[float, ArrayLike], Union[float, ArrayLike]]:
    #     """
    #     Consume a Bluesky run from tiled and emit the relevant x and y for the agent.

    #     Parameters
    #     ----------
    #     run : BlueskyRun

    #     Returns
    #     -------
    #     independent_var :
    #         The independent variable of the measurement
    #     dependent_var :
    #         The measured data, processed for relevance
    #     """
    #     ...

    @abstractmethod
    def tell(self, x, y) -> Dict[str, ArrayLike]:
        """
        Tell the agent about some new data
        Parameters
        ----------
        x :
            Independent variable for data observed
        y :
            Dependent variable for data observed

        Returns
        -------
        dict
            Dictionary to be unpacked or added to a document

        """
        ...

    @abstractmethod
    def tell_many(self, x, y) -> Dict[str, ArrayLike]:
        """
        Tell the agent about some new data
        Parameters
        ----------
        x :
            Independent variable for data observed
        y :
            Dependent variable for data observed

        Returns
        -------
        dict
            Dictionary to be unpacked or added to a document

        """
        ...

    @abstractmethod
    def ask(self, batch_size: int) -> Tuple[Sequence[Dict[str, ArrayLike]], Sequence[ArrayLike]]:
        """
        Ask the agent for a new batch of points to measure.

        Parameters
        ----------
        batch_size : int
            Number of new points to measure

        Returns
        -------
        docs : Sequence[dict]
            Documents of key metadata from the ask approach for each point in next_points.
            Must be length of batch size.
        next_points : Sequence
            Sequence of independent variables of length batch size
        """
        ...

    def report(self, **kwargs) -> Dict[str, ArrayLike]:
        """
        Create a report given the data observed by the agent.
        This could be potentially implemented in the base class to write document stream.
        Additional functionality for converting the report dict into an image or formatted report is
        the duty of the child class.
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Short string name"""
        return "agent"


class IntensityAgent(Agent):
    """A simple naive agent that cycles samples sequentially in environment space"""

    def tell(self, x, y):
        return x, np.sum(y, axis=1)
    
    def ask(self, batch_size):
        raise NotImplementedError
    
    def tell_many(self, x, y):
        raise NotImplementedError
    
from sklearn.decomposition import KernelPCA
class KPCAAgent(Agent):
    """A simple naive agent that cycles samples sequentially in environment space"""

    def __init__(self, **kwargs):
        print("Creating PCAAgent")
        self.kwargs = kwargs
        if kwargs.get('n_components') is None:
            raise ValueError("Please enter a non-None value for n_components of PCA.")
        kwargs['kernel'] = 'rbf' if kwargs.get('kernel') is None else kwargs.get('kernel')

    def tell(self, x, y):
        print(f"PCAAgent told about new data: {x.shape=}, {y.shape=}")
        if len(y) < self.kwargs.get('n_components'):
            self.PCA = KernelPCA(n_components=None, **{k:v for k,v in self.kwargs.items() if k!='n_components'})
            return x, self.PCA.fit_transform(y)
        
        self.PCA = KernelPCA(**self.kwargs)
        return x, self.PCA.fit_transform(y)
    
    def ask(self, batch_size):
        raise NotImplementedError
    
    def tell_many(self, x, y):
        raise NotImplementedError
    

# from sklearn.decomposition import PCA
from cuml import IncrementalPCA as PCA
class PCAAgent(Agent):
    """A simple naive agent that cycles samples sequentially in environment space"""

    def __init__(self, **kwargs):
        print("Creating PCAAgent")
        self.kwargs = kwargs
        self.n_components = kwargs.get('n_components') if kwargs.get('n_components') is not None else 50
        # if kwargs.get('n_components') is None:
        #     raise ValueError("Please enter a non-None value for n_components of PCA.")
        

    def tell(self, x, y):
        print(f"PCAAgent told about new data: {x.shape=}, {y.shape=}")
        start = time.perf_counter_ns()
        # if len(y) < self.kwargs.get('n_components'):
        #     self.PCA = PCA(n_components=None, **{k:v for k,v in self.kwargs.items() if k!='n_components'})
        #     return x, self.PCA.fit_transform(y)
        
        self.PCA = PCA(n_components=min(self.n_components, len(y)))
        result = self.PCA.fit_transform(y.astype(np.float32))
        end = time.perf_counter_ns()
        print(f"PCA took {(end-start)/1e6:.02f}ms to fit.")
        return x, result
    
    def ask(self, batch_size):
        raise NotImplementedError
    
    def tell_many(self, x, y):
        raise NotImplementedError

from cuml import UMAP
class UMAPAgent(Agent):
    def __init__(self, **kwargs):
        print("Creating UMAPAgent")
        self.kwargs = kwargs
        self.n_components = kwargs.get('n_components') if kwargs.get('n_components') is not None else 2

    def tell(self, x, y):
        print(f"UMAPAgent told about new data: {x.shape=}, {y.shape=}")
        start = time.perf_counter_ns()
        self.UMAP = UMAP(n_components=self.n_components)
        new_y = self.UMAP.fit_transform(y.astype(np.float32))
        end = time.perf_counter_ns()
        print(f"UMAP took {(end-start)/1e6:.02f}ms to fit.")
        return x, new_y
    
    def ask(self, batch_size):
        raise NotImplementedError
    
    def tell_many(self, x, y):
        raise NotImplementedError

from cuml.cluster.hdbscan import HDBSCAN
class HDBSCANAgent(Agent):
    def __init__(self, *args, **kwargs):
        print("Creating HDBSCANAgent")
        self.args = args
        self.kwargs = kwargs

    def tell(self, x, y):
        print(f"HDBSCANAgent told about new data: {x.shape=}, {y.shape=}")
        start = time.perf_counter_ns()
        if len(y) < 2:
            self.labels = np.zeros(len(y))
            new_y = np.zeros(len(y))
        else:
            self.HDBSCAN = HDBSCAN(**self.kwargs)
            self.labels = self.HDBSCAN.fit(y.astype(np.float32))
            new_y = self.HDBSCAN.labels_ + 1
            # print(new_y)
        end = time.perf_counter_ns()
        print(f"HDBSCAN took {(end-start)/1e6:.02f}ms to fit.")
        return x, new_y
    
    def ask(self, batch_size):
        raise NotImplementedError
    
    def tell_many(self, x, y):
        raise NotImplementedError


# from sklearn.cluster import KMeans
from cuml.cluster import KMeans
class KMeansAgent(Agent):
    """A simple naive agent that cycles samples sequentially in environment space"""

    def __init__(self, *args, **kwargs):
        print("Creating KMeansAgent")
        self.args = args
        self.kwargs = kwargs
        self.n_clusters = kwargs['n_clusters']
        self.experiment_id = kwargs.get('experiment_id')
        self.KMeans = None

    def tell(self, x, y):
        print(f"KMeansAgent told about new data: {x.shape=}, {y.shape=}")
        start = time.perf_counter_ns()
        if len(y) < self.n_clusters:
            self.labels = np.zeros(len(y))
            new_y = np.zeros(len(y))
        else:
            self.KMeans = KMeans(n_clusters=self.n_clusters, n_init=20)#, max_iter=300, n_init=10)
            self.KMeans.fit(y.astype(np.float32))
            self.labels = self.KMeans.labels_
            new_y = self.labels
        self.x = x
        self.y = new_y
        end = time.perf_counter_ns()
        print(f"KMeans took {(end-start)/1e6:.02f}ms to fit.")
        return x, new_y
    
    def ask(self, batch_size):
        raise NotImplementedError
    
    def tell_many(self, x, y):
        raise NotImplementedError
    
    def report(self, meshgrid):
        print("Reporting KMeansAgent")
        if self.KMeans is None:
            print("No KMeans model to report")
            return
        
        label_grid = griddata(self.x, self.y, (meshgrid[0], meshgrid[1]), method='nearest')
        # tasks.image_report(experiment_id=self.experiment_id, matrix=label_grid, name='kmeans_labels', extra_data = dict(n_measured=self.n_clusters))
        tasks.image_report(experiment_id=self.experiment_id, 
                            matrix=label_grid, name='kmeans_labels',
                            extra_data = dict(n_measured=len(self.y)),
                            )


from torch import nn
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=1000, latent_dim=3, device='cuda'):
        super(VAE, self).__init__()
        self.device = device
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
            )
        
        # latent mean and variance 
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)

        self.mean_var_decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
            )
     
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)      
        z = mean + var*epsilon
        return z

    def decode(self, x):
        x = self.mean_var_decoder(x)
        return self.decoder(x)

    def forward(self, x):
        # print(f"VAE forward: {x.shape=}")
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar


def vae_loss(x, x_hat, mean, log_var):
    # reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction='mean')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD


class DKLModel(gpytorch.models.ExactGP):
        
        def __init__(self, train_x, train_y, likelihood):
            super(DKLModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.MultitaskMean(
                gpytorch.means.ConstantMean(), num_tasks=train_y.shape[-1]
            )
            self.covar_module = gpytorch.kernels.MultitaskKernel(
                gpytorch.kernels.RBFKernel(), num_tasks=train_y.shape[-1], rank=1
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class DKLAgent_old(Agent):
    def __init__(self, input_bounds, input_min_spacings, experiment_id):
        print("Creating Classification GP Agent")
        self.input_bounds = input_bounds
        self.input_min_spacings = input_min_spacings
        self.inputs=None
        self.targets=None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("THE DEVICE BEING USED IS: ", self.device)
        p = [np.arange(low, high, delta) for (low, high), delta in zip(self.input_bounds, self.input_min_spacings)]
        # p = [np.linspace(low, high, 128) for (low, high) in self.input_bounds]
        self.meshgrid_points = np.meshgrid(*p)
        self.plotting_positions = torch.stack([torch.tensor(arr.flatten(),dtype=torch.float32).to(self.device) for arr in self.meshgrid_points],dim=1)
        p = [np.linspace(low, high, 4) for (low, high) in self.input_bounds]
        self.scipy_meshgrid = np.meshgrid(*p)
        self.optimize_positions = torch.stack([torch.tensor(arr.flatten(),dtype=torch.float32).to(self.device) for arr in self.scipy_meshgrid],dim=1)
        self.experiment_id = experiment_id

    def tell(self, x, y):
        print(f"GPytorchAgent told about new data: {x.shape=}, {y.shape=}")
        if True:
            self.inputs = torch.atleast_2d(torch.tensor(x, device=self.device, dtype=torch.float32))
            self.targets = torch.atleast_1d(torch.tensor(y, device=self.device, dtype=torch.float32))
        start = time.perf_counter_ns()
        self.fit()
        end = time.perf_counter_ns()
        print(f"GP took {(end-start)/1e6:.02f}ms to fit.")
        return x, y

    def fit(self):# model, likelihood, train_x, train_y):
        train_x = self.inputs
        train_y = self.targets
        # normalize train_y to be -1 to 1
        train_y = (train_y - train_y.mean()) / train_y.std()
        training_iterations = 50

        print("Fitting VAE")
        vae = VAE(input_dim=train_y.shape[1], hidden_dim=1000, latent_dim=3, device=self.device).to(self.device)
        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-8)
        vae.train()
        for i in range(training_iterations):
            optimizer.zero_grad()
            # print(f"train_y.shape={train_y.shape}, train_y.dtype={train_y.dtype}")
            x_hat, mean, logvar = vae(train_y)
            # print(f"{x_hat.shape=}, {mean.shape=}, {logvar.shape=}")
            # print(f"{x_hat.mean()}")
            # print(f"{mean.mean()}")
            # print(f"{logvar.mean()}")
            loss = vae_loss(train_y, x_hat, mean, logvar)
            loss.backward()
            optimizer.step()
            # print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        vae.eval()
        self.vae = vae
        optimizer = None

        print("VAE fitted")
    
        print("Fitting GP")
        encoded_y = self.vae.encode(train_y)[0].detach().clone()
        print(f"{encoded_y.shape=}")

        # likelihood = DirichletClassificationLikelihood(train_y, learn_additional_noise=False)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=encoded_y.shape[-1])
        model = DKLModel(train_x, encoded_y, likelihood)

        model.to(self.device)
        likelihood.to(self.device)

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        
        
        print("Training GP")
        for i in range(training_iterations):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, encoded_y)

            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
            optimizer.step()

        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():# , gpytorch.settings.max_root_decomposition_size(100):
            self.model = model
            self.likelihood = likelihood

    # def find_next_position()

    def ask(self, batch_size):
        print("Sampling...")
        max_n_samples = 10_000
        if len(self.plotting_positions) > max_n_samples:
            test_x = self.plotting_positions[np.random.choice(len(self.plotting_positions), max_n_samples)]
        else:
            test_x = self.plotting_positions
        
        print(f"Evaluating model on {len(test_x)} positons...")
        evaluation = self.model(test_x)
        # evaluation.sample(torch.Size((256,)))
        test_x = test_x.detach().cpu().numpy()
        # preds = evaluation.mean.detach().cpu().numpy()
        # preds = self.vae.decode(test_x).detach().cpu().numpy()
        # stds = evaluation.stddev.sum(0).detach().cpu().numpy()
        # print(f'{preds.shape=}')
        means_and_variances = evaluation.mean.detach()#.cpu().numpy()
        means = self.vae.mean_var_decoder(means_and_variances).detach().cpu().numpy()
        print(f'{means.shape=}')
        print(means)
        stds = evaluation.stddev[:,0].detach().cpu().numpy()
        # stds_all = self.likelihood(evaluation).loc.detach().cpu().numpy()
        # stds = stds_all.prod(axis=0)
        # pred_samples = evaluation.sample(torch.Size((256,))).exp()
        # info = (pred_samples / pred_samples.sum(-2, keepdim=True))
        # probabilities = info.mean(0).detach().cpu().numpy()
        # stds = info.std(0)
        # stds /= stds.sum(len(stds.shape[1:]), keepdim=True)
        # # stds = stds.pow(2)
        # # # The stds here actually represent the uncertainty on the proabbility of each label.
        # stds = stds.sum(0).detach().cpu().numpy()
        # stds = stds.sum(0).detach().cpu().numpy()
        # partial = tasks.plot_griddata.s(xs=test_x, ys=np.copy(stds))
        grid_stds = griddata(test_x, stds, (self.meshgrid_points[0], self.meshgrid_points[1]), method='nearest')
        tasks.image_report(experiment_id=self.experiment_id, 
                            matrix=grid_stds, name='uncertainties',
                            extra_data = dict(n_measured=len(self.inputs)),
                            )
        grid_0 = griddata(test_x, means[:,0], (self.meshgrid_points[0], self.meshgrid_points[1]), method='nearest')
        tasks.image_report(experiment_id=self.experiment_id, 
                            matrix=grid_0, name='vae_0',
                            extra_data = dict(n_measured=len(self.inputs)),
                            )
        grid_1 = griddata(test_x, means[:,1], (self.meshgrid_points[0], self.meshgrid_points[1]), method='nearest')
        tasks.image_report(experiment_id=self.experiment_id, 
                            matrix=grid_1, name='vae_1',
                            extra_data = dict(n_measured=len(self.inputs)),
                            )
        grid_2 = griddata(test_x, means[:,2], (self.meshgrid_points[0], self.meshgrid_points[1]), method='nearest')
        tasks.image_report(experiment_id=self.experiment_id, 
                            matrix=grid_2, name='vae_2',
                            extra_data = dict(n_measured=len(self.inputs)),
                            )
        # plt.imshow(stds.reshape(self.meshgrid_points[0].shape), cmap='terrain', origin='lower')
        
        # plt.scatter(*local_maxima.T, marker='o', color='k')
        # plt.imshow(grid_stds, cmap='terrain', origin='lower', extent=np.ravel(self.input_bounds))
        # cb = plt.colorbar() 

        # # stds_all = stds_all.detach().cpu().numpy()
        # # probability = stds
        # # means = evaluation.mean.detach().cpu().numpy()
        # # stds = stds**3
        choice_indices = np.argsort(stds.flatten())
        num_most_uncertain_to_measure_first = batch_size
        most_uncertain_indices = choice_indices[-num_most_uncertain_to_measure_first:][::-1]
        p = stds[:-num_most_uncertain_to_measure_first]
        if p.sum()!=0:
            p /= p.sum()
            probabalistic_indices = np.random.choice(choice_indices[:-num_most_uncertain_to_measure_first], 
                                                size=batch_size-num_most_uncertain_to_measure_first,
                                                replace=False,
                                                p=p)
            selected_indices =  np.concatenate([most_uncertain_indices, probabalistic_indices])
        else:
            selected_indices = choice_indices[-batch_size:][::-1]
        # most_uncertain_indices = np.argsort(stds.flatten())
        # next_positions = list(self.all_possible_positions[most_uncertain_indices[-batch_size:]].detach().cpu().numpy())
        
        next_positions = test_x[selected_indices]
        # # plt.scatter(*next_positions.T, marker='x', color='r')
        # partial.apply_async(args=[], kwargs=dict(grid_points=self.meshgrid_points,\
        #                     scatter_xs=next_positions.T[0],\
        #                     scatter_ys=next_positions.T[1],\
        #                     bounds=self.input_bounds,\
        #                     filename='stds.png'))
        # # plt.savefig(f'stds.png')
        # # plt.clf()
        # print("Next positions selected...")
        # print("Plotting")
        # grid_inputs = griddata(self.inputs.cpu(), self.targets.cpu(), (self.meshgrid_points[0], self.meshgrid_points[1]), method='nearest')
        # # plt.imshow(evaluation.loc.max(0)[1].reshape(self.meshgrid_points[0].shape), cmap='terrain', origin='lower')
        # plt.imshow(grid_inputs, cmap='terrain', origin='lower')
        # cb = plt.colorbar() 
        # plt.savefig(f'clustering_outputs.png')
        # cb.remove()
        # grid_evaluations = griddata(test_x, evaluation.loc.max(0)[1].cpu(), (self.meshgrid_points[0], self.meshgrid_points[1]), method='nearest')
        # tasks.image_report(experiment_id=self.experiment_id,
        #                    matrix=grid_evaluations, name='predictions',
        #                    extra_data = dict(n_measured=len(self.inputs))
        #                 )
        # # plt.imshow(evaluation.loc.max(0)[1].reshape(self.meshgrid_points[0].shape), cmap='terrain', origin='lower')
        # plt.scatter(*self.inputs.cpu().numpy().T, marker='.', s=1, c='r')
        # plt.imshow(grid_evaluations, cmap='terrain', origin='lower', extent=np.ravel(self.input_bounds))
        # cb = plt.colorbar() 
        # plt.savefig(f'predictions.png')
        # plt.clf()
        return next_positions


    
    def tell_many(self, x, y):
        raise NotImplementedError
    

class DKLAgent(Agent):
    def __init__(self, input_bounds, input_min_spacings, experiment_id, data_ids):
        print("Creating Classification GP Agent")
        self.intensity_factor = 1
        self.umap_factor = 1
        self.data_ids = data_ids
        self.input_bounds = input_bounds
        self.input_min_spacings = input_min_spacings
        self.inputs=None
        self.targets=None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("THE DEVICE BEING USED IS: ", self.device)
        p = [np.arange(low, high, delta) for (low, high), delta in zip(self.input_bounds, self.input_min_spacings)]
        # p = [np.linspace(low, high, 128) for (low, high) in self.input_bounds]
        self.meshgrid_points = np.meshgrid(*p)
        self.plotting_positions = torch.stack([torch.tensor(arr.flatten(),dtype=torch.float32).to(self.device) for arr in self.meshgrid_points],dim=1)
        p = [np.linspace(low, high, 4) for (low, high) in self.input_bounds]
        self.scipy_meshgrid = np.meshgrid(*p)
        self.optimize_positions = torch.stack([torch.tensor(arr.flatten(),dtype=torch.float32).to(self.device) for arr in self.scipy_meshgrid],dim=1)
        self.experiment_id = experiment_id

    def tell(self, x, y):
        print(f"GPytorchAgent told about new data: {x.shape=}, {y.shape=}")
        if True:
            self.inputs = torch.atleast_2d(torch.tensor(x, device=self.device, dtype=torch.float32))
            self.targets = torch.atleast_1d(torch.tensor(y, device=self.device, dtype=torch.float32))
            self.intensities = self.targets.sum(axis=1)
            self.intensities = self.intensities / self.intensities.max()
        start = time.perf_counter_ns()
        self.fit()
        end = time.perf_counter_ns()
        print(f"GP took {(end-start)/1e6:.02f}ms to fit.")
        return x, y

    def fit(self):# model, likelihood, train_x, train_y):
        train_x = self.inputs
        train_y = self.targets
        # train_y = (train_y - train_y.mean()) / train_y.std()
        training_iterations = 100

        print("Fitting UMAP")
        umap_model = UMAP(n_components=3)
        encoded_y = umap_model.fit(train_y).transform(train_y)
        encoded_y = torch.as_tensor(encoded_y, device=self.device)#, dtype=torch.float32)
        print(f"UMAP fitted: {type(encoded_y)}")
        tasks.save_report(experiment_id=self.experiment_id, name='umap_coords', 
                          data={
                                'umap_coords': encoded_y.cpu().numpy().tolist(),
                                'xy_coords': self.inputs.cpu().numpy().tolist(),
                                'n_measured': len(self.inputs),
                                'data_ids': self.data_ids,
                            })
        # scaled_encoded_y = encoded_y.cpu().numpy()
        # scaled_encoded_y = (255 * (scaled_encoded_y - scaled_encoded_y.min(axis=0)[np.newaxis, ...]) / (scaled_encoded_y.max(axis=0)[np.newaxis, ...] - scaled_encoded_y.min(axis=0)[np.newaxis, ...])).astype(np.uint8)
        # app.send_task("celery_workers.tasks.image_report", args=(
        #                     self.experiment_id,
        #                     'UMAP as RGB',
        #                     self.inputs.cpu().numpy(),
        #                     encoded_y.cpu().numpy(),
        #                     (self.meshgrid_points[0], self.meshgrid_points[1]),
        #                     "linear",
        #                     dict(n_measured=len(self.inputs)),
        #                     ))
        
    
        print("Fitting GP")
        # encoded_y = self.vae.encode(train_y)[0].detach().clone()
        # print(f"{encoded_y.shape=}")

        # likelihood = DirichletClassificationLikelihood(train_y, learn_additional_noise=False)
        gp_targets = torch.cat([encoded_y, self.intensities.unsqueeze(-1)], dim=-1)
        print(f"{encoded_y.shape=}, {self.intensities.shape=}, {gp_targets.shape=}")
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=encoded_y.shape[-1] + 1)
        
        # gp_targets = encoded_y
        model = DKLModel(train_x, gp_targets, likelihood)
        # print(model)
        model.to(self.device)
        likelihood.to(self.device)

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        
        
        print("Training GP")
        for i in range(training_iterations):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, gp_targets)

            loss.backward()
            # print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
            optimizer.step()

        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():# , gpytorch.settings.max_root_decomposition_size(100):
            self.model = model
            self.likelihood = likelihood

    # def find_next_position()

    def ask(self, batch_size):
        print("Sampling...")
        max_n_samples = 10_000
        if len(self.plotting_positions) >= max_n_samples:
            test_x = self.plotting_positions[np.random.choice(len(self.plotting_positions), max_n_samples)]
        else:
            test_x = self.plotting_positions
        
        print(f"Evaluating model on {len(test_x)} positons...")
        evaluation = self.likelihood(self.model(test_x))
        # evaluation.sample(torch.Size((256,)))
        test_x = test_x.detach().cpu().numpy()
        # preds = evaluation.mean.detach().cpu().numpy()
        # preds = self.vae.decode(test_x).detach().cpu().numpy()
        # stds = evaluation.stddev.sum(0).detach().cpu().numpy()
        # print(f'{preds.shape=}')
        # means_and_variances = evaluation.mean.detach()#.cpu().numpy()
        # means = self.vae.mean_var_decoder(means_and_variances).detach().cpu().numpy()
        means = evaluation.mean.detach().cpu().numpy()
        intensity_mean = means[:,-1]
        means = means[:,:-1]
        print(f'{means.shape=}')
        # print(means)
        stds = evaluation.stddev
        intensity_std = stds[:,-1].detach().cpu().numpy()
        stds -= stds.min(axis=0, keepdim=True).values
        stds /= stds.max(axis=0, keepdim=True).values
        stds = stds.sum(axis=1).detach().cpu().numpy() / 4

        # stds = stds * self.umap_factor + self.intensity_factor * np.clip(intensity_mean,0,1)
        stds = stds * np.clip(intensity_mean,0,1)
        # stds = stds/std_std
        # stds = 

        # stds = evaluation.stddev.sum(axis=1).detach().cpu().numpy()
        # stds_all = self.likelihood(evaluation).loc.detach().cpu().numpy()
        # stds = stds_all.prod(axis=0)
        # pred_samples = evaluation.sample(torch.Size((256,))).exp()
        # info = (pred_samples / pred_samples.sum(-2, keepdim=True))
        # probabilities = info.mean(0).detach().cpu().numpy()
        # stds = info.std(0)
        # stds /= stds.sum(len(stds.shape[1:]), keepdim=True)
        # # stds = stds.pow(2)
        # # # The stds here actually represent the uncertainty on the proabbility of each label.
        # stds = stds.sum(0).detach().cpu().numpy()
        # stds = stds.sum(0).detach().cpu().numpy()
        # partial = tasks.plot_griddata.s(xs=test_x, ys=np.copy(stds))
        xi = (self.meshgrid_points[0], self.meshgrid_points[1])
        method = 'nearest'
        # grid_stds = griddata(test_x, stds, (self.meshgrid_points[0], self.meshgrid_points[1]), method='linear')
        app.send_task("celery_workers.tasks.image_report", args=(
                            self.experiment_id,
                            'Acquisition function',
                            test_x,
                            stds,
                            xi,
                            method,
                            dict(n_measured=len(self.inputs)),
                            ))
        app.send_task("celery_workers.tasks.image_report", args=(
                            self.experiment_id,
                            'GP Intensities',
                            test_x,
                            intensity_mean,
                            xi,
                            method,
                            dict(n_measured=len(self.inputs)),
                            ))
        app.send_task("celery_workers.tasks.image_report", args=(
                            self.experiment_id,
                            'GP Intensity Uncertainty',
                            test_x,
                            intensity_std,
                            xi,
                            method,
                            dict(n_measured=len(self.inputs)),
                            ))
        # grid_0 = griddata(test_x, means[:,0], (self.meshgrid_points[0], self.meshgrid_points[1]), method='linear')
        app.send_task("celery_workers.tasks.image_report", args=(
                            self.experiment_id,
                            'UMAP x',
                            test_x,
                            means[:,0],
                            xi,
                            method,
                            dict(n_measured=len(self.inputs)),
                            ))
        # grid_1 = griddata(test_x, means[:,1], (self.meshgrid_points[0], self.meshgrid_points[1]), method='linear')
        app.send_task("celery_workers.tasks.image_report", args=(
                            self.experiment_id,
                            'UMAP y',
                            test_x,
                            means[:,1],
                            xi,
                            method,
                            dict(n_measured=len(self.inputs)),
                            ))
        # grid_2 = griddata(test_x, means[:,2], (self.meshgrid_points[0], self.meshgrid_points[1]), method='linear')
        app.send_task("celery_workers.tasks.image_report", args=(
                            self.experiment_id,
                            'UMAP z',
                            test_x,
                            means[:,2],
                            xi,
                            method,
                            dict(n_measured=len(self.inputs)),
                            ))
        
        app.send_task("celery_workers.tasks.image_report", args=(
                            self.experiment_id,
                            'UMAP interpolated coords',
                            test_x,
                            means,
                            xi,
                            method,
                            dict(n_measured=len(self.inputs)),
                            ))
        scaled_means = (255 * (means - means.min(axis=0)[np.newaxis, ...]) / (means.max(axis=0)[np.newaxis, ...] - means.min(axis=0)[np.newaxis, ...])).astype(np.uint8)
        # print(f"Scaled means shape={means.shape}")
        # grid_rgb = griddata(test_x, scaled_means, (self.meshgrid_points[0], self.meshgrid_points[1]), method='linear')
        # app.tasks.image_report(experiment_id=self.experiment_id,
        app.send_task("celery_workers.tasks.image_report", args=(
                            self.experiment_id,
                            'GP est. UMAP as RGB',
                            test_x,
                            scaled_means,
                            xi,
                            method,
                            dict(n_measured=len(self.inputs)),
                            ))
        # tasks.image_report(experiment_id=self.experiment_id, 
                            # name='rgb',
                            # points=test_x,
                            # values=scaled_means,
                            # xi=xi,
                            # method=method,
                            # extra_data = dict(n_measured=len(self.inputs)),
                            # )
        high_stds = stds
        # high_stds = np.clip(stds, np.percentile(stds, 90), None)
        # if len(high_stds) < batch_size:
        #     high_stds = np.clip(stds, np.percentile(stds, 50), None)
        # high_stds[high_stds == high_stds.min()] = 0
        # app.send_task("celery_workers.tasks.image_report", args=(
        #                     self.experiment_id,
        #                     'Clipped Uncertainties',
        #                     test_x,
        #                     high_stds,
        #                     xi,
        #                     method,
        #                     dict(n_measured=len(self.inputs)),
        #                     ))
        choice_indices = np.argsort(high_stds.flatten())

        num_most_uncertain_to_measure_first = 1
        most_uncertain_indices = choice_indices[-num_most_uncertain_to_measure_first:][::-1]
        p = high_stds[:-num_most_uncertain_to_measure_first]
        if p.sum()!=0:
            p /= p.sum()
            probabalistic_indices = np.random.choice(choice_indices[:-num_most_uncertain_to_measure_first], 
                                                size=batch_size-num_most_uncertain_to_measure_first,
                                                replace=False,
                                                p=p)
            selected_indices =  np.concatenate([most_uncertain_indices, probabalistic_indices])
        else:
            selected_indices = choice_indices[-batch_size:][::-1]
        next_positions = test_x[selected_indices]
        return next_positions


    
    def tell_many(self, x, y):
        raise NotImplementedError


class DirichletGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_classes):
        super(DirichletGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(batch_shape=torch.Size((num_classes,)))
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=torch.Size((num_classes,))),
            batch_shape=torch.Size((num_classes,)),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GPytorchAgent(Agent):
    def __init__(self, input_bounds, input_min_spacings, experiment_id):
        print("Creating Classification GP Agent")
        self.input_bounds = input_bounds
        self.input_min_spacings = input_min_spacings
        self.inputs=None
        self.targets=None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("THE DEVICE BEING USED IS: ", self.device)
        p = [np.arange(low, high, delta) for (low, high), delta in zip(self.input_bounds, self.input_min_spacings)]
        # p = [np.linspace(low, high, 128) for (low, high) in self.input_bounds]
        self.meshgrid_points = np.meshgrid(*p)
        self.plotting_positions = torch.stack([torch.tensor(arr.flatten(),dtype=torch.float32).to(self.device) for arr in self.meshgrid_points],dim=1)
        p = [np.linspace(low, high, 4) for (low, high) in self.input_bounds]
        self.scipy_meshgrid = np.meshgrid(*p)
        self.optimize_positions = torch.stack([torch.tensor(arr.flatten(),dtype=torch.float32).to(self.device) for arr in self.scipy_meshgrid],dim=1)
        self.experiment_id = experiment_id
        # print(f"Total number of possible position is ~{self.all_possible_positions.shape}")

    def tell(self, x, y):
        print(f"GPytorchAgent told about new data: {x.shape=}, {y.shape=}")
        # return np.ones(self.decicion_space_shape), None
        if True: #self.inputs is None:
            self.inputs = torch.atleast_2d(torch.tensor(x, device=self.device, dtype=torch.float32))
            self.targets = torch.atleast_1d(torch.tensor(y, device=self.device, dtype=torch.int32))
        # else:
        #     self.inputs = torch.cat([self.inputs, torch.atleast_2d(torch.tensor(x, device=self.device, dtype=torch.float32))], dim=0)
        #     self.targets = torch.cat([self.targets, torch.atleast_1d(torch.tensor(y, device=self.device, dtype=torch.int32))], dim=0)
        #     self.inputs.to(self.device)
        #     self.targets.to(self.device)
        start = time.perf_counter_ns()
        self.fit()
        end = time.perf_counter_ns()
        print(f"GP took {(end-start)/1e6:.02f}ms to fit.")
        return x, y
        # self.surrogate_model.set_train_data(self.inputs, self.targets, strict=False)
        # return dict(independent_variable=x, observable=y, cache_len=len(self.targets))

    def fit(self):# model, likelihood, train_x, train_y):
        print("Fitting GP")
        train_x = self.inputs
        train_y = self.targets

        # likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=train_y.shape[1])
        # model = MultitaskGPModel(train_x, train_y, likelihood)
        likelihood = DirichletClassificationLikelihood(train_y, learn_additional_noise=False)
        # likelihood = DirichletClassificationLikelihood(train_y, learn_additional_noise=True)
        model = DirichletGPModel(train_x, likelihood.transformed_targets, likelihood, num_classes=likelihood.num_classes)

        model.to(self.device)
        likelihood.to(self.device)

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        
        training_iterations = 50
        print("Training GP")
        for i in range(training_iterations):
        # iterator = tqdm.tqdm(range(training_iterations))
        # for i in iterator:
            optimizer.zero_grad()
            output = model(train_x)
            # loss = -mll(output, train_y)
            loss = -mll(output, likelihood.transformed_targets).sum()
            loss.backward()
            # print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
            optimizer.step()

        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():# , gpytorch.settings.max_root_decomposition_size(100):
            self.model = model
            self.likelihood = likelihood

    # def find_next_position()

    def ask(self, batch_size):
        max_n_samples = 10_000
        if len(self.plotting_positions) > max_n_samples:
            test_x = self.plotting_positions[np.random.choice(len(self.plotting_positions), max_n_samples)]
        else:
            test_x = self.plotting_positions
        
        print(f"Evaluating model on {len(test_x)} positons...")
        evaluation = self.model(test_x)
        test_x = test_x.detach().cpu().numpy()
        # stds_all = self.likelihood(evaluation).loc.detach().cpu().numpy()
        # stds = stds_all.prod(axis=0)
        print("Sampling...")
        pred_samples = evaluation.sample(torch.Size((256,))).exp()
        info = (pred_samples / pred_samples.sum(-2, keepdim=True))
        # probabilities = info.mean(0).detach().cpu().numpy()
        stds = info.std(0)
        # stds /= stds.sum(len(stds.shape[1:]), keepdim=True)
        # # stds = stds.pow(2)
        # # # The stds here actually represent the uncertainty on the proabbility of each label.
        # stds = stds.sum(0).detach().cpu().numpy()
        stds = stds.sum(0).detach().cpu().numpy()
        partial = tasks.plot_griddata.s(xs=test_x, ys=np.copy(stds))
        grid_stds = griddata(test_x, stds, (self.meshgrid_points[0], self.meshgrid_points[1]), method='nearest')
        tasks.image_report(experiment_id=self.experiment_id, 
                            matrix=grid_stds, name='uncertainties',
                            extra_data = dict(n_measured=len(self.inputs)),
                            )
        # plt.imshow(stds.reshape(self.meshgrid_points[0].shape), cmap='terrain', origin='lower')
        
        # plt.scatter(*local_maxima.T, marker='o', color='k')
        # plt.imshow(grid_stds, cmap='terrain', origin='lower', extent=np.ravel(self.input_bounds))
        # cb = plt.colorbar() 

        # # stds_all = stds_all.detach().cpu().numpy()
        # # probability = stds
        # # means = evaluation.mean.detach().cpu().numpy()
        # # stds = stds**3
        choice_indices = np.argsort(stds.flatten())
        num_most_uncertain_to_measure_first = 1
        most_uncertain_indices = choice_indices[-num_most_uncertain_to_measure_first:][::-1]
        p = stds[:-num_most_uncertain_to_measure_first]
        if p.sum()!=0:
            p /= p.sum()
            probabalistic_indices = np.random.choice(choice_indices[:-num_most_uncertain_to_measure_first], 
                                                size=batch_size-num_most_uncertain_to_measure_first,
                                                replace=False,
                                                p=p)
            selected_indices =  np.concatenate([most_uncertain_indices, probabalistic_indices])
        else:
            selected_indices = choice_indices[-batch_size:][::-1]
        # most_uncertain_indices = np.argsort(stds.flatten())
        # next_positions = list(self.all_possible_positions[most_uncertain_indices[-batch_size:]].detach().cpu().numpy())
        
        next_positions = test_x[selected_indices]
        # plt.scatter(*next_positions.T, marker='x', color='r')
        partial.apply_async(args=[], kwargs=dict(grid_points=self.meshgrid_points,\
                            scatter_xs=next_positions.T[0],\
                            scatter_ys=next_positions.T[1],\
                            bounds=self.input_bounds,\
                            filename='stds.png'))
        # plt.savefig(f'stds.png')
        # plt.clf()
        print("Next positions selected...")
        print("Plotting")
        grid_inputs = griddata(self.inputs.cpu(), self.targets.cpu(), (self.meshgrid_points[0], self.meshgrid_points[1]), method='nearest')
        # plt.imshow(evaluation.loc.max(0)[1].reshape(self.meshgrid_points[0].shape), cmap='terrain', origin='lower')
        plt.imshow(grid_inputs, cmap='terrain', origin='lower')
        cb = plt.colorbar() 
        plt.savefig(f'clustering_outputs.png')
        cb.remove()
        grid_evaluations = griddata(test_x, evaluation.loc.max(0)[1].cpu(), (self.meshgrid_points[0], self.meshgrid_points[1]), method='nearest')
        tasks.image_report(experiment_id=self.experiment_id,
                           matrix=grid_evaluations, name='predictions',
                           extra_data = dict(n_measured=len(self.inputs))
                        )
        # plt.imshow(evaluation.loc.max(0)[1].reshape(self.meshgrid_points[0].shape), cmap='terrain', origin='lower')
        plt.scatter(*self.inputs.cpu().numpy().T, marker='.', s=1, c='r')
        plt.imshow(grid_evaluations, cmap='terrain', origin='lower', extent=np.ravel(self.input_bounds))
        cb = plt.colorbar() 
        plt.savefig(f'predictions.png')
        plt.clf()
        # cb.remove()
        
        # cb.remove()
        # # for i, std in enumerate(stds_all):
        # #     interp_stds = griddata(test_x, std, (self.meshgrid_points[0], self.meshgrid_points[1]), method='nearest')
        # #     # plt.scatter(self.meshgrid_points[0], self.meshgrid_points[1], c=interp_stds, cmap='terrain')
        # #     plt.imshow(interp_stds, cmap='terrain', origin='lower')
        # #     cb = plt.colorbar()
        # #     # plt.colorbar()
        # #     plt.savefig(f'stds_{i:02d}.png')
        # #     cb.remove()
        return next_positions


    
    def tell_many(self, x, y):
        raise NotImplementedError

def minimize_func(x, model, device):
    test_x = torch.unsqueeze(torch.tensor(x, device=device, dtype=torch.float32), 0)
    evaluation = model(test_x)
    pred_samples = evaluation.sample(torch.Size((256,))).exp()
    info = (pred_samples / pred_samples.sum(-2, keepdim=True))
    stds = info.std(0).sum(0).detach().cpu().numpy()
    # print(stds_all)
    # stds_all /= stds_all.sum(len(stds_all.shape[1:]), keepdim=True)
    # stds_all = stds_all.pow(2)
    # stds = stds_all.sum(0).detach().cpu().numpy()
    # print(f"{x}: {-stds[0]}")
    return -stds[0]

# https://docs.gpytorch.ai/en/latest/examples/04_Variational_and_Approximate_GPs/Non_Gaussian_Likelihoods.html#Setting-up-the-classification-model
# Maybe I can use https://docs.gpytorch.ai/en/stable/examples/03_Multitask_Exact_GPs/ModelList_GP_Regression.html
    # as well? It can have one model for variational classification and another for regressions
    # Can use regression on dim. reduced data as a 'look for this' kind of spectra.

# First we should just try to get a variational/approximate GP working on the classifacation part
# This github comment has good info on the softmax likelihood function https://github.com/cornellius-gp/gpytorch/issues/1001

# from gpytorch.models import ApproximateGP
# from gpytorch.variational import CholeskyVariationalDistribution
# from gpytorch.variational import VariationalStrategy

# class GPClassificationModel(ApproximateGP):
#     def __init__(self, train_x):
#         variational_distribution = CholeskyVariationalDistribution(num_inducing_points=train_x.size(0))
#         variational_strategy = VariationalStrategy(
#             self, inducing_points=train_x, variational_distribution=variational_distribution, learn_inducing_locations=True
#         )
#         super(GPClassificationModel, self).__init__(variational_strategy)
#         self.mean_module = gpytorch.means.ConstantMean()
#         self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
#         return latent_pred

# class VariationalGPytorchAgent(Agent):
#     def __init__(self, input_bounds, input_min_spacings, experiment_id):
#         print("Creating GPytorchAgent")
#         self.input_bounds = input_bounds
#         self.input_min_spacings = input_min_spacings
#         self.inputs=None
#         self.targets=None
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         p = [np.arange(low, high, delta) for (low, high), delta in zip(self.input_bounds, self.input_min_spacings)]
#         self.meshgrid_points = np.meshgrid(*p)
#         self.all_possible_positions = torch.stack([torch.tensor(arr.flatten(),dtype=torch.float32).to(self.device) for arr in self.meshgrid_points],dim=1)
#         self.experiment_id = experiment_id
#         print(f"Total number of possible position is ~{self.all_possible_positions.shape}")

#     def tell(self, x, y):
#         print(f"GPytorchAgent told about new data: {x.shape=}, {y.shape=}")
#         # return np.ones(self.decicion_space_shape), None
#         if self.inputs is None:
#             self.inputs = torch.atleast_2d(torch.tensor(x, device=self.device, dtype=torch.float32))
#             self.targets = torch.atleast_1d(torch.tensor(y, device=self.device, dtype=torch.int32))
#         else:
#             self.inputs = torch.cat([self.inputs, torch.atleast_2d(torch.tensor(x, device=self.device, dtype=torch.float32))], dim=0)
#             self.targets = torch.cat([self.targets, torch.atleast_1d(torch.tensor(y, device=self.device, dtype=torch.int32))], dim=0)
#             self.inputs.to(self.device)
#             self.targets.to(self.device)
#         self.fit()
#         return x, y
#         # self.surrogate_model.set_train_data(self.inputs, self.targets, strict=False)
#         # return dict(independent_variable=x, observable=y, cache_len=len(self.targets))

#     def fit(self):# model, likelihood, train_x, train_y):
#         print("Fitting GP")
#         train_x = self.inputs
#         train_y = self.targets

#         # likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=train_y.shape[1])
#         # model = MultitaskGPModel(train_x, train_y, likelihood)
#         # likelihood = DirichletClassificationLikelihood(train_y, learn_additional_noise=True)
#         # likelihood = DirichletClassificationLikelihood(train_y, learn_additional_noise=True)
#         # model = DirichletGPModel(train_x, likelihood.transformed_targets, likelihood, num_classes=likelihood.num_classes)
#         model = GPClassificationModel(train_x)
#         # I cannot use Bernoulli likelihood because I have more than 2 classes. Softmax seems most appropriate
#         likelihood = gpytorch.likelihoods.SoftmaxLikelihood()

#         model.to(self.device)
#         likelihood.to(self.device)

#         # Find optimal model hyperparameters
#         model.train()
#         likelihood.train()

#         # Use the adam optimizer
#         optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

#         # "Loss" for GPs - the marginal log likelihood
#         # mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
#         mll = gpytorch.mlls.VariationalELBO(likelihood, model, train_y.numel())
        
#         training_iterations = 50
#         print("Training GP")
#         for i in range(training_iterations):
#         # iterator = tqdm.tqdm(range(training_iterations))
#         # for i in iterator:
#             optimizer.zero_grad()
#             output = model(train_x)
#             loss = -mll(output, train_y)
#             loss.backward()
#             print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
#             optimizer.step()

#         model.eval()
#         likelihood.eval()

#         with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(100):
#             self.model = model
#             self.likelihood = likelihood

#     # def find_next_position()

#     def ask(self, batch_size):
#         print("Asking GP")
#         # return 1,1
#         # random_positions = np.random.choice(len(self.all_possible_positions),size=2500,replace=False)
#         # test_x = self.all_possible_positions[random_positions]
#         test_x = self.all_possible_positions
#         print(f"Evaluating model on {len(test_x)} positons...")
#         evaluation = self.model(test_x)
#         test_x = test_x.detach().cpu().numpy()
#         # stds_all = self.likelihood(evaluation).loc.detach().cpu().numpy()
#         # stds = stds_all.prod(axis=0)
#         print("Sampling...")
#         pred_samples = evaluation.sample(torch.Size((64,))).exp()
#         info = (pred_samples / pred_samples.sum(-2, keepdim=True))
#         # probabilities = info.mean(0).detach().cpu().numpy()
#         stds_all = info.std(0).pow(3)
#         # The stds here actually represent the uncertainty on the proabbility of each label.
#         stds = stds_all.sum(0).detach().cpu().numpy()
#         stds_all = stds_all.detach().cpu().numpy()
#         # probability = stds
#         # means = evaluation.mean.detach().cpu().numpy()
#         # stds = stds**3
#         selected_indices = np.random.choice(len(test_x), size=batch_size, replace=False, p=stds/np.sum(stds))
#         # selected_indices = np.argsort(stds.flatten())[-batch_size:][::-1]
#         # most_uncertain_indices = np.argsort(stds.flatten())
#         # next_positions = list(self.all_possible_positions[most_uncertain_indices[-batch_size:]].detach().cpu().numpy())
#         next_positions = list(test_x[selected_indices])
#         print("Next positions selected...")
#         print("Plotting")
#         grid_evaluations = griddata(test_x, evaluation.loc.max(0)[1], (self.meshgrid_points[0], self.meshgrid_points[1]), method='nearest')
#         # plt.imshow(evaluation.loc.max(0)[1].reshape(self.meshgrid_points[0].shape), cmap='terrain', origin='lower')
#         plt.imshow(grid_evaluations, cmap='terrain', origin='lower')
#         cb = plt.colorbar() 
#         plt.savefig(f'predictions.png')
#         cb.remove()
#         grid_stds = griddata(test_x, stds, (self.meshgrid_points[0], self.meshgrid_points[1]), method='nearest')
#         # plt.imshow(stds.reshape(self.meshgrid_points[0].shape), cmap='terrain', origin='lower')
#         plt.imshow(grid_stds, cmap='terrain', origin='lower')
#         cb = plt.colorbar() 
#         plt.savefig(f'stds.png')
#         cb.remove()
#         for i, std in enumerate(stds_all):
#             interp_stds = griddata(test_x, std, (self.meshgrid_points[0], self.meshgrid_points[1]), method='nearest')
#             # plt.scatter(self.meshgrid_points[0], self.meshgrid_points[1], c=interp_stds, cmap='terrain')
#             plt.imshow(interp_stds, cmap='terrain', origin='lower')
#             cb = plt.colorbar()
#             # plt.colorbar()
#             plt.savefig(f'stds_{i:02d}.png')
#             cb.remove()
#         return next_positions

#         # raise NotImplementedError
    
#     def tell_many(self, x, y):
#         raise NotImplementedError