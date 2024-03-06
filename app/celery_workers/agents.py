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
import matplotlib.pyplot as plt


from typing import Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, TypedDict, Union
from numpy.typing import ArrayLike

from abc import ABC, abstractmethod

import time

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

        #     new_y = np.zeros((len(self.KMeans.labels_), self.kwargs.get('n_clusters')))
        #     for i, label in enumerate(self.KMeans.labels_):
        #         new_y[i,label] = 1
        end = time.perf_counter_ns()
        print(f"KMeans took {(end-start)/1e6:.02f}ms to fit.")
        return x, new_y
    
    def ask(self, batch_size):
        raise NotImplementedError
    
    def tell_many(self, x, y):
        raise NotImplementedError


# class MultitaskGPModel(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood):
#         super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.MultitaskMean(
#             gpytorch.means.ConstantMean(), num_tasks=train_y.shape[1]
#         )
#         self.covar_module = gpytorch.kernels.MultitaskKernel(
#             gpytorch.kernels.RBFKernel(), num_tasks=train_y.shape[1], rank=1
#         )

#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
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
    """A simple naive agent that cycles samples sequentially in environment space"""

    def __init__(self, input_bounds, input_min_spacings, experiment_id):
        print("Creating GPytorchAgent")
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
        
        training_iterations = 100
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

        # fit = basinhopping(minimize_func, np.array([0.5, 0.5]), minimizer_kwargs=dict(bounds=self.input_bounds, method='L-BFGS-B', args=(self.model, self.device)))
        # score = fit.fun
        # local_maxima = np.array([fit.x])
        # print(f"Most uncertain found: {fit.x} | {score}")
        # local_maxima = []
        # scores = []
        # for x0 in self.optimize_positions:
        #     # print(f"{local_maxima=}")
        #     fit = minimize(minimize_func, x0=x0.cpu(), args=(self.model, self.device), bounds=self.input_bounds, method='L-BFGS-B', options=dict(eps=.5))
        #     # print("Minimize returned: ", fit)
        #     x = fit.x
        #     # print(x0)
        #     # print(x)
        #     x = [(x[i]//self.input_min_spacings[i])*self.input_min_spacings[i] for i in range(2)]
        #     if x not in local_maxima:
        #         local_maxima.append(x)
        #         scores.append(fit.fun)
        # local_maxima = np.array(local_maxima)
        # local_maxima = local_maxima[np.argsort(scores)]
        # print(f"Asking GP... Currently have {len(self.inputs)} inputs and {len(self.targets)} targets.")
        # # return 1,1
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
        tasks.image_report(experiment_id=self.experiment_id, matrix=grid_stds, name='uncertainties')
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
        tasks.image_report(experiment_id=self.experiment_id, matrix=grid_evaluations, name='predictions')
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