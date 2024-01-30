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
    

from sklearn.decomposition import PCA
class PCAAgent(Agent):
    """A simple naive agent that cycles samples sequentially in environment space"""

    def __init__(self, **kwargs):
        print("Creating PCAAgent")
        self.kwargs = kwargs
        if kwargs.get('n_components') is None:
            raise ValueError("Please enter a non-None value for n_components of PCA.")

    def tell(self, x, y):
        print(f"PCAAgent told about new data: {x.shape=}, {y.shape=}")
        if len(y) < self.kwargs.get('n_components'):
            self.PCA = PCA(n_components=None, **{k:v for k,v in self.kwargs.items() if k!='n_components'})
            return x, self.PCA.fit_transform(y)
        
        self.PCA = PCA(**self.kwargs)
        return x, self.PCA.fit_transform(y)
    
    def ask(self, batch_size):
        raise NotImplementedError
    
    def tell_many(self, x, y):
        raise NotImplementedError


from sklearn.cluster import KMeans
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
        if len(y) < self.n_clusters:
            self.labels = np.zeros(len(y))
            new_y = np.zeros(len(y))
        else:
            self.KMeans = KMeans(n_clusters=self.n_clusters)
            self.KMeans.fit(y)
            self.labels = self.KMeans.labels_

            new_y = np.zeros((len(self.KMeans.labels_), self.kwargs.get('n_clusters')))
            for i, label in enumerate(self.KMeans.labels_):
                new_y[i,label] = 1
        return x, new_y
    
    def ask(self, batch_size):
        raise NotImplementedError
    
    def tell_many(self, x, y):
        raise NotImplementedError


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=train_y.shape[1]
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=train_y.shape[1], rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class GPytorchAgent(Agent):
    """A simple naive agent that cycles samples sequentially in environment space"""

    def __init__(self, input_bounds, input_min_spacings, id):
        print("Creating GPytorchAgent")
        self.id = id
        self.input_bounds = input_bounds
        self.input_min_spacings = input_min_spacings
        self.inputs=None
        self.targets=None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        p = [np.arange(low, high, delta) for (low, high), delta in zip(self.input_bounds, self.input_min_spacings)]
        self.meshgrid_points = np.meshgrid(*p)
        self.all_possible_positions = torch.stack([torch.tensor(arr.flatten(),dtype=torch.float32).to(self.device) for arr in self.meshgrid_points],dim=1)
        print(f"Total number of possible position is ~{self.all_possible_positions.shape}")

    def tell(self, x, y):
        print(f"GPytorchAgent told about new data: {x.shape=}, {y.shape=}")
        # return np.ones(self.decicion_space_shape), None
        if self.inputs is None:
            self.inputs = torch.atleast_2d(torch.tensor(x, device=self.device, dtype=torch.float32))
            self.targets = torch.atleast_1d(torch.tensor(y, device=self.device, dtype=torch.float32))
        else:
            self.inputs = torch.cat([self.inputs, torch.atleast_2d(torch.tensor(x, device=self.device, dtype=torch.float32))], dim=0)
            self.targets = torch.cat([self.targets, torch.atleast_1d(torch.tensor(y, device=self.device, dtype=torch.float32))], dim=0)
            self.inputs.to(self.device)
            self.targets.to(self.device)
        self.fit()
        return x, y
        # self.surrogate_model.set_train_data(self.inputs, self.targets, strict=False)
        # return dict(independent_variable=x, observable=y, cache_len=len(self.targets))

    def fit(self):# model, likelihood, train_x, train_y):
        print("Fitting GP")
        # indices = np.arange(train_x.shape[0])
        # max(int(np.ceil(train_x.shape[0]*0.2)), min(train_x.shape[0], 100))
        # print(min(train_x.shape[0], min(100,int(np.ceil(train_x.shape[0]*0.2)))))
        # indices = np.random.choice(indices, size=min(train_x.shape[0], 1000), replace=False)
        # train_x = train_x[indices]
        # train_y = train_y[indices]
        train_x = self.inputs
        train_y = self.targets
        # if torch.cuda.is_available():
        #     train_x = train_x.cuda()
        #     train_y = train_y.cuda()

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=train_y.shape[1])
        model = MultitaskGPModel(train_x, train_y, likelihood)
        
        # if torch.cuda.is_available():
        #     model = model.cuda()
        #     likelihood = likelihood.cuda()
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
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
            optimizer.step()

        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(100):
            self.model = model
            self.likelihood = likelihood

    def ask(self, batch_size):
        print("Asking GP")
        # return 1,1
        random_positions = np.random.choice(len(self.all_possible_positions),size=1000,replace=False)
        test_x = self.all_possible_positions[random_positions]
        evaluation = self.model(test_x)
        test_x = test_x.detach().cpu().numpy()
        stds = self.likelihood(evaluation).stddev[:,:].detach().cpu().numpy().sum(axis=1)
        # means = evaluation.mean.detach().cpu().numpy()
        most_uncertain_indices = np.argsort(stds.flatten())
        next_positions = list(self.all_possible_positions[most_uncertain_indices[:batch_size]].detach().cpu().numpy())
        # interp_stds = griddata(test_x, stds, (self.meshgrid_points[0], self.meshgrid_points[1]), method='nearest')
        # print(f'{means.shape=}')
        # interp_0 = griddata(test_x, means[:,0], (self.meshgrid_points[0], self.meshgrid_points[1]), method='nearest')
        # interp_1 = griddata(test_x, means[:,1], (self.meshgrid_points[0], self.meshgrid_points[1]), method='nearest')
        # interp_2 = griddata(test_x, means[:,2], (self.meshgrid_points[0], self.meshgrid_points[1]), method='nearest')
        # np.save(f'save/stds/run_{len(self.inputs)}.npy', interp_stds)
        # np.save(f'save/c0/run_{len(self.inputs)}.npy', interp_0)
        # np.save(f'save/c1/run_{len(self.inputs)}.npy', interp_1)
        # np.save(f'save/c2/run_{len(self.inputs)}.npy', interp_2)
        # np.save(f'save/stds/run_{len(self.inputs)}.npy', np.array([random_positions, stds]).T)
        # return 1,1
        return next_positions

        # raise NotImplementedError
    
    def tell_many(self, x, y):
        raise NotImplementedError

    
# class MangerAgent(Agent):
#     def __init__(self, max_count, input_bounds, input_min_spacings, n_independent_keys, dependent_shape, re_manager):
#         """Load the model, set up the necessary bits"""
#         self.id = re_manager.id
#         self.save_positions_path = f'save/ManagerAgent/pos/{self.id:010}.txt'
#         self.save_flat_targets_path = f'save/ManagerAgent/flat_target/{self.id:010}.txt'
#         self.dependent_shape = dependent_shape
#         # self.dependent_shape = (300,300)
#         self.re_manager = re_manager
#         self.max_count = max_count
#         self.count = 0
#         self.min_count_to_make_decision = 10
#         self.input_bounds = np.array(input_bounds)
#         self.input_min_spacings = input_min_spacings
#         # self.initial_points = []
#         # The shape below is found by computing (upper-lower)/spacing
#         # for each bound at the same time and rounding up as an int
#         # self.decision_space = np.ones(np.ceil(np.diff(input_bounds, axis=1).flatten()/input_min_spacings).astype(int))
#         self.inputs = np.nan * np.zeros((max_count, n_independent_keys))
#         self.targets = np.nan * np.zeros((max_count, *self.dependent_shape))
#         self.flat_targets = np.nan * np.zeros((max_count, np.multiply(*self.dependent_shape)))

#         # self.pipeline = [PCAAgent(n_components=5), KMeansAgent(n_clusters=4), GPytorchAgent(input_bounds=self.input_bounds, input_min_spacings=self.input_min_spacings)]
#         self.pipeline = [KMeansAgent(n_clusters=4, id=self.id), GPytorchAgent(input_bounds=self.input_bounds, input_min_spacings=self.input_min_spacings, id=self.id)]
#         # self.pipeline = [IntensityAgent(), SingleTaskGPAgent(bounds=torch.tensor(input_bounds))]

#     # def update_from_config(self):
#     #     config = self.read_config()
#     #     if not self.config_changed:
#     #         return
#     #     self.pipeline = 

    
#     def tell(self, x, y):
#         print("ManagerAgent was told about new data")
#         # print(x)
#         """Tell the Manger about something new"""
#         if self.count + 1 == self.max_count:
#             return
#         # print(f'{y[0].values.shape}')
#         resized_y = resize(y[0][0].values, self.dependent_shape, anti_aliasing=False)
#         self.flat_y = resized_y.flatten()
#         self.inputs[self.count+1] = np.concatenate([i.values for i in x])
#         self.flat_targets[self.count] = self.flat_y
#         self.targets[self.count] = resized_y

#         self.x = self.inputs[self.count]
#         self.y = resized_y

        
#         # plt.imshow(resized_y, origin='lower')
#         # plt.savefig(f'save/imgs/img_{self.count:03}.png')
#         # np.save(f'save/raw/raw_{self.count:03}.npy', resized_y)

#         self.count += 1

#         if self.count < self.min_count_to_make_decision:
#             return

#         # if self.count < self.n_count_before_thinking:
#         #     return
        
#         self.update_decision_space()
        
#     def tell_many(self, xs, ys):
#         """The adaptive plan uses the tell_many method by default, just pass it along."""
#         self.tell(xs,ys)

#     def is_already_measured(self, pos):
#         return (np.less(np.abs(np.array(pos) - self.inputs[:self.count]), self.input_min_spacings)).all()

#     def get_random_position(self):
#         if self.count == 0:
#             return [np.random.uniform(low,high) for low, high in self.input_bounds] 

#         allowed_random_attempts = 500
#         for i in range(allowed_random_attempts):
#             random_position = [np.random.uniform(low,high) for low, high in self.input_bounds] 
#             print(f"Positions Measured So Far {self.count}:")
#             # print(self.inputs[:self.count])
#             print("Next position:")
#             print(random_position)
#             already_measured = self.is_already_measured(random_position)
#             # too_close = 
#             print("Was it already measured?")
#             print(already_measured)
#             if already_measured:
#                 continue
#             print(f"Took {i+1} tries to get a valid random coordinate.")
#             return random_position

#     def ask(self, n):
#         """Ask the Manager for a new command"""
#         print("ManagerAgent was asked for a new point")
#         if n != 1:
#             raise NotImplementedError
#         if self.count == self.max_count:
#             raise ValueError("The Manager has already been told about the maximum number of points.")
#         if self.re_manager.abort:
#             raise ValueError("The Manager has been told to abort.")
        
#         print("Manager was asked for a point")
#         if self.count < self.min_count_to_make_decision:
#             pos = self.get_random_position()
#             return pos

#         next_point = self.make_decision()
#         self.next_point = [np.clip(val, high, low) for val, (high, low) in zip(next_point, self.input_bounds)]
#         print(f"Manager suggests point: {self.next_point}")
#         return self.next_point
    
#     def update_decision_space(self):
#         if self.count < self.min_count_to_make_decision:
#             return 
#         x, y = self.pipeline[0].tell(self.inputs[1:self.count], self.flat_targets[:self.count-1])
#         # print("--"*20)
#         # print(x)
#         # print("--"*20)
#         # print(y)
#         # print("--"*20)
#         for node in self.pipeline[1:]:
#             x, y = node.tell(x,y)
#             # print("--"*20)
#             # print(x)
#             # print("--"*20)
#             # print(y)
#             # print("--"*20)
#         # self.save_info()
#         # labels = self.pipeline[0].labels
#         # np.save('save/labels.npy', labels)
#         # np.save('save/positions.npy', self.inputs[:self.count])
#         # np.save(f'save/pca/pc{self.count}.npy',self.pipeline[0].PCA.components_)

#         # if y is not None:
#         #     raise ValueError("The final pipeline node should have returned (decision_space, None)... Something is wrong.")
        
#         # self.decision_space = x/np.sum(x)


#     def make_decision(self):
#         # flat_p = self.decision_space.flatten()
#         # i_choice = np.random.choice(np.arange(len(flat_p)), p=flat_p)
#         # i_choice = np.unravel_index(i_choice, self.decision_space.shape)
#         # frac_position = np.array(i_choice) / np.array(self.decision_space.shape)
#         # choice_pos = self.input_bounds[:,0] + frac_position * np.diff(self.input_bounds, axis=1).flatten()
#         # return choice_pos
#         # self.get_random_position()
#         choice_position = self.pipeline[-1].ask(1)
#         return choice_position

#     def save_info(self):
#         with open(self.save_positions_path, 'a') as f:
#                 np.savetxt(f, self.x)
#         with open(self.save_flat_targets_path, 'a') as f:
#                 np.savetxt(f, self.flat_y.T)
#         for node in self.pipeline:
#             node.save_info()
