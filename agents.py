import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import torch

from bluesky_adaptive.recommendations import NoRecommendation

from typing import Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, TypedDict, Union
from numpy.typing import ArrayLike

from abc import ABC, abstractmethod

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

    def tell(self, x, y):
        print(f"KMeansAgent told about new data: {x.shape=}, {y.shape=}")
        if len(y) < self.kwargs.get('n_clusters'):
            return x, np.zeros(len(y))
        
        self.KMeans = KMeans(*self.args, **self.kwargs)
        self.KMeans.fit(y)
        return x, self.KMeans.labels_

    def ask(self, batch_size):
        raise NotImplementedError
    
    def tell_many(self, x, y):
        raise NotImplementedError


class GPytorchAgent(Agent):
    """A simple naive agent that cycles samples sequentially in environment space"""

    def __init__(self, input_bounds, input_min_spacings, decision_space_shape):
        print("Creating GPytorchAgent")
        self.input_bouds = input_bounds
        self.input_min_spacings = input_min_spacings
        self.decicion_space_shape = decision_space_shape
        self.inputs=None
        self.targets=None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def tell(self, x, y):
        print(f"GPytorchAgent told about new data: {x.shape=}, {y.shape=}")
        # return np.ones(self.decicion_space_shape), None
        if self.inputs is None:
            self.inputs = torch.atleast_2d(torch.tensor(x, device=self.device))
            self.targets = torch.atleast_1d(torch.tensor(y, device=self.device))
        else:
            self.inputs = torch.cat([self.inputs, torch.atleast_2d(torch.tensor(x, device=self.device))], dim=0)
            self.targets = torch.cat([self.targets, torch.atleast_1d(torch.tensor(y, device=self.device))], dim=0)
            self.inputs.to(self.device)
            self.targets.to(self.device)
        self.surrogate_model.set_train_data(self.inputs, self.targets, strict=False)
        # return dict(independent_variable=x, observable=y, cache_len=len(self.targets))

    def ask(self, batch_size):
        raise NotImplementedError
    
    def tell_many(self, x, y):
        raise NotImplementedError
    

import importlib
from abc import ABC
from logging import getLogger
from typing import Callable, Optional, Tuple

import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import AcquisitionFunction, UpperConfidenceBound
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

class SingleTaskGPAgent():
    def __init__(
        self,
        *,
        bounds: torch.Tensor,
        gp: SingleTaskGP = None,
        device: torch.device = None,
        out_dim=1,
        partial_acq_function: Optional[Callable] = None,
        num_restarts: int = 10,
        raw_samples: int = 20,
        **kwargs,
    ):
        """Single Task GP based Bayesian Optimization

        Parameters
        ----------
        bounds : torch.Tensor
            A `2 x d` tensor of lower and upper bounds for each column of `X`
        gp : SingleTaskGP, optional
            GP surrogate model to use, by default uses BoTorch default
        device : torch.device, optional
            Device, by default cuda if avail
        out_dim : int, optional
            Dimension of output predictions by surrogate model, by default 1
        partial_acq_function : Optional[Callable], optional
            Partial acquisition function that will take a single argument of a conditioned surrogate model.
            By default UCB with beta at 0.1
        num_restarts : int, optional
            Number of restarts for optimizing the acquisition function, by default 10
        raw_samples : int, optional
            Number of samples used to instantiate the initial conditions of the acquisition function optimizer.
            For a discussion of num_restarts vs raw_samples, see:
            https://github.com/pytorch/botorch/issues/366
            Defaults to 20.
        """
        super().__init__(**kwargs)
        self.inputs = None
        self.targets = None

        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else torch.device(device)
        )
        self.bounds = torch.tensor(bounds, device=self.device).view(2, -1)
        if gp is None:
            dummy_x, dummy_y = torch.randn(2, self.bounds.shape[-1], device=self.device), torch.randn(
                2, out_dim, device=self.device
            )
            gp = SingleTaskGP(dummy_x, dummy_y)

        self.surrogate_model = gp
        self.mll = ExactMarginalLogLikelihood(self.surrogate_model.likelihood, self.surrogate_model)

        self.surrogate_model.to(self.device)
        self.mll.to(self.device)

        if partial_acq_function is None:
            self._partial_acqf = lambda gp: UpperConfidenceBound(gp, beta=0.1)
            self.acqf_name = "UpperConfidenceBound"
        else:
            self._partial_acqf = partial_acq_function
            self.acqf_name = "custom"
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples

    # def server_registrations(self) -> None:
    #     super().server_registrations()
    #     self._register_method("update_acquisition_function")

    # def update_acquisition_function(self, acqf_name, **kwargs):
    #     module = importlib.import_module("botorch.acquisition")
    #     self.acqf_name = acqf_name
    #     self._partial_acqf = lambda gp: getattr(module, acqf_name)(gp, **kwargs)
    #     self.close_and_restart()

    def start(self, *args, **kwargs):
        _md = dict(acqf_name=self.acqf_name)
        self.metadata.update(_md)
        super().start(*args, **kwargs)

    def tell(self, x, y):
        if self.inputs is None:
            self.inputs = torch.atleast_2d(torch.tensor(x, device=self.device))
            self.targets = torch.atleast_1d(torch.tensor(y, device=self.device))
        else:
            self.inputs = torch.cat([self.inputs, torch.atleast_2d(torch.tensor(x, device=self.device))], dim=0)
            self.targets = torch.cat([self.targets, torch.atleast_1d(torch.tensor(y, device=self.device))], dim=0)
            self.inputs.to(self.device)
            self.targets.to(self.device)
        self.surrogate_model.set_train_data(self.inputs, self.targets, strict=False)
        # return dict(independent_variable=x, observable=y, cache_len=len(self.targets))
        return x, y

    # def report(self):
    #     """Fit GP, and construct acquisition function.
    #     Document retains state dictionary.
    #     """
    #     fit_gpytorch_mll(self.mll)
    #     acqf = self._partial_acqf(self.surrogate_model)
    #     return dict(
    #         latest_data=self.tell_cache[-1],
    #         cache_len=self.inputs.shape[0],
    #         **{
    #             "STATEDICT-" + ":".join(key.split(".")): val.detach().cpu().numpy()
    #             for key, val in acqf.state_dict().items()
    #         },
    #     )

    def ask(self, batch_size=1):
        """Fit GP, optimize acquisition function, and return next points.
        Document retains candidate, acquisition values, and state dictionary.
        """
        if batch_size > 1:
            # logger.warning(f"Batch size greater than 1 is not implemented. Reducing {batch_size} to 1.")
            batch_size = 1
        fit_gpytorch_mll(self.mll)
        acqf = self._partial_acqf(self.surrogate_model)
        acqf.to(self.device)
        candidate, acq_value = optimize_acqf(
            acq_function=acqf,
            bounds=self.bounds,
            q=batch_size,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
        )
        return (
            [
                dict(
                    candidate=candidate.detach().cpu().numpy(),
                    acquisition_value=acq_value.detach().cpu().numpy(),
                    # latest_data=self.tell_cache[-1],
                    # cache_len=self.inputs.shape[0],
                    **{
                        "STATEDICT-" + ":".join(key.split(".")): val.detach().cpu().numpy()
                        for key, val in acqf.state_dict().items()
                    },
                )
            ],
            torch.atleast_1d(candidate).detach().cpu().numpy(),
        )

    # def remodel_from_report(self, run: BlueskyRun, idx: int = None) -> Tuple[AcquisitionFunction, SingleTaskGP]:
    #     idx = -1 if idx is None else idx
    #     keys = [key for key in run.report["data"].keys() if key.split("-")[0] == "STATEDICT"]
    #     state_dict = {".".join(key[10:].split(":")): torch.tensor(run.report["data"][key][idx]) for key in keys}
    #     acqf = self._partial_acqf(self.surrogate_model)
    #     acqf.load_state_dict(state_dict)
    #     acqf.to(self.device)
    #     return acqf, acqf.model


class MangerAgent(Agent):
    def __init__(self, max_count, input_bounds, input_min_spacings, n_independent_keys, dependent_shape):
        """Load the model, set up the necessary bits"""

        self.max_count = max_count
        self.count = 0
        self.min_count_to_make_decision = 1000
        self.input_bounds = np.array(input_bounds)
        self.input_min_spacings = input_min_spacings
        # The shape below is found by computing (upper-lower)/spacing
        # for each bound at the same time and rounding up as an int
        self.decision_space = np.ones(np.ceil(np.diff(input_bounds, axis=1).flatten()/input_min_spacings).astype(int))
        self.inputs = np.nan * np.zeros((max_count, n_independent_keys))
        self.targets = np.nan * np.zeros((max_count, *dependent_shape))

        self.pipeline = [PCAAgent(n_components=10), KMeansAgent(n_clusters=3), SingleTaskGPAgent(bounds=torch.tensor(input_bounds))]
        # self.pipeline = [IntensityAgent(), SingleTaskGPAgent(bounds=torch.tensor(input_bounds))]

    # def update_from_config(self):
    #     config = self.read_config()
    #     if not self.config_changed:
    #         return
    #     self.pipeline = 

    
    def tell(self, x, y):
        # print(x)
        """Tell the Manger about something new"""
        if self.count + 1 == self.max_count:
            return
        if self.count < self.min_count_to_make_decision:
            return
        
        self.inputs[self.count] = np.concatenate([i.values for i in x])
        self.targets[self.count] = y[0].values

        self.count += 1

        # if self.count < self.n_count_before_thinking:
        #     return
        
        self.update_decision_space()
        
    def tell_many(self, xs, ys):
        """The adaptive plan uses the tell_many method by default, just pass it along."""
        self.tell(xs,ys)

    def ask(self, n):
        """Ask the Manager for a new command"""
        if n != 1:
            raise NotImplementedError
        if self.count == self.max_count:
            raise NoRecommendation
        print("Manager was asked for a point")
        if self.count < self.min_count_to_make_decision:
            return [np.random.uniform(low,high) for low, high in self.input_bounds]
        next_point = self.make_decision()
        self.next_point = [np.clip(val, high, low) for val, (high, low) in zip(next_point, self.input_bounds)]
        print(f"Manager suggests point: {self.next_point}")
        return self.next_point
    
    def update_decision_space(self):
        
        x, y = self.pipeline[0].tell(self.inputs[:self.count], self.targets[:self.count].reshape(self.count, -1))
        print("--"*20)
        print(x)
        print("--"*20)
        print(y)
        print("--"*20)
        for node in self.pipeline[1:]:
            x, y = node.tell(x,y)
            print("--"*20)
            print(x)
            print("--"*20)
            print(y)
            print("--"*20)

        # if y is not None:
        #     raise ValueError("The final pipeline node should have returned (decision_space, None)... Something is wrong.")
        
        # self.decision_space = x/np.sum(x)


    def make_decision(self):
        # flat_p = self.decision_space.flatten()
        # i_choice = np.random.choice(np.arange(len(flat_p)), p=flat_p)
        # i_choice = np.unravel_index(i_choice, self.decision_space.shape)
        # frac_position = np.array(i_choice) / np.array(self.decision_space.shape)
        # choice_pos = self.input_bounds[:,0] + frac_position * np.diff(self.input_bounds, axis=1).flatten()
        # return choice_pos
        choice_position = self.pipeline[-1].ask(1)
        return choice_position
