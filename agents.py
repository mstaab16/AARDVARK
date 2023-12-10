import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

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
        return x, self.KMeans.fit_transform(y)

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
    
    def tell(self, x, y):
        print(f"GPytorchAgent told about new data: {x.shape=}, {y.shape=}")
        return np.ones(self.decicion_space_shape), None
        # if self.inputs is None:
        #     self.inputs = torch.atleast_2d(torch.tensor(x, device=self.device))
        #     self.targets = torch.atleast_1d(torch.tensor(y, device=self.device))
        # else:
        #     self.inputs = torch.cat([self.inputs, torch.atleast_2d(torch.tensor(x, device=self.device))], dim=0)
        #     self.targets = torch.cat([self.targets, torch.atleast_1d(torch.tensor(y, device=self.device))], dim=0)
        #     self.inputs.to(self.device)
        #     self.targets.to(self.device)
        # self.surrogate_model.set_train_data(self.inputs, self.targets, strict=False)
        # return dict(independent_variable=x, observable=y, cache_len=len(self.targets))

    def ask(self, batch_size):
        raise NotImplementedError
    
    def tell_many(self, x, y):
        raise NotImplementedError


class MangerAgent(Agent):
    def __init__(self, max_count, input_bounds, input_min_spacings, n_independent_keys, dependent_shape):
        """Load the model, set up the necessary bits"""

        self.max_count = max_count
        self.count = 0
        self.input_bounds = np.array(input_bounds)
        self.input_min_spacings = input_min_spacings
        # The shape below is found by computing (upper-lower)/spacing
        # for each bound at the same time and rounding up as an int
        self.decision_space = np.ones(np.ceil(np.diff(input_bounds, axis=1).flatten()/input_min_spacings).astype(int))
        self.inputs = np.nan * np.zeros((max_count, n_independent_keys))
        self.targets = np.nan * np.zeros((max_count, *dependent_shape))

        self.pipeline = [PCAAgent(n_components=50), KMeansAgent(n_clusters=6), GPytorchAgent(input_bounds, input_min_spacings, self.decision_space.shape)]

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
        self.next_point = self.make_decision()
        print(f"Manager suggests point: {self.next_point}")
        return self.next_point
    
    def update_decision_space(self):
        x, y = self.pipeline[0].tell(self.inputs[:self.count], self.targets[:self.count].reshape(self.count, -1))
        for node in self.pipeline[1:]:
            x, y = node.tell(x,y)

        if y is not None:
            raise ValueError("The final pipeline node should have returned (decision_space, None)... Something is wrong.")
        
        self.decision_space = x/np.sum(x)


    def make_decision(self):
        flat_p = self.decision_space.flatten()
        i_choice = np.random.choice(np.arange(len(flat_p)), p=flat_p)
        i_choice = np.unravel_index(i_choice, self.decision_space.shape)
        frac_position = np.array(i_choice) / np.array(self.decision_space.shape)
        choice_pos = self.input_bounds[:,0] + frac_position * np.diff(self.input_bounds, axis=1).flatten()
        return choice_pos
