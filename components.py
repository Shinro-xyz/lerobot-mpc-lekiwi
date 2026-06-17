from abc import ABC, abstractmethod
from typing import Any

class Controller(ABC):
    """
    Abstract base class for all controllers.
    """
    @abstractmethod
    def compute(self,*args:Any,**kwargs:Any)-> Any:
        """
        Compute the control action.

        Args:
            *args: Positional arguments for computation.
            **kwargs: Keyword arguments for computation.

        Returns:
            The computed control action.
        """
        pass

    def reset(self,*args:Any,**kwargs:Any)-> Any:
        """
        Reset the controller to its initial state.

        Args:
            *args: Positional arguments for reset.
            **kwargs: Keyword arguments for reset.

        Returns:
            None.
        """
        pass

class Plant(ABC):
    """
    Abstract base class for a system plant.
    """
    @abstractmethod
    def get_state(self, *args:Any, **kwargs:Any)->Any:
        """
        Get the current state of the plant.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            The current state.
        """
        pass

    @abstractmethod
    def get_model(self, *args:Any, **kwargs:Any)->Any:
        """
        Get the mathematical model of the plant.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            The plant model.
        """
        pass

    @abstractmethod
    def step(self, *args:Any, **kwargs:Any)->Any:
        """
        Perform a single time step simulation or execution of the plant.

        Args:
            *args: Positional arguments (e.g., control input).
            **kwargs: Keyword arguments.

        Returns:
            The next state or result of the step.
        """
        pass