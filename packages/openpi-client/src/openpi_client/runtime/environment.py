import abc


class Environment(abc.ABC):
    """An Environment represents the robot and the environment it inhabits.

    The primary contract of environments is that they can be queried for observations
    about their state, and have actions applied to them to change that state.
    """

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset the environment to its initial state.

        This will be called once before starting each episode.
        """

    @abc.abstractmethod
    def done(self) -> bool:
        """Allow the environment to signal that the task is done.

        This will be called after each step. It should return `True` if the task is
        done (either successfully or unsuccessfully), and `False` otherwise.
        """

    @abc.abstractmethod
    def get_observation(self) -> dict:
        """Query the environment for the current state."""

    @abc.abstractmethod
    def apply_action(self, action: dict) -> None:
        """Take an action in the environment."""
