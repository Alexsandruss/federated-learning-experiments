from typing import List
import numpy as np
from sklearn.model_selection import train_test_split
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor


class SGDRegSD(ShardDescriptor):
    """Shard descriptor class."""

    def __init__(self, rank: int, n_samples: int = 10, noise: float = 0.15, test_size: float = 0.25) -> None:
        """
        Initialize SGDReg Shard Descriptor.
        """
        np.random.seed(rank)  # Setting seed for reproducibility
        self.test_size = test_size
        self.n_samples = max(n_samples, 5)

        coef = np.array([2.564, 0.576, 1.675, 1.019])
        bias = np.array([1.657])

        x = np.random.uniform(size=(n_samples, coef.shape[0]))
        x += noise * np.random.uniform(size=(n_samples, coef.shape[0]))
        y = np.matmul(x, coef) + bias

        self.data = np.concatenate((x, y.reshape(y.shape[0], 1)), axis=1)

    def get_dataset(self, dataset_type: str) -> np.ndarray:
        """
        Return a shard dataset by type.
        A simple list with elements (x, y) implemets the Shard Dataset interface.
        """
        train_data, test_data = train_test_split(self.data, test_size=self.test_size, random_state=42)
        if dataset_type == 'train':
            return train_data
        elif dataset_type == 'val':
            return test_data
        else:
            pass

    @property
    def sample_shape(self) -> List[str]:
        """Return the sample shape info."""
        (*x, _) = self.data[0]
        return [str(i) for i in np.array(x, ndmin=1).shape]

    @property
    def target_shape(self) -> List[str]:
        """Return the target shape info."""
        (*_, y) = self.data[0]
        return [str(i) for i in np.array(y, ndmin=1).shape]

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return 'Allowed dataset types are `train` and `val`'
