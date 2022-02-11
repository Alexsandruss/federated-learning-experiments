from typing import List
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor


class SGDClsSD(ShardDescriptor):
    """Shard descriptor class."""

    def __init__(self, rank: int, n_samples: int = 10, noise: float = 0.15, test_size: float = 0.25) -> None:
        """
        Initialize SGDCls Shard Descriptor.
        """
        np.random.seed(rank)  # Setting seed for reproducibility
        self.test_size = test_size
        self.n_samples = max(n_samples, 5)

        x, y = make_classification(n_samples=self.n_samples * 10, n_features=4, n_informative=4, n_redundant=0, n_classes=3, random_state=42)
        x, y = x[n_samples * rank:n_samples * (rank + 1)], y[n_samples * rank:n_samples * (rank + 1)]
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
