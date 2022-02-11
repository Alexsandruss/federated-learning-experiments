from openfl.plugins.frameworks_adapters.framework_adapter_interface import (
    FrameworkAdapterPluginInterface,
)


class CustomFrameworkAdapter(FrameworkAdapterPluginInterface):
    """Framework adapter plugin class."""

    @staticmethod
    def get_tensor_dict(model, optimizer=None):
        """Extract weights from a model."""
        return {'w': model.weights}

    @staticmethod
    def set_tensor_dict(model, tensor_dict, optimizer=None, device='cpu'):
        """Load weights to a model."""
        model.weights = tensor_dict['w']
