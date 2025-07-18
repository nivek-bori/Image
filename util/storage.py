import time


# singleton class for one time, lazy loading of expensive resources
class ResourceLoader:
    _instance = None  # cls instance

    _resources = {}  # loaded resources
    _load_functions = {}  # resource loading functions

    def __new__(cls):
        if cls._instance is None:  # load new instance
            cls._instance = super(ResourceLoader, cls).__new__(cls)
        return cls._instance

    def set_load_function(self, resource_name, load_function):
        if not isinstance(resource_name, str) or not resource_name:
            raise ValueError(f'invalid resource_name {resource_name}')
        if not callable(load_function):
            raise TypeError('load_function must be a callable function.')

        self._load_functions[resource_name] = load_function

    def get_resource(self, resource_name):
        if not isinstance(resource_name, str) or not resource_name:
            raise ValueError(f'invalid resource_name {resource_name}')

        if resource_name not in self._resources:
            if resource_name not in self._load_functions:
                raise ValueError(f'no resource loader for resource {resource_name}')

            self._resources[resource_name] = self._load_functions[resource_name]()

        return self._resources[resource_name]

    def reload(self, resource_name=None):
        # resource name specified
        if resource_name and resource_name in self._resources:
            self._resources[resource_name] = self._load_functions[
                resource_name
            ]()  # reload resource

        # resource name unspecified -> all resources
        else:
            for resource_name in self._resources.keys():
                self._resources[resource_name] = self._load_functions[
                    resource_name
                ]()  # reload resource
