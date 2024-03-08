import importlib.util


def load_module(package_name, module_name):
    """
    Dynamically loads a module from a package.

    Parameters:
    - package_name (str): The name of the package.
    - module_name (str): The name of the module.

    Returns:
    - module: The loaded module.
    """
    # Construct the full package path
    package_path = package_name.replace('.', '/')

    # Construct the path to the module within the package
    module_path = f"{package_path}/{module_name}.py"

    # Create a spec for the module
    spec = importlib.util.spec_from_file_location(module_name, module_path)

    # Load the module using the spec
    module = importlib.util.module_from_spec(spec)

    # Execute the module code
    spec.loader.exec_module(module)

    return module
