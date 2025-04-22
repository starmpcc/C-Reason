import importlib
import os

LAUNCHER_REGISTRY = {}

__all__ = "Launcher"


def register_launcher(name):
    def register_launcher(cls):
        if name in LAUNCHER_REGISTRY:
            raise ValueError("Cannot register duplicate Launcher ({})".format(name))
        LAUNCHER_REGISTRY[name] = cls

        return cls

    return register_launcher


def import_launcher(launcher_dir, namespace):
    for file in os.listdir(launcher_dir):
        path = os.path.join(launcher_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            launcher_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + launcher_name)


# automatically import any Python files in the launcher/ directory
launcher_dir = os.path.dirname(__file__)
import_launcher(launcher_dir, "src.launcher")
