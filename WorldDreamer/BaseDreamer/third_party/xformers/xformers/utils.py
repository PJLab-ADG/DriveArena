# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import importlib
import os
import sys
from collections import namedtuple
from dataclasses import fields
from typing import Any, Callable, Dict, List

Item = namedtuple("Item", ["constructor", "config"])


# credit: snippet used in ClassyVision (and probably other places)
def import_all_modules(root: str, base_module: str) -> List[str]:
    modules: List[str] = []
    for file in os.listdir(root):
        if file.endswith((".py", ".pyc")) and not file.startswith("_"):
            module = file[: file.find(".py")]
            if module not in sys.modules:
                module_name = ".".join([base_module, module])
                importlib.import_module(module_name)
                modules.append(module_name)

    return modules


def get_registry_decorator(
    class_registry, name_registry, reference_class, default_config
) -> Callable[[str, Any], Callable[[Any], Any]]:
    def register_item(name: str, config: Any = default_config):
        """Registers a subclass.

        This decorator allows xFormers to instantiate a given subclass
        from a configuration file, even if the class itself is not part of the
        xFormers library."""

        def register_cls(cls):
            if name in class_registry:
                raise ValueError("Cannot register duplicate item ({})".format(name))
            if not issubclass(cls, reference_class):
                raise ValueError(
                    "Item ({}: {}) must extend the base class: {}".format(
                        name, cls.__name__, reference_class.__name__
                    )
                )
            if cls.__name__ in name_registry:
                raise ValueError(
                    "Cannot register item with duplicate class name ({})".format(
                        cls.__name__
                    )
                )

            class_registry[name] = Item(constructor=cls, config=config)
            name_registry.add(cls.__name__)
            return cls

        return register_cls

    return register_item


def generate_matching_config(superset: Dict[str, Any], config_class: Any) -> Any:
    """Given a superset of the inputs and a reference config class,
    return exactly the needed config"""

    # Extract the required fields
    field_names = list(map(lambda x: x.name, fields(config_class)))
    subset = {k: v for k, v in superset.items() if k in field_names}

    # The missing fields get Noned
    for k in field_names:
        if k not in subset.keys():
            subset[k] = None

    return config_class(**subset)
