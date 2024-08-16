from functools import partial
import logging
import warnings
from typing import Callable, List, TypeVar, Any
import hydra
from omegaconf import DictConfig
from lightning import Callback
from lightning.pytorch.loggers import Logger

log = logging.getLogger(__name__)

T = TypeVar("T", Logger, Callback)


def multi_instantiate(
    cfg: DictConfig, context: dict[str, Any] | None = None
) -> List[T]:
    """Instantiates multiple classes from config.

    Instantiates multiple classes from a config object. The configuration object
    has the following structure: dict[submodule_name: str, conf: DictConfig]. The
    conf contains the configuration for the submodule and adheres to one of the
    following schema:

    1. dict[_target_: str, ...] where _target_ key specifies the class to be
         instantiated and the rest of the keys are the arguments to be passed to
         the class constructor (usual hydra config schema).

    2. dict[conf: DictConfig, dependencies: dict[str, str]] where conf contains
         the configuration for the submodule as defined in (1) missing requested
         dependencies and dependencies is a mapping from the class arguments to
         the keys in the context dictionary.

    3. dict[conf: Callable, dependencies: DictConfig] where conf is a callable
         acts as a factory function for the class and dependencies is a mapping
         from the class arguments to the keys in the context dictionary.

    Args:
        cfg: A DictConfig object containing configurations.
        context: A dictionary containing dependencies that the individual
          classes might request from.

    Returns:
        A list of instantiated classes.

    Raises:
        TypeError: If the config is not a DictConfig.
    """

    instances: List[T] = []

    if not cfg:
        warnings.warn("No configs found! Skipping...")
        return instances

    if not isinstance(cfg, DictConfig):
        raise TypeError("Config must be a DictConfig!")

    for _, conf in cfg.items():
        target_name = conf.__class__.__name__
        instance_dependencies = {}
        # Resolve dependencies if requested
        if isinstance(conf, DictConfig) and "dependencies" in conf:
            if "conf" not in conf:
                raise TypeError(
                    "Invalid config schema. If dependencies are requested, then "
                    "the config must contain a 'conf' section specifying the "
                    "class arguments not included in the dependencies."
                )
            # Resolve dependencies if any specified
            if conf.dependencies:
                if context:
                    instance_dependencies = {
                        kwarg: context[key] for kwarg, key in conf.dependencies.items()
                    }
                else:
                    raise ValueError(
                        "The config requests dependencies, but none were provided."
                    )
            conf: DictConfig | Callable = conf.conf
            # pylint: disable-next=protected-access
            target_name = conf.func.__name__ if callable(conf) else conf._target_

        log.info("Instantiating an object of type <%s>", target_name)
        if isinstance(conf, partial):
            instances.append(conf(**instance_dependencies))
        elif isinstance(conf, DictConfig):
            if "_target_" in conf:
                instances.append(
                    hydra.utils.instantiate(
                        conf,
                        **instance_dependencies,
                        _recursive_=False,
                    )
                )
            else:
                raise ValueError(
                    f"Invalid config schema ({conf}). The config must contain "
                    "a '_target_' key specifying the class to be instantiated."
                )
        # Object has already been instantiated
        else:
            instances.append(conf)

    return instances


def instantiate_with_context(
    cfg: DictConfig,
    context: dict[str, Any] | None = None,
) -> Any:
    if not cfg:
        warnings.warn("No configs found! Skipping...")
        return None

    if not isinstance(cfg, DictConfig):
        raise TypeError("Config must be a DictConfig!")

    if cfg.dependencies:
        if context:
            dependencies = {
                kwarg: context[key] for kwarg, key in cfg.dependencies.items()
            }
        else:
            raise ValueError(
                "The config requests dependencies, but none were provided."
            )
    else:
        dependencies = {}

    return cfg.conf(**dependencies)
