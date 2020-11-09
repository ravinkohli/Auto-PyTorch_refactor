from typing import NamedTuple, Dict, Any


class FitRequirement(NamedTuple):
    name: str
    type: object


def replace_prefix_in_config_dict(config: Dict[str, Any], prefix: str, replace: str = "") -> Dict[str, Any]:
    """
    Replace the prefix in all keys with the specified replacement string (the empty string by
    default to remove the prefix from the key). The functions makes sure that the prefix is a proper config
    prefix by checking if it ends with ":", if not it appends ":" to the prefix.

    :param config: config dictionary where the prefixed of the keys should be replaced
    :param prefix: prefix to be replaced in each key
    :param replace: the string to replace the prefix with
    :return: updated config dictionary
    """
    # make sure that prefix ends with the config separator ":"
    if not prefix.endswith(":"):
        prefix = prefix + ":"
    # only replace first occurrence of the prefix
    return {k.replace(prefix, replace, 1): v
            for k, v in config.items() if
            k.startswith(prefix)}
