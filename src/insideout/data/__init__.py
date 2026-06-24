"""Data-loading subpackage.

Functions
---------
load_survey_data
    Load a CSV survey file and partition columns by YAML config.
load_fc_data
    Load fMRI functional-connectivity data from a ``.mat`` file.
"""

from .loaders import load_fc_data, load_survey_data

__all__ = ["load_fc_data", "load_survey_data"]

if __name__ == "__main__":
    import doctest

    doctest.testmod()
