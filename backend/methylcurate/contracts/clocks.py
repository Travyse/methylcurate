__all__ = []

from pydantic import BaseModel
from typing import Literal, Optional, List

# Call pyaging here to get the clock names. Filter by metadata for humans and methylation. Then remove any mention of CpGPT

MethylationClocks = Literal[
    "altumage",
    "dunedinpace",
    "dnamic",
    "dnamphenoage",
    "grimage",
    "grimage2",
    "horvath2013",
    "hannum",
    "intrinclock",
    "pcgrimage",
    "pchannum",
    "pchorvath2013",
    "pcphenoage",
    "pcskinandblood",
    "skinandblood",
    "systemsage",
    "systemsageblood",
    "corticalage",
    "pcbrainage",
    "systemsagebrain",
    "systemsageheart",
    "systemsagehormone",
    "systemsageimmune",
    "systemsageinflammation",
    "systemsagekidney",
    "systemsagekidney",
    "systemsageliver",
    "systemsagelung",
    "systemsagemetabolic",
    "systemsagemusculoskeletal",
    "zhangblup",
    "zhangen",
    "zhangmortality",
]


class MethylationAgingClock(BaseModel):
    """
    Model representing a methylation aging clock, including its name and any relevant metadata.

    Attributes:
        clock_name (MethylationClocks): The name of the methylation aging clock.
    """

    clock_name: MethylationClocks


class PredictionInput(BaseModel):
    """
    Prediction input model for methylation aging clock predictions, including a list of clocks to predict and the imputation strategy for missing data.

    Attributes:
        clock_list (List[MethylationAgingClock]): A list of methylation aging clocks to predict.
        imputer_strategy (Literal["mean", "median", "constant", "knn"]): The strategy to use for imputing missing data, with a default value of "knn".
    """

    clock_list: List[MethylationAgingClock]
    imputer_strategy: Literal["mean", "median", "constant", "knn"] = "knn"


class ClockPredictionResult(BaseModel):
    """
    Prediction result model for a methylation aging clock, including the clock name and various performance metrics.

    Attributes:
        clock_name (MethylationClocks): The name of the methylation aging clock.
        mean_squared_error (float): The mean squared error of the predictions for this clock.
        median_absolute_error (float): The median absolute error of the predictions for this clock.
        mean_absolute_error (float): The mean absolute error of the predictions for this clock.
        r_squared (float): The R-squared value of the predictions for this clock.
    """

    clock_name: MethylationClocks
    mean_squared_error: float
    median_absolute_error: float
    mean_absolute_error: float
    r_squared: float


class PredictionResult(BaseModel):
    """
    Prediction result model for multiple methylation aging clocks, including a list of individual clock prediction results.

    Attributes:
        clock_results (List[ClockPredictionResult]): A list of prediction results for each methylation aging clock.
    """

    clock_results: List[ClockPredictionResult]
