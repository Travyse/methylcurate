__all__ = ["detect_data_type", "convert_data_type"]
import pandas as pd
import numpy as np
from typing import List, Tuple
from ...contracts.qc import PreprocessDataInput, PreprocessDataResult, PreprocessClippingInput


def detect_data_type(data_matrix: pd.DataFrame) -> str:  # TODO Fix the input
    """Detect whether methylation data is in beta-value or M-value representation.

    Evaluates the proportion of non-NA values that fall within [0, 1].
    If more than 95% are in that range the data is classified as "beta";
    otherwise it is classified as "m".

    Args:
        data_matrix: Methylation data frame (samples × probes).

    Returns:
        ``"beta"`` if >95% of values lie in [0, 1], otherwise ``"m"``.
    """
    non_na = data_matrix.notna()
    in_range = (data_matrix >= 0) & (data_matrix <= 1)

    result = (in_range & non_na).sum().sum() / non_na.sum().sum()

    print(
        f"Total values: {data_matrix.size}, Non-NA values: {non_na.sum().sum()}, Values in [0,1]: {(in_range & non_na).sum().sum()}"
    )
    print(f"Proportion of values that fall within beta range: {result}")
    if result > 0.95:
        return "beta"
    else:
        return "m"


def _convert_beta_to_m(beta_matrix: np.ndarray, clipping: PreprocessClippingInput = None) -> np.ndarray:
    """Convert beta-values to M-values, optionally clipping extreme values first."""
    if clipping is not None:
        lower = clipping.lower_bound if clipping.lower_bound is not None else 0.001
        upper = clipping.upper_bound if clipping.upper_bound is not None else 0.999
        clipped_data = np.clip(beta_matrix, lower, upper)
    else:
        clipped_data = beta_matrix
    beta_values = np.log2(clipped_data / (1 - clipped_data))
    return beta_values


def _convert_m_to_beta(m_matrix: np.ndarray) -> np.ndarray:
    """Convert M-values to beta-values via the logistic function."""
    beta_values = 1 / (1 + np.power(2, -m_matrix))
    return beta_values


def convert_data_type(
    data_conversion_input: PreprocessDataInput, data_df: pd.DataFrame
) -> Tuple[PreprocessDataResult, pd.DataFrame]:
    """Convert methylation data between beta-value and M-value representations.

    When the source and target types differ the appropriate conversion
    function is applied and a new DataFrame is returned.  Clipping is
    forwarded to the underlying conversion helper.

    Args:
        data_conversion_input: Conversion specification (from/to types
            and optional clipping bounds).
        data_df: Methylation data frame (samples × probes).

    Returns:
        A tuple of (conversion result, converted DataFrame).

    Raises:
        ValueError: If the requested conversion pair is unsupported.
    """
    if data_conversion_input.from_type == data_conversion_input.to_type:
        return PreprocessDataResult(data_type=data_conversion_input.to_type), data_df

    if data_conversion_input.from_type == "beta" and data_conversion_input.to_type == "m":
        converted_array = _convert_beta_to_m(data_df.to_numpy(), data_conversion_input.clipping)
    elif data_conversion_input.from_type == "m" and data_conversion_input.to_type == "beta":
        converted_array = _convert_m_to_beta(data_df.to_numpy())
    else:
        raise ValueError(
            f"Unsupported conversion from {data_conversion_input.from_type} to {data_conversion_input.to_type}"
        )

    converted_df = pd.DataFrame(converted_array, index=data_df.index, columns=data_df.columns)
    return PreprocessDataResult(data_type=data_conversion_input.to_type), converted_df
