__all__ = ["detect_data_type", "convert_data_type"]

import numpy as np
import pandas as pd

from ...contracts.qc import PreprocessClippingInput, PreprocessDataInput, PreprocessDataResult


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

    print(f"Total values: {data_matrix.size}, Non-NA values: {non_na.sum().sum()}, Values in [0,1]: {(in_range & non_na).sum().sum()}")
    print(f"Proportion of values that fall within beta range: {result}")
    if result > 0.95:
        return "beta"
    else:
        return "m"


def _convert_beta_to_m(beta_matrix: np.ndarray, clipping: PreprocessClippingInput | None = None) -> np.ndarray:
    if clipping is not None:
        lower = clipping.lower_bound if clipping.lower_bound is not None else 0.001
        upper = clipping.upper_bound if clipping.upper_bound is not None else 0.999
        np.clip(beta_matrix, lower, upper, out=beta_matrix)
    return np.log2(beta_matrix / (1.0 - beta_matrix))


def _convert_m_to_beta(m_matrix: np.ndarray) -> np.ndarray:
    np.negative(m_matrix, out=m_matrix)
    np.power(2.0, m_matrix, out=m_matrix)
    return 1.0 / (1.0 + m_matrix)


def convert_data_type(data_conversion_input: PreprocessDataInput, data_df: pd.DataFrame) -> tuple[PreprocessDataResult, pd.DataFrame]:
    if data_conversion_input.from_type == data_conversion_input.to_type:
        return PreprocessDataResult(data_type=data_conversion_input.to_type), data_df

    idx = data_df.index
    cols = data_df.columns
    arr = data_df.to_numpy(dtype=np.float32, copy=True)

    if data_conversion_input.from_type == "beta" and data_conversion_input.to_type == "m":
        converted = _convert_beta_to_m(arr, data_conversion_input.clipping)
    elif data_conversion_input.from_type == "m" and data_conversion_input.to_type == "beta":
        converted = _convert_m_to_beta(arr)
    else:
        raise ValueError(f"Unsupported conversion from {data_conversion_input.from_type} to {data_conversion_input.to_type}")

    return PreprocessDataResult(data_type=data_conversion_input.to_type), pd.DataFrame(converted, index=idx, columns=cols)
