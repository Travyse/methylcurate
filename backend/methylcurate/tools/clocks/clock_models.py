__all__ = ["CorticalAge", "PCBrainAge"]

import pickle

import numpy as np
import pandas as pd


def _impute_clock_data(required_cpgs, dnam, default_imputation, user_imputation=None):
    """Align and impute missing CpG values for a clock model.

    Args:
        required_cpgs: List of CpG identifiers required by the model.
        dnam: Methylation DataFrame (samples × probes).
        default_imputation: Reference mean values for CpGs when no user
            imputation is provided.
        user_imputation: Optional user-provided CpG mean values.

    Returns:
        NumPy float32 array with no NaN values.

    Raises:
        ValueError: If required CpGs are missing and no imputation
            reference is available.
    """
    X = dnam.reindex(columns=required_cpgs).to_numpy(dtype=np.float32, copy=True)

    if user_imputation is not None:
        imp = pd.Series(user_imputation.iloc[0]).reindex(required_cpgs).to_numpy(dtype=np.float32)
    elif default_imputation is not None:
        imp = pd.Series(default_imputation.iloc[0]).reindex(required_cpgs).to_numpy(dtype=np.float32)
    else:
        imp = None
    mask = np.isnan(X)
    if mask.any():
        col_means = np.nanmean(X, axis=0)

        if imp is None and np.isnan(col_means).any():
            raise ValueError("Missing required CpGs and no imputation reference was provided.")

        fill_vals = col_means if imp is None else np.where(np.isnan(col_means), imp, col_means)

        if np.isnan(fill_vals).any():
            raise ValueError("Imputation reference is missing required CpG values.")

        X[mask] = fill_vals[np.where(mask)[1]]

    return X


class CorticalAge:
    def __init__(self, coefs, intercept=0.577682570446177, default_imputation=None):
        """
        Initializes the CorticalAge model with necessary weights and parameters.

        :param coefs: pd.DataFrame (Model coefficients)
        :param intercept: float (Model intercept)
        :param default_imputation: pd.Series, optional (Reference mean values for CpGs)
        """
        self.coefs = coefs
        self.intercept = intercept
        self.default_imputation = default_imputation
        self.required_cpgs = self.coefs.index.tolist()

    def impute_data(self, dnam, user_imputation=None):
        """Align and impute missing CpG values.

        Delegates to the shared ``_impute_clock_data`` utility.

        Args:
            dnam: Methylation DataFrame (samples × probes).
            user_imputation: Optional user-provided CpG mean values.

        Returns:
            NumPy float32 array with no NaN values.
        """
        return _impute_clock_data(
            self.required_cpgs,
            dnam,
            self.default_imputation,
            user_imputation=user_imputation,
        )

    def predict(self, dnam, pheno=None, user_imputation=None):
        """
        Runs the PCBrainAge prediction pipeline.

        :param dnam: pd.DataFrame (Samples as rows, CpGs as columns)
        :param pheno: pd.DataFrame, optional (Phenotype data to append results to)
        :param user_imputation: dict/Series, optional (Custom imputation values)
        """

        def _anti_transformation(pred, adult_age=20):
            pred = np.asarray(pred)
            return np.where(pred < 0, (1 + adult_age) * np.exp(pred) - 1, (1 + adult_age) * pred + adult_age)

        # Step 1: Impute and Align
        clean_dnam = self.impute_data(dnam, user_imputation=user_imputation)
        # Step 2: Matrix Math
        prediction = ((clean_dnam @ self.coefs.values) + self.intercept).squeeze()
        prediction = _anti_transformation(prediction)
        # Step 3: Return logic
        if pheno is not None:
            # Ensure we don't modify the original dataframe passed in
            pheno_copy = pheno.copy()
            pheno_copy["corticalage"] = np.ravel(prediction)
            return pheno_copy

        return pd.Series(np.ravel(prediction), index=dnam.index, name="corticalage")

    def dump_state(self, path):
        """
        Saves the model state to a file.

        :param path: str (File path to save the model)
        """
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "coefs": self.coefs,
                    "intercept": self.intercept,
                    "default_imputation": self.default_imputation,
                    "required_cpgs": self.required_cpgs,
                },
                f,
            )

    @classmethod
    def load_state(cls, path):
        """
        Loads the model state from a file.

        :param path: str (File path to load the model from)
        """
        state = None
        with open(path, "rb") as f:
            state = pickle.load(f)
        return cls(state["coefs"], intercept=state["intercept"], default_imputation=state["default_imputation"])


class PCBrainAge:
    def __init__(self, rotation, center, coefs, intercept, default_imputation=None):
        """
        Initializes the PCBrainAge model with necessary weights and parameters.

        :param rotation: pd.DataFrame (Rows: CpGs, Cols: PCs)
        :param center: pd.Series or np.array (Mean values for centering)
        :param coefs: np.array (Model coefficients)
        :param intercept: float (Model intercept)
        :param default_imputation: pd.Series, optional (Reference mean values for CpGs)
        """
        self.rotation = rotation
        self.center = center
        self.coefs = coefs
        self.intercept = intercept
        self.default_imputation = default_imputation
        self.required_cpgs = rotation.index.tolist()

    def impute_data(self, dnam, user_imputation=None):
        """Align and impute missing CpG values.

        Delegates to the shared ``_impute_clock_data`` utility.

        Args:
            dnam: Methylation DataFrame (samples × probes).
            user_imputation: Optional user-provided CpG mean values.

        Returns:
            NumPy float32 array with no NaN values.
        """
        return _impute_clock_data(
            self.required_cpgs,
            dnam,
            self.default_imputation,
            user_imputation=user_imputation,
        )

    def predict(self, dnam, pheno=None, user_imputation=None):
        """
        Runs the PCBrainAge prediction pipeline.

        :param dnam: pd.DataFrame (Samples as rows, CpGs as columns)
        :param pheno: pd.DataFrame, optional (Phenotype data to append results to)
        :param user_imputation: dict/Series, optional (Custom imputation values)
        """
        # Step 1: Impute and Align
        clean_dnam = self.impute_data(dnam, user_imputation=user_imputation)
        # Step 2: Matrix Math
        # Subtract centers -> Matrix Multiply by Rotation -> Multiply by Coefs -> Add Intercept
        centered_matrix = clean_dnam - np.array(self.center)
        pc_scores = centered_matrix @ self.rotation.values
        prediction = (pc_scores @ self.coefs.T + self.intercept).squeeze()

        # Step 3: Return logic
        if pheno is not None:
            # Ensure we don't modify the original dataframe passed in
            pheno_copy = pheno.copy()
            pheno_copy["pcbrainage"] = np.ravel(prediction)
            return pheno_copy

        return pd.Series(np.ravel(prediction), index=dnam.index, name="pcbrainage")

    def dump_state(self, path):
        """
        Saves the model state to a file.

        :param path: str (File path to save the model)
        """
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "center": self.center,
                    "rotation": self.rotation,
                    "coefs": self.coefs,
                    "intercept": self.intercept,
                    "default_imputation": self.default_imputation,
                    "required_cpgs": self.required_cpgs,
                },
                f,
            )

    @classmethod
    def load_state(cls, path):
        """
        Loads the model state from a file.

        :param path: str (File path to load the model from)
        """
        state = None
        with open(path, "rb") as f:
            state = pickle.load(f)
        return cls(
            state["rotation"],
            state["center"],
            state["coefs"],
            state["intercept"],
            default_imputation=state["default_imputation"],
        )
