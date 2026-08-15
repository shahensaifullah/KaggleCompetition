from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OrdinalEncoder
)


def build_preprocessor(
    # Missing-value settings
    numeric_strategy="median",
    categorical_strategy="constant",
    numeric_fill_value=0,
    categorical_fill_value="Unknown",
    add_missing_indicator=True,

    # Numeric scaling settings
    scaling="standard",

    # Category encoding settings
    encoding="onehot",

    # Column rules
    drop_columns=None,
    remainder="drop",

    # One-hot encoding settings
    min_frequency=None,
    max_categories=None
):
    """
    Create a reusable scikit-learn preprocessor for pandas DataFrames.

    Parameters
    ----------
    numeric_strategy : str
        "mean", "median", "most_frequent", or "constant"

    categorical_strategy : str
        "most_frequent" or "constant"

    numeric_fill_value : int or float
        Used only when numeric_strategy="constant".

    categorical_fill_value : str
        Used only when categorical_strategy="constant".

    add_missing_indicator : bool
        If True, creates extra 0/1 columns marking values that were NaN.

    scaling : str or None
        "standard" -> mean 0, standard deviation 1
        "minmax"   -> scales values between 0 and 1
        "robust"   -> better when numeric columns contain outliers
        None       -> no scaling

    encoding : str
        "onehot" -> creates separate 0/1 columns for categories
        "ordinal" -> converts categories to integer codes

    drop_columns : list[str] or None
        Columns to exclude, such as IDs, names, target leakage columns,
        free-text fields, or high-cardinality identifiers.

    remainder : str
        "drop" or "passthrough".
        Usually keep "drop" when automatic selectors handle every column.

    min_frequency : int, float, or None
        For one-hot encoding: groups rare categories together.
        Example: min_frequency=10 means categories occurring fewer than
        10 times become an infrequent category.

    max_categories : int or None
        Limits one-hot categories per feature.
    """

    drop_columns = drop_columns or []

    # Detect data type dynamically, but exclude unwanted columns.
    def numeric_selector(dataframe):
        cols = dataframe.select_dtypes(include="number").columns
        return [col for col in cols if col not in drop_columns]

    def categorical_selector(dataframe):
        cols = dataframe.select_dtypes(
            include=["object", "category", "bool", "string"]
        ).columns
        return [col for col in cols if col not in drop_columns]

    # ----- Numeric: impute -> optional scale -----
    numeric_imputer_args = {
        "strategy": numeric_strategy,
        "add_indicator": add_missing_indicator
    }

    if numeric_strategy == "constant":
        numeric_imputer_args["fill_value"] = numeric_fill_value

    numeric_steps = [
        ("imputer", SimpleImputer(**numeric_imputer_args))
    ]

    scaler_map = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler()
    }

    if scaling is not None:
        if scaling not in scaler_map:
            raise ValueError(
                "scaling must be 'standard', 'minmax', 'robust', or None."
            )

        numeric_steps.append(
            ("scaler", scaler_map[scaling])
        )

    numeric_pipeline = Pipeline(steps=numeric_steps)

    # ----- Categorical: impute -> encode -----
    categorical_imputer_args = {
        "strategy": categorical_strategy
    }

    if categorical_strategy == "constant":
        categorical_imputer_args["fill_value"] = categorical_fill_value

    if encoding == "onehot":
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=min_frequency,
            max_categories=max_categories
        )

    elif encoding == "ordinal":
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )

    else:
        raise ValueError(
            "encoding must be 'onehot' or 'ordinal'."
        )

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(**categorical_imputer_args)),
        ("encoder", encoder)
    ])

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_selector),
            ("categorical", categorical_pipeline, categorical_selector)
        ],
        remainder=remainder
    )

"""
at the end
create a model 
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

and create a pipeline to merge model and the preprocessors
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

now train the model using fit.
pipeline.fit(X_train, y_train)
"""