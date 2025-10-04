import re
import pandas as pd

def uncamelcase_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert camelCase column names to snake_case."""
    df.columns = [re.sub(r'(?<!^)(?=[A-Z])', '_', word).lower() for word in df.columns]
    return df