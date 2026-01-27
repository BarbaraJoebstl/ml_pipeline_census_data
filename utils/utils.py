import re


def normalize_column_name(col_name: str) -> str:
    """
    Converts CSV column names to valid Python identifiers:
    - Lowercases everything
    - Replaces non-alphanumeric characters (like '-', ' ') with '_'
    """
    col_name = col_name.strip().lower()  # lowercase & strip spaces
    col_name = re.sub(r"[^0-9a-zA-Z]+", "_", col_name)  # replace non-alphanumeric with '_'
    col_name = re.sub(r"_+", "_", col_name)  # collapse multiple underscores
    col_name = col_name.strip("_")  # remove leading/trailing underscores
    return col_name
