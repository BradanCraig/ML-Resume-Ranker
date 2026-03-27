import pandas as pd
from pathlib import Path
import ast
import numpy as np
from datetime import datetime


def parse_list(column: pd.Series):
    """Convert 'string lists' back to list."""
    return column.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)


def compare_skills(resume_skills: list[str] | int, job_skills: list[str] | int):
    """Return number of skills that match the description."""
    # This method will require more levels of language processing
    if resume_skills == "-1" or job_skills == "-1":
        return -1
    resume = set([s.lower() for s in resume_skills])
    job = set(job_skills.lower().split())
    return len([skill for skill in resume if skill in job])


def convert_to_datetime(date_str: str):
    """Converts a stringified list of dates into a list of pandas datetime objects."""

    if not isinstance(date_str, str):
        raise ValueError()

    date_norm = date_str.title().replace(".", "")

    dt = None

    for fmt in ("%b %Y", "%B %Y", "%m/%Y", "%Y", "end"):
        if fmt == "end":
            print(f"date str not handles {date_str}")
            dt = "-1"
        else:
            try:
                dt = pd.to_datetime(date_norm, format=fmt)
                break
            except (ValueError, TypeError):
                continue

    return dt


def convert_dates(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Converts string dates to datetime objects and gets
    amount of time in the field."""
    new_df = df.copy(deep=True)
    # Replace bad values
    current_date_values = [
        "Till Date",
        "Ongoing",
        "Current",
        "Present",
        "∞",
    ]

    # itterate through and convert values
    for i, start in enumerate(df[col_name]):
        start = ast.literal_eval(start)
        if isinstance(start, list):
            new_list = []
            for date in start:
                if date == "-1":
                    continue
                elif date == "N/A" or date == "None" or date is None:
                    new_df.at[i, col_name] = "-1"
                elif date.title() in current_date_values:
                    new_df.at[i, col_name] = datetime.now().strftime("%m %Y")
                else:
                    try:
                        new_list.append(convert_to_datetime(date))
                    except:
                        print(date)
            new_df.at[i, col_name] = new_list
        else:
            print(start)
            break
        # put in a list for consistency

    return new_df


def load_dataset(path: Path):
    df = pd.read_csv(path)
    print(df.columns)
    print(df["start_dates"][:30])
    print(df["end_dates"][:30])
    # Clean Data
    df["skills"] = parse_list(df["skills"])
    df["degree_names"] = parse_list(df["degree_names"])
    df["major_field_of_studies"] = parse_list(df["major_field_of_studies"])
    df["positions"] = parse_list(df["positions"])
    df["certification_skills"] = parse_list(df["certification_skills"])

    # Convert Strings To Floats

    # Replace Nan Values
    df.fillna("-1", inplace=True)
    # this value might be better to change to a different value

    # Add column for the number of matching skills
    matching_skills = [
        compare_skills(
            df.iloc[i]["related_skils_in_job"], df.iloc[i]["skills_required"]
        )
        for i, _ in enumerate(df.iterrows())
    ]
    df["matching_skills"] = matching_skills

    df["start_dates"] = df["start_dates"].astype("object")
    df["end_dates"] = df["end_dates"].astype("object")
    df = convert_dates(df, "start_dates")
    df = convert_dates(df, "end_dates")


def main():
    path = Path("resume_data.csv")
    load_dataset(path)


if __name__ == "__main__":
    main()
