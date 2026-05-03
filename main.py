import pandas as pd
from pathlib import Path
import ast
import numpy as np
from datetime import datetime
import re
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


DEGREE_RANK = {
    "high school": 0,
    "associate": 1,
    "bachelor": 2,
    "master": 3,
    "phd": 4,
    "doctorate": 4,
}


DEGREE_KEYWORDS = {
    "high school": ["high school", "ged"],
    "associate": ["associate", "associates"],
    "bachelor": [
        "bachelor",
        "bachelors",
        "b.sc",
        "bs",
        "b.s",
        "ba",
        "b.a",
    ],
    "master": [
        "master",
        "masters",
        "m.sc",
        "ms",
        "m.s",
        "mba",
        "m.a",
    ],
    "phd": ["phd", "doctorate", "ph.d"],
}


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
            # print(f"date str not handles {date_str}")
            dt = "-1"
        else:
            try:
                dt = pd.to_datetime(date_norm, format=fmt)
                break
            except (ValueError, TypeError):
                continue

    return dt


def normalize_text(text):
    """Normalize text for comparisons."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_degree_level(text):
    """Find the highest degree level mentioned in text."""

    text = normalize_text(text)
    highest = -1

    for degree, keywords in DEGREE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                highest = max(highest, DEGREE_RANK[degree])

    return highest


def determine_education_match(resume_degrees, job_requirement):
    """
    Compare resume education against job education requirement.

    Returns:
        1  -> requirement met
        0  -> requirement not met
       -1 -> unknown / missing data
    """

    if resume_degrees == "-1" or job_requirement == "-1":
        return -1

    if not isinstance(resume_degrees, list):
        return -1

    job_level = extract_degree_level(job_requirement)

    if job_level == -1:
        return -1

    highest_resume_level = -1

    for degree in resume_degrees:
        level = extract_degree_level(str(degree))
        highest_resume_level = max(highest_resume_level, level)

    return int(highest_resume_level >= job_level)


def certification_overlap(resume_certs, job_text):
    """
    Counts how many certifications/keywords from the resume
    appear in the job description.
    """

    if resume_certs == "-1" or job_text == "-1":
        return -1

    if not isinstance(resume_certs, list):
        return -1

    job_text = normalize_text(job_text)

    matches = 0

    for cert in resume_certs:
        cert = normalize_text(str(cert))

        if cert and cert in job_text:
            matches += 1

    return matches


COMMON_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "for",
    "with",
    "in",
    "on",
    "of",
    "is",
    "are",
    "be",
    "as",
    "by",
    "at",
    "from",
    "this",
    "that",
}


def tokenize(text):
    """Simple tokenizer."""

    text = normalize_text(text)
    tokens = text.split()

    return [token for token in tokens if token not in COMMON_STOPWORDS]


def keyword_overlap(resume_text, job_text):
    """
    Calculates keyword overlap ratio between resume and job posting.
    """

    if resume_text == "-1" or job_text == "-1":
        return -1

    resume_tokens = set(tokenize(resume_text))
    job_tokens = set(tokenize(job_text))

    if len(job_tokens) == 0:
        return 0

    overlap = resume_tokens.intersection(job_tokens)

    return round(len(overlap) / len(job_tokens), 3)


def extract_top_keywords(text, top_n=10):
    """
    Extract most common keywords from text.
    """

    tokens = tokenize(text)
    counter = Counter(tokens)

    return [word for word, _ in counter.most_common(top_n)]


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
                        pass
            new_df.at[i, col_name] = new_list
        else:
            break
        # put in a list for consistency

    return new_df


def compute_experience(start_list, end_list):
    """Subtract start dates from end dates and return total years of experience."""
    if not isinstance(start_list, list) or not isinstance(end_list, list):
        return 0

    total_days = 0
    for start, end in zip(start_list, end_list):
        if isinstance(start, datetime) and isinstance(end, datetime):
            total_days += (end - start).days

    # convert days to years
    return round(total_days / 365.25, 2)  # accounts for leap years


def compare_job_description(df: pd.DataFrame):

    model = SentenceTransformer("all-MiniLM-L6-v2")
    similarities = []
    for i in tqdm(range(len(df))):
        resume_embed = model.encode(df["career_objective"][i])
        job_embed = model.encode(df["responsibilities"][i])
        similarity = cosine_similarity([resume_embed], [job_embed])[0][0]
        similarities.append(similarity)

    return similarities


def load_dataset(path: Path):
    df = pd.read_csv(path)
    # print(df.columns)
    # print(df["start_dates"][:30])
    # print(df["end_dates"][:30])
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
    df["matching_skills_in_job"] = matching_skills
    df["matching_skills"] = [
        compare_skills(df.iloc[i]["skills"], df.iloc[i]["skills_required"])
        for i, _ in enumerate(df.iterrows())
    ]
    df["start_dates"] = df["start_dates"].astype("object")
    df["end_dates"] = df["end_dates"].astype("object")
    df = convert_dates(df, "start_dates")
    df = convert_dates(df, "end_dates")

    df["experience_years"] = df.apply(
        lambda row: compute_experience(row["start_dates"], row["end_dates"]), axis=1
    )
    print(df.columns)
    # Education Matching
    df["education_match"] = df.apply(
        lambda row: determine_education_match(
            row["degree_names"],
            row["educationaL_requirements"],
        ),
        axis=1,
    )

    # Certification Overlap
    df["certification_overlap"] = df.apply(
        lambda row: certification_overlap(
            row["certification_skills"],
            row["skills_required"],
        ),
        axis=1,
    )

    df["objective_job_similarity"] = compare_job_description(df)

    return df[
        [
            "objective_job_similarity",
            "experience_years",
            "education_match",
            "certification_overlap",
            "matching_skills",
            "matched_score",
        ]
    ].copy()


def main():
    path = Path("resume_data.csv")
    new_df = load_dataset(path)
    print(new_df)
    new_df.to_csv("dataset.csv", index=False)


if __name__ == "__main__":
    main()
