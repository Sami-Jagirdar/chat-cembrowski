import logging
import os
from pathlib import Path
import sys

# This is a workaround for now to be able to import from src/ until we package this properly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) 

from src.data.vectordb import (
    get_openai_client,
    get_qdrant_client,
)

from src.retrieval.query_engine import QueryEngine


logging.basicConfig(level=logging.INFO)


def main() -> None:
    qdrant_client = get_qdrant_client()
    openai_client = get_openai_client()

    engine = QueryEngine(
        qdrant_client=qdrant_client,
        openai_client=openai_client,
        top_k=10,
    )

    questions = [
        "Who is the corresponding author of the paper on seasonal variation in hemoglobin A1c across both hemispheres, and what is their email?",
        "In what journal, volume, and issue was the study on seasonal HbA1c variation in both hemispheres published?",
        "Which five locations contributed HbA1c data to the seasonal variation study by Higgins et al.?",
        "In the Higgins et al. seasonal HbA1c study, what HbA1c analyzer was used in Marshfield?",
        "In the Higgins et al. seasonal HbA1c study, what HbA1c method/analyzer was used in Calgary?",
        "In the seasonal HbA1c variation study comparing both hemispheres, what data period was used for Edmonton?",
        "In the seasonal HbA1c variation study comparing both hemispheres, what data period was used for Singapore?",
        "According to the mean monthly maximum temperature table in the Higgins et al. seasonal HbA1c study, what is the January temperature in Edmonton?",
        "According to the mean monthly maximum temperature table in the Higgins et al. seasonal HbA1c study, what is the July temperature in Melbourne?",
        "In the Higgins et al. seasonal HbA1c study, what is the annual range of mean monthly maximum temperatures in Singapore, and what is the magnitude of that variation?",
        "In the Higgins et al. seasonal HbA1c study, what is the temperature variation over the year for Marshfield, Calgary, and Edmonton?",
        "In the Higgins et al. seasonal HbA1c study, what is the difference between the lowest and highest HbA1c mean/median value for each of Marshfield, Edmonton, Melbourne, and Singapore?",
        "In the Higgins et al. seasonal HbA1c study, what is the mean of the means for HbA1c in Singapore and Melbourne, and what was the range?",
        "In the Higgins et al. seasonal HbA1c study, what is the median HbA1c range reported for Calgary?",
        "What HbA1c treatment target do the American and Canadian Diabetes Associations advocate, and what is the more stringent target mentioned in the Higgins et al. seasonal HbA1c study?",
        "What analytical imprecision goal for HbA1c testing has the National Academy of Clinical Biochemistry set, as cited in the Higgins et al. seasonal HbA1c paper?",
        "What is the reported biological variation range for HbA1c cited in the Higgins et al. seasonal HbA1c paper?",
        "According to the Higgins et al. seasonal HbA1c study, in which months are the highest HbA1c values seen in Melbourne, and in which months are they lowest?",
        "According to the Higgins et al. seasonal HbA1c study, in Edmonton and Marshfield, when do higher and lower HbA1c values occur?",
        "In the Higgins et al. seasonal HbA1c study, what p-value was reported when comparing the mean HbA1c data from Singapore and Melbourne?",
        "In the Higgins et al. seasonal HbA1c study, what did the authors hypothesize might explain why Calgary's HbA1c mean of means (6.6%) is lower than the other locations?",
        "What HbA1c difference between summer and winter did Tseng and associates report in East Orange, New Jersey, and what was the temperature difference there?",
        "What did Ishii and colleagues report regarding seasonal HbA1c variation in Japan, and what did they attribute it to?",
        "Whose 1985 study first reported lower HbA1c values in June and July?",
        "What did Garde and colleagues report in Copenhagen that the authors of the Higgins et al. seasonal HbA1c study describe as at variance with the rest of the literature?",
        "In the Higgins et al. seasonal HbA1c study, did the authors track changes in individual patients' HbA1c values over the data period?",
        "In the Higgins et al. seasonal HbA1c study, did the authors observe a clear seasonal variation at all five sites?",
        "In the Higgins et al. seasonal HbA1c study, what tool was used to compute the mean and median HbA1c values?",
    ]

    answers_gt = [
    "Trefor Higgins, at DynaLIFEDx, Edmonton; email trefor.higgins@dynalifedx.com.",
    "Journal of Diabetes Science and Technology, Volume 3, Issue 4, July 2009.",
    "Edmonton (Canada), Calgary (Canada), Singapore, Melbourne (Australia), and Marshfield, Wisconsin.",
    "Tosoh 2+2.",
    "Roche Tina-quant performed on a Roche Integra 700 analyzer.",
    "February 2002 through January 2004.",
    "January 2006 through December 2007.",
    "-8.2 °C.",
    "13.0 °C.",
    "29.6 °C to 31.7 °C, a variation of 2.0 °C.",
    "33.7 °C, 26.8 °C, and 31.2 °C respectively.",
    "0.4 for Marshfield, 0.3 for Edmonton, 0.2 for Melbourne, and 0.1 for Singapore.",
    "7.3%, with a range of 7.1% to 7.5%.",
    "5.9% to 6.3%.",
    "A target of 7% or lower, with ≤6.0% considered if it can be safely achieved.",
    "Ideally less than 3%.",
    "Approximately 1.7% to 3%.",
    "Highest in winter months June through September; lowest December through March.",
    "Higher values November through March; lower values June through October.",
    "p = 0.29 (i.e., the two are not significantly different).",
    "That HbA1c in Calgary is ordered on many individuals without diabetes (used more as a screening test).",
    "A difference of 0.22 in HbA1c, with a temperature difference of 26.2 °C.",
    "A 0.5% difference between winter and spring/autumn in type 2 diabetes patients, attributed to increased caloric intake and decreased physical activity in winter.",
    "Mortensen.",
    "That in healthy women, HbA1c was higher in September and October and lower in December and January (opposite seasonal pattern).",
    "No — the paper explicitly states no attempt was made to track changes in an individual's HbA1c over the data period.",
    "No — clear seasonal variation as described by others was not seen except for the Marshfield data; Calgary showed no seasonal variation over the study period.",
    "Microsoft Excel.",
    ]


    for question_number in range(len(questions)):
        print("\n" + "=" * 80)
        print(f"QUESTION:\n{questions[question_number]}")
        print("=" * 80)

        answer = engine.query(questions[question_number])

        print("\nANSWER:\n")
        print(answer, "\n")

        print(f"GROUND TRUTH ANSWER: \n{answers_gt[question_number]}")
        print()


if __name__ == "__main__":
    main()