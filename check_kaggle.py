"""Check competition status and submission limits."""
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# Check competition details
try:
    result = api.competition_list_cli(search='rogii-wellbore')
    print("Competition list:", result)
except Exception as e:
    print("list error:", e)

# Check submissions
try:
    subs = api.competitions_submissions_cli(competition='rogii-wellbore-geology-prediction')
    print("\nSubmissions:")
    for s in subs[:5]:
        print(f"  {s}")
except Exception as e:
    print("submissions error:", e)
