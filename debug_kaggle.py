"""Debug Kaggle submission."""
import kaggle
from kaggle.api import kaggle_api_extended
import inspect

# Check what's available in kaggle API
print(dir(kaggle_api_extended))
api_class = [x for x in dir(kaggle_api_extended) if 'api' in x.lower() or 'Api' in x]
print("API classes:", api_class)
