# -*- coding: utf-8 -*-
"""
Shared crime-category taxonomy used across the paper scripts
(01_clean_data.py, 02_fit_models.py, 03_forecast_and_evaluate.py).

CATEGORY_ORDER is fixed everywhere so full-city and district outputs
stay comparable. The Stan-facing id mapping is 1-indexed
(CATEGORY_TO_ID); scripts that instead index into 0-based numpy arrays
(e.g. 03_forecast_and_evaluate.py) build their own 0-indexed mapping
locally rather than reusing this one -- the two conventions serve
different consumers and shouldn't be mixed.

Display labels (short names for maps, long names for tables) differ
script to script and are defined locally where used.
"""

CATEGORY_ORDER = [
    "Vehicle_Theft",
    "Vandalism_Disorder",
    "Burglary",
    "Violent_Retaliatory",
]

CATEGORY_TO_ID = {cat: i + 1 for i, cat in enumerate(CATEGORY_ORDER)}
ID_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_ID.items()}
