# -*- coding: utf-8 -*-
"""
Shared crime-category taxonomy used across the all scripts.

CATEGORY_ORDER is fixed everywhere so full-city and district outputs stay comparable. The Stan-facing id mapping is 1-indexed
(CATEGORY_TO_ID). Note there is a script that instead index into 0-based numpy arrays (e.g. 05_evaluate_likelihood.py) with 
their own 0-indexed mapping locally rather than reusing this one.

Author: Persia Luca (2026), Università della Svizzera italiana, Lugano, Switzerland
"""

CATEGORY_ORDER = ["Vehicle_Theft", "Vandalism_Disorder", "Burglary", "Violent_Retaliatory"]
CATEGORY_TO_ID = {cat: i + 1 for i, cat in enumerate(CATEGORY_ORDER)}
ID_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_ID.items()}

