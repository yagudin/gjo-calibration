import json

import streamlit as st
from google.cloud import firestore
from google.oauth2 import service_account

from gjo_requests import request_forecasts, request_resolutions

firestore_info = json.loads(st.secrets["firestore_info"])
credentials = service_account.Credentials.from_service_account_info(firestore_info)
db = firestore.Client(credentials=credentials, project="gjo-calibration")


def get_forecasts(uid, questions, platform_url, headers, cookies):
    platform = "gjo" if platform_url == "https://www.gjopen.com" else "cset"

    db_forecasts = db.collection(f"users_{platform}").document(uid).get().to_dict()
    db_forecasts = dict() if db_forecasts is None else db_forecasts

    missing_forecasts_qs = list(set(questions) - set(db_forecasts))
    missing_forecasts = request_forecasts(
        uid, missing_forecasts_qs, platform_url, headers, cookies
    )

    if missing_forecasts:
        if not db_forecasts:
            db.collection(f"users_{platform}").add({}, uid)
        db.collection(f"users_{platform}").document(uid).update(missing_forecasts)

    return {key: value for key, value in {**db_forecasts, **missing_forecasts}.items() if key in questions}


def get_resolutions(questions, platform_url, headers, cookies):
    platform = "gjo" if platform_url == "https://www.gjopen.com" else "cset"

    db_resolutions = db.collection(f"questions_{platform}").document("resolutions").get().to_dict()
    db_resolutions = dict() if db_resolutions is None else db_resolutions
    relevant_resolutions = {
        key: value for key, value in db_resolutions.items() if key in set(questions)
    }

    missing_resolutions_qs = list(set(questions) - set(relevant_resolutions))
    missing_resolutions = request_resolutions(
        missing_resolutions_qs, platform_url, headers, cookies
    )

    if missing_resolutions:
        db.collection(f"questions_{platform}").document("resolutions").update(missing_resolutions)

    return {**relevant_resolutions, **missing_resolutions}
