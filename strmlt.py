import numpy as np
import pandas as pd
import streamlit as st
import uncurl
from calibration import overconfidence
from firebase_requests import get_forecasts, get_resolutions
from gjo_requests import get_resolved_questions
from plotting import plotly_calibration, plotly_calibration_odds

if __name__ == "__main__":
    st.set_page_config(page_title="How calibrated are you?", page_icon="🦊")
    st.title("🦊 How calibrated are you?")

    # ---

    # if st.checkbox('I am new! Show me instructions.'):
    #     st.write("""
    #         Hey!
    #     """)

    # ---

    platform = st.selectbox(
        "Which platform are you using?",
        ["Good Judgement Open", "CSET Foretell"],
    )
    platform_url = {
        "Good Judgement Open": "https://www.gjopen.com",
        "CSET Foretell": "https://www.cset-foretell.com",
    }[platform]

    uid = st.number_input("What is your user ID?", min_value=1, value=28899)
    uid = str(uid)

    curl_value = """curl 'https://www.gjopen.com/' \\
  -H 'authority: www.gjopen.com' \\
  -H 'cache-control: max-age=0' \\
  -H 'sec-ch-ua: " Not A;Brand";v="99", "Chromium";v="90", "Google Chrome";v="90"' \\
  -H 'sec-ch-ua-mobile: ?0' \\
  -H 'dnt: 1' \\
  -H 'upgrade-insecure-requests: 1' \
  -H 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36' \
  -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9' \
  -H 'sec-fetch-site: none' \\
  -H 'sec-fetch-mode: navigate' \\
  -H 'sec-fetch-user: ?1' \\
  -H 'sec-fetch-dest: document' \
  -H 'accept-language: en-US,en;q=0.9,ru;q=0.8' \\
  -H 'cookie: a-very-long-mysterious-string' \\
  --compressed"""
    curl_command = st.text_area(
        "Ugh... Gimme your cURL info...", value=curl_value
    )
    curl_command = "".join(curl_command.split("\\\n"))
    
    if curl_command != curl_value:

        curl_content = uncurl.parse_context(curl_command)
        headers, cookies = curl_content.headers, curl_content.cookies

        # ---

        questions = get_resolved_questions(uid, platform_url, headers, cookies)

        st.write(f"{len(questions)} questions you forecasted on have resolved.")

        # ---
        # TODO: Make a progress bar..?

        forecasts = get_forecasts(uid, questions, platform_url, headers, cookies)
        resolutions = get_resolutions(questions, platform_url, headers, cookies)

        # ---

        num_forecasts = sum(len(f) for f in forecasts.values())
        st.write(
            f"On these {len(questions)} questions you've made {num_forecasts} forecasts."
        )

        flatten = lambda t: [item for sublist in t for item in sublist]
        y_true = flatten(resolutions[q]["y_true"] for q in questions for _ in forecasts[q])
        y_pred = flatten(f["y_pred"] for q in questions for f in forecasts[q])

        # Note that I am "double counting" each prediction.
        if st.checkbox("Drop last"):
            y_true = flatten(
                resolutions[q]["y_true"][:-1] for q in questions for _ in forecasts[q]
            )
            y_pred = flatten(f["y_pred"][:-1] for q in questions for f in forecasts[q])

        y_true, y_pred = np.array(y_true), np.array(y_pred)

        st.write(f"Which gives us {len(y_pred)} datapoints to work with.")

        # ---

        strategy = st.selectbox(
            "Which binning stranegy do you prefer?",
            ["uniform", "quantile"],
        )

        recommended_n_bins = int(np.sqrt(len(y_pred))) if strategy == "quantile" else 20 + 1
        n_bins = st.number_input(
            "How many bins do you want me to display?",
            min_value=1,
            value=recommended_n_bins,
        )

        fig = plotly_calibration(y_true, y_pred, n_bins=n_bins, strategy=strategy)
        st.plotly_chart(fig, use_container_width=True)

        overconf = overconfidence(y_true, y_pred)
        st.write(f"Your over/under- confidence score is {overconf:.2f}.")

        # ---

        fig = plotly_calibration_odds(y_true, y_pred, n_bins=n_bins, strategy=strategy)
        st.plotly_chart(fig, use_container_width=True)
