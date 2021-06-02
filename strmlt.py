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

    st.sidebar.header("Welcome!")

    st.sidebar.write("Good calibration is vital for good judgemental forecasting. "
             "When a calibrated forecaster predicts 70% on 10 questions, we actually expect "
             "around 7 of these to resolve positively. Unfortunately, there is "
             "no easy way to see which fraction of our 70% forecasts resolves "
             "positively on Good Judgement Open. Hence, I made this web app.")

    st.sidebar.subheader("On cURL")

    st.sidebar.write("I use your cookies for gathering information from GJO: which questions did you forecast on; what did you forecast on; how did they resolve.")

    st.sidebar.write("I do not use them for other purposes, neither do I store them. The code is on [github](https://github.com/yagudin/gjo-calibration).")

    st.sidebar.write("""
        1. Go to e.g [gjopen.com/questions](https://www.gjopen.com/questions) in a new tab in Chrome or in Firefox.
        2. Press `Ctrl + Shift + I`, and then navigate to the "Network" tab. 
        3. Click on “Reload” or reload the page.
        4. Right click on the first request which loads the "questions" document. Click Copy, then "copy as cURL". Paste the results here.
    """)

    st.sidebar.write("Nuño Sempere made [video instructions](https://www.youtube.com/watch?v=_G3FNzYNPCs) for an earlier version of the web app.")

    # st.sidebar.subheader("On plots and methodology")

    # st.sidebar.write("""
    #     - I generate two calibration curves: one in linear space and another one in 'odds' space (hopefully it will be easier to see how well calibrated you are around probabilities close to 0 and 1).
    #     - I generate plots with a modified [sklearn.calibration.calibration_curve](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html), basically it groups points into bins and computes the proportions of samples resolving positively and the mean predicted probabilities.
    #     - The confidence intervals are a standart deviations wide.
    #     - If you hover over a datapoint you can see precise coordinates (x, y) and number of samples (N) contributing to it.
    # """)

    st.sidebar.subheader("Authorship and acknowledgments")

    st.sidebar.write("This web app was built by [Misha Yagudin](https://twitter.com/mishayagudin). I am grateful to [Nuño Sempere](https://nunosempere.github.io/) for providing feedback.")

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
  -H 'sec-ch-ua: "something-something-about-your-browser"' \\
  -H 'sec-ch-ua-mobile: ?0' \\
  -H 'dnt: 1' \\
  -H 'upgrade-insecure-requests: 1' \
  -H 'user-agent: Mozilla/5.0 something-something-about-your-PC' \
  -H 'accept: text/html...' \
  -H 'sec-fetch-site: none' \\
  -H 'sec-fetch-mode: navigate' \\
  -H 'sec-fetch-user: ?1' \\
  -H 'sec-fetch-dest: document' \\
  -H 'accept-language: en-US,en;q=0.9,ru;q=0.8' \\
  -H 'cookie: a-very-long-mysterious-string' \\
  --compressed"""
    curl_command = st.text_area(
        "Om Nom Nom Nom... Paste cURL here, if confused see the sidebar for the instructions.", value=curl_value
    )
    
    if curl_command == curl_value:
        st.warning('Please input your cURL (see the sidebar for the instructions :-)')
        st.stop()

    try:
        curl_command = curl_command.replace("\\", "")
        curl_content = uncurl.parse_context(curl_command)
        headers, cookies = curl_content.headers, curl_content.cookies
    except SystemExit:
        st.warning("It seems like something is wrong with the cURL you provided: see the sidebar for the instructions.")
        st.stop()

    # ---

    with st.spinner('Loading resolved questions...'):
        questions = get_resolved_questions(uid, platform_url, headers, cookies)

    st.write(f"- {len(questions)} questions you forecasted on have resolved.")

    # ---
    # TODO: Make a progress bar..?

    with st.spinner('Loading your forecasts...'):
        forecasts = get_forecasts(uid, questions, platform_url, headers, cookies)
    
    with st.spinner("Loading questions's resolutions..."):
        resolutions = get_resolutions(questions, platform_url, headers, cookies)

    # ---

    num_forecasts = sum(len(f) for f in forecasts.values())
    st.write(
        f"- You've made {num_forecasts} forecasts on these {len(questions)} questions."
    )

    flatten = lambda t: [item for sublist in t for item in sublist]
    # y_true = flatten(resolutions[q]["y_true"] for q in questions for _ in forecasts[q])
    # y_pred = flatten(f["y_pred"] for q in questions for f in forecasts[q])

    # Note that I am "double counting" each prediction.
    # if st.checkbox("Drop last"):
    y_true = flatten(
        resolutions[q]["y_true"][:-1] for q in questions for _ in forecasts[q]
    )
    y_pred = flatten(f["y_pred"][:-1] for q in questions for f in forecasts[q])

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    order = np.arange(len(y_true))
    np.random.default_rng(0).shuffle(order)
    y_true, y_pred = y_true[order], y_pred[order]


    st.write(f"- Which gives us {len(y_pred)} datapoints to work with.")

    # ---

    strategy_select = st.selectbox(
        "Which binning stranegy do you prefer?",
        [
            "I want bins to have identical widths",
            "I want bins to have the same number of samples",
        ],
    )
    strategy = {
        "I want bins to have identical widths": "uniform",
        "I want bins to have the same number of samples": "quantile",
    }[strategy_select]

    recommended_n_bins = int(np.sqrt(len(y_pred))) if strategy == "quantile" else 20 + 1
    n_bins = st.number_input(
        "How many bins do you want me to display?",
        min_value=1,
        value=recommended_n_bins,
    )

    # ---

    fig = plotly_calibration(y_true, y_pred, n_bins=n_bins, strategy=strategy)
    st.plotly_chart(fig, use_container_width=True)

    fig = plotly_calibration_odds(y_true, y_pred, n_bins=n_bins, strategy=strategy)
    st.plotly_chart(fig, use_container_width=True)

    # overconf = overconfidence(y_true, y_pred)
    # st.write(f"Your over/under- confidence score is {overconf:.2f}.")

    # get_resolutions(list(range(, platform_url, headers, cookies)
