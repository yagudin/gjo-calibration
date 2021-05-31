import asyncio
import logging
import re
from itertools import count

import aiohttp
import aioitertools
import requests
import streamlit as st
from bs4 import BeautifulSoup

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)


@st.cache
def get_resolved_questions(uid, platform_url, headers, cookies):
    logging.info(
        f"[ ] get_resolved_questions for uid={uid}, platform_url={platform_url}"
    )

    questions = []  # [question_id]

    for page_num in count(1):
        url = f"{platform_url}/memberships/{uid}/scores/?page={page_num}"
        page = requests.get(url, headers=headers, cookies=cookies).text

        extracted_qs = re.findall("/questions/(\d+)", page)
        questions.extend(extracted_qs)

        if not extracted_qs:
            break

    logging.info(
        f"[X] get_resolved_questions for uid={uid}, platform_url={platform_url}"
    )

    return questions


async def get_question_resolution(qid, platform_url, session):
    logging.info(
        f"[ ] get_question_resolution for qid={qid}, platform_url={platform_url}"
    )

    url = f"{platform_url}/questions/{qid}"

    async with session.get(url) as resp:
        if resp.status != 200:
            logging.error(
                f"get_question_resolution for uid={uid}, platform_url={platform_url} | "
                f"resp.status == {resp.status} → {resp.reason}"
            )

        page = await resp.text()

        soup = BeautifulSoup(page, "html.parser")
        soup = soup.find_all("div", {"id": "prediction-interface-container"})[0]

        binary = soup.find_all("div", {"class": "binary-probability-value"})
        if binary:
            y_true = (0, 1) if re.search("Yes", binary[1].text) is None else (1, 0)
        else:
            tables = soup.find_all("table")
            y_true = tuple(len(tr.findAll("i")) for tr in tables[0].findAll("tr")[1:])

        logging.info(
            f"[X] get_question_resolution for uid={uid}, platform_url={platform_url}"
        )
        return {"y_true": y_true}


def _extract_forecasts_from_page(page):
    soup = BeautifulSoup(page, "html.parser")
    soup_predictions = soup.find_all("div", {"class": "prediction-values"})
    predictions = [re.findall("\n\s*(\d+)%", p_tag.text) for p_tag in soup_predictions]
    predictions = [tuple(int(prob) / 100 for prob in pred) for pred in predictions]
    predictions = [
        (pred[0], 1 - pred[0]) if len(pred) == 1 else pred for pred in predictions
    ]

    # I search for a line containing "made a forecast"
    # I search for the next line containig <span data-localizable-timestamp="[^"]*">
    # And graab a timestamp from it
    timestamps = []
    looking_for_a_forecast = True
    for line in page.split("\n"):
        if looking_for_a_forecast:
            hit = re.findall("made a forecast", line)
            if hit:
                looking_for_a_forecast = False

        else:
            hit = re.findall('<span data-localizable-timestamp="([^"]+)">', line)
            if hit:
                timestamps.extend(hit)
                looking_for_a_forecast = True

    if len(timestamps) != len(predictions):
        logging.error(
            f"In _extract_forecasts_from_page with uid={uid}, qid={qid}, page_num={page_num} "
            f"got different number of predictions ({len(timestamps)}) and timestamps ({len(predictions)})."
        )

    return [
        {"y_pred": pred, "timestamp": timestamp}
        for pred, timestamp in zip(predictions, timestamps)
    ]


async def get_forecasts_on_the_question(uid, qid, platform_url, session):
    logging.info(
        f"[ ] get_forecasts_on_the_question for uid={uid}, qid={qid}, platform_url={platform_url}"
    )

    forecasts = []  # [{"y_pred": (probs, ...), "timestamp": timestamp}, ...]

    for page_num in count(1):
        url = f"{platform_url}/questions/{qid}/prediction_sets?membership_id={uid}&page={page_num}"

        async with session.get(url) as resp:
            if resp.status != 200:
                logging.error(
                    f"get_forecasts_on_the_question for uid={uid}, qid={qid}, platform_url={platform_url} | "
                    f"resp.status == {resp.status} → {resp.reason}"
                )

            page = await resp.text()

            extracted_forecasts = _extract_forecasts_from_page(page)
            forecasts.extend(extracted_forecasts)

            if not extracted_forecasts:
                break

    logging.info(
        f"[X] get_forecasts_on_the_question for uid={uid}, qid={qid}, platform_url={platform_url}"
    )
    return forecasts


# ---


async def async_get_forecasts(uid, questions, platform_url, headers, cookies):
    async with aiohttp.ClientSession(headers=headers, cookies=cookies) as session:
        forecasts_list = await aioitertools.asyncio.gather(
            *[
                get_forecasts_on_the_question(uid, q, platform_url, session)
                for q in questions
            ],
            limit=5,
        )
        return {q: forecasts_list[i] for i, q in enumerate(questions)}


async def async_get_resolutions(questions, platform_url, headers, cookies):
    async with aiohttp.ClientSession(headers=headers, cookies=cookies) as session:
        resolutions_list = await aioitertools.asyncio.gather(
            *[get_question_resolution(q, platform_url, session) for q in questions],
            limit=5,
        )
        return {q: resolutions_list[i] for i, q in enumerate(questions)}


def request_forecasts(uid, missing_forecasts_qs, platform_url, headers, cookies):
    return asyncio.run(
        async_get_forecasts(uid, missing_forecasts_qs, platform_url, headers, cookies)
    )


def request_resolutions(missing_resolutions_qs, platform_url, headers, cookies):
    return asyncio.run(
        async_get_resolutions(missing_resolutions_qs, platform_url, headers, cookies)
    )
