
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import re
import cv2
import numpy as np
import os
from urllib.parse import urlparse

app = FastAPI(title="URL & QR Phishing Analysis API")

# Load model and feature order
model = joblib.load("url_qr_phishing_model.pkl")
FEATURE_NAMES = joblib.load("url_qr_phishing_features.pkl")


# ---------- URL FEATURE EXTRACTION ----------

SUSPICIOUS_TLDS = {".tk", ".ml", ".ga", ".cf", ".gq"}

URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl",
    "t.co", "ow.ly", "is.gd",
    "buff.ly", "adf.ly"
}


def extract_features(url):
    url = str(url).strip()
    parsed = urlparse(url)

    domain = parsed.netloc.lower().split(":")[0]

    url_length = len(url)

    has_ip_address = int(bool(
        re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", domain)
    ))

    num_dots = url.count(".")
    num_hyphens = url.count("-")
    num_at_symbols = url.count("@")

    has_https = int(parsed.scheme.lower() == "https")

    domain_parts = domain.split(".")
    num_subdomains = max(0, len(domain_parts) - 2)

    has_suspicious_tld = int(
        any(domain.endswith(tld) for tld in SUSPICIOUS_TLDS)
    )

    has_url_shortener = int(domain in URL_SHORTENERS)

    num_special_chars = sum(
        1 for char in url
        if char in "!#$%&'()*+,/:;=?@[]_~"
    )

    has_redirect_pattern = int(
        url.lower().count("http://") > 1
    )

    domain_length = len(domain)

    return [
        url_length,
        has_ip_address,
        num_dots,
        num_hyphens,
        num_at_symbols,
        has_https,
        num_subdomains,
        has_suspicious_tld,
        has_url_shortener,
        num_special_chars,
        has_redirect_pattern,
        domain_length
    ]


def predict_url(url):
    extracted = extract_features(url)

    features_df = pd.DataFrame(
        [extracted],
        columns=FEATURE_NAMES
    )

    prediction = model.predict(features_df)[0]
    probabilities = model.predict_proba(features_df)[0]

    suspicious_index = list(model.classes_).index(1)
    suspicious_probability = probabilities[suspicious_index]

    final_prediction = (
        "SUSPICIOUS" if prediction == 1 else "SAFE"
    )

    return {
        "prediction": final_prediction,
        "suspicious_probability": round(
            float(suspicious_probability), 4
        )
    }


# ---------- URL INPUT ----------

class URLRequest(BaseModel):
    url: str


@app.post("/analyze-url")
def analyze_url(request: URLRequest):
    result = predict_url(request.url)

    return {
        "url": request.url,
        **result
    }


# ---------- QR INPUT ----------

@app.post("/analyze-qr")
async def analyze_qr(file: UploadFile = File(...)):

    contents = await file.read()

    image_array = np.frombuffer(
        contents,
        np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    detector = cv2.QRCodeDetector()

    decoded_data, points, _ = detector.detectAndDecode(image)

    if not decoded_data:
        raise HTTPException(
            status_code=400,
            detail="Could not detect or decode a QR code."
        )

    result = predict_url(decoded_data)

    return {
        "decoded_content": decoded_data,
        **result
    }


@app.get("/")
def home():
    return {
        "message": "URL and QR Phishing Analysis API is running"
    }
