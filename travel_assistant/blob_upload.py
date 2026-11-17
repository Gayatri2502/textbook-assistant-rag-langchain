"""Short-lived, path-scoped grants for private textbook uploads."""
import base64
import hashlib
import hmac
import json
import os
import time
from urllib.request import Request, urlopen
from uuid import uuid4


def token():
    return os.getenv("BLOB_READ_WRITE_TOKEN", "")


def new_upload(limit):
    secret = token()
    path = f"textbooks/{uuid4().hex}.pdf"
    options = {
        "pathname": path,
        "maximumSizeInBytes": limit,
        "allowedContentTypes": ["application/pdf"],
        "addRandomSuffix": False,
        "allowOverwrite": False,
        "validUntil": int(time.time() * 1000) + 30 * 60 * 1000,
        "ttlDays": 1,
    }
    payload = base64.b64encode(json.dumps(options).encode())
    signature = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    signed = base64.b64encode(signature.encode() + b"." + payload).decode()
    return {"path": path, "token": f"vercel_blob_client_{secret.split('_')[3]}_{signed}"}


def open_pdf(path):
    secret = token()
    store = secret.split("_")[3]
    url = f"https://{store.lower()}.private.blob.vercel-storage.com/{path}"
    return urlopen(Request(url, headers={"Authorization": f"Bearer {secret}"}), timeout=60)
