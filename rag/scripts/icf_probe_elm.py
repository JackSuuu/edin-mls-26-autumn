"""Check ELM authentication without generating an answer or printing credentials."""
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def main():
    config = {}
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                config[key.strip()] = value.strip().strip("\"'")
    key = os.getenv("ELM_API_KEY") or config.get("ELM_API_KEY")
    if not key:
        raise SystemExit("ELM key is not configured")
    base = os.getenv("ELM_BASE_URL") or config.get("ELM_BASE_URL", "https://elm.edina.ac.uk/api/v1")
    model = os.getenv("ELM_MODEL") or config.get("ELM_MODEL", "utter-project/EuroLLM-22B-Instruct-2512")
    request = Request(base.rstrip("/") + "/models", headers={"Authorization": "Bearer " + key})
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
            models = {item.get("id") for item in payload.get("data", [])}
            print("ELM models HTTP:", response.status)
            print("Course model listed:", model in models)
    except HTTPError as exc:
        raise SystemExit("ELM probe HTTP status: " + str(exc.code)) from None
    except URLError:
        raise SystemExit("ELM probe could not connect") from None


if __name__ == "__main__":
    main()
