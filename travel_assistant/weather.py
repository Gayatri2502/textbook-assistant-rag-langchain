"""On-demand Open-Meteo lookups, with trace boundaries around each API call."""
import httpx

from .trace import Trace


def geocode(client, location):
    response = client.get("https://geocoding-api.open-meteo.com/v1/search",
                          params={"name": location, "count": 1, "format": "json"})
    response.raise_for_status()
    matches = response.json().get("results", [])
    if not matches:
        raise ValueError("Location not found")
    return matches[0]


def conditions(client, place):
    response = client.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": place["latitude"], "longitude": place["longitude"],
        "current": "temperature_2m,precipitation,weather_code,wind_speed_10m", "timezone": "auto",
    })
    response.raise_for_status()
    data = response.json()
    return {
        "location": ", ".join(str(place[k]) for k in ("name", "admin1", "country") if place.get(k)),
        "current": data["current"], "units": data.get("current_units", {}),
        "timezone": data.get("timezone"), "source": "https://open-meteo.com/",
        "description": "Modelled current conditions, not a direct observation",
    }


def weather_pipeline(location, trace):
    with httpx.Client(timeout=10.0) as client:
        place = yield from trace.step(
            "geocode", lambda: geocode(client, location),
            lambda value: {"name": value["name"], "country": value.get("country")},
            provider="Open-Meteo Geocoding", city=location,
        )
        return (yield from trace.step(
            "weather", lambda: conditions(client, place),
            lambda value: {"location": value["location"], "current": value["current"], "units": value["units"],
                           "timezone": value["timezone"], "source": value["source"]},
            provider="Open-Meteo Forecast",
        ))


def current_weather(location):
    pipeline = weather_pipeline(location, Trace())
    while True:
        try:
            next(pipeline)
        except StopIteration as done:
            return done.value
