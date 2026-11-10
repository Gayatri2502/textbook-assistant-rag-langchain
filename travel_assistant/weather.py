"""On-demand current weather from Open-Meteo (modelled conditions)."""
import httpx


def current_weather(location: str) -> dict:
    with httpx.Client(timeout=10.0) as client:
        response = client.get('https://geocoding-api.open-meteo.com/v1/search', params={'name': location, 'count': 1, 'format': 'json'})
        response.raise_for_status()
        matches = response.json().get('results', [])
        if not matches:
            raise ValueError('Location not found')
        place = matches[0]
        response = client.get('https://api.open-meteo.com/v1/forecast', params={'latitude': place['latitude'], 'longitude': place['longitude'], 'current': 'temperature_2m,precipitation,weather_code,wind_speed_10m', 'timezone': 'auto'})
        response.raise_for_status()
        data = response.json()
    return {'location': ', '.join(str(place[k]) for k in ('name', 'admin1', 'country') if place.get(k)), 'current': data['current'], 'units': data.get('current_units', {}), 'timezone': data.get('timezone'), 'source': 'https://open-meteo.com/', 'description': 'Modelled current conditions, not a direct observation'}
