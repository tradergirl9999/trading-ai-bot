import requests


def get_weather(location: str = "auto") -> str:
    try:
        url = f"https://wttr.in/{location}?format=j1"
        r = requests.get(url, timeout=8, headers={"User-Agent": "curl/7.0"})
        data = r.json()
        current = data["current_condition"][0]
        area = data["nearest_area"][0]
        city = area["areaName"][0]["value"]
        country = area["country"][0]["value"]

        desc = current["weatherDesc"][0]["value"]
        temp_c = current["temp_C"]
        temp_f = current["temp_F"]
        feels_c = current["FeelsLikeC"]
        humidity = current["humidity"]
        wind_kmh = current["windspeedKmph"]
        wind_dir = current["winddir16Point"]
        visibility = current["visibility"]
        uv = current["uvIndex"]

        today = data["weather"][0]
        max_c = today["maxtempC"]
        min_c = today["mintempC"]
        sunrise = today["astronomy"][0]["sunrise"]
        sunset = today["astronomy"][0]["sunset"]

        return (
            f"Weather in {city}, {country}: {desc}\n"
            f"Temperature: {temp_c}°C / {temp_f}°F (feels like {feels_c}°C)\n"
            f"Today's range: {min_c}°C – {max_c}°C\n"
            f"Humidity: {humidity}% | Wind: {wind_kmh} km/h {wind_dir}\n"
            f"Visibility: {visibility} km | UV Index: {uv}\n"
            f"Sunrise: {sunrise} | Sunset: {sunset}"
        )
    except Exception as e:
        return f"Weather unavailable: {e}"
