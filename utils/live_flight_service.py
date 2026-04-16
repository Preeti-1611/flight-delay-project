"""
Live Flight Service — fetches REAL flight data from AviationStack API
and live radar positions from FlightRadar24.
"""
import requests
import os
from datetime import datetime

# ── AviationStack free-tier key ──────────────────────────────────────
# Sign up at https://aviationstack.com/signup/free (no credit card)
# Then paste your key below or set env var AVIATIONSTACK_API_KEY
AVIATIONSTACK_KEY = os.environ.get(
    "AVIATIONSTACK_API_KEY",
    "ee53fe95ba731e144386e6e56990b580"  # <— replace with your real key
)

# Indian airport IATA codes we care about
INDIAN_AIRPORTS = {
    "DEL": "Delhi, Delhi",
    "BOM": "Mumbai, Maharashtra",
    "BLR": "Bengaluru, Karnataka",
    "MAA": "Chennai, Tamil Nadu",
    "HYD": "Hyderabad, Telangana",
    "CCU": "Kolkata, West Bengal",
    "COK": "Kochi, Kerala",
    "PNQ": "Pune, Maharashtra",
    "AMD": "Ahmedabad, Gujarat",
    "JAI": "Jaipur, Rajasthan",
    "GOI": "Goa, Goa",
    "LKO": "Lucknow, Uttar Pradesh",
}

IATA_TO_CITY = INDIAN_AIRPORTS            # code → full name
CITY_TO_IATA = {v: k for k, v in INDIAN_AIRPORTS.items()}  # full name → code


class LiveFlightService:
    """Wraps AviationStack + FlightRadar24 into one service."""

    BASE_URL = "http://api.aviationstack.com/v1"

    def __init__(self):
        self.api_key = AVIATIONSTACK_KEY
        # FlightRadar24 (best-effort; may fail silently)
        try:
            from FlightRadar24 import FlightRadar24API
            self.fr_api = FlightRadar24API()
        except Exception:
            self.fr_api = None
        self.india_bounds = "36.00,6.00,68.00,98.00"

    # ------------------------------------------------------------------ #
    #  AviationStack – real scheduled / live flights                      #
    # ------------------------------------------------------------------ #
    def get_real_flights(self, dep_iata=None, arr_iata=None,
                         airline_iata=None, flight_status=None,
                         limit=100):
        """
        Fetch real flight records from AviationStack.
        Returns a list of dicts with normalized keys.
        """
        if not self.api_key or self.api_key == "YOUR_KEY_HERE":
            return self._fallback_flights(dep_iata, arr_iata)

        params = {"access_key": self.api_key, "limit": limit}
        if dep_iata:
            params["dep_iata"] = dep_iata
        if arr_iata:
            params["arr_iata"] = arr_iata
        if airline_iata:
            params["airline_iata"] = airline_iata
        if flight_status:
            params["flight_status"] = flight_status   # active / landed / scheduled / cancelled

        try:
            resp = requests.get(f"{self.BASE_URL}/flights", params=params, timeout=10)
            if resp.status_code != 200:
                print(f"AviationStack HTTP {resp.status_code}")
                return self._fallback_flights(dep_iata, arr_iata)

            body = resp.json()
            if "error" in body:
                print(f"AviationStack error: {body['error']}")
                return self._fallback_flights(dep_iata, arr_iata)

            raw = body.get("data", [])
            flights = []
            for r in raw:
                dep = r.get("departure", {})
                arr = r.get("arrival", {})
                flt = r.get("flight", {})
                airl = r.get("airline", {})

                flights.append({
                    "flight_iata": flt.get("iata", "N/A"),
                    "flight_number": flt.get("number", ""),
                    "airline_name": airl.get("name", "Unknown"),
                    "airline_iata": airl.get("iata", ""),
                    "flight_status": r.get("flight_status", "unknown"),
                    "flight_date": r.get("flight_date", ""),
                    # departure
                    "dep_airport": dep.get("airport", "N/A"),
                    "dep_iata": dep.get("iata", ""),
                    "dep_scheduled": dep.get("scheduled", ""),
                    "dep_estimated": dep.get("estimated", ""),
                    "dep_actual": dep.get("actual", ""),
                    "dep_delay": dep.get("delay"),          # minutes or None
                    "dep_terminal": dep.get("terminal", ""),
                    "dep_gate": dep.get("gate", ""),
                    # arrival
                    "arr_airport": arr.get("airport", "N/A"),
                    "arr_iata": arr.get("iata", ""),
                    "arr_scheduled": arr.get("scheduled", ""),
                    "arr_estimated": arr.get("estimated", ""),
                    "arr_actual": arr.get("actual", ""),
                    "arr_delay": arr.get("delay"),          # minutes or None
                    "arr_terminal": arr.get("terminal", ""),
                    "arr_gate": arr.get("gate", ""),
                    # live position (only when flight_status == "active")
                    "latitude": (r.get("live") or {}).get("latitude"),
                    "longitude": (r.get("live") or {}).get("longitude"),
                    "altitude": (r.get("live") or {}).get("altitude"),
                    "speed_horizontal": (r.get("live") or {}).get("speed_horizontal"),
                    "is_ground": (r.get("live") or {}).get("is_ground", False),
                    "data_source": "AviationStack (Live)"
                })
            return flights
        except Exception as e:
            print(f"AviationStack error: {e}")
            return self._fallback_flights(dep_iata, arr_iata)

    # ------------------------------------------------------------------ #
    #  FlightRadar24 – live radar positions                                #
    # ------------------------------------------------------------------ #
    def get_radar_flights(self, limit=300):
        """Return live radar blips from FlightRadar24 for India."""
        if not self.fr_api:
            return []
        try:
            flights = self.fr_api.get_flights(bounds=self.india_bounds)
            out = []
            for f in flights[:limit]:
                out.append({
                    "callsign": f.callsign,
                    "latitude": f.latitude,
                    "longitude": f.longitude,
                    "heading": f.heading,
                    "altitude": f.altitude,
                    "ground_speed": f.ground_speed,
                    "aircraft_code": f.aircraft_code,
                    "airline_iata": f.airline_iata,
                    "origin": f.origin_airport_iata,
                    "destination": f.destination_airport_iata
                })
            return out
        except Exception as e:
            print(f"FlightRadar24 error: {e}")
            return []

    # ------------------------------------------------------------------ #
    #  Fallback – simulated Indian flights when API key is missing         #
    # ------------------------------------------------------------------ #
    def _fallback_flights(self, dep_iata=None, arr_iata=None):
        """Generate realistic-looking demo data so the page is never blank."""
        import random
        now = datetime.now()
        airlines = [
            ("IndiGo", "6E"), ("Air India", "AI"), ("SpiceJet", "SG"),
            ("Vistara", "UK"), ("AirAsia India", "I5"), ("Akasa Air", "QP"),
            ("Alliance Air", "9I"),
        ]
        codes = list(INDIAN_AIRPORTS.keys())
        flights = []
        for i in range(40):
            al_name, al_iata = random.choice(airlines)
            dep = random.choice(codes)
            arr = random.choice([c for c in codes if c != dep])
            if dep_iata and dep != dep_iata:
                continue
            if arr_iata and arr != arr_iata:
                continue

            hour = random.randint(5, 22)
            minute = random.choice([0, 15, 30, 45])
            dep_delay = random.choice([None, 0, 0, 0, 5, 10, 15, 25, 45, 60])
            arr_delay = (dep_delay + random.randint(-5, 10)) if dep_delay else random.choice([None, 0, 0, 5])
            status = random.choice(["scheduled", "active", "landed", "landed", "landed"])
            flights.append({
                "flight_iata": f"{al_iata}{random.randint(100,999)}",
                "flight_number": f"{random.randint(100,999)}",
                "airline_name": al_name,
                "airline_iata": al_iata,
                "flight_status": status,
                "flight_date": now.strftime("%Y-%m-%d"),
                "dep_airport": INDIAN_AIRPORTS[dep].split(",")[0],
                "dep_iata": dep,
                "dep_scheduled": f"{now.strftime('%Y-%m-%d')}T{hour:02d}:{minute:02d}:00+05:30",
                "dep_estimated": None,
                "dep_actual": f"{now.strftime('%Y-%m-%d')}T{hour:02d}:{minute + (dep_delay or 0) % 60:02d}:00+05:30" if status != "scheduled" else None,
                "dep_delay": dep_delay,
                "dep_terminal": random.choice(["T1", "T2", "T3", ""]),
                "dep_gate": random.choice(["A1", "B4", "C12", "D7", ""]),
                "arr_airport": INDIAN_AIRPORTS[arr].split(",")[0],
                "arr_iata": arr,
                "arr_scheduled": None,
                "arr_estimated": None, "arr_actual": None,
                "arr_delay": arr_delay,
                "arr_terminal": random.choice(["T1", "T2", ""]),
                "arr_gate": "",
                "latitude": None, "longitude": None,
                "altitude": None, "speed_horizontal": None,
                "is_ground": False,
                "data_source": "Simulated Demo Data"
            })
        return flights


# Quick smoke-test
if __name__ == "__main__":
    svc = LiveFlightService()
    print("── AviationStack flights ──")
    for f in svc.get_real_flights(dep_iata="DEL")[:3]:
        print(f"  {f['flight_iata']}  {f['dep_iata']}→{f['arr_iata']}  "
              f"status={f['flight_status']}  dep_delay={f['dep_delay']}  "
              f"source={f['data_source']}")
    print("── Radar flights ──")
    radar = svc.get_radar_flights(limit=5)
    for r in radar:
        print(f"  {r['callsign']}  alt={r['altitude']}  spd={r['ground_speed']}")
