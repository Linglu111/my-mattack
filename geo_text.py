from dataclasses import dataclass
from typing import Iterable, List, Optional

import yaml


COUNTRY_TEMPLATES = [
    "a photo taken in {}",
    "a street view image from {}",
    "a geographic scene in {}",
]

CITY_TEMPLATES = [
    "a street view photo in {}",
    "a city scene in {}",
    "an urban image from {}",
]

COUNTRY_EVIDENCE_TEMPLATES = [
    "architecture commonly found in {}",
    "street signs and buildings from {}",
    "roads, storefronts, and urban layout in {}",
    "outdoor visual landmarks from {}",
]

CITY_EVIDENCE_TEMPLATES = [
    "landmarks and streets in {}",
    "downtown architecture in {}",
    "storefronts and road signs in {}",
    "urban layout and local visual cues from {}",
]

COUNTRY_VOCAB = [
    "United States", "China", "Japan", "France", "Germany", "United Kingdom",
    "Italy", "Spain", "Canada", "Australia", "Brazil", "Mexico", "South Korea",
    "India", "Russia", "Netherlands", "Belgium", "Switzerland", "Austria",
    "Sweden", "Norway", "Denmark", "Finland", "Ireland", "Portugal", "Greece",
    "Turkey", "Poland", "Czech Republic", "Hungary", "Romania", "Ukraine",
    "Thailand", "Vietnam", "Malaysia", "Singapore", "Indonesia", "Philippines",
    "Taiwan", "Hong Kong", "United Arab Emirates", "Saudi Arabia", "Israel",
    "Egypt", "Morocco", "South Africa", "Argentina", "Chile", "Colombia",
    "Peru", "New Zealand", "Iceland", "Luxembourg", "Croatia", "Slovenia",
    "Slovakia", "Estonia", "Latvia", "Lithuania", "Bulgaria", "Serbia",
    "Georgia", "Qatar", "Kuwait", "Jordan", "Lebanon", "Pakistan",
    "Bangladesh", "Sri Lanka", "Nepal", "Cambodia", "Laos", "Myanmar",
    "Kenya", "Tanzania", "Nigeria", "Ghana", "Ethiopia", "Uruguay",
    "Costa Rica", "Panama",
]

CITY_VOCAB = [
    "New York City", "Los Angeles", "San Francisco", "Chicago", "Washington DC",
    "Boston", "Seattle", "Miami", "Las Vegas", "Honolulu", "Toronto",
    "Vancouver", "Montreal", "Calgary", "Mexico City", "Cancun", "Guadalajara",
    "Sao Paulo", "Rio de Janeiro", "Brasilia", "Buenos Aires", "Santiago",
    "Bogota", "Lima", "Montevideo", "San Jose", "Panama City",
    "London", "Manchester", "Edinburgh", "Dublin", "Paris", "Lyon",
    "Marseille", "Nice", "Berlin", "Munich", "Hamburg", "Frankfurt",
    "Cologne", "Amsterdam", "Rotterdam", "Brussels", "Antwerp", "Zurich",
    "Geneva", "Lucerne", "Vienna", "Salzburg", "Madrid", "Barcelona",
    "Seville", "Valencia", "Lisbon", "Porto", "Rome", "Milan", "Venice",
    "Florence", "Naples", "Athens", "Istanbul", "Ankara", "Prague",
    "Budapest", "Warsaw", "Krakow", "Bucharest", "Kyiv", "Lviv",
    "Stockholm", "Gothenburg", "Oslo", "Bergen", "Copenhagen", "Helsinki",
    "Reykjavik", "Tallinn", "Riga", "Vilnius", "Zagreb", "Split",
    "Ljubljana", "Bratislava", "Belgrade", "Sofia", "Tbilisi",
    "Moscow", "Saint Petersburg", "Sochi", "Dubai", "Abu Dhabi", "Doha",
    "Riyadh", "Jeddah", "Kuwait City", "Amman", "Beirut", "Tel Aviv",
    "Jerusalem", "Cairo", "Alexandria", "Marrakesh", "Casablanca",
    "Cape Town", "Johannesburg", "Nairobi", "Dar es Salaam", "Lagos",
    "Abuja", "Accra", "Addis Ababa", "Tokyo", "Osaka", "Kyoto",
    "Yokohama", "Sapporo", "Fukuoka", "Seoul", "Busan", "Incheon",
    "Beijing", "Shanghai", "Guangzhou", "Shenzhen", "Chengdu", "Hangzhou",
    "Xi'an", "Chongqing", "Hong Kong", "Taipei", "Kaohsiung", "Singapore",
    "Bangkok", "Chiang Mai", "Phuket", "Hanoi", "Ho Chi Minh City",
    "Da Nang", "Kuala Lumpur", "Penang", "Jakarta", "Bali", "Surabaya",
    "Manila", "Cebu", "New Delhi", "Mumbai", "Bangalore", "Chennai",
    "Kolkata", "Jaipur", "Agra", "Islamabad", "Karachi", "Lahore",
    "Dhaka", "Colombo", "Kathmandu", "Phnom Penh", "Siem Reap",
    "Vientiane", "Yangon", "Mandalay", "Sydney", "Melbourne", "Brisbane",
    "Perth", "Adelaide", "Auckland", "Wellington", "Queenstown",
    "Christchurch", "Doha West Bay", "Sharjah", "Muscat", "Almaty",
    "Astana", "Tashkent", "Samarkand", "Baku", "Yerevan", "Valletta",
    "Monaco", "Luxembourg City", "Macau", "Puerto Vallarta", "Marrakesh Medina",
    "Santorini", "Mykonos", "Ibiza", "Granada", "Bilbao", "Bordeaux",
    "Strasbourg", "Nuremberg", "Dresden", "Heidelberg", "Oxford",
    "Cambridge", "Bath", "York", "Bristol", "Liverpool", "Glasgow",
    "Belfast", "Cork", "Galway", "Bruges", "Ghent", "Lausanne",
    "Interlaken", "Innsbruck", "Graz", "Bologna", "Turin", "Verona",
]


@dataclass
class GeoVocab:
    countries: List[str]
    cities: List[str]
    country_templates: List[str]
    city_templates: List[str]
    country_evidence_templates: List[str]
    city_evidence_templates: List[str]


def _merge_unique(base: Iterable[str], extra: Iterable[str]) -> List[str]:
    values = []
    seen = set()
    for item in list(base) + list(extra):
        if item not in seen:
            values.append(item)
            seen.add(item)
    return values


def load_geo_vocab(vocab_path: Optional[str] = None) -> GeoVocab:
    vocab = {
        "countries": list(COUNTRY_VOCAB),
        "cities": list(CITY_VOCAB),
        "country_templates": list(COUNTRY_TEMPLATES),
        "city_templates": list(CITY_TEMPLATES),
        "country_evidence_templates": list(COUNTRY_EVIDENCE_TEMPLATES),
        "city_evidence_templates": list(CITY_EVIDENCE_TEMPLATES),
    }
    if vocab_path:
        with open(vocab_path, "r", encoding="utf-8") as handle:
            user_vocab = yaml.safe_load(handle) or {}
        replace_defaults = bool(user_vocab.get("replace_defaults", False))
        for key in vocab:
            if key in user_vocab and user_vocab[key] is not None:
                values = list(user_vocab[key])
                vocab[key] = values if replace_defaults else _merge_unique(vocab[key], values)

    return GeoVocab(
        countries=vocab["countries"],
        cities=vocab["cities"],
        country_templates=vocab["country_templates"],
        city_templates=vocab["city_templates"],
        country_evidence_templates=vocab["country_evidence_templates"],
        city_evidence_templates=vocab["city_evidence_templates"],
    )


def expand_prompts(labels: List[str], templates: List[str]) -> List[str]:
    return [template.format(label) for label in labels for template in templates]
