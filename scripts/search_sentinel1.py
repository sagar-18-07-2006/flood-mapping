import requests
from datetime import datetime
from pathlib import Path

URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

# Bengaluru study area (WGS84)
polygon = (
    "POLYGON((77.40 12.75,77.85 12.75,"
    "77.85 13.20,77.40 13.20,77.40 12.75))"
)

spatial = (
    "OData.CSC.Intersects("
    f"area=geography'SRID=4326;{polygon}')"
)

windows = {
    "PRE-FLOOD": ("2022-08-01", "2022-09-04"),
    "POST-FLOOD": ("2022-09-04", "2022-09-16"),
}

session = requests.Session()
session.headers.update({
    "User-Agent": "BengaluruFloodMappingStudentProject/1.0",
    "Accept": "application/json",
})

def search(label, start, end):
    print(f"\n{'=' * 65}")
    print(f"{label}: {start} to {end}")
    print("=" * 65)

    filters = [
        "Collection/Name eq 'SENTINEL-1'",
        "startswith(Name,'S1A_')",
        "contains(Name,'IW_GRDH_1SDV_')",
        spatial,
        f"ContentDate/Start ge {start}T00:00:00.000Z",
        f"ContentDate/Start lt {end}T00:00:00.000Z",
    ]

    params = {
        "$filter": " and ".join(filters),
        "$orderby": "ContentDate/Start asc",
        "$top": 100,
        "$expand": "Attributes",
    }

    try:
        response = session.get(URL, params=params, timeout=90)
        response.raise_for_status()
        products = response.json().get("value", [])

    except requests.RequestException as exc:
        print("Catalogue request failed:", exc)
        if getattr(exc, "response", None) is not None:
            print(exc.response.text[:1000])
        return []

    print("Products found:", len(products))

    results = []

    for product in products:
        name = product.get("Name", "")
        attrs = product.get("Attributes", [])

        metadata = {}
        for attr in attrs:
            key = attr.get("Name", "")
            value = attr.get("Value")
            metadata[key] = value

        date = product.get("ContentDate", {}).get("Start", "")

        relevant = {
            k: v for k, v in metadata.items()
            if any(word in k.lower() for word in
                   ["orbit", "polar", "producttype", "timeliness"])
        }

        item = {
            "name": name,
            "date": date,
            "id": product.get("Id"),
            "metadata": relevant,
        }
        results.append(item)

        print("\nProduct:", name)
        print("Acquisition:", date)
        print("Metadata:", relevant)

    return results


all_results = {}

for label, (start, end) in windows.items():
    all_results[label] = search(label, start, end)

print("\n\nSEARCH SUMMARY")
for label, products in all_results.items():
    print(f"{label}: {len(products)} products")

# Save results for later inspection
import json

output = Path("data/processed/sentinel1_catalogue_results.json")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(all_results, indent=2))

print("\nSaved catalogue results:", output)
