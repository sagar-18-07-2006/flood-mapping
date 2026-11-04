import requests
import getpass
import sys
from pathlib import Path

CATALOGUE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/"
    "CDSE/protocol/openid-connect/token"
)
DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/odata/v1/Products"

PRODUCTS = [
    "S1A_IW_GRDH_1SDV_20220812T004026_20220812T004051_044512_054FD9_2CA3.SAFE",
    "S1A_IW_GRDH_1SDV_20220905T004028_20220905T004053_044862_055BB0_85F8.SAFE",
]

OUT = Path("data/raw/sentinel1")
OUT.mkdir(parents=True, exist_ok=True)

session = requests.Session()
session.headers.update({"User-Agent": "BengaluruFloodMapping/1.0"})

def get_product(name):
    response = session.get(
        CATALOGUE,
        params={"$filter": f"Name eq '{name}'"},
        timeout=60,
    )
    response.raise_for_status()
    products = response.json().get("value", [])

    if not products:
        raise RuntimeError(f"Product not found: {name}")

    return products[0]

print("Looking up the two Sentinel-1 products...\n")

selected = []
for name in PRODUCTS:
    product = get_product(name)
    size = product.get("ContentLength", 0)

    selected.append(product)

    print("Product:", name)
    print("ID:", product["Id"])
    print("Size:", f"{size / (1024**3):.2f} GB" if size else "Not reported")
    print()

username = input("Copernicus account email: ").strip()
password = getpass.getpass("Copernicus password (hidden): ")

print("\nAuthenticating with Copernicus...")

response = session.post(
    TOKEN_URL,
    data={
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": username,
        "password": password,
    },
    timeout=60,
)

if not response.ok:
    print("Authentication failed:", response.status_code)
    print(response.text[:500])
    sys.exit(1)

token = response.json()["access_token"]
session.headers["Authorization"] = f"Bearer {token}"

print("Authentication successful.")

total_size = sum(p.get("ContentLength", 0) for p in selected)
print(f"Total catalogue size: {total_size / (1024**3):.2f} GB")

confirm = input("Download both products? (y/n): ").strip().lower()
if confirm != "y":
    print("Download cancelled.")
    sys.exit(0)

for product in selected:
    name = product["Name"]
    product_id = product["Id"]

    # Store the archive as a ZIP; extract it after download.
    filename = name.removesuffix(".SAFE") + ".zip"
    destination = OUT / filename

    if destination.exists() and destination.stat().st_size > 0:
        print(f"\nAlready exists, skipping: {destination}")
        continue

    url = f"{DOWNLOAD_URL}({product_id})/$value"

    print(f"\nDownloading: {name}")

    try:
        with session.get(
            url,
            stream=True,
            timeout=(30, 180),
            allow_redirects=True,
        ) as r:
            r.raise_for_status()
            expected = int(r.headers.get("Content-Length", 0))
            downloaded = 0

            with destination.open("wb") as f:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        if expected:
                            percent = downloaded / expected * 100
                            print(
                                f"\r{downloaded / 1024**3:.2f} GB "
                                f"({percent:.1f}%)",
                                end="",
                                flush=True,
                            )
                        else:
                            print(
                                f"\r{downloaded / 1024**3:.2f} GB",
                                end="",
                                flush=True,
                            )

        print(f"\nSaved: {destination}")

    except Exception as exc:
        destination.unlink(missing_ok=True)
        print(f"\nDownload failed: {exc}")
        print("You can retry after checking your connection.")
        sys.exit(1)

print("\nBoth Sentinel-1 downloads completed.")
