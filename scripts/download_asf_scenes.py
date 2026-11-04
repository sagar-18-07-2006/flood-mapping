from pathlib import Path
from getpass import getpass
import asf_search as asf
import zipfile
import time

URL = (
    "https://datapool.asf.alaska.edu/GRD_HD/SA/"
    "S1A_IW_GRDH_1SDV_20220812T004026_20220812T004051_"
    "044512_054FD9_2CA3.zip"
)

OUT = Path("data/raw/sentinel1/pre_flood")
OUT.mkdir(parents=True, exist_ok=True)

filename = URL.rsplit("/", 1)[-1]
target = OUT / filename

token = getpass("Earthdata token (hidden): ")
session = asf.ASFSession().auth_with_token(token)

for attempt in range(1, 4):
    print(f"\nDownload attempt {attempt}/3")

    try:
        if target.exists():
            target.unlink()

        asf.download_urls(
            urls=[URL],
            path=str(OUT),
            session=session,
            processes=1
        )

        print("Checking ZIP integrity...")
        with zipfile.ZipFile(target) as archive:
            bad_file = archive.testzip()

        if bad_file is None:
            print("SUCCESS: ZIP integrity verified.")
            break

        raise RuntimeError(f"Corrupt archive member: {bad_file}")

    except Exception as exc:
        print("Attempt failed:", exc)
        if target.exists():
            target.unlink()

        if attempt == 3:
            raise

        time.sleep(10)
