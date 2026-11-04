from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path("data/interim/sentinel1")

for manifest in sorted(ROOT.glob("*/*.SAFE/manifest.safe")):
    print("\n" + "=" * 75)
    print(manifest.parent.name)
    print("=" * 75)

    tree = ET.parse(manifest)
    root = tree.getroot()

    for element in root.iter():
        tag = element.tag.split("}")[-1]

        if tag in {
            "startTime",
            "stopTime",
            "relativeOrbitNumber",
            "absoluteOrbitNumber",
            "pass",
            "mode",
            "productType",
            "transmitterReceiverPolarisation",
            "footprint"
        }:
            value = element.text.strip() if element.text else ""
            if value:
                print(f"{tag}: {value}")

    measurements = manifest.parent / "measurement"
    print("\nMeasurement files:")

    for file in sorted(measurements.glob("*")):
        print(f"  {file.name} ({file.stat().st_size / 1e6:.1f} MB)")
