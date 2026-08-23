"""Put ``src`` on the path so tests run against the checkout, not an install."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
