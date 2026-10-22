"""BlogForge AI entry point.

    python main.py                        interactive menu
    python main.py --topic "How Agentic AI Works"
    python main.py --demo                 simulated demonstration (no keys, no network)
    python main.py --check                configuration check
"""

import sys

from app.cli import main

if __name__ == "__main__":
    sys.exit(main())