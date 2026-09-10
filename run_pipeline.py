"""Run the complete reproducible Volve forecast pipeline."""

import subprocess
import sys


def main():
    for script in ("01_prepare.py", "02_arps.py", "03_ml.py", "04_economics.py"):
        print(f"\n--- {script} ---")
        subprocess.run([sys.executable, script], check=True)


if __name__ == "__main__":
    main()
