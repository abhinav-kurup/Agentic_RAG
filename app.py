"""Entry point — launches the Streamlit frontend (API-backed)."""
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "frontend/app.py", "--server.port=8501"],
        check=False,
    )
