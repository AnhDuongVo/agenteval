import subprocess
import sys
from pathlib import Path


def test_workshop_automated_check():
    script = Path(__file__).resolve().parents[1] / "examples/workshop.py"
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "Workshop checks passed" in result.stdout
