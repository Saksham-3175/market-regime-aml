"""Ensure project root is on sys.path so `src` is importable during tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
