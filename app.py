"""Public entrypoint. No server credentials or private datasets in the UI."""
import os
import sys
from pathlib import Path
os.environ['FIA_PUBLIC_MODE'] = 'true'
here = Path(__file__).resolve().parent
root = here if (here/'football_ai').is_dir() else here.parent
sys.path.insert(0, str(root))
from football_ai.simple_dashboard import main
main()
