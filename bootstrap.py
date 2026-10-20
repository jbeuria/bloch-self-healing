"""Load this flat source folder as the self_healing package from any clone name."""
import importlib.util
from pathlib import Path
import sys


def activate():
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    existing = sys.modules.get('self_healing')
    if existing is None:
        spec = importlib.util.spec_from_file_location(
            'self_healing', root / '__init__.py', submodule_search_locations=[str(root)])
        module = importlib.util.module_from_spec(spec)
        sys.modules['self_healing'] = module
        spec.loader.exec_module(module)
    elif Path(existing.__file__).resolve() != root / '__init__.py':
        raise RuntimeError('A different self_healing package is already loaded')
    return root
