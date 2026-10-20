import logging
import subprocess
import sys
from pathlib import Path

from griptape_nodes.node_library.advanced_node_library import AdvancedNodeLibrary
from griptape_nodes.node_library.library_registry import Library, LibrarySchema

logger = logging.getLogger("face_parsing_library")


class FaceParsingLibraryAdvanced(AdvancedNodeLibrary):
    def before_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        logger.info(f"Loading '{library_data.name}' library...")
        submodule_path = self._init_submodule()
        if not self._is_installed(submodule_path):
            # Runtime dependencies come from the manifest's pip_dependencies; upstream ships no requirements.txt.
            self._write_installed_sentinel(submodule_path)
        # Always re-apply sys.path, it does not persist across engine restarts.
        self._install_package(submodule_path)

    def after_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        logger.info(f"Finished loading '{library_data.name}' library")

    def _get_library_root(self) -> Path:
        return Path(__file__).parent

    def _init_submodule(self) -> Path:
        library_root = self._get_library_root()
        submodule_dir = library_root / "face_parsing_pytorch"
        if submodule_dir.exists() and any(submodule_dir.iterdir()):
            logger.info("Submodule already initialized")
            return submodule_dir
        subprocess.check_call(["git", "-C", str(library_root.parent), "submodule", "update", "--init", "--recursive"])
        if not submodule_dir.exists() or not any(submodule_dir.iterdir()):
            msg = f"Submodule init failed: {submodule_dir}"
            raise RuntimeError(msg)
        logger.info("Submodule initialized successfully")
        return submodule_dir

    def _get_submodule_commit(self, submodule_path: Path) -> str:
        """Return the HEAD commit SHA of the submodule (the version pinned by the library author)."""
        return subprocess.check_output(["git", "-C", str(submodule_path), "rev-parse", "HEAD"], text=True).strip()

    def _get_installed_sentinel(self) -> Path:
        return self._get_library_root() / ".installed_commit"

    def _write_installed_sentinel(self, submodule_path: Path) -> None:
        self._get_installed_sentinel().write_text(self._get_submodule_commit(submodule_path))

    def _is_installed(self, submodule_path: Path) -> bool:
        """For sys.path installs, the commit sentinel is the only durable install signal.

        sys.path mutations do not survive across engine restarts and cannot be observed
        from a venv subprocess, so we rely entirely on the committed-version sentinel.
        """
        sentinel = self._get_installed_sentinel()
        if not sentinel.exists():
            return False
        return sentinel.read_text().strip() == self._get_submodule_commit(submodule_path)

    def _install_package(self, submodule_path: Path) -> None:
        """Put the upstream modules (model.py, resnet.py) on sys.path; the repo is not pip-installable."""
        if str(submodule_path) not in sys.path:
            sys.path.insert(0, str(submodule_path))
        logger.info(f"Added {submodule_path} to sys.path")
