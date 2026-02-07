"""Get version information for Julia and related packages."""

import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class VersionInfo:
    """Version information for Julia and related packages."""

    julia: str
    jutul: str
    jutuldarcy: str

    def format_markdown(self) -> str:
        """Format version info as markdown."""
        return f"Versions: Julia {self.julia} | Jutul {self.jutul} | JutulDarcy {self.jutuldarcy}"


def get_version_info(project_dir: Optional[str] = None) -> Optional[VersionInfo]:
    """Get Julia, Jutul, and JutulDarcy version information.

    Args:
        project_dir: Julia project directory. If None, uses current directory.

    Returns:
        VersionInfo object, or None if Julia is not available.
    """
    import os

    if project_dir is None:
        project_dir = os.getcwd()

    # Output format: "VERSION|JUTUL_VERSION|JUTULDARCY_VERSION"
    # Regex parsing handles any startup messages that precede this
    julia_code = '''
print(VERSION)
print("|")
try
    using Jutul
    print(pkgversion(Jutul))
catch
    print("N/A")
end
print("|")
try
    using JutulDarcy
    print(pkgversion(JutulDarcy))
catch
    print("N/A")
end
'''

    try:
        result = subprocess.run(
            [
                "julia",
                f"--project={project_dir}",
                "-e",
                julia_code,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=project_dir,
            timeout=120,
        )

        if result.returncode != 0:
            return None

        # Parse "X.Y.Z|X.Y.Z|X.Y.Z" format from output
        # The output might have startup messages before it, so find the pattern
        import re

        # Look for version pattern: "number|something|something" at end of output
        match = re.search(r"(\d+\.\d+\.\d+)\|([\w./-]+)\|([\w./-]+)\s*$", result.stdout)
        if match:
            return VersionInfo(
                julia=match.group(1),
                jutul=match.group(2),
                jutuldarcy=match.group(3),
            )

    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass

    return None
