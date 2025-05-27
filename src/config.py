from pathlib import Path


class Config:
    """Configuration class for setting up environment variables, Spark session."""

    def __init__(self, env: str = "dev") -> None:
        """Initialize the Config class with environment variables and travel month."""
        self.env = env
        self.project_root = self.get_project_root()

    def get_project_root(self) -> str:
        """Get the root directory of the project."""
        return str(Path(__file__).parents[1])
