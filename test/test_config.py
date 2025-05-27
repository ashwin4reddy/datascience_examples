from src.config import Config


def test_config_init() -> None:
    """Test initializing the Config class."""
    config = Config()
    assert config.env == "dev"
