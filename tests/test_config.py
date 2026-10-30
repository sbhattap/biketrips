from biketrips.config import load_config


def test_config_resolves_paths():
    cfg = load_config()
    assert cfg["seed"] == 42
    assert cfg["data"]["trips_csv"].name == "21_1000w.csv"
    assert cfg["data"]["raw_dir"].is_absolute()
