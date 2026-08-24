from datetime import date

from artifacts.flattrade_day_cache import (
    decode_active_strikes,
    load_day_cache,
    save_day_cache,
)


def test_day_cache_round_trips_deduplicated_rows(tmp_path):
    target = date(2026, 8, 13)
    active = {("CE", 560): {24250}, ("PE", 560): {24450}}
    contracts = {
        ("CE", 24250): {
            "token": "45100",
            "tsym": "NIFTY18AUG26C24250",
            "dname": "NIFTY 18AUG26 24250 CE",
            "rows": [
                {"time": "13-08-2026 09:15:00", "close": 200.0},
                {"time": "13-08-2026 09:15:00", "close": 201.0},
            ],
        }
    }

    destination = save_day_cache(
        tmp_path,
        target,
        [{"time": "13-08-2026 09:15:00", "close": 24300.0}],
        active,
        contracts,
    )
    loaded = load_day_cache(tmp_path, target)

    assert destination.exists()
    assert loaded["contracts"]["CE:24250"]["rows"] == [
        {"time": "13-08-2026 09:15:00", "close": 201.0}
    ]
    assert decode_active_strikes(loaded["active_strikes"]) == active
