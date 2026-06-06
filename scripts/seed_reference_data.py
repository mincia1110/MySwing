"""Seed script for reference data (professional baseball benchmarks).

Populates the reference_data table with benchmark metrics for different playing levels.
Reference data is used for swing evaluation comparisons (Requirements 7.2, 7.3).

Usage:
    python -m scripts.seed_reference_data

Metrics included:
    - bat_speed: Bat speed at impact zone (km/h)
    - launch_angle: Bat angle at impact (degrees)
    - hip_shoulder_separation: Hip-shoulder separation angle (degrees)
    - hand_path_efficiency: Direct distance / actual path ratio (0-1)
    - attack_angle: Bat angle through hitting zone (degrees)
"""

from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db.models import ReferenceDataTable
from app.db.session import get_engine, sync_session_factory

# Reference data organized by level and metric
# Each entry: (metric_name, min_value, max_value, optimal_min, optimal_max, source)
REFERENCE_DATA: dict[str, dict[str, list[tuple[str, float, float, float, float, str]]]] = {
    "professional": {
        "adult": [
            (
                "bat_speed",
                110.0,
                130.0,
                115.0,
                125.0,
                "Professional baseball average data",
            ),
            (
                "attack_angle",
                5.0,
                25.0,
                10.0,
                20.0,
                "Professional baseball average data",
            ),
            (
                "hip_shoulder_separation",
                30.0,
                60.0,
                40.0,
                55.0,
                "Professional baseball average data",
            ),
            (
                "hand_path_efficiency",
                0.70,
                0.95,
                0.80,
                0.90,
                "Professional baseball average data",
            ),
            (
                "attack_angle",
                5.0,
                15.0,
                8.0,
                12.0,
                "Professional baseball average data",
            ),
        ],
    },
    "college": {
        "adult": [
            (
                "bat_speed",
                100.0,
                120.0,
                105.0,
                115.0,
                "College baseball average data",
            ),
            (
                "attack_angle",
                5.0,
                25.0,
                10.0,
                20.0,
                "College baseball average data",
            ),
            (
                "hip_shoulder_separation",
                25.0,
                55.0,
                35.0,
                50.0,
                "College baseball average data",
            ),
            (
                "hand_path_efficiency",
                0.65,
                0.90,
                0.75,
                0.85,
                "College baseball average data",
            ),
            (
                "attack_angle",
                5.0,
                15.0,
                8.0,
                12.0,
                "College baseball average data",
            ),
        ],
    },
    "high_school": {
        "adult": [
            (
                "bat_speed",
                85.0,
                110.0,
                90.0,
                105.0,
                "High school baseball average data",
            ),
            (
                "attack_angle",
                3.0,
                25.0,
                8.0,
                18.0,
                "High school baseball average data",
            ),
            (
                "hip_shoulder_separation",
                20.0,
                50.0,
                30.0,
                45.0,
                "High school baseball average data",
            ),
            (
                "hand_path_efficiency",
                0.60,
                0.85,
                0.70,
                0.80,
                "High school baseball average data",
            ),
            (
                "attack_angle",
                3.0,
                15.0,
                6.0,
                12.0,
                "High school baseball average data",
            ),
        ],
        "youth": [
            (
                "bat_speed",
                70.0,
                95.0,
                75.0,
                90.0,
                "High school youth baseball average data",
            ),
            (
                "attack_angle",
                3.0,
                22.0,
                8.0,
                16.0,
                "High school youth baseball average data",
            ),
            (
                "hip_shoulder_separation",
                15.0,
                45.0,
                25.0,
                40.0,
                "High school youth baseball average data",
            ),
            (
                "hand_path_efficiency",
                0.55,
                0.80,
                0.65,
                0.75,
                "High school youth baseball average data",
            ),
            (
                "attack_angle",
                3.0,
                14.0,
                5.0,
                11.0,
                "High school youth baseball average data",
            ),
        ],
    },
    "recreational": {
        "adult": [
            (
                "bat_speed",
                70.0,
                100.0,
                80.0,
                95.0,
                "Recreational baseball average data",
            ),
            (
                "attack_angle",
                0.0,
                30.0,
                8.0,
                18.0,
                "Recreational baseball average data",
            ),
            (
                "hip_shoulder_separation",
                15.0,
                45.0,
                25.0,
                40.0,
                "Recreational baseball average data",
            ),
            (
                "hand_path_efficiency",
                0.50,
                0.80,
                0.60,
                0.75,
                "Recreational baseball average data",
            ),
            (
                "attack_angle",
                0.0,
                18.0,
                5.0,
                12.0,
                "Recreational baseball average data",
            ),
        ],
    },
}


def get_seed_records() -> list[dict]:
    """Generate the list of reference data records for seeding.

    Returns:
        List of dictionaries ready for bulk insert.
    """
    records = []
    for level, age_groups in REFERENCE_DATA.items():
        for age_group, metrics in age_groups.items():
            for metric_name, min_val, max_val, opt_min, opt_max, source in metrics:
                records.append(
                    {
                        "level": level,
                        "age_group": age_group,
                        "metric_name": metric_name,
                        "min_value": min_val,
                        "max_value": max_val,
                        "optimal_min": opt_min,
                        "optimal_max": opt_max,
                        "source": source,
                    }
                )
    return records


def seed_reference_data() -> None:
    """Insert or update reference data in the database.

    Uses PostgreSQL upsert (INSERT ... ON CONFLICT UPDATE) to handle
    re-running the seed script safely.
    """
    records = get_seed_records()
    session = sync_session_factory()

    try:
        for record in records:
            stmt = pg_insert(ReferenceDataTable).values(**record)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_reference_data_key",
                set_={
                    "min_value": stmt.excluded.min_value,
                    "max_value": stmt.excluded.max_value,
                    "optimal_min": stmt.excluded.optimal_min,
                    "optimal_max": stmt.excluded.optimal_max,
                    "source": stmt.excluded.source,
                },
            )
            session.execute(stmt)

        session.commit()
        print(f"Successfully seeded {len(records)} reference data records.")
    except Exception as e:
        session.rollback()
        print(f"Error seeding reference data: {e}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    seed_reference_data()
