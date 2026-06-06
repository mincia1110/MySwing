"""Root conftest for Hypothesis settings profile."""

from hypothesis import HealthCheck, settings

# Register Hypothesis profiles
settings.register_profile(
    "ci",
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "dev",
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "debug",
    max_examples=10,
    suppress_health_check=[HealthCheck.too_slow],
)
