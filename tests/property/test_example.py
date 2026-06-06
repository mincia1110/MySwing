"""Example property-based test to verify Hypothesis setup."""

import pytest
from hypothesis import given
from hypothesis import strategies as st


@pytest.mark.property
@given(file_ext=st.sampled_from(["mp4", "mov", "avi"]))
def test_supported_formats_are_lowercase(file_ext: str) -> None:
    """Property: All supported format strings are lowercase.

    **Validates: Requirements 1.1**
    """
    assert file_ext == file_ext.lower()
    assert file_ext in {"mp4", "mov", "avi"}
