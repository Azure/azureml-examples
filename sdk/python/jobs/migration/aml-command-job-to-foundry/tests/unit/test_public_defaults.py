from __future__ import annotations

import pytest

from foundrytrainingjob import constants, dataset


def test_storage_connection_default_requires_public_configuration() -> None:
    assert constants.contains_placeholder_token(
        constants.DEFAULT_STORAGE_CONNECTION_NAME
    )

    with pytest.raises(
        ValueError, match="FOUNDRY_TRAININGJOB__STORAGE_CONNECTION_NAME"
    ):
        dataset._resolve_connection_name(connection_name=None)


def test_explicit_storage_connection_is_accepted() -> None:
    assert dataset._resolve_connection_name(connection_name="project-storage") == (
        "project-storage"
    )
