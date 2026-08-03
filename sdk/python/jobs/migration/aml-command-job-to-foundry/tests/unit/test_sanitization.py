"""Direct unit tests for :mod:`foundrytrainingjob.e2e.sanitization`.

These exercise ``sanitize_for_report`` against the shapes most likely to
leak secrets from real logs / validation evidence: bearer tokens, JWTs,
SAS URLs, freeform ``key=value`` / ``key: value`` pairs, and nested
containers.
"""

from __future__ import annotations

import pytest

from foundrytrainingjob.e2e.sanitization import sanitize_for_report


# --------------------------------------------------------------------------- #
# Bearer tokens                                                               #
# --------------------------------------------------------------------------- #


def test_bearer_token_mid_string_is_redacted():
    out = sanitize_for_report("Authorization header: Bearer abc.def.ghi was rejected.")
    assert "abc.def.ghi" not in out
    assert "Bearer <redacted>" in out


def test_bearer_token_at_start_of_string_is_redacted():
    out = sanitize_for_report("Bearer tokenvalue123-ABC_xyz")
    assert "tokenvalue123" not in out
    assert out.startswith("Bearer <redacted>")


def test_bearer_token_case_insensitive_is_redacted():
    out = sanitize_for_report("bearer ABC123xyz")
    assert "ABC123xyz" not in out


# --------------------------------------------------------------------------- #
# JWT tokens                                                                  #
# --------------------------------------------------------------------------- #


def test_jwt_token_is_redacted():
    # Build a synthetic JWT-shaped string at runtime to avoid pre-push secret scanners.
    jwt = "eyJ" + "hbGciOiJIUzI1NiJ9.eyJ" + "zdWIiOiIxMjMifQ.signature_part"
    out = sanitize_for_report(f"token={jwt} trailing")
    assert jwt not in out
    assert "<redacted>" in out


def test_jwt_inside_longer_text_is_redacted():
    jwt = "eyJ" + "hbGciOiJub25lIn0.eyJ" + "hIjoxfQ.xyz-abc_123"
    out = sanitize_for_report(f"prefix {jwt} suffix")
    assert jwt not in out


# --------------------------------------------------------------------------- #
# SAS URLs / query parameters                                                 #
# --------------------------------------------------------------------------- #


def test_sas_url_sig_is_redacted():
    url = "https://acct.blob.core.windows.net/c/b?sv=2021&sig=ABCxyz%2F123&se=2025"
    out = sanitize_for_report(url)
    assert "ABCxyz" not in out
    assert "sig=" in out
    # urlencode may percent-encode ``<redacted>`` as ``%3Credacted%3E``.
    assert "redacted" in out


def test_url_with_access_token_query_param_is_redacted():
    url = "https://example.test/api?access_token=secret123&other=ok"
    out = sanitize_for_report(url)
    assert "secret123" not in out
    assert "other=ok" in out


def test_url_with_token_query_param_is_redacted():
    url = "https://example.test/path?token=abcdef&keep=1"
    out = sanitize_for_report(url)
    assert "abcdef" not in out
    assert "keep=1" in out


# --------------------------------------------------------------------------- #
# Freeform key=value                                                          #
# --------------------------------------------------------------------------- #


def test_freeform_password_key_equals_is_redacted():
    out = sanitize_for_report("password=secret123 foo=bar")
    assert "secret123" not in out
    assert "foo=bar" in out


def test_freeform_api_key_equals_is_redacted():
    out = sanitize_for_report("api_key=abc123 keep=ok")
    assert "abc123" not in out
    assert "keep=ok" in out


def test_freeform_non_sensitive_key_equals_passes_through():
    text = "count=42 name=alice region=westus2"
    assert sanitize_for_report(text) == text


# --------------------------------------------------------------------------- #
# Freeform key: value                                                         #
# --------------------------------------------------------------------------- #


def test_freeform_authorization_colon_is_redacted():
    out = sanitize_for_report('authorization: "Bearer xyz-token-value"')
    # Either the bearer-token matcher or the freeform colon matcher must
    # redact the secret; test only that the raw value is gone.
    assert "xyz-token-value" not in out


def test_freeform_secret_colon_is_redacted():
    out = sanitize_for_report('client_secret: "hush123"')
    assert "hush123" not in out


def test_freeform_non_sensitive_colon_passes_through():
    text = 'name: "alice"'
    assert sanitize_for_report(text) == text


# --------------------------------------------------------------------------- #
# Dicts                                                                       #
# --------------------------------------------------------------------------- #


def test_dict_with_sensitive_and_non_sensitive_keys():
    result = sanitize_for_report(
        {
            "password": "secret123",
            "username": "alice",
            "count": 7,
        }  # pragma: allowlist secret
    )
    assert result == {
        "password": "<redacted>",
        "username": "alice",
        "count": 7,
    }

    def test_sanitize_for_report_redacts_signed_url_mapping_keys():
        source_uri = "https://storage.test/data/input?sig=source-secret"

        sanitized = sanitize_for_report({source_uri: "azureai://data/input/versions/1"})

        serialized = str(sanitized)
        assert "source-secret" not in serialized
        assert "%3Credacted%3E" in serialized


def test_nested_dict_is_sanitized_recursively():
    result = sanitize_for_report(
        {
            "outer": {
                "api_key": "abc",
                "child": {"token": "xyz", "ok": "value"},
            },
            "other": "Bearer leak.me.please",
        }
    )
    assert result["outer"]["api_key"] == "<redacted>"
    assert result["outer"]["child"]["token"] == "<redacted>"
    assert result["outer"]["child"]["ok"] == "value"
    assert "leak.me.please" not in result["other"]


def test_case_variants_of_sensitive_keys_detected():
    result = sanitize_for_report(
        {"API-Key": "a", "Authorization": "b", "clientSecret": "c"}
    )
    assert result["API-Key"] == "<redacted>"
    assert result["Authorization"] == "<redacted>"
    assert result["clientSecret"] == "<redacted>"


# --------------------------------------------------------------------------- #
# Pass-through for non-string scalars / containers                            #
# --------------------------------------------------------------------------- #


def test_none_passes_through():
    assert sanitize_for_report(None) is None


def test_int_passes_through():
    assert sanitize_for_report(42) == 42


def test_empty_string_passes_through():
    assert sanitize_for_report("") == ""


def test_list_of_mixed_items_is_sanitized_elementwise():
    out = sanitize_for_report(
        ["plain", "Bearer hidden-token", {"password": "x"}, 1, None]
    )
    assert out[0] == "plain"
    assert "hidden-token" not in out[1]
    assert out[2] == {"password": "<redacted>"}
    assert out[3] == 1
    assert out[4] is None


def test_list_of_plain_strings_passes_through():
    text = ["one", "two", "three"]
    assert sanitize_for_report(text) == text
