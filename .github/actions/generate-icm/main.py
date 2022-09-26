import argparse
import contextlib
from datetime import datetime
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from typing import Any, Literal, Optional, TypeAlias, TypeVar, overload
import requests
import yaml
from pydantic import BaseModel
from xml.etree import ElementTree

T = TypeVar("T")
U = TypeVar("U")

RoutingOptions: TypeAlias = Literal["None", "SkipTransferOnUpdate"]


class SOAPFault(Exception):
    """SOAP 1.2 Fault"""
    def __init__(
        self,
        *,
        code: str,
        reason: str,
        role: Optional[str] = None,
        node: Optional[str],
        detail: Optional[str] = None
    ):
        self.code = code
        self.reason = reason
        self.role = role
        self.node = node
        self.detail = detail


class AuthCert(BaseModel):
    cert: Path
    private_key: Path


class AddOrUpdateIncident2Args(BaseModel):
    routingOptions: RoutingOptions = "None"
    incident: dict
    connectorId: str


class AddOrUpdateIncident2Result(BaseModel):
    IncidentId: int
    Status: Literal["Invalid", "AddedNew", "UpdatedExisting",
                    "DidNotChangeExisting", "AlertSourceUpdatesPending",
                    "UpdateToHoldingNotAllowed", "Discarded"]


class ActionArgs(BaseModel):
    host: str
    auth: AuthCert
    args: AddOrUpdateIncident2Args


def sorted_dict(val: T) -> T:
    """Returns a copy of dict `val` where key-values are inserted in 
    alphabetical order. If `val` is not a dict, it's returned unchanged

    Args:
        val (T): Any value

    Returns:
        T: Either the same value passed in if not a dict, or a copy 
           with keys inserted in alphabetical order
    """
    if not isinstance(val, dict):
        return val

    return {
        k: sorted_dict(v)
        for k, v in sorted(val.items(), key=lambda kv: kv[0])
    }


@overload
def mergeLeft(left: T, right: U) -> U:
    ...


def mergeLeft(left: dict, right: dict) -> dict:
    if not (isinstance(left, dict) and isinstance(right, dict)):
        return right
    for k, v in right.items():
        left[k] = mergeLeft(left.get(k, None), v)

    return left


def string_to_tempfile(s: str):
    """Writes a utf-8 encoded string into a temporary file

    Args:
        s (str): Base64 encoded string

    Returns:
        NamedTemporaryFile: Temp File, must be closed by caller
    """
    tmp = NamedTemporaryFile()
    tmp.write(bytes(s, encoding="utf-8"))
    tmp.seek(0)
    return tmp


def toXML(obj: dict, xsi_namespace="i") -> str:
    """Serializes a dict into an XML string

    Args:
        obj (dict): dict to serialize
        xsi_namespace (str, optional): Prefix for the Schema Instance Namespace. Defaults to "i".

    Returns:
        str: xml string representing the dict
    """
    ATTRIBUTESKEY = "$attributes"
    NAMESPACEKEY = "$namespace"

    def tag(
        name: str,
        value: str,
        *,
        attributes: dict = {},
        namespace: str = None
    ) -> str:
        """Surrounds a string in xml tags

        Args:
            name (str): Name of tag
            value (str): Value of tag
            attributes (dict, optional): Dict of attributes for xml element. Defaults to {}.
            namespace (str, optional): Namespace prefix for the tag. Defaults to None.

        Returns:
            str: xml string
        """
        attributes_str = ' '.join(f'{k}="{v}"' for k, v in attributes.items())
        ns = f"{namespace}:" if namespace else ""
        if attributes_str:
            attributes_str = ' ' + attributes_str
        if value:
            return f'<{ns}{name}{attributes_str}>{value}</{ns}{name}>'
        return f'<{ns}{name}{attributes_str} />'

    def serialize_dict(val: dict, namespace: str = None) -> str:
        ns = val.pop(NAMESPACEKEY, namespace)
        return ''.join(serialize(k, v, ns) for k, v in val.items())

    def serialize(name: str, val: Any, namespace: str = None) -> str:
        """Serializes a key value pair into an XML element

        Args:
            name (str): Tag name for element
            val (Any): Value to serialize
            namespace (str, optional): Namespace to prepend to child tags
            of value. Defaults to None.

        Returns:
            str: XML string
        """
        t = lambda v, a={}: tag(name, v, attributes=a, namespace=namespace)
        if isinstance(val, dict):
            attrs = val.pop(ATTRIBUTESKEY, {})
            return t(serialize_dict(val, namespace), attrs)
        if isinstance(val, list):
            return t(
                ''.join(serialize_dict(v, namespace=namespace) for v in val)
            )
        elif isinstance(val, bool):
            return t(str(val).lower())
        elif val is None:
            return t(None, {f"{xsi_namespace}:nil": "true"})
        else:
            return t(str(val))

    return serialize_dict(obj)


@contextlib.contextmanager
def parseArgs():
    """Parses the github action arguments. Must be invoke with `with` statement

    Yields:
        _type_: ActionArgs
    """
    def getInput(name: str, *, required=False):
        val = os.environ.get(f'INPUT_{name.replace(" ", "_").upper()}', "")
        if not val and required:
            raise ValueError(f"Missing required parameter: {name}")
        return val

    certificate = getInput("certificate", required=True)
    private_key = getInput("private_key", required=True)
    connectorId = getInput("connector_id", required=True)
    host = getInput("host", required=True)
    body = yaml.load(getInput("args", required=True), yaml.Loader)

    with string_to_tempfile(certificate) as cert_file, string_to_tempfile(
        private_key
    ) as key_file:
        yield ActionArgs(
            host=host,
            auth=AuthCert(
                cert=Path(cert_file.name), private_key=Path(key_file.name)
            ),
            args=AddOrUpdateIncident2Args(**body, connectorId=connectorId)
        )


def add_or_update_incident_2(
    host: str, auth: AuthCert, args: AddOrUpdateIncident2Args
):
    now = datetime.utcnow().isoformat()
    soap_ns = "http://www.w3.org/2003/05/soap-envelope"
    icm_ns = "http://schemas.datacontract.org/2004/07/Microsoft.AzureAd.Icm.Types"
    baseIncident = {
        "OccurringLocation": {},
        "RaisingLocation": {},
        "Source":
            {
                "CreatedBy": "Monitor",
                "Origin": "Monitor",
                "SourceId": args.connectorId,
                "CreateDate": now,
                "ModifiedDate": now,
            },
        "Severity": 4,
    }
    incident = sorted_dict(
        {
            "$attributes":
                {
                    "xmlns:b": icm_ns,
                    "xmlns:i": "http://www.w3.org/2001/XMLSchema-instance"
                },
            "$namespace": "b",
            **mergeLeft(baseIncident, args.incident)
        }
    )
    soap_message = {
        "s:Envelope":
            {
                "$attributes":
                    {
                        "xmlns:s": soap_ns,
                        "xmlns:a": "http://www.w3.org/2005/08/addressing",
                    },
                "s:Header":
                    {
                        "a:Action":
                            "http://tempuri.org/IConnectorIncidentManager/AddOrUpdateIncident2",
                        "a:To":
                            f"https://{host}/Connector3/ConnectorIncidentManager.svc"
                    },
                "s:Body":
                    {
                        # Key order is significant, must be in alphabetical
                        # order
                        "AddOrUpdateIncident2":
                            {
                                "connectorId": args.connectorId,
                                "incident": incident,
                                "routingOptions": args.routingOptions,
                                "$attributes": {
                                    "xmlns": "http://tempuri.org/"
                                },
                            }
                    }
            }
    }

    response = requests.post(
        f"https://{host}/Connector3/ConnectorIncidentManager.svc",
        cert=(auth.cert, auth.private_key),
        headers={'Content-Type': 'application/soap+xml; charset=utf-8'},
        data='<?xml version="1.0" encoding="UTF-8"?>\n' + toXML(soap_message)
    )

    xml_result = ElementTree.fromstring(response.text)
    fault = xml_result.find(f".//{{{soap_ns}}}Fault")

    if fault:

        def innerXML(element: Optional[ElementTree.Element]) -> Optional[str]:
            return element and (
                element.text or
                ''.join(ElementTree.tostring(e, 'unicode') for e in element)
            )

        raise SOAPFault(
            code=innerXML(fault.find(f".//{{{soap_ns}}}Code")),
            reason=innerXML(fault.find(f".//{{{soap_ns}}}Reason")),
            role=innerXML(fault.find(f".//{{{soap_ns}}}Role")),
            detail=innerXML(fault.find(f".//{{{soap_ns}}}Detail")),
            node=innerXML(fault.find(f".//{{{soap_ns}}}Node")),
        )

    return AddOrUpdateIncident2Result(
        IncidentId=int(xml_result.findtext(f'.//{{{icm_ns}}}IncidentId')),
        Status=xml_result.findtext(f'.//{{{icm_ns}}}Status')
    )


def main():
    # Utility functions to report status to Github Action Runner
    core = {
        "debug": lambda s: print(f"::debug::{s}"),
        "error": lambda s: print(f"::error::{s}")
    }

    with parseArgs() as args:
        try:
            add_or_update_incident_2(**vars(args))
        except SOAPFault as e:
            core["debug"](e.detail)
            core["error"](e.reason)
            sys.exit(1)


if __name__ == "__main__":
    main()
