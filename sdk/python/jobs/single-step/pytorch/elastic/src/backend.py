"""
Rendezvous backend for Azure Table Storage.
"""
import base64
import binascii
from typing import Optional, Tuple

from azure.data.tables import TableServiceClient, TableClient
from azure.core import MatchConditions
from azure.core.exceptions import ResourceNotFoundError, ResourceModifiedError

from torch.distributed.elastic.rendezvous import (
    RendezvousConnectionError,
    RendezvousStateError,
)
from torch.distributed.elastic.rendezvous.dynamic_rendezvous import (
    RendezvousBackend,
    Token,
)

from utils import get_table_name


class AzureRendezvousBackend(RendezvousBackend):
    """Represents an Azure Table Storage-based rendezvous backend."""

    _prefix = "torchelastic"  # Prefix for the table name

    _run_id: str
    _table_service_client: TableServiceClient
    _table_client: TableClient
    _table_name: str
    _partition_key: str

    def __init__(
        self,
        run_id: str,
        table_service_client: TableServiceClient,
    ) -> None:
        self._run_id = run_id
        self._table_service_client = table_service_client
        self._table_name = get_table_name(self._prefix, run_id)
        self._table_client = self._table_service_client.create_table_if_not_exists(self._table_name)
        self._partition_key = "state"

    @property
    def name(self) -> str:
        """See base class."""
        return "azuretable"

    def get_state(self) -> Optional[Tuple[bytes, Token]]:
        """See base class."""
        try:
            entity = self._table_client.get_entity('state', 'state')
            etag = entity.metadata['etag']
            state = base64.b64decode(entity['State'])
            return state, etag
        except ResourceNotFoundError:
            return None

    def set_state(self, state: bytes, token: Optional[Token] = None) -> Optional[Tuple[bytes, Token, bool]]:
        entity = {"PartitionKey": "state", "RowKey": "state", "State": base64.b64encode(state).decode()}
        try:
            if token:
                upsert_result = self._table_client.update_entity(
                    entity, etag=token, match_condition=MatchConditions.IfNotModified
                )
            else:
                upsert_result = self._table_client.upsert_entity(entity)
            return state, upsert_result["etag"], True
        except Exception as e:
            # print(f"Exception writing to table:")
            # traceback.print_exc()
            pass
        curr_state = self.get_state()
        return curr_state[0], curr_state[1], False

    def _decode_state(self, entity: dict) -> Tuple[bytes, Token]:
        """Decodes the state from the entity and returns it along with the token."""
        base64_state = entity["State"]

        try:
            state = base64.b64decode(base64_state)
        except binascii.Error as exc:
            raise RendezvousStateError("The state object is corrupt. See inner exception for details.") from exc

        return state, entity.metadata["etag"]
