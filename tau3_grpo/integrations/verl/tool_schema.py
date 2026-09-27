"""Opt-in lossless schema serialization for the pinned veRL tool registry."""

from copy import deepcopy

from pydantic import PrivateAttr, model_serializer
from verl.tools.schemas import OpenAIFunctionToolSchema


class FullToolSchema(OpenAIFunctionToolSchema):
    """Keep veRL's typed parser view, but render the complete official schema.

    The pinned schema model discards JSON Schema extras, including array items
    and $defs. The original payload is therefore carried separately through the
    tool config. Execution still uses the official environment's validation.
    """

    _original: dict = PrivateAttr()

    @model_serializer(mode="plain")
    def serialize_original(self):
        return deepcopy(self._original)

    @classmethod
    def from_payload(cls, payload, registered):
        normalized = deepcopy(payload)
        normalized["function"]["parameters"].setdefault("required", [])
        expected = OpenAIFunctionToolSchema.model_validate(normalized)
        if expected.model_dump() != registered.model_dump():
            raise ValueError("full schema payload differs from the registered tool schema")
        schema = cls.model_validate(normalized)
        schema._original = deepcopy(payload)
        return schema
