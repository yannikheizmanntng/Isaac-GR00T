from __future__ import annotations

import argparse
from enum import Enum
from typing import Any, Union, get_args, get_origin, get_type_hints

from pydantic_core import PydanticUndefined

from ._args import PydanticArgsBase, AdditionalArgsBase


class ArgsParser:
    def __init__(self, Args: type[PydanticArgsBase]) -> None:
        self._ArgsType = Args

    def _get_alias_mapping(self) -> dict[str, str]:
        alias_mapping = {}
        for field_name, field_type_str in self._ArgsType.__annotations__.items():
            field_info = self._ArgsType.model_fields[field_name]
            if field_info.alias:
                alias_mapping[field_info.alias] = field_name
        return alias_mapping

    def _parse_additional_args(self, arg_list: list[str]) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        for item in arg_list:
            key, val = item.split("=", 1)  # split only once in case val contains '='

            # accumulate duplicates into lists (enables list[str] via repeated keys)
            if key in parsed:
                if isinstance(parsed[key], list):
                    parsed[key].append(val)
                else:
                    parsed[key] = [parsed[key], val]
            else:
                parsed[key] = val

        return parsed

    def _is_list_type(self, hint: Any) -> bool:
        origin = get_origin(hint)
        if origin is list:
            return True
        if origin is Union:
            return any(get_origin(arg) is list for arg in get_args(hint))
        return False

    def _coerce_additional_args_types(
        self,
        additional_model: type[AdditionalArgsBase],
        parsed: dict[str, Any],
    ) -> dict[str, Any]:
        hints = get_type_hints(additional_model)

        # Ensure any fields annotated as list[...] (or Optional[list[...]]) are lists
        for key, hint in hints.items():
            if self._is_list_type(hint):
                if key in parsed and not isinstance(parsed[key], list):
                    parsed[key] = [parsed[key]]

        return parsed

    def _create_args_from_namespace(self, args: argparse.Namespace):
        args_dict = vars(args)

        for alias, field_name in self._get_alias_mapping().items():
            if alias in args_dict:
                args_dict[field_name] = args_dict.pop(alias)

        for field_name, field_type in get_type_hints(self._ArgsType).items():
            if isinstance(field_type, type) and issubclass(field_type, AdditionalArgsBase):
                args_value = args_dict.get(field_name)
                if isinstance(args_value, list):
                    parsed_dict = self._parse_additional_args(args_value)
                    parsed_dict = self._coerce_additional_args_types(field_type, parsed_dict)
                    args_dict[field_name] = field_type(**parsed_dict)
                elif isinstance(args_value, field_type):
                    continue
                elif isinstance(args_value, dict):
                    parsed_dict = self._coerce_additional_args_types(field_type, args_value)
                    args_dict[field_name] = field_type(**parsed_dict)

        args_dict = {k: v for k, v in args_dict.items() if v is not PydanticUndefined}
        return self._ArgsType.model_validate(args_dict)

    def parse(self):
        parser = argparse.ArgumentParser()

        for field_name, field_type in get_type_hints(self._ArgsType).items():
            field_info = self._ArgsType.model_fields[field_name]

            choices = None
            if isinstance(field_type, type) and issubclass(field_type, Enum):
                choices = [item.value for item in field_type]

            aliases = [f"--{field_name}"]
            if field_info.alias:
                aliases.append(f"-{field_info.alias}")

            if isinstance(field_type, type) and issubclass(field_type, AdditionalArgsBase):
                parser.add_argument(
                    *aliases,
                    type=str,
                    nargs="+",
                    default=field_info.default,
                    help=field_info.description,
                )
            else:
                parser.add_argument(
                    *aliases,
                    type=str,
                    choices=choices,
                    default=field_info.default,
                    help=field_info.description,
                )

        args = parser.parse_args()
        return self._create_args_from_namespace(args)