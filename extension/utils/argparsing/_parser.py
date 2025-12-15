from __future__ import annotations
import argparse
from enum import Enum
from pydantic_core import PydanticUndefined
from typing import Any, get_type_hints

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
        parsed_dict = {}
        for item in arg_list:
            key, val = item.split("=")
            parsed_dict[key] = val
        return parsed_dict

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
                    args_dict[field_name] = field_type(**parsed_dict)
                elif isinstance(args_value, field_type):
                    continue
                elif isinstance(args_value, dict):
                    args_dict[field_name] = field_type(**args_value)

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