from __future__ import annotations
import inspect
from typing import Any, Callable, Optional, Union
from pydantic import BaseModel


class PydanticArgsBase(BaseModel):
    class Config:
        validate_by_name = True

    def _as_flattened_dict(self) -> dict[str, Any]:
        flattened_dict: dict[str, Any] = {}
        for key, value in self.model_dump().items():
            if isinstance(value, BaseModel):
                for nested_key, nested_value in value.model_dump().items():
                    if nested_key in flattened_dict:
                        raise ValueError(f"Duplicate key found: '{nested_key}'")
                    flattened_dict[nested_key] = nested_value
            elif isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    if nested_key in flattened_dict:
                        raise ValueError(f"Duplicate key found: '{nested_key}'")
                    flattened_dict[nested_key] = nested_value
            else:
                if key in flattened_dict:
                    raise ValueError(f"Duplicate key found: '{key}'")
                flattened_dict[key] = value
        return flattened_dict

    def call(self, method_or_class: Union[Callable[..., Any], type], **kwargs) -> Any:
        if inspect.isclass(method_or_class):
            signature = inspect.signature(method_or_class.__init__)
        else:
            signature = inspect.signature(method_or_class)
        args = self._as_flattened_dict()
        method_params = signature.parameters
        filtered_args = {key: value for key, value in args.items() if key in method_params}
        filtered_args.update({key: value for key, value in kwargs.items() if key in method_params})
        return method_or_class(**filtered_args)

    def save(self, path: Optional[str] = None) -> None:
        if path is None:
            path = "."
        with open(f"{path}/args.json", "w") as f:
            f.write(self.model_dump_json(indent=4))


class AdditionalArgsBase(BaseModel):
    pass
