from __future__ import annotations
from typing import List, Optional
from pydantic import Field

from extension.utils.argparsing import AdditionalArgsBase


class InferenceArgs(AdditionalArgsBase):
    model_path: str = Field(
        default="/home/innovation-hacking/heizmany/Isaac-GR00T/models/GR00T-N1.5-3B_20260210-200439/checkpoint-100000",
        description="Path to the model checkpoint directory.",
    )
    port: int = Field(
        default=5555,
        description="The port number for the server.",
    )
    host: str = Field(
        default="localhost",
        description="The host address for the server.",
    )
    server: bool = Field(
        default=True,
        description="Whether to run the server.",
    )
    client: bool = Field(
        default=False,
        description="Whether to run the client.",
    )
    denoising_steps: int = Field(
        default=4,
        description="The number of denoising steps to use.",
    )
    api_token: Optional[str] = Field(
        default=None,
        description="API token for authentication. If not provided, authentication is disabled.",
    )
    http_server: bool = Field(
        default=False,
        description="Whether to run it as HTTP server. Default is ZMQ server.",
    )

    def to_cli(self) -> List[str]:
        argv: List[str] = []
        argv += ["--model-path", self.model_path]
        argv += ["--host", self.host]
        argv += ["--port", str(self.port)]
        argv += ["--denoising-steps", str(self.denoising_steps)]
        if self.server:
            argv += ["--server"]
        if self.client:
            argv += ["--client"]
        if self.http_server:
            argv += ["--http-server"]
        if self.api_token:
            argv += ["--api-token", self.api_token]
        return argv
