import re
from pathlib import Path

from pydantic import SecretStr

from jev_rag.agent.contracts import ProviderError


def load_credentials(workspace: Path, environment: dict, provider: str) -> SecretStr:
    """Read only ignored credential files, after live authorization. Never log values.

    This small literal parser intentionally rejects shell expansion and executable
    dotenv syntax. Process environment takes precedence over .env.local over .env.
    """
    if provider == "deepseek":
        names = ("DEEPSEEK_API_KEY", "LLM_API_KEY")
    elif provider == "jev":
        names = ("TYPESAFE_API_KEY", "JEV_API_KEY")
    else:
        names = ("LLM_API_KEY",)
    values = {}
    for filename in (".env", ".env.local"):
        path = workspace / filename
        if not path.exists():
            continue
        if path.is_symlink() or path.resolve().parent != workspace.resolve():
            raise ProviderError("credential_file_must_be_inside_workspace")
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = re.fullmatch(r"([A-Z_][A-Z0-9_]*)\s*=\s*(.*)", line)
            if not match or match[1] not in names:
                continue
            value = match[2].strip()
            if len(value) >= 2 and value[0] in {"'", '"'} and value[-1] == value[0]:
                value = value[1:-1]
            if any(char in value for char in ("$", "`", "\n", "\r", " ", "\t")):
                raise ProviderError("credential_value_must_be_literal_without_whitespace")
            values[match[1]] = value
    values.update({key: environment[key] for key in names if environment.get(key)})
    for name in names:
        if values.get(name):
            return SecretStr(values[name])
    raise ProviderError("credential_missing_set_ignored_env_local")
