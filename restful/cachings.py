from redis import Redis
from valkey import Valkey
from typing import _ProtocolMeta
from restful.configs import settings, Settings


""" caching_connector """
def _caching_connector(db_type: str, env_var: Settings) -> _ProtocolMeta:
  if db_type == 'redis':
    return Redis(
      host     = env_var.CACHING_HOST,
      port     = env_var.CACHING_PORT,
      password = env_var.CACHING_PASS,
      ssl      = True
    )
  
  elif db_type == 'valkey':
    return Valkey(
      host     = env_var.CACHING_HOST,
      port     = env_var.CACHING_PORT,
      password = env_var.CACHING_PASS,
      ssl      = True
    )

  # default
  else:
    return Redis(
      host     = env_var.CACHING_HOST,
      port     = env_var.CACHING_PORT,
      password = env_var.CACHING_PASS,
      ssl      = True
    )

connector: _ProtocolMeta = _caching_connector(
  db_type = settings.CACHING_TYPE, env_var = settings)

async def close_caching_connector() -> None:
    global connector
    if connector: await connector.close()
