from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OPENAI_API_KEY: str

    SUPABASE_URL: str
    # service_role: every table has RLS on and this service has no end-user
    # session, so the anon key would silently return zero rows.
    SUPABASE_SERVICE_KEY: str

    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_PASSWORD: str = ""
    REDIS_DB: int = 0

    # Pinned. service.py rejects any request whose org_id / workflow_id does
    # not match these — this deployment serves exactly the RelayN workflow.
    RELAYN_ORG_ID: str
    RELAYN_WORKFLOW_ID: str

    # Shared with relayn_gateway, which holds the same value in the
    # AGENT_SECRET_* variable this service is registered under
    # (agent_services.hmac_secret_env). Empty refuses every /v1 request:
    # the signature of "" is computable by anyone, so it fails closed.
    GATEWAY_HMAC_SECRET: str = ""

    # Base URL of the shared relayn_services deployment. Optional — the KB
    # ingest script is self-contained and does not call it; kept for parity.
    RELAYN_SERVICES_URL: str = "http://localhost:8001"

    CAL_API_KEY: str = ""
    CAL_EVENT_TYPE_ID: int = 0
    CAL_TIMEZONE: str = "Asia/Kathmandu"

    @property
    def REDIS_URL(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
