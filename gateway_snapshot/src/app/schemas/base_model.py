from pydantic import BaseModel, ConfigDict


class StrictBaseModel(BaseModel):
    """Base model that forbids extra/unknown fields."""
    model_config = ConfigDict(extra="forbid")


class IgnoreExtraBaseModel(BaseModel):
    """Base model that ignores unknown fields"""
    model_config = ConfigDict(extra="ignore")


class AllowExtraBaseModel(BaseModel):
    """Keeps and returns unknown fields"""
    model_config = ConfigDict(extra="allow")
