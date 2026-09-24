"""Immutable, validated hardware snapshots attached to native steering beams."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .configuration import GeometryConfig


class FrozenGeometry(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid', strict=True, allow_inf_nan=False)

    def replace(self, **changes):
        """Return a validated replacement; nested dictionaries update sections."""
        data = self.model_dump()
        for key, value in changes.items():
            if isinstance(value, dict) and isinstance(data.get(key), dict):
                data[key].update(value)
            else:
                data[key] = value.model_dump() if isinstance(value, BaseModel) else value
        return type(self).model_validate(data)


class AssemblyGeometry(FrozenGeometry):
    enabled: bool


class FieldGeometry(FrozenGeometry):
    width_mm: float = Field(gt=0)
    height_mm: float = Field(gt=0)
    material: str
    outer_width_mm: float = Field(gt=0)
    outer_height_mm: float = Field(gt=0)
    side_margin_mm: float = Field(ge=0)


class MLCGeometry(FrozenGeometry):
    enabled: bool
    source_to_center_mm: float = Field(gt=0)
    thickness_mm: float = Field(gt=0)
    round_tip_enabled: bool
    tip_radius_mm: float = Field(gt=0)
    tip_center_z_offset_mm: float


class JawGeometry(FrozenGeometry):
    enabled: bool
    source_to_center_mm: float = Field(gt=0)
    thickness_mm: float = Field(gt=0)


class SlitGeometry(FrozenGeometry):
    enabled: bool
    type: Literal['slits']
    material: str
    source_to_center_mm: float = Field(gt=0)
    thickness_mm: float = Field(gt=0)
    width_mm: float = Field(gt=0)
    height_mm: float = Field(gt=0)
    rotation_x_deg: float
    rotation_y_deg: float
    brass_frame_thickness_x_mm: float = Field(gt=0)
    brass_frame_thickness_y_mm: float = Field(gt=0)
    slit_count: int = Field(gt=0, le=10001)
    slit_entrance_width_mm: float = Field(gt=0)
    slit_entrance_ctc_mm: float = Field(gt=0)
    blade_thickness_mm: float = Field(gt=0)
    lateral_shift_mm: float = 0.0


class BeamGeometry(FrozenGeometry):
    assembly: AssemblyGeometry
    field: FieldGeometry
    mlc: MLCGeometry
    jaws: JawGeometry
    aperture: SlitGeometry

    @model_validator(mode='after')
    def check_configuration(self):
        self.as_config().validate()
        # Resolve containment, focusing and stage separation at the actual SAD
        # later, at the steering/preparation boundary.
        return self

    @classmethod
    def from_config(cls, config=None):
        return cls.model_validate((config or GeometryConfig.load()).data)

    def as_config(self):
        return GeometryConfig.from_data(self.model_dump())
