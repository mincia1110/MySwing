"""SQLAlchemy ORM models for the Baseball Swing Analysis service.

Defines the database tables as specified in the design document's Database Schema section.
Maps to Requirements 2.7, 2.8, 7.2, 7.3.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class UserTable(Base):
    """Users table - stores registered user accounts."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    profile: Mapped["UserProfileTable"] = relationship(
        back_populates="user", uselist=False, lazy="selectin"
    )
    videos: Mapped[list["VideoTable"]] = relationship(back_populates="user", lazy="selectin")
    analyses: Mapped[list["AnalysisTable"]] = relationship(back_populates="user", lazy="selectin")


class UserProfileTable(Base):
    """User profiles table - stores physical and batting characteristics.

    Required fields (Requirement 2.1): height, bat_length, batting_direction
    Optional fields (Requirements 2.3, 2.4): weight, camera_direction, age_group, level, bat_weight
    """

    __tablename__ = "user_profiles"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False
    )

    # Required fields
    height: Mapped[float] = mapped_column(Float, nullable=False)  # cm, 100-220
    bat_length: Mapped[float] = mapped_column(Float, nullable=False)  # inches, 24-36
    batting_direction: Mapped[str] = mapped_column(
        String(10), nullable=False
    )  # "left" or "right"

    # Optional recommended fields
    weight: Mapped[float | None] = mapped_column(Float, nullable=True)  # kg
    camera_direction: Mapped[str | None] = mapped_column(
        String(10), nullable=True
    )  # "front", "side", "rear"

    # Optional fields
    age_group: Mapped[str | None] = mapped_column(String(50), nullable=True)
    level: Mapped[str | None] = mapped_column(
        String(20), nullable=True
    )  # "professional", "college", "high_school", "recreational"
    bat_weight: Mapped[float | None] = mapped_column(Float, nullable=True)  # oz, 16-36

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    user: Mapped["UserTable"] = relationship(back_populates="profile")


class VideoTable(Base):
    """Videos table - stores uploaded video file metadata."""

    __tablename__ = "videos"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    file_key: Mapped[str] = mapped_column(String(500), nullable=False)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    resolution_width: Mapped[int] = mapped_column(Integer, nullable=False)
    resolution_height: Mapped[int] = mapped_column(Integer, nullable=False)
    frame_rate: Mapped[float] = mapped_column(Float, nullable=False)
    format: Mapped[str] = mapped_column(String(10), nullable=False)
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    user: Mapped["UserTable"] = relationship(back_populates="videos")
    analyses: Mapped[list["AnalysisTable"]] = relationship(back_populates="video", lazy="selectin")
    quality_checks: Mapped[list["QualityCheckTable"]] = relationship(
        back_populates="video", lazy="selectin"
    )


class AnalysisTable(Base):
    """Analyses table - tracks analysis job status and lifecycle."""

    __tablename__ = "analyses"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    video_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending"
    )  # pending, preprocessing, analyzing, evaluating, generating_report, completed, failed
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    user: Mapped["UserTable"] = relationship(back_populates="analyses")
    video: Mapped["VideoTable"] = relationship(back_populates="analyses")
    result: Mapped["AnalysisResultTable"] = relationship(
        back_populates="analysis", uselist=False, lazy="selectin"
    )


class AnalysisResultTable(Base):
    """Analysis results table - stores the complete analysis output as JSONB.

    Maps to Requirements 7.2, 7.3 for structured analysis data storage.
    """

    __tablename__ = "analysis_results"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analyses.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    biomechanics_data: Mapped[dict] = mapped_column(JSONB, nullable=False)
    swing_phases_data: Mapped[dict] = mapped_column(JSONB, nullable=False)
    evaluations_data: Mapped[dict] = mapped_column(JSONB, nullable=False)
    improvements_data: Mapped[dict] = mapped_column(JSONB, nullable=False)
    drill_recommendations: Mapped[dict] = mapped_column(JSONB, nullable=False)
    overlay_video_key: Mapped[str | None] = mapped_column(String(500), nullable=True)
    processing_time_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    analysis: Mapped["AnalysisTable"] = relationship(back_populates="result")


class QualityCheckTable(Base):
    """Quality checks table - stores video quality validation results.

    Maps to Requirement 9.8 for quality summary display.
    """

    __tablename__ = "quality_checks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    video_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False
    )
    brightness_status: Mapped[str] = mapped_column(
        String(10), nullable=False
    )  # "pass" or "warning"
    framing_status: Mapped[str] = mapped_column(String(10), nullable=False)
    resolution_status: Mapped[str] = mapped_column(String(10), nullable=False)
    frame_rate_stability_status: Mapped[str] = mapped_column(String(10), nullable=False)
    details: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    checked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    video: Mapped["VideoTable"] = relationship(back_populates="quality_checks")


class ReferenceDataTable(Base):
    """Reference data table - stores professional player benchmark data.

    Maps to Requirements 7.2, 7.3 for level-appropriate reference comparisons.
    Includes data for different levels: professional, college, high_school, recreational.
    """

    __tablename__ = "reference_data"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    level: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # "professional", "college", "high_school", "recreational"
    age_group: Mapped[str] = mapped_column(
        String(50), nullable=False, default="adult"
    )
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)
    min_value: Mapped[float] = mapped_column(Float, nullable=False)
    max_value: Mapped[float] = mapped_column(Float, nullable=False)
    optimal_min: Mapped[float] = mapped_column(Float, nullable=False)
    optimal_max: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str | None] = mapped_column(String(255), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        UniqueConstraint("level", "age_group", "metric_name", name="uq_reference_data_key"),
    )
