"""
feature_extraction.py

note about the importance of dt and resampling: Different touchscreens have different sampling rates
and therefore the density of points in time can vary widely across samples. To understand if the user 
is drawing a long stroke slowly or a short stroke quickly, we need to capture the timing information as a delta time
from point to point. This allows the model to learn patterns in the timing of strokes and implements 
equidistant resampling to ensure that the spatial distribution of points is consistent across samples. 


Paper-congruent raw-point feature extraction for online handwriting.

This module implements the raw-touch-point branch described in:

    Victor Carbune et al.,
    "Fast Multi-language LSTM-based Online Handwriting Recognition",
    Section 2.1.1, arXiv:1902.10525.

It intentionally does NOT implement the paper's Bézier-curve representation as per alannas instruction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Mapping, Optional, Sequence, Tuple
import argparse
import json
import math

import numpy as np

try:
    from .inkml_parser import InkMLParser, InkPoint, InkSample, InkStroke
except ImportError:
    from inkml_parser import InkMLParser, InkPoint, InkSample, InkStroke


class FeatureExtractionError(ValueError):
    """Raised when a sample cannot be converted into valid model features."""


PenStateMode = Literal["last_point_up", "explicit_up_event"]


@dataclass(frozen=True)
class FeatureExtractionConfig:
    """Configuration for the paper-compatible raw-point pipeline."""

    spatial_step: float = 0.05
    surrogate_area_scale: float = 1.20
    min_coordinate_scale: float = 1e-6
    monotonic_time_tolerance: float = 1e-9
    require_monotonic_time: bool = True
    preserve_single_point_strokes: bool = True
    pen_state_mode: PenStateMode = "last_point_up"

    def validate(self) -> None:
        if not math.isfinite(self.spatial_step) or self.spatial_step <= 0:
            raise FeatureExtractionError("spatial_step must be positive and finite.")
        if (
            not math.isfinite(self.surrogate_area_scale)
            or self.surrogate_area_scale < 1.0
        ):
            raise FeatureExtractionError(
                "surrogate_area_scale must be finite and at least 1.0."
            )
        if self.min_coordinate_scale <= 0:
            raise FeatureExtractionError("min_coordinate_scale must be positive.")
        if self.monotonic_time_tolerance < 0:
            raise FeatureExtractionError(
                "monotonic_time_tolerance cannot be negative."
            )
        if self.pen_state_mode not in {"last_point_up", "explicit_up_event"}:
            raise FeatureExtractionError(
                f"Unsupported pen_state_mode: {self.pen_state_mode!r}."
            )


@dataclass(frozen=True)
class NormalizationParameters:
    """Coordinate transform applied to one complete ink sample."""

    x_origin: float
    y_area_min: float
    scale: float
    used_known_writing_area: bool


@dataclass(frozen=True)
class FeatureSequence:
    """Variable-length model input produced from one InkSample."""

    features: np.ndarray
    sequence_length: int
    label: Optional[str]
    sample_id: str
    normalized_strokes: Tuple[InkStroke, ...]
    resampled_strokes: Tuple[InkStroke, ...]
    normalization: NormalizationParameters
    metadata: Mapping[str, object] = field(default_factory=dict)

    FEATURE_NAMES: Tuple[str, ...] = (
        "delta_x",
        "delta_y",
        "delta_t",
        "pen_state",
        "new_stroke",
    )

    def __post_init__(self) -> None:
        if self.features.ndim != 2 or self.features.shape[1] != 5:
            raise FeatureExtractionError(
                f"features must have shape [T, 5], got {self.features.shape}."
            )
        if self.sequence_length != self.features.shape[0]:
            raise FeatureExtractionError(
                "sequence_length must equal features.shape[0]."
            )
        if not np.isfinite(self.features).all():
            raise FeatureExtractionError("features contain NaN or infinity.")


class PaperFeatureExtractor:
    """Create [dx, dy, dt, p, n] features from an :class:`InkSample`.

    Paper-congruent behavior:
    - x and y are scaled isometrically using writing-area height.
    - x is shifted so the first point has x = 0.
    - the writing-area y range maps to [0, 1].
    - when the area is unknown, a 20%-expanded surrogate vertical area is used.
    - strokes are spatially resampled at delta = 0.05 by default.
    - x, y, and t are linearly interpolated during spatial resampling.
    - deltas are computed across the full flattened sequence and are not reset
      at stroke boundaries.
    """

    def __init__(self, config: Optional[FeatureExtractionConfig] = None) -> None:
        self.config = config or FeatureExtractionConfig()
        self.config.validate()

    def transform(self, sample: InkSample) -> FeatureSequence:
        """Run the complete feature-extraction pipeline."""

        self._validate_sample(sample)
        time_shifted = self._shift_time_to_zero(sample)
        normalized_strokes, normalization = self._normalize_coordinates(time_shifted)
        resampled_strokes = tuple(
            self._resample_stroke(stroke) for stroke in normalized_strokes
        )
        features = self._encode_delta_features(resampled_strokes)

        metadata: Dict[str, object] = {
            "feature_names": FeatureSequence.FEATURE_NAMES,
            "spatial_step": self.config.spatial_step,
            "surrogate_area_scale": self.config.surrogate_area_scale,
            "pen_state_mode": self.config.pen_state_mode,
            "original_stroke_count": len(sample.strokes),
            "original_point_count": sample.point_count,
            "resampled_point_count_without_optional_up_events": sum(
                len(stroke.points) for stroke in resampled_strokes
            ),
        }

        return FeatureSequence(
            features=features,
            sequence_length=int(features.shape[0]),
            label=sample.label,
            sample_id=sample.sample_id,
            normalized_strokes=normalized_strokes,
            resampled_strokes=resampled_strokes,
            normalization=normalization,
            metadata=metadata,
        )

    def _validate_sample(self, sample: InkSample) -> None:
        flat_points = [point for stroke in sample.strokes for point in stroke.points]
        if not flat_points:
            raise FeatureExtractionError("Sample contains no points.")

        for point in flat_points:
            if not all(math.isfinite(value) for value in (point.x, point.y, point.t)):
                raise FeatureExtractionError(
                    f"Sample {sample.sample_id!r} contains non-finite values."
                )

        if self.config.require_monotonic_time:
            times = np.asarray([point.t for point in flat_points], dtype=np.float64)
            differences = np.diff(times)
            bad = differences < -self.config.monotonic_time_tolerance
            if np.any(bad):
                index = int(np.flatnonzero(bad)[0])
                raise FeatureExtractionError(
                    "Timestamps are not globally monotonic. "
                    f"Negative jump at flattened points {index}->{index + 1}: "
                    f"{times[index]} -> {times[index + 1]}. "
                    "Do not silently repair per-stroke-local clocks unless the "
                    "dataset format explicitly requires it."
                )

    def _shift_time_to_zero(self, sample: InkSample) -> InkSample:
        first_time = sample.strokes[0].points[0].t
        shifted_strokes = tuple(
            InkStroke(
                points=tuple(
                    InkPoint(point.x, point.y, point.t - first_time)
                    for point in stroke.points
                ),
                stroke_id=stroke.stroke_id,
            )
            for stroke in sample.strokes
        )

        return InkSample(
            strokes=shifted_strokes,
            label=sample.label,
            sample_id=sample.sample_id,
            source_path=sample.source_path,
            writing_area=sample.writing_area,
            metadata=sample.metadata,
        )

# Most important -> this takes the raw ink sample and normalizes the coordinates according to the paper's specifications. 
# Paper: Fast Multi Language LSTM based Online Handwriting.pdf - found in ~/Research Papers and References/Model
# It handles both known and unknown writing areas, applying appropriate scaling and 
# shifting to ensure the features are consistent with the model's expectations.
# We need normalization to ensure that not only the modle can learn effectively but also for 
# edit functionality in the future. The input will be normalized but we can always reconstruct the absolute coordinates 
# to change the ink and then re-apply the feature extraction to het the new modified features for the model. 

    def _normalize_coordinates(
        self, sample: InkSample
    ) -> Tuple[Tuple[InkStroke, ...], NormalizationParameters]:
        all_points = [point for stroke in sample.strokes for point in stroke.points]
        x_origin = all_points[0].x

        if sample.writing_area is not None and sample.writing_area.height > 0:
            y_area_min = sample.writing_area.min_y
            scale = sample.writing_area.height
            used_known_area = True
        else:
            observed_y = np.asarray([point.y for point in all_points], dtype=np.float64)
            observed_min = float(np.min(observed_y))
            observed_max = float(np.max(observed_y))
            observed_height = observed_max - observed_min

            if observed_height < self.config.min_coordinate_scale:
                observed_x = np.asarray([point.x for point in all_points], dtype=np.float64)
                observed_width = float(np.max(observed_x) - np.min(observed_x))
                scale = max(observed_width, 1.0, self.config.min_coordinate_scale)
                y_area_min = observed_min - 0.5 * scale
            else:
                scale = observed_height * self.config.surrogate_area_scale
                total_padding = scale - observed_height
                y_area_min = observed_min - 0.5 * total_padding

            used_known_area = False

        if not math.isfinite(scale) or scale < self.config.min_coordinate_scale:
            raise FeatureExtractionError(
                f"Invalid coordinate scale {scale} for sample {sample.sample_id!r}."
            )

        normalized_strokes: List[InkStroke] = []
        for stroke in sample.strokes:
            normalized_points = tuple(
                InkPoint(
                    x=(point.x - x_origin) / scale,
                    y=(point.y - y_area_min) / scale,
                    t=point.t,
                )
                for point in stroke.points
            )
            normalized_strokes.append(
                InkStroke(normalized_points, stroke_id=stroke.stroke_id)
            )

        parameters = NormalizationParameters(
            x_origin=float(x_origin),
            y_area_min=float(y_area_min),
            scale=float(scale),
            used_known_writing_area=used_known_area,
        )
        return tuple(normalized_strokes), parameters

    def _resample_stroke(self, stroke: InkStroke) -> InkStroke:
        points = stroke.points

        if len(points) == 1:
            if not self.config.preserve_single_point_strokes:
                raise FeatureExtractionError(
                    "Single-point stroke encountered while "
                    "preserve_single_point_strokes=False."
                )
            return stroke

        values = np.asarray(
            [[point.x, point.y, point.t] for point in points],
            dtype=np.float64,
        )
        spatial_deltas = np.diff(values[:, :2], axis=0)
        segment_lengths = np.linalg.norm(spatial_deltas, axis=1)
        cumulative = np.concatenate(
            [np.asarray([0.0], dtype=np.float64), np.cumsum(segment_lengths)]
        )
        total_length = float(cumulative[-1])

        if total_length < self.config.min_coordinate_scale:
            if not self.config.preserve_single_point_strokes:
                raise FeatureExtractionError(
                    "Zero-length stroke encountered while "
                    "preserve_single_point_strokes=False."
                )
            representative = InkPoint(
                x=float(values[0, 0]),
                y=float(values[0, 1]),
                t=float(values[-1, 2]),
            )
            return InkStroke((representative,), stroke_id=stroke.stroke_id)

        keep_indices: List[int] = [0]
        for index in range(1, len(cumulative)):
            if cumulative[index] > cumulative[keep_indices[-1]] + self.config.min_coordinate_scale:
                keep_indices.append(index)
            else:
                keep_indices[-1] = index

        cumulative_unique = cumulative[keep_indices]
        values_unique = values[keep_indices]

        targets = np.arange(
            0.0,
            total_length,
            self.config.spatial_step,
            dtype=np.float64,
        )
        if targets.size == 0 or not np.isclose(targets[-1], total_length):
            targets = np.append(targets, total_length)
        else:
            targets[-1] = total_length

        x_resampled = np.interp(targets, cumulative_unique, values_unique[:, 0])
        y_resampled = np.interp(targets, cumulative_unique, values_unique[:, 1])
        t_resampled = np.interp(targets, cumulative_unique, values_unique[:, 2])

        resampled_points = tuple(
            InkPoint(float(x), float(y), float(t))
            for x, y, t in zip(x_resampled, y_resampled, t_resampled)
        )
        return InkStroke(resampled_points, stroke_id=stroke.stroke_id)

    def _encode_delta_features(self, strokes: Sequence[InkStroke]) -> np.ndarray:
        absolute_rows: List[Tuple[float, float, float, float, float]] = []

        for stroke in strokes:
            point_count = len(stroke.points)
            for point_index, point in enumerate(stroke.points):
                new_stroke = 1.0 if point_index == 0 else 0.0

                if self.config.pen_state_mode == "last_point_up":
                    pen_state = 0.0 if point_index == point_count - 1 else 1.0
                else:
                    pen_state = 1.0

                absolute_rows.append(
                    (point.x, point.y, point.t, pen_state, new_stroke)
                )

            if self.config.pen_state_mode == "explicit_up_event":
                endpoint = stroke.points[-1]
                absolute_rows.append(
                    (endpoint.x, endpoint.y, endpoint.t, 0.0, 0.0)
                )

        if not absolute_rows:
            raise FeatureExtractionError("No points remain after resampling.")

        absolute = np.asarray(absolute_rows, dtype=np.float64)
        features = np.zeros((absolute.shape[0], 5), dtype=np.float32)

        features[1:, 0:3] = np.diff(absolute[:, 0:3], axis=0).astype(np.float32)
        features[:, 3] = absolute[:, 3].astype(np.float32)
        features[:, 4] = absolute[:, 4].astype(np.float32)

        tiny_negative = (
            (features[:, 2] < 0)
            & (features[:, 2] >= -self.config.monotonic_time_tolerance)
        )
        features[tiny_negative, 2] = 0.0

        if np.any(features[:, 2] < -self.config.monotonic_time_tolerance):
            first_bad = int(np.flatnonzero(features[:, 2] < 0)[0])
            raise FeatureExtractionError(
                f"Negative delta_t remained after preprocessing at row {first_bad}."
            )

        return features

    @staticmethod
    def reconstruct_absolute_deltas(
        features: np.ndarray,
        *,
        initial_y: float = 0.0,
        initial_t: float = 0.0,
    ) -> np.ndarray:
        """Reconstruct relative absolute x, y, and t from the first 3 columns."""

        array = np.asarray(features)
        if array.ndim != 2 or array.shape[1] < 3:
            raise FeatureExtractionError("features must have shape [T, >=3].")

        reconstructed = np.cumsum(array[:, :3], axis=0, dtype=np.float64)
        reconstructed[:, 1] += initial_y
        reconstructed[:, 2] += initial_t
        return reconstructed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract paper-compatible [dx, dy, dt, p, n] features."
    )
    parser.add_argument("inkml", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--default-time-unit", default="s")
    parser.add_argument("--spatial-step", type=float, default=0.05)
    parser.add_argument("--surrogate-area-scale", type=float, default=1.20)
    parser.add_argument(
        "--pen-state-mode",
        choices=("last_point_up", "explicit_up_event"),
        default="last_point_up",
    )
    args = parser.parse_args()

    ink_parser = InkMLParser(
        require_time=True,
        default_time_unit=args.default_time_unit,
        convert_time_to_seconds=True,
    )
    sample = ink_parser.parse(args.inkml)

    extractor = PaperFeatureExtractor(
        FeatureExtractionConfig(
            spatial_step=args.spatial_step,
            surrogate_area_scale=args.surrogate_area_scale,
            pen_state_mode=args.pen_state_mode,
        )
    )
    sequence = extractor.transform(sample)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=sequence.features,
        sequence_length=np.asarray(sequence.sequence_length, dtype=np.int64),
        label=np.asarray(sequence.label or ""),
        sample_id=np.asarray(sequence.sample_id),
        feature_names=np.asarray(sequence.FEATURE_NAMES),
    )

    summary = {
        "sample_id": sequence.sample_id,
        "label": sequence.label,
        "feature_shape": list(sequence.features.shape),
        "feature_names": list(sequence.FEATURE_NAMES),
        "normalization": {
            "x_origin": sequence.normalization.x_origin,
            "y_area_min": sequence.normalization.y_area_min,
            "scale": sequence.normalization.scale,
            "used_known_writing_area": sequence.normalization.used_known_writing_area,
        },
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
