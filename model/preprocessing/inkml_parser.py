"""
inkml_parser.py

OBjective: Create an InkMl parser that converts inkML documents into 
usable representations for the models pipeline. 

Object-oriented InkML parser for the InkCalc online handwriting pipeline.

what it does? :
converts an inkml file into
    - absolute x coordinates
    - absolute y coordinates
    - absolute timestamps in seconds
    - original stroke boundaries
    - the ground-truth label, when present
    - a known writing area, when the InkML document exposes one

    it does not perform model feature extraction INTENTIONALLY. That is 
    handled in the feature_extraction.py module.
    coordinate normalization, spatial resampling, and creation of [dx, dy, dt, p] is done in
    feature extraction not here


The implementation supports:
    - XML namespaces
    - <traceFormat> channel declarations
    - X/Y/T channel reordering
    - common InkML absolute, first-difference, and second-difference prefixes
      (!, ', and ")
    - time-unit conversion to seconds
    - simple writing-area metadata extraction
    - label extraction from <annotation> elements

Only the Python standard library is required. If defusedxml is installed, it
is used automatically for safer XML parsing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import argparse
import json
import math
import re

try:
    from defusedxml import ElementTree as ET  # type: ignore
except ImportError:
    from xml.etree import ElementTree as ET


_NUMBER_TOKEN_RE = re.compile(
    r"""(?P<prefix>[!'\"]?)(?P<number>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"""
)


class InkMLParseError(ValueError):
    """Inkml parsing error cannot be done safely."""


@dataclass(frozen=True)
class InkPoint:
    """A single absolute online-ink point."""

    x: float
    y: float
    t: float


@dataclass(frozen=True)
class InkStroke:
    """A pen stroke containing points in recorded order."""

    points: Tuple[InkPoint, ...]
    stroke_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.points:
            raise InkMLParseError("InkStroke cannot be empty.")


@dataclass(frozen=True)
class WritingArea:
    """Known writing-area bounds, when available in the InkML metadata."""

    min_x: float
    min_y: float
    width: float
    height: float

    #max x is not necessarily the "right" of the writing area, since some InkML documents may use a coordinate system with x increasing to the left. The parser does not
    #assume any particular orientation and simply reports the bounds as given.
    @property
    def max_x(self) -> float:
        return self.min_x + self.width

    # max y is not necessarily the "top" of the writing area, since some InkML documents
    # may use a coordinate system with y increasing downwards. The parser does not
    # assume any particular orientation and simply reports the bounds as given.
    @property
    def max_y(self) -> float:
        return self.min_y + self.height

    def validate(self) -> None:
        values = (self.min_x, self.min_y, self.width, self.height)
        if not all(math.isfinite(v) for v in values):
            raise InkMLParseError("Writing-area values must be finite.")
        if self.width <= 0 or self.height <= 0:
            raise InkMLParseError("Writing-area width and height must be positive.")


# raw inkML sample before and model specific pre processing.
# This is the direct output of the InkMLParser and the input to the feature extraction module.
@dataclass(frozen=True)
class InkSample:
    """Parsed InkML sample before model-specific preprocessing."""

    strokes: Tuple[InkStroke, ...]
    label: Optional[str]
    sample_id: str
    source_path: str
    writing_area: Optional[WritingArea] = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.strokes:
            raise InkMLParseError("InkSample must contain at least one stroke.")

    @property
    def point_count(self) -> int:
        return sum(len(stroke.points) for stroke in self.strokes)


@dataclass(frozen=True)
class ChannelDefinition:
    """Description of one InkML trace channel."""

    name: str
    units: Optional[str] = None
    data_type: Optional[str] = None


@dataclass
class _ChannelState:
    """Internal decoder state for InkML derivative encodings."""

    mode: int = 0
    previous_value: float = 0.0
    previous_delta: float = 0.0
    initialized: bool = False


class InkMLParser:
    """Parse InkML files into :class:`InkSample` objects."""

    _TIME_UNIT_TO_SECONDS: Dict[str, float] = {
        "s": 1.0,
        "sec": 1.0,
        "secs": 1.0,
        "second": 1.0,
        "seconds": 1.0,
        "ms": 1e-3,
        "msec": 1e-3,
        "millisecond": 1e-3,
        "milliseconds": 1e-3,
        "us": 1e-6,
        "µs": 1e-6,
        "usec": 1e-6,
        "microsecond": 1e-6,
        "microseconds": 1e-6,
        "ns": 1e-9,
        "nsec": 1e-9,
        "nanosecond": 1e-9,
        "nanoseconds": 1e-9,
    }

    _X_NAMES = {"x", "xcoord", "xcoordinate"}
    _Y_NAMES = {"y", "ycoord", "ycoordinate"}
    _T_NAMES = {"t", "time", "timestamp"}

    def __init__(
        self,
        *,
        require_time: bool = True,
        default_time_unit: str = "s",
        convert_time_to_seconds: bool = True,
    ) -> None:
        self.require_time = require_time
        self.default_time_unit = default_time_unit
        self.convert_time_to_seconds = convert_time_to_seconds

    def parse(self, path: str | Path) -> InkSample:
        """Parse one InkML file."""

        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(source)

        try:
            root = ET.parse(str(source)).getroot()
        except Exception as exc:
            raise InkMLParseError(f"Failed to parse XML from {source}: {exc}") from exc

        channels = self._parse_trace_format(root)
        traces = [element for element in root.iter() if self._local_name(element.tag) == "trace"]
        if not traces:
            raise InkMLParseError(f"No <trace> elements found in {source}.")

        if not channels:
            channels = self._infer_channels_from_trace_text(traces[0].text or "")

        channel_index = self._resolve_required_channel_indices(channels)
        time_scale = self._time_scale(channels[channel_index["t"]].units) if "t" in channel_index else 1.0

        strokes: List[InkStroke] = []
        for trace_number, trace_element in enumerate(traces):
            trace_text = trace_element.text or ""
            rows = self._decode_trace_rows(trace_text, len(channels))
            if not rows:
                continue

            points: List[InkPoint] = []
            for row in rows:
                x = row[channel_index["x"]]
                y = row[channel_index["y"]]
                if "t" in channel_index:
                    t = row[channel_index["t"]] * time_scale
                else:
                    t = float(len(points))

                if not all(math.isfinite(value) for value in (x, y, t)):
                    raise InkMLParseError(
                        f"Non-finite point found in trace {trace_number} of {source}."
                    )
                points.append(InkPoint(float(x), float(y), float(t)))

            if points:
                stroke_id = trace_element.attrib.get("id")
                strokes.append(InkStroke(tuple(points), stroke_id=stroke_id))

        if not strokes:
            raise InkMLParseError(f"All traces were empty in {source}.")

        label = self._extract_label(root)
        writing_area = self._extract_writing_area(root)
        metadata = dict(self._extract_metadata(root))
        metadata["channel_order"] = ",".join(channel.name for channel in channels)
        if "t" in channel_index:
            metadata["time_unit_output"] = (
                "s"
                if self.convert_time_to_seconds
                else (channels[channel_index["t"]].units or self.default_time_unit)
            )
        else:
            metadata["time_unit_output"] = "synthetic-index"

        return InkSample(
            strokes=tuple(strokes),
            label=label,
            sample_id=source.stem,
            source_path=str(source.resolve()),
            writing_area=writing_area,
            metadata=metadata,
        )

    def _parse_trace_format(self, root) -> List[ChannelDefinition]:
        for element in root.iter():
            if self._local_name(element.tag) != "traceFormat":
                continue

            channels: List[ChannelDefinition] = []
            for child in element:
                if self._local_name(child.tag) != "channel":
                    continue
                name = child.attrib.get("name", "").strip()
                if not name:
                    continue
                channels.append(
                    ChannelDefinition(
                        name=name,
                        units=child.attrib.get("units"),
                        data_type=child.attrib.get("type"),
                    )
                )
            if channels:
                return channels
        return []

    def _infer_channels_from_trace_text(self, trace_text: str) -> List[ChannelDefinition]:
        chunks = self._split_trace_into_point_chunks(trace_text)
        if not chunks:
            raise InkMLParseError("Cannot infer channels from an empty trace.")

        token_count = len(self._parse_numeric_tokens(chunks[0]))
        if token_count >= 3:
            return [
                ChannelDefinition("X"),
                ChannelDefinition("Y"),
                ChannelDefinition("T", units=self.default_time_unit),
            ] + [ChannelDefinition(f"channel_{index}") for index in range(3, token_count)]
        if token_count == 2 and not self.require_time:
            return [ChannelDefinition("X"), ChannelDefinition("Y")]

        raise InkMLParseError(
            "No <traceFormat> was declared and the trace does not contain "
            "enough values to infer X, Y, and T."
        )

    def _resolve_required_channel_indices(
        self, channels: Sequence[ChannelDefinition]
    ) -> Dict[str, int]:
        normalized = [self._normalize_channel_name(channel.name) for channel in channels]

        def find(names: set[str]) -> Optional[int]:
            for index, name in enumerate(normalized):
                if name in names:
                    return index
            return None

        x_index = find(self._X_NAMES)
        y_index = find(self._Y_NAMES)
        t_index = find(self._T_NAMES)

        if x_index is None and len(channels) >= 1:
            x_index = 0
        if y_index is None and len(channels) >= 2:
            y_index = 1
        if t_index is None and len(channels) >= 3:
            t_index = 2

        if x_index is None or y_index is None:
            raise InkMLParseError("Could not resolve X and Y channels.")
        if self.require_time and t_index is None:
            raise InkMLParseError(
                "Could not resolve a T/time channel. Pass require_time=False "
                "only if synthetic timing is intentionally acceptable."
            )

        result = {"x": x_index, "y": y_index}
        if t_index is not None:
            result["t"] = t_index
        return result

    def _decode_trace_rows(self, trace_text: str, channel_count: int) -> List[List[float]]:
        chunks = self._split_trace_into_point_chunks(trace_text)
        if not chunks:
            return []

        states = [_ChannelState() for _ in range(channel_count)]
        decoded_rows: List[List[float]] = []

        for point_number, chunk in enumerate(chunks):
            tokens = self._parse_numeric_tokens(chunk)
            if not tokens:
                continue
            if len(tokens) != channel_count:
                raise InkMLParseError(
                    f"Point {point_number} contains {len(tokens)} values, but "
                    f"traceFormat declares {channel_count} channels. Raw point: {chunk!r}"
                )

            point_default_mode: Optional[int] = None
            if tokens[0][0]:
                point_default_mode = self._prefix_to_mode(tokens[0][0])

            row: List[float] = []
            for channel_index, (prefix, numeric_value) in enumerate(tokens):
                state = states[channel_index]

                if prefix:
                    mode = self._prefix_to_mode(prefix)
                elif point_default_mode is not None:
                    mode = point_default_mode
                else:
                    mode = state.mode

                state.mode = mode

                if mode == 0:
                    absolute = numeric_value
                    delta = absolute - state.previous_value if state.initialized else 0.0
                elif mode == 1:
                    delta = numeric_value
                    absolute = state.previous_value + delta if state.initialized else delta
                elif mode == 2:
                    delta = state.previous_delta + numeric_value
                    absolute = state.previous_value + delta if state.initialized else delta
                else:  # pragma: no cover
                    raise InkMLParseError(f"Unsupported InkML derivative mode: {mode}")

                state.previous_value = absolute
                state.previous_delta = delta
                state.initialized = True
                row.append(float(absolute))

            decoded_rows.append(row)

        return decoded_rows

    @staticmethod
    def _split_trace_into_point_chunks(trace_text: str) -> List[str]:
        normalized = trace_text.strip()
        if not normalized:
            return []

        if "," in normalized or ";" in normalized:
            chunks = re.split(r"[,;]", normalized)
        else:
            lines = [line.strip() for line in normalized.splitlines() if line.strip()]
            chunks = lines if len(lines) > 1 else [normalized]

        return [chunk.strip() for chunk in chunks if chunk.strip()]

    @staticmethod
    def _parse_numeric_tokens(chunk: str) -> List[Tuple[str, float]]:
        matches = list(_NUMBER_TOKEN_RE.finditer(chunk))
        if not matches:
            return []

        cursor = 0
        for match in matches:
            gap = chunk[cursor : match.start()]
            if gap.strip():
                raise InkMLParseError(f"Unexpected trace token content: {gap!r}")
            cursor = match.end()
        if chunk[cursor:].strip():
            raise InkMLParseError(f"Unexpected trace token content: {chunk[cursor:]!r}")

        return [
            (match.group("prefix"), float(match.group("number")))
            for match in matches
        ]

    @staticmethod
    def _prefix_to_mode(prefix: str) -> int:
        if prefix == "!":
            return 0
        if prefix == "'":
            return 1
        if prefix == '"':
            return 2
        raise InkMLParseError(f"Unknown InkML value prefix: {prefix!r}")

    def _time_scale(self, declared_unit: Optional[str]) -> float:
        if not self.convert_time_to_seconds:
            return 1.0

        unit = (declared_unit or self.default_time_unit).strip().lower()
        try:
            return self._TIME_UNIT_TO_SECONDS[unit]
        except KeyError as exc:
            supported = ", ".join(sorted(self._TIME_UNIT_TO_SECONDS))
            raise InkMLParseError(
                f"Unsupported time unit {unit!r}. Supported aliases: {supported}"
            ) from exc

    def _extract_label(self, root) -> Optional[str]:
        preferred_types = ("normalizedlabel", "truth", "label", "transcription", "groundtruth")
        candidates: Dict[str, str] = {}
        fallback: Optional[str] = None

        for element in root.iter():
            if self._local_name(element.tag) != "annotation":
                continue
            text = "".join(element.itertext()).strip()
            if not text:
                continue
            annotation_type = element.attrib.get("type", "").strip().lower()
            if annotation_type:
                candidates.setdefault(annotation_type, text)
            elif fallback is None:
                fallback = text

        for preferred in preferred_types:
            if preferred in candidates:
                return candidates[preferred]

        return next(iter(candidates.values()), fallback)

    def _extract_writing_area(self, root) -> Optional[WritingArea]:
        candidate_names = {"activearea", "writingarea", "canvas", "capturearea"}

        for element in root.iter():
            if self._local_name(element.tag).lower() not in candidate_names:
                continue

            attributes = {key.lower(): value for key, value in element.attrib.items()}
            min_x = self._first_float(attributes, ("x", "left", "minx"), default=0.0)
            min_y = self._first_float(attributes, ("y", "top", "miny"), default=0.0)
            width = self._first_float(attributes, ("width", "w"))
            height = self._first_float(attributes, ("height", "h"))

            if (width is None or height is None) and "size" in attributes:
                size_values = [
                    value for _, value in self._parse_numeric_tokens(attributes["size"])
                ]
                if len(size_values) >= 2:
                    width = width if width is not None else size_values[0]
                    height = height if height is not None else size_values[1]

            if width is None or height is None:
                continue

            area = WritingArea(
                min_x=float(min_x or 0.0),
                min_y=float(min_y or 0.0),
                width=float(width),
                height=float(height),
            )
            try:
                area.validate()
            except InkMLParseError:
                continue
            return area

        return None

    def _extract_metadata(self, root) -> Dict[str, str]:
        metadata: Dict[str, str] = {}
        for element in root.iter():
            if self._local_name(element.tag) != "annotation":
                continue
            annotation_type = element.attrib.get("type", "").strip()
            text = "".join(element.itertext()).strip()
            if annotation_type and text:
                metadata.setdefault(annotation_type, text)
        return metadata

    @staticmethod
    def _first_float(
        attributes: Mapping[str, str],
        names: Iterable[str],
        *,
        default: Optional[float] = None,
    ) -> Optional[float]:
        for name in names:
            if name in attributes:
                try:
                    return float(attributes[name])
                except ValueError:
                    return default
        return default

    @staticmethod
    def _normalize_channel_name(name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", name.strip().lower())

    @staticmethod
    def _local_name(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]


def _sample_to_json_dict(sample: InkSample) -> Dict[str, object]:
    return {
        "sample_id": sample.sample_id,
        "source_path": sample.source_path,
        "label": sample.label,
        "point_count": sample.point_count,
        "stroke_count": len(sample.strokes),
        "writing_area": asdict(sample.writing_area) if sample.writing_area else None,
        "metadata": dict(sample.metadata),
        "strokes": [
            {
                "stroke_id": stroke.stroke_id,
                "points": [asdict(point) for point in stroke.points],
            }
            for stroke in sample.strokes
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse an InkML file.")
    parser.add_argument("inkml", type=Path)
    parser.add_argument(
        "--default-time-unit",
        default="s",
        help="Used only when the InkML T channel does not declare units.",
    )
    parser.add_argument(
        "--allow-missing-time",
        action="store_true",
        help="Allow files without a T channel by synthesizing integer timestamps.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path for a full parsed JSON representation.",
    )
    args = parser.parse_args()

    ink_parser = InkMLParser(
        require_time=not args.allow_missing_time,
        default_time_unit=args.default_time_unit,
    )
    sample = ink_parser.parse(args.inkml)

    summary = {
        "sample_id": sample.sample_id,
        "label": sample.label,
        "strokes": len(sample.strokes),
        "points": sample.point_count,
        "writing_area_known": sample.writing_area is not None,
        "metadata": dict(sample.metadata),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(_sample_to_json_dict(sample), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
