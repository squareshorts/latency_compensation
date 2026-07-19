"""Causal prediction-only tracker adapters for delayed detector boxes.

The adapters expose only ``update(t, detections)`` followed by
``forecast(track, t + delta)`` and therefore cannot ingest target-time
detections. ByteTrackAdapter implements two-stage high/low-confidence
association. OCSortAdapter adds observation-centric velocity direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment


def box_to_state(box: np.ndarray) -> np.ndarray:
    box = np.asarray(box, dtype=float)
    width = max(float(box[2] - box[0]), 1.0)
    height = max(float(box[3] - box[1]), 1.0)
    return np.asarray([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, width, height], dtype=float)


def state_to_box(state: np.ndarray, width: int = 1550, height: int = 2048) -> np.ndarray:
    cx, cy, w, h = np.asarray(state, dtype=float)[:4]
    w, h = max(float(w), 1.0), max(float(h), 1.0)
    return np.asarray([
        np.clip(cx - w / 2, 0, width - 1), np.clip(cy - h / 2, 0, height - 1),
        np.clip(cx + w / 2, 0, width - 1), np.clip(cy + h / 2, 0, height - 1),
    ], dtype=float)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = np.maximum(a[:2], b[:2])
    x2, y2 = np.minimum(a[2:], b[2:])
    intersection = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return intersection / max(area_a + area_b - intersection, 1e-12)


@dataclass
class TrackState:
    track_id: int
    class_id: int
    timestamp_ns: int
    state: np.ndarray
    covariance: np.ndarray = field(default_factory=lambda: np.eye(8, dtype=float) * 100.0)
    observations: list[tuple[int, np.ndarray]] = field(default_factory=list)
    hits: int = 1

    @classmethod
    def create(cls, track_id: int, class_id: int, timestamp_ns: int, box: np.ndarray) -> "TrackState":
        measurement = box_to_state(box)
        return cls(track_id, class_id, int(timestamp_ns), np.r_[measurement, np.zeros(4)],
                   observations=[(int(timestamp_ns), np.asarray(box, dtype=float).copy())])

    def predict_to(self, timestamp_ns: int) -> None:
        timestamp_ns = int(timestamp_ns)
        dt = max((timestamp_ns - self.timestamp_ns) / 1e9, 0.0)
        transition = np.eye(8)
        transition[:4, 4:] = np.eye(4) * dt
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + np.eye(8) * max(1.0, 4.0 * dt)
        self.timestamp_ns = timestamp_ns

    def update(self, box: np.ndarray, timestamp_ns: int, observation_correction: bool = False) -> None:
        timestamp_ns = int(timestamp_ns)
        if timestamp_ns != self.timestamp_ns:
            self.predict_to(timestamp_ns)
        measurement = box_to_state(box)
        observation = np.asarray(box, dtype=float).copy()
        if observation_correction and self.observations:
            previous_ts, previous_box = self.observations[-1]
            dt = max((timestamp_ns - previous_ts) / 1e9, 1e-6)
            self.state[4:] = (measurement - box_to_state(previous_box)) / dt
            self.state[:4] = measurement
            self.covariance = np.eye(8) * 9.0
        else:
            h = np.c_[np.eye(4), np.zeros((4, 4))]
            s = h @ self.covariance @ h.T + np.eye(4) * 9.0
            gain = self.covariance @ h.T @ np.linalg.inv(s)
            self.state += gain @ (measurement - h @ self.state)
            self.covariance = (np.eye(8) - gain @ h) @ self.covariance
        self.observations.append((timestamp_ns, observation))
        self.observations = self.observations[-4:]
        self.hits += 1

    def forecast(self, timestamp_ns: int, width: int = 1550, height: int = 2048) -> np.ndarray:
        dt = max((int(timestamp_ns) - self.timestamp_ns) / 1e9, 0.0)
        return state_to_box(self.state[:4] + self.state[4:] * dt, width=width, height=height)

    @property
    def velocity_direction(self) -> np.ndarray:
        velocity = self.state[4:6].astype(float)
        norm = float(np.linalg.norm(velocity))
        return velocity / norm if norm > 1e-9 else np.zeros(2)


def _associate(tracks: list[TrackState], detection_indices: list[int], boxes: list[np.ndarray],
               classes: list[int], minimum_iou: float, direction_weight: float = 0.0):
    if not tracks or not detection_indices:
        return [], list(range(len(tracks))), detection_indices.copy()
    cost = np.full((len(tracks), len(detection_indices)), 1e6)
    for ti, track in enumerate(tracks):
        predicted = track.forecast(track.timestamp_ns)
        previous_center = box_to_state(track.observations[-1][1])[:2]
        for di, detection_index in enumerate(detection_indices):
            if classes[detection_index] != track.class_id:
                continue
            overlap = box_iou(predicted, boxes[detection_index])
            if overlap < minimum_iou:
                continue
            score = overlap
            if direction_weight and np.linalg.norm(track.velocity_direction) > 0:
                displacement = box_to_state(boxes[detection_index])[:2] - previous_center
                norm = float(np.linalg.norm(displacement))
                if norm > 1e-9:
                    score += direction_weight * float(np.dot(displacement / norm, track.velocity_direction))
            cost[ti, di] = -score
    rows, columns = linear_sum_assignment(cost)
    matches, matched_tracks, matched_detections = [], set(), set()
    for row, column in zip(rows, columns):
        if cost[row, column] >= 1e5:
            continue
        detection_index = detection_indices[column]
        matches.append((int(row), int(detection_index)))
        matched_tracks.add(int(row))
        matched_detections.add(int(detection_index))
    return (matches, [i for i in range(len(tracks)) if i not in matched_tracks],
            [i for i in detection_indices if i not in matched_detections])


class ByteTrackAdapter:
    """Prediction-only ByteTrack-style two-stage association adapter."""

    def __init__(self, high_threshold: float = 0.5, low_threshold: float = 0.25, maximum_gap_ms: int = 600):
        self.high_threshold, self.low_threshold = float(high_threshold), float(low_threshold)
        self.maximum_gap_ns = int(maximum_gap_ms * 1_000_000)
        self.tracks: list[TrackState] = []
        self.next_track_id = 1

    def update(self, timestamp_ns: int, detections: Iterable[dict]) -> dict[int, TrackState]:
        detections, timestamp_ns = list(detections), int(timestamp_ns)
        self.tracks = [t for t in self.tracks if timestamp_ns - t.observations[-1][0] <= self.maximum_gap_ns]
        for track in self.tracks:
            track.predict_to(timestamp_ns)
        boxes = [np.asarray([d["x1"], d["y1"], d["x2"], d["y2"]], float) for d in detections]
        classes = [int(d["class_id"]) for d in detections]
        scores = [float(d.get("confidence", 1.0)) for d in detections]
        high = [i for i, score in enumerate(scores) if score >= self.high_threshold]
        low = [i for i, score in enumerate(scores) if self.low_threshold <= score < self.high_threshold]
        assigned: dict[int, TrackState] = {}
        first, unmatched_track_indices, unmatched_high = _associate(self.tracks, high, boxes, classes, 0.20)
        for track_index, detection_index in first:
            track = self.tracks[track_index]
            track.update(boxes[detection_index], timestamp_ns)
            assigned[detection_index] = track
        remaining_tracks = [self.tracks[index] for index in unmatched_track_indices]
        second, _, _ = _associate(remaining_tracks, low, boxes, classes, 0.10)
        for local_track_index, detection_index in second:
            track = remaining_tracks[local_track_index]
            track.update(boxes[detection_index], timestamp_ns)
            assigned[detection_index] = track
        for detection_index in unmatched_high:
            track = TrackState.create(self.next_track_id, classes[detection_index], timestamp_ns, boxes[detection_index])
            self.next_track_id += 1
            self.tracks.append(track)
            assigned[detection_index] = track
        return assigned


class OCSortAdapter:
    """Prediction-only observation-centric SORT adapter."""

    def __init__(self, minimum_iou: float = 0.10, direction_weight: float = 0.20, maximum_gap_ms: int = 600):
        self.minimum_iou, self.direction_weight = float(minimum_iou), float(direction_weight)
        self.maximum_gap_ns = int(maximum_gap_ms * 1_000_000)
        self.tracks: list[TrackState] = []
        self.next_track_id = 1

    def update(self, timestamp_ns: int, detections: Iterable[dict]) -> dict[int, TrackState]:
        detections, timestamp_ns = list(detections), int(timestamp_ns)
        self.tracks = [t for t in self.tracks if timestamp_ns - t.observations[-1][0] <= self.maximum_gap_ns]
        for track in self.tracks:
            track.predict_to(timestamp_ns)
        boxes = [np.asarray([d["x1"], d["y1"], d["x2"], d["y2"]], float) for d in detections]
        classes = [int(d["class_id"]) for d in detections]
        matches, _, unmatched = _associate(self.tracks, list(range(len(detections))), boxes, classes,
                                            self.minimum_iou, self.direction_weight)
        assigned: dict[int, TrackState] = {}
        for track_index, detection_index in matches:
            track = self.tracks[track_index]
            track.update(boxes[detection_index], timestamp_ns, observation_correction=True)
            assigned[detection_index] = track
        for detection_index in unmatched:
            track = TrackState.create(self.next_track_id, classes[detection_index], timestamp_ns, boxes[detection_index])
            self.next_track_id += 1
            self.tracks.append(track)
            assigned[detection_index] = track
        return assigned
