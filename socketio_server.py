import base64
import struct
import time
import zlib
from typing import Optional

import eventlet
import numpy as np
import socketio


MAGIC_V2 = b"IPV2"
FLAG_HAS_DEPTH = 1 << 0
FLAG_HAS_CONFIDENCE = 1 << 1
FLAG_IS_SMOOTHED_DEPTH = 1 << 2
FLAG_COMPRESSED = 1 << 3
FLAG_DEPTH_CLIPPED = 1 << 4
FLAG_DEPTH_DOWNSAMPLED = 1 << 5
FLAG_CONFIDENCE_FILTERED = 1 << 6

DEPTH_FORMAT_FLOAT32_RAW = 1
DEPTH_FORMAT_FLOAT32_ZLIB = 2
CONFIDENCE_FORMAT_U8_RAW = 1
CONFIDENCE_FORMAT_U8_ZLIB = 2


class DataPacket:
    def __init__(self, transform_matrix: np.ndarray, timestamp: float, payload_nbytes: int):
        self.transform_matrix = transform_matrix.copy()
        self.timestamp = float(timestamp)
        self.payload_nbytes = int(payload_nbytes)

    def __str__(self):
        return f"Translation: {self.transform_matrix[:3, 3]}, Timestamp: {self.timestamp:.3f}"


class DataPacketV2:
    def __init__(
        self,
        transform_matrix: np.ndarray,
        pose_timestamp: float,
        depth_timestamp: float,
        intrinsics: np.ndarray,
        camera_width: int,
        camera_height: int,
        depth_width: int,
        depth_height: int,
        depth_format: int,
        confidence_format: int,
        depth_downsample_factor: int,
        min_confidence_level: int,
        depth_clip_max_meters: float,
        is_smoothed_depth: bool,
        is_compressed: bool,
        is_depth_clipped: bool,
        is_depth_downsampled: bool,
        is_confidence_filtered: bool,
        depth_map: Optional[np.ndarray],
        confidence_map: Optional[np.ndarray],
        payload_nbytes: int,
    ):
        self.transform_matrix = transform_matrix.copy()
        self.pose_timestamp = float(pose_timestamp)
        self.depth_timestamp = float(depth_timestamp)
        self.intrinsics = intrinsics.copy()
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        self.depth_width = int(depth_width)
        self.depth_height = int(depth_height)
        self.depth_format = int(depth_format)
        self.confidence_format = int(confidence_format)
        self.depth_downsample_factor = int(depth_downsample_factor)
        self.min_confidence_level = int(min_confidence_level)
        self.depth_clip_max_meters = float(depth_clip_max_meters)
        self.is_smoothed_depth = bool(is_smoothed_depth)
        self.is_compressed = bool(is_compressed)
        self.is_depth_clipped = bool(is_depth_clipped)
        self.is_depth_downsampled = bool(is_depth_downsampled)
        self.is_confidence_filtered = bool(is_confidence_filtered)
        self.depth_map = depth_map
        self.confidence_map = confidence_map
        self.payload_nbytes = int(payload_nbytes)

    def __str__(self):
        mode = "smoothed" if self.is_smoothed_depth else "raw"
        return (
            f"Translation: {self.transform_matrix[:3, 3]}, "
            f"PoseTimestamp: {self.pose_timestamp:.3f}, "
            f"DepthTimestamp: {self.depth_timestamp:.3f}, "
            f"Depth: {self.depth_width}x{self.depth_height} ({mode})"
        )


def _safe_fps(current: float, previous: float) -> float:
    if previous <= 0 or current <= previous:
        return 0.0
    return 1.0 / (current - previous)


def decode_data(encoded_str: str) -> DataPacket:
    data_bytes = base64.b64decode(encoded_str)

    transform_matrix = np.zeros((4, 4), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            transform_matrix[i, j] = struct.unpack(
                "f", data_bytes[4 * (4 * i + j) : 4 * (4 * i + j + 1)]
            )[0]
    transform_matrix = transform_matrix.T

    timestamp = struct.unpack("d", data_bytes[64:72])[0]
    return DataPacket(transform_matrix, timestamp, len(data_bytes))


def _decode_depth_payload(
    payload: bytes, depth_format: int, depth_width: int, depth_height: int
) -> np.ndarray:
    expected_nbytes = depth_width * depth_height * 4
    if depth_format == DEPTH_FORMAT_FLOAT32_RAW:
        decompressed = payload
    elif depth_format == DEPTH_FORMAT_FLOAT32_ZLIB:
        decompressed = zlib.decompress(payload)
    else:
        raise ValueError(f"Unsupported depth format: {depth_format}")

    if len(decompressed) != expected_nbytes:
        raise ValueError(
            f"Depth payload size mismatch: got={len(decompressed)}, expected={expected_nbytes}"
        )
    return np.frombuffer(decompressed, dtype="<f4").reshape((depth_height, depth_width))


def _decode_confidence_payload(
    payload: bytes, confidence_format: int, depth_width: int, depth_height: int
) -> np.ndarray:
    expected_nbytes = depth_width * depth_height
    if confidence_format == CONFIDENCE_FORMAT_U8_RAW:
        decompressed = payload
    elif confidence_format == CONFIDENCE_FORMAT_U8_ZLIB:
        decompressed = zlib.decompress(payload)
    else:
        raise ValueError(f"Unsupported confidence format: {confidence_format}")

    if len(decompressed) != expected_nbytes:
        raise ValueError(
            f"Confidence payload size mismatch: got={len(decompressed)}, expected={expected_nbytes}"
        )
    return np.frombuffer(decompressed, dtype=np.uint8).reshape((depth_height, depth_width))


def decode_data_v2(encoded_str: str) -> DataPacketV2:
    data_bytes = base64.b64decode(encoded_str)
    min_header_size = (
        4  # magic
        + 1  # version
        + 1  # flags
        + 2  # reserved
        + 8  # pose ts
        + 8  # depth ts
        + 16 * 4  # transform
        + 9 * 4  # intrinsics
        + 4 * 4  # camera/depth width/height
        + 1  # depth format
        + 1  # confidence format
        + 1  # downsample factor
        + 1  # min confidence
        + 2  # reserved2
        + 4  # clip max meters
        + 4  # depth bytes len
        + 4  # confidence bytes len
    )
    if len(data_bytes) < min_header_size:
        raise ValueError(f"V2 packet too short: {len(data_bytes)}")

    offset = 0

    magic = data_bytes[offset : offset + 4]
    offset += 4
    if magic != MAGIC_V2:
        raise ValueError(f"Unexpected V2 magic: {magic!r}")

    version = data_bytes[offset]
    offset += 1
    if version != 2:
        raise ValueError(f"Unsupported V2 version: {version}")

    flags = data_bytes[offset]
    offset += 1

    offset += 2  # reserved

    pose_timestamp = struct.unpack_from("<d", data_bytes, offset)[0]
    offset += 8
    depth_timestamp = struct.unpack_from("<d", data_bytes, offset)[0]
    offset += 8

    transform_flat = struct.unpack_from("<16f", data_bytes, offset)
    offset += 16 * 4
    transform_matrix = np.array(transform_flat, dtype=np.float32).reshape(4, 4).T

    intrinsics_flat = struct.unpack_from("<9f", data_bytes, offset)
    offset += 9 * 4
    intrinsics = np.array(intrinsics_flat, dtype=np.float32).reshape(3, 3).T

    camera_width, camera_height, depth_width, depth_height = struct.unpack_from(
        "<IIII", data_bytes, offset
    )
    offset += 4 * 4

    depth_format = data_bytes[offset]
    offset += 1
    confidence_format = data_bytes[offset]
    offset += 1

    depth_downsample_factor = data_bytes[offset]
    offset += 1
    min_confidence_level = data_bytes[offset]
    offset += 1

    offset += 2  # reserved2

    depth_clip_max_meters = struct.unpack_from("<f", data_bytes, offset)[0]
    offset += 4

    depth_bytes_len, confidence_bytes_len = struct.unpack_from("<II", data_bytes, offset)
    offset += 8

    remaining = len(data_bytes) - offset
    expected_remaining = depth_bytes_len + confidence_bytes_len
    if remaining < expected_remaining:
        raise ValueError(
            f"Invalid V2 lengths: remaining={remaining}, expected={expected_remaining}"
        )

    depth_raw = data_bytes[offset : offset + depth_bytes_len]
    offset += depth_bytes_len
    confidence_raw = data_bytes[offset : offset + confidence_bytes_len]
    offset += confidence_bytes_len

    has_depth = (flags & FLAG_HAS_DEPTH) != 0
    has_confidence = (flags & FLAG_HAS_CONFIDENCE) != 0
    is_smoothed_depth = (flags & FLAG_IS_SMOOTHED_DEPTH) != 0
    is_compressed = (flags & FLAG_COMPRESSED) != 0
    is_depth_clipped = (flags & FLAG_DEPTH_CLIPPED) != 0
    is_depth_downsampled = (flags & FLAG_DEPTH_DOWNSAMPLED) != 0
    is_confidence_filtered = (flags & FLAG_CONFIDENCE_FILTERED) != 0

    depth_map = None
    if has_depth and depth_bytes_len > 0:
        depth_map = _decode_depth_payload(depth_raw, depth_format, depth_width, depth_height)

    confidence_map = None
    if has_confidence and confidence_bytes_len > 0:
        confidence_map = _decode_confidence_payload(
            confidence_raw, confidence_format, depth_width, depth_height
        )

    return DataPacketV2(
        transform_matrix=transform_matrix,
        pose_timestamp=pose_timestamp,
        depth_timestamp=depth_timestamp,
        intrinsics=intrinsics,
        camera_width=camera_width,
        camera_height=camera_height,
        depth_width=depth_width,
        depth_height=depth_height,
        depth_format=depth_format,
        confidence_format=confidence_format,
        depth_downsample_factor=depth_downsample_factor,
        min_confidence_level=min_confidence_level,
        depth_clip_max_meters=depth_clip_max_meters,
        is_smoothed_depth=is_smoothed_depth,
        is_compressed=is_compressed,
        is_depth_clipped=is_depth_clipped,
        is_depth_downsampled=is_depth_downsampled,
        is_confidence_filtered=is_confidence_filtered,
        depth_map=depth_map,
        confidence_map=confidence_map,
        payload_nbytes=len(data_bytes),
    )


class UmiFTReferenceEvaluator:
    """
    Runtime evaluator aligned with UMI-FT stream assumptions:
    - main RGB/pose around 60 Hz
    - depth around 30 Hz
    - near-surface emphasis via clipping (e.g. 0.5m)
    """

    def __init__(
        self,
        target_pose_hz: float = 60.0,
        target_depth_hz: float = 30.0,
        summary_interval_sec: float = 5.0,
    ):
        self.target_pose_hz = float(target_pose_hz)
        self.target_depth_hz = float(target_depth_hz)
        self.summary_interval_sec = float(summary_interval_sec)

        now = time.time()
        self.start_wall = now
        self.last_summary_wall = now

        self.v1_count = 0
        self.v2_count = 0
        self.total_payload_bytes = 0

        self.prev_v1_ts = 0.0
        self.prev_pose_ts = 0.0
        self.prev_depth_ts = 0.0

        self.v1_dt_sum = 0.0
        self.v1_dt_count = 0
        self.pose_dt_sum = 0.0
        self.pose_dt_count = 0
        self.depth_dt_sum = 0.0
        self.depth_dt_count = 0

        self.depth_valid_count = 0
        self.depth_total_count = 0
        self.depth_near_count = 0

        self.conf_hist = np.zeros(4, dtype=np.int64)

    @staticmethod
    def _rate_from_intervals(dt_sum: float, dt_count: int) -> float:
        if dt_sum <= 0 or dt_count <= 0:
            return 0.0
        return dt_count / dt_sum

    def _maybe_report(self):
        now = time.time()
        if now - self.last_summary_wall < self.summary_interval_sec:
            return
        self.last_summary_wall = now

        uptime = max(1e-6, now - self.start_wall)
        bw_mbps = (self.total_payload_bytes / uptime) / 1e6

        v1_rate = self._rate_from_intervals(self.v1_dt_sum, self.v1_dt_count)
        pose_rate = self._rate_from_intervals(self.pose_dt_sum, self.pose_dt_count)
        depth_rate = self._rate_from_intervals(self.depth_dt_sum, self.depth_dt_count)

        depth_valid_ratio = (
            self.depth_valid_count / self.depth_total_count if self.depth_total_count > 0 else 0.0
        )
        near_ratio = (
            self.depth_near_count / self.depth_valid_count if self.depth_valid_count > 0 else 0.0
        )

        print(
            "[eval][umi-ft-ref] "
            f"uptime={uptime:.1f}s "
            f"pkts(v1/v2)=({self.v1_count}/{self.v2_count}) "
            f"rate(v1/pose/depth)=({v1_rate:.2f}/{pose_rate:.2f}/{depth_rate:.2f})Hz "
            f"target(60/30)=({pose_rate / self.target_pose_hz * 100:.1f}%/"
            f"{depth_rate / self.target_depth_hz * 100:.1f}%) "
            f"bw={bw_mbps:.2f}MB/s "
            f"depth_valid={depth_valid_ratio * 100:.1f}% "
            f"near_surface={near_ratio * 100:.1f}% "
            f"conf(0/1/2/3+)={tuple(int(x) for x in self.conf_hist)}"
        )

    def update_v1(self, packet: DataPacket):
        self.v1_count += 1
        self.total_payload_bytes += packet.payload_nbytes

        if self.prev_v1_ts > 0 and packet.timestamp > self.prev_v1_ts:
            dt = packet.timestamp - self.prev_v1_ts
            if dt < 1.0:
                self.v1_dt_sum += dt
                self.v1_dt_count += 1
        self.prev_v1_ts = packet.timestamp
        self._maybe_report()

    def update_v2(self, packet: DataPacketV2):
        self.v2_count += 1
        self.total_payload_bytes += packet.payload_nbytes

        if self.prev_pose_ts > 0 and packet.pose_timestamp > self.prev_pose_ts:
            dt = packet.pose_timestamp - self.prev_pose_ts
            if dt < 1.0:
                self.pose_dt_sum += dt
                self.pose_dt_count += 1
        self.prev_pose_ts = packet.pose_timestamp

        if self.prev_depth_ts > 0 and packet.depth_timestamp > self.prev_depth_ts:
            dt = packet.depth_timestamp - self.prev_depth_ts
            if dt < 1.0:
                self.depth_dt_sum += dt
                self.depth_dt_count += 1
        self.prev_depth_ts = packet.depth_timestamp

        if packet.depth_map is not None:
            valid = np.isfinite(packet.depth_map) & (packet.depth_map > 0)
            self.depth_valid_count += int(np.count_nonzero(valid))
            self.depth_total_count += int(packet.depth_map.size)

            clip_ref = packet.depth_clip_max_meters if packet.depth_clip_max_meters > 0 else 0.5
            near_surface = valid & (packet.depth_map <= clip_ref + 1e-6)
            self.depth_near_count += int(np.count_nonzero(near_surface))

        if packet.confidence_map is not None:
            flat = packet.confidence_map.reshape(-1)
            self.conf_hist[0] += int(np.count_nonzero(flat == 0))
            self.conf_hist[1] += int(np.count_nonzero(flat == 1))
            self.conf_hist[2] += int(np.count_nonzero(flat == 2))
            self.conf_hist[3] += int(np.count_nonzero(flat >= 3))

        self._maybe_report()


# Create a Socket.IO server
sio = socketio.Server()

# Create a WSGI app
app = socketio.WSGIApp(sio)


evaluator = UmiFTReferenceEvaluator()
prev_time_v1 = 0.0
prev_pose_time_v2 = 0.0
prev_depth_time_v2 = 0.0


@sio.event
def connect(sid, environ):
    print("Client connected", sid)


@sio.event
def disconnect(sid):
    print("Client disconnected", sid)


@sio.on("update")
def handle_message(sid, data):
    global prev_time_v1
    packet = decode_data(data)
    fps = _safe_fps(packet.timestamp, prev_time_v1)
    prev_time_v1 = packet.timestamp

    print(f"[v1] {packet}, fps: {fps:.2f}, payload: {packet.payload_nbytes} bytes")
    evaluator.update_v1(packet)


@sio.on("update_v2")
def handle_message_v2(sid, data):
    global prev_pose_time_v2, prev_depth_time_v2
    try:
        packet = decode_data_v2(data)
    except Exception as exc:
        print(f"[v2] decode error: {exc}")
        return

    pose_fps = _safe_fps(packet.pose_timestamp, prev_pose_time_v2)
    depth_fps = _safe_fps(packet.depth_timestamp, prev_depth_time_v2)
    prev_pose_time_v2 = packet.pose_timestamp
    prev_depth_time_v2 = packet.depth_timestamp

    depth_summary = "depth=none"
    if packet.depth_map is not None:
        valid = np.isfinite(packet.depth_map) & (packet.depth_map > 0)
        if np.any(valid):
            depth_min = float(np.min(packet.depth_map[valid]))
            depth_max = float(np.max(packet.depth_map[valid]))
            depth_summary = f"depth_range=[{depth_min:.3f}, {depth_max:.3f}]m"
        else:
            depth_summary = "depth_range=[nan, nan]"

    conf_summary = ""
    if packet.confidence_map is not None:
        hist = np.bincount(packet.confidence_map.reshape(-1), minlength=4)
        conf_summary = (
            f", conf(0/1/2/3+)=({int(hist[0])}/{int(hist[1])}/{int(hist[2])}/{int(hist[3:].sum())})"
        )

    print(
        f"[v2] {packet}, pose_fps: {pose_fps:.2f}, depth_fps: {depth_fps:.2f}, "
        f"{depth_summary}, payload: {packet.payload_nbytes} bytes, "
        f"fmt(d/c)=({packet.depth_format}/{packet.confidence_format}), "
        f"ds={packet.depth_downsample_factor}, clip={packet.depth_clip_max_meters:.2f}, "
        f"minConf={packet.min_confidence_level}, compressed={packet.is_compressed}, "
        f"clipped={packet.is_depth_clipped}, downsampled={packet.is_depth_downsampled}, "
        f"confFiltered={packet.is_confidence_filtered}{conf_summary}"
    )

    evaluator.update_v2(packet)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    eventlet.wsgi.server(eventlet.listen(("", 5555)), app)
