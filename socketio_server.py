import base64
import struct
import time

import eventlet
import numpy as np
import socketio


MAGIC_V2 = b"IPV2"
FLAG_HAS_DEPTH = 1 << 0
FLAG_HAS_CONFIDENCE = 1 << 1
FLAG_IS_SMOOTHED_DEPTH = 1 << 2


class DataPacket:
    def __init__(self, transform_matrix: np.ndarray, timestamp):
        self.transform_matrix = transform_matrix.copy()
        self.timestamp = timestamp

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
        is_smoothed_depth: bool,
        depth_map: np.ndarray | None,
        confidence_map: np.ndarray | None,
    ):
        self.transform_matrix = transform_matrix.copy()
        self.pose_timestamp = float(pose_timestamp)
        self.depth_timestamp = float(depth_timestamp)
        self.intrinsics = intrinsics.copy()
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        self.depth_width = int(depth_width)
        self.depth_height = int(depth_height)
        self.is_smoothed_depth = bool(is_smoothed_depth)
        self.depth_map = depth_map
        self.confidence_map = confidence_map

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


def decode_data(encoded_str):
    # Decode the base64 string to bytes
    data_bytes = base64.b64decode(encoded_str)
    
    transform_matrix = np.zeros((4, 4))
    # Unpack transform matrix (16 floats)
    for i in range(4):
        for j in range(4):
            transform_matrix[i, j] = struct.unpack('f', data_bytes[4 * (4 * i + j):4 * (4 * i + j + 1)])[0]
    # The transform matrix is stored in column-major order in swift, so we need to transpose it in python
    transform_matrix = transform_matrix.T
    
    # Unpack timestamp (1 double)
    timestamp = struct.unpack('d', data_bytes[64:72])[0]

    return DataPacket(transform_matrix, timestamp)


def decode_data_v2(encoded_str):
    data_bytes = base64.b64decode(encoded_str)
    min_header_size = 4 + 1 + 1 + 2 + 8 + 8 + 16 * 4 + 9 * 4 + 4 * 4 + 1 + 1 + 4 + 4
    if len(data_bytes) < min_header_size:
        raise ValueError(f"V2 packet too short: {len(data_bytes)}")

    offset = 0
    magic = data_bytes[offset:offset + 4]
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

    camera_width, camera_height, depth_width, depth_height = struct.unpack_from("<IIII", data_bytes, offset)
    offset += 4 * 4

    depth_format = data_bytes[offset]
    offset += 1
    confidence_format = data_bytes[offset]
    offset += 1

    depth_bytes_len, confidence_bytes_len = struct.unpack_from("<II", data_bytes, offset)
    offset += 8

    remaining = len(data_bytes) - offset
    expected_remaining = depth_bytes_len + confidence_bytes_len
    if remaining < expected_remaining:
        raise ValueError(
            f"Invalid V2 lengths: remaining={remaining}, expected={expected_remaining}"
        )

    depth_raw = data_bytes[offset:offset + depth_bytes_len]
    offset += depth_bytes_len
    confidence_raw = data_bytes[offset:offset + confidence_bytes_len]
    offset += confidence_bytes_len

    has_depth = (flags & FLAG_HAS_DEPTH) != 0
    has_confidence = (flags & FLAG_HAS_CONFIDENCE) != 0
    is_smoothed_depth = (flags & FLAG_IS_SMOOTHED_DEPTH) != 0

    depth_map = None
    if has_depth and depth_bytes_len > 0:
        if depth_format != 1:
            raise ValueError(f"Unsupported depth format: {depth_format}")
        expected_depth_len = depth_width * depth_height * 4
        if depth_bytes_len != expected_depth_len:
            raise ValueError(
                f"Depth byte size mismatch: got={depth_bytes_len}, expected={expected_depth_len}"
            )
        depth_map = np.frombuffer(depth_raw, dtype="<f4").reshape((depth_height, depth_width))

    confidence_map = None
    if has_confidence and confidence_bytes_len > 0:
        if confidence_format != 1:
            raise ValueError(f"Unsupported confidence format: {confidence_format}")
        expected_conf_len = depth_width * depth_height
        if confidence_bytes_len != expected_conf_len:
            raise ValueError(
                f"Confidence byte size mismatch: got={confidence_bytes_len}, expected={expected_conf_len}"
            )
        confidence_map = np.frombuffer(confidence_raw, dtype=np.uint8).reshape((depth_height, depth_width))

    return DataPacketV2(
        transform_matrix=transform_matrix,
        pose_timestamp=pose_timestamp,
        depth_timestamp=depth_timestamp,
        intrinsics=intrinsics,
        camera_width=camera_width,
        camera_height=camera_height,
        depth_width=depth_width,
        depth_height=depth_height,
        is_smoothed_depth=is_smoothed_depth,
        depth_map=depth_map,
        confidence_map=confidence_map,
    )


# Create a Socket.IO server
sio = socketio.Server()

# Create a WSGI app
app = socketio.WSGIApp(sio)

# Event handler for new connections
@sio.event
def connect(sid, environ):
    print("Client connected", sid)

# Event handler for disconnections
@sio.event
def disconnect(sid):
    print("Client disconnected", sid)

prev_time_v1 = 0.0
package_cnt_v1 = 0
prev_pose_time_v2 = 0.0
prev_depth_time_v2 = 0.0
package_cnt_v2 = 0


# Event handler for messages on 'update' channel
@sio.on('update')
def handle_message(sid, data):
    # Assuming data is base64-encoded from the client
    global prev_time_v1, package_cnt_v1
    structured_data = decode_data(data)
    fps = _safe_fps(structured_data.timestamp, prev_time_v1)
    print(f"[v1] {structured_data}, fps: {fps:.2f}")
    prev_time_v1 = structured_data.timestamp
    package_cnt_v1 += 1
    # Process data here as needed


@sio.on('update_v2')
def handle_message_v2(sid, data):
    global prev_pose_time_v2, prev_depth_time_v2, package_cnt_v2
    try:
        packet = decode_data_v2(data)
    except Exception as exc:
        print(f"[v2] decode error: {exc}")
        return

    pose_fps = _safe_fps(packet.pose_timestamp, prev_pose_time_v2)
    depth_fps = _safe_fps(packet.depth_timestamp, prev_depth_time_v2)
    prev_pose_time_v2 = packet.pose_timestamp
    prev_depth_time_v2 = packet.depth_timestamp
    package_cnt_v2 += 1

    depth_summary = "depth=none"
    if packet.depth_map is not None:
        valid = np.isfinite(packet.depth_map)
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
        f"{depth_summary}{conf_summary}"
    )

# Run the server
if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    eventlet.wsgi.server(eventlet.listen(('', 5555)), app)
