# Protocol V2 Specification

## 1. Scope
`update_v2` is the Socket.IO event for depth-enabled streaming in this repository.  
It carries pose, depth, confidence, camera intrinsics, and preprocessing metadata.

This document fixes the current on-wire binary layout used by:
- iOS sender: `iPhoneVIO/SocketClient.swift` (`DataPacketV2`)
- Python receiver: `socketio_server.py` (`decode_data_v2`)

## 2. Transport
- Event name: `update_v2`
- Payload type on socket: Base64 string
- Base64-decoded bytes: little-endian binary packet

## 3. Packet Layout (Version 2)
Order is exact and contiguous.

| Offset | Size | Type | Name | Notes |
|---|---:|---|---|---|
| 0 | 4 | bytes | `magic` | ASCII `"IPV2"` |
| 4 | 1 | `uint8` | `version` | fixed `2` |
| 5 | 1 | `uint8` | `flags` | bit flags, see section 4 |
| 6 | 2 | `uint16` | `reserved` | fixed `0` |
| 8 | 8 | `float64` | `pose_timestamp` | AR frame timestamp |
| 16 | 8 | `float64` | `depth_timestamp` | depth capture timestamp |
| 24 | 64 | `16 x float32` | `transform` | 4x4, column-major from Swift |
| 88 | 36 | `9 x float32` | `intrinsics` | 3x3, column-major from Swift |
| 124 | 4 | `uint32` | `camera_width` | camera image width |
| 128 | 4 | `uint32` | `camera_height` | camera image height |
| 132 | 4 | `uint32` | `depth_width` | depth map width (post preprocessing) |
| 136 | 4 | `uint32` | `depth_height` | depth map height (post preprocessing) |
| 140 | 1 | `uint8` | `depth_format` | enum, see section 5 |
| 141 | 1 | `uint8` | `confidence_format` | enum, see section 5 |
| 142 | 1 | `uint8` | `depth_downsample_factor` | metadata only |
| 143 | 1 | `uint8` | `min_confidence_level` | metadata only |
| 144 | 2 | `uint16` | `reserved2` | fixed `0` |
| 146 | 4 | `float32` | `depth_clip_max_meters` | metadata only |
| 150 | 4 | `uint32` | `depth_bytes_len` | payload byte count |
| 154 | 4 | `uint32` | `confidence_bytes_len` | payload byte count |
| 158 | `depth_bytes_len` | bytes | `depth_payload` | raw or zlib |
| ... | `confidence_bytes_len` | bytes | `confidence_payload` | raw or zlib |

## 4. Flags
`flags` is `uint8` bitmask:

| Bit | Mask | Name | Meaning |
|---:|---:|---|---|
| 0 | `0x01` | `has_depth` | depth payload exists |
| 1 | `0x02` | `has_confidence` | confidence payload exists |
| 2 | `0x04` | `smoothed_depth` | source uses `smoothedSceneDepth` |
| 3 | `0x08` | `compressed` | at least one payload compressed |
| 4 | `0x10` | `depth_clipped` | clipping was applied |
| 5 | `0x20` | `depth_downsampled` | downsample was applied |
| 6 | `0x40` | `confidence_filtered` | confidence threshold filter applied |
| 7 | `0x80` | reserved | currently unused |

## 5. Format Enums
`depth_format`:
- `1`: float32 meters, raw bytes
- `2`: float32 meters, zlib-compressed bytes

`confidence_format`:
- `0`: not present
- `1`: uint8 raw bytes
- `2`: uint8 zlib-compressed bytes

## 6. Matrix Convention
- `transform` and `intrinsics` are serialized from Swift memory order (column-major).
- Python decoder reshapes then transposes (`.T`) to standard row-major NumPy view.

## 7. Payload Interpretation
Depth:
- decoded dtype: `float32`
- shape: `(depth_height, depth_width)`
- unit: meters
- invalid/filtered pixels may be `0`

Confidence:
- decoded dtype: `uint8`
- shape: `(depth_height, depth_width)`
- typical levels: `0, 1, 2` (higher values treated as `3+` in server stats)

## 8. Compatibility
- `update` (v1 pose-only) remains supported in parallel.
- Receivers should route by event name:
  - `update` -> v1 decoder
  - `update_v2` -> this spec

## 9. Validation Checklist
- Validate `magic == "IPV2"` and `version == 2`.
- Validate minimum header size before reading variable sections.
- Validate `depth_bytes_len` and `confidence_bytes_len` against available bytes.
- Validate decompressed byte counts:
  - depth: `depth_width * depth_height * 4`
  - confidence: `depth_width * depth_height`

## 10. Current Producer Defaults
Current iOS defaults in this repository:
- depth source: `smoothedSceneDepth` when supported, else `sceneDepth`
- clip max: `0.5m`
- downsample factor: `2`
- min confidence level: `1`
- zlib compression: enabled
