# Depth Integration Plan (MVP First)

## 1. Goal
- Add depth data collection and transport to the existing iPhoneVIO pipeline.
- Keep pose streaming functional and backward compatible during migration.
- Use minimal, reviewable commits.

## 2. Current State
- iOS app currently sends pose (`4x4 transform`) + `timestamp` only.
- AR session does not enable depth frame semantics.
- Python server decodes only pose payload.

Relevant files:
- `iPhoneVIO/ARSessionManager.swift`
- `iPhoneVIO/SocketClient.swift`
- `socketio_server.py`

## 3. Scope
### In scope (MVP)
- Enable ARKit scene depth on supported devices.
- Capture:
  - pose (`camera.transform`)
  - frame timestamp
  - depth map (`sceneDepth` or `smoothedSceneDepth`)
  - optional confidence map
  - camera intrinsics and camera image resolution metadata
- Define and ship a new packet format (`v2`).
- Add Python decoder for the new packet.

### Out of scope (later)
- Model training changes.
- Advanced depth denoise/upsample on-device.
- Large-scale recording and dataset tooling.

## 4. API/Platform Constraints
- Depth from ARKit world tracking requires LiDAR-capable devices.
- Must check capability before enabling:
  - `ARWorldTrackingConfiguration.supportsFrameSemantics(...)`
- Depth map is metric (meters) and typically lower resolution than RGB.
- Camera intrinsics are in camera image pixel units; must be scaled if depth resolution is used directly.

## 5. Data Design (Packet v2)
Use a new event name, for example `update_v2`, while keeping existing `update` unchanged.

### 5.1 Header
- `magic` (4 bytes): `"IPV2"`
- `version` (1 byte): `2`
- `flags` (1 byte):
  - bit0: has_depth
  - bit1: has_confidence
  - bit2: smoothed_depth
- `reserved` (2 bytes)

### 5.2 Core Fields
- `pose_timestamp` (Float64)
- `depth_timestamp` (Float64)
- `transform` (16 x Float32)
- `intrinsics` (9 x Float32)
- `camera_width`, `camera_height` (UInt32 each)
- `depth_width`, `depth_height` (UInt32 each)
- `depth_format` (UInt8): `1 = float32_meters`
- `confidence_format` (UInt8): `1 = uint8_012`, `0 = none`
- `depth_bytes_len` (UInt32)
- `confidence_bytes_len` (UInt32)
- `depth_bytes`
- `confidence_bytes` (optional)

### 5.3 Encoding Notes
- Use little-endian consistently on both sides.
- Keep depth as raw `float32` for first version.
- If bandwidth becomes limiting, add optional compression in a later commit.

## 6. iOS Changes
## 6.1 AR session setup
- In `setupARSession()`:
  - check `supportsFrameSemantics([.sceneDepth, .smoothedSceneDepth])`
  - enable one semantic (default to `.smoothedSceneDepth` for stability)
  - if unsupported: continue pose-only mode

## 6.2 Per-frame capture
- In `session(_:didUpdate:)`:
  - read `frame.camera.transform`, `frame.timestamp`
  - read selected `ARDepthData` from `frame.smoothedSceneDepth` or `frame.sceneDepth`
  - extract:
    - `depthMap` (`CVPixelBuffer`)
    - optional `confidenceMap`
    - `camera.intrinsics`
    - `camera.imageResolution`
  - convert pixel buffers to linear byte arrays

## 6.3 Transport layer
- Add `DataPacketV2` next to old packet struct.
- Add `sendDataV2(...)` in `SocketClient`.
- Emit `update_v2`.
- Keep existing `update` path temporarily for rollback.

## 7. Python Server Changes
## 7.1 Decoder
- Add `decode_data_v2(encoded_str_or_bytes)`:
  - parse header
  - parse metadata
  - validate lengths
  - decode depth to `np.float32` shaped `(H, W)`
  - decode confidence to `np.uint8` shaped `(H, W)` if present

## 7.2 Runtime handling
- Register handler for `update_v2`.
- Keep current `update` handler for compatibility.
- Print/log:
  - pose FPS
  - depth FPS
  - depth min/max
  - confidence histogram (optional)

## 8. Validation Checklist
- LiDAR device:
  - `update_v2` arrives continuously
  - depth shape is stable
  - pose and depth timestamps are monotonic
- Non-LiDAR device:
  - app does not crash
  - falls back to pose-only path
- Decoder unit test:
  - synthetic packet roundtrip test (Swift->Python layout compatibility)

## 9. Commit Plan (Minimal)
- Commit 1: current workspace snapshot (already done).
- Commit 2: this plan document only.
- Commit 3: iOS depth capture + packet v2 sender.
- Commit 4: Python v2 decoder + dual-path server handlers.
- Commit 5 (optional): bandwidth optimization (compression/downsample).

## 10. Notes from UMI-FT Paper Mapping
- The paper pipeline uses synchronized RGB/depth/pose and handles differing stream rates.
- For this repository, the immediate equivalent is:
  - preserve per-modal timestamps
  - do not assume depth and RGB run at the same rate
  - carry intrinsics + resolution metadata with each depth packet
