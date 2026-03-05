//
//  ARSessionManager.swift
//  iPhoneVIO
//
//  Created by David Gao on 4/26/24.
//

import Foundation
import ARKit
import Combine
import CoreVideo

class ViewController: UIViewController, ARSessionDelegate, ObservableObject {
    @Published var displayString: String = ""

    let session = ARSession()
    let socketClient = SocketClient()
    var hostIP: String = "192.168.123.18"

    var hostPort: Int = 5555
    var prevTimestamp: Double = 0.0
    private var depthStreamingEnabled: Bool = false
    private var usingSmoothedDepth: Bool = false
    // UMI-FT style depth preprocessing:
    // - emphasize near-surface geometry by clipping depth
    // - optional downsampling for lower streaming bandwidth
    private let depthClipMaxMeters: Float = 0.5
    private let depthDownsampleFactor: Int = 2
    private let minConfidenceLevel: UInt8 = 1
    private let enableDepthCompression: Bool = true

    private struct DepthPayload {
        let depthBytes: Data
        let confidenceBytes: Data
        let width: UInt32
        let height: UInt32
        let isClipped: Bool
        let isDownsampled: Bool
        let isConfidenceFiltered: Bool
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        setupARSession()
        subscribeToActionStream()
    }
    
    func setupARSession() {
        socketClient.connect(hostIP: hostIP, hostPort: hostPort)
        self.publishPose = true
        session.delegate = self
        let configuration = ARWorldTrackingConfiguration()

        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
            configuration.frameSemantics.insert(.smoothedSceneDepth)
            depthStreamingEnabled = true
            usingSmoothedDepth = true
            print("Depth streaming enabled with smoothedSceneDepth")
        } else if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            configuration.frameSemantics.insert(.sceneDepth)
            depthStreamingEnabled = true
            usingSmoothedDepth = false
            print("Depth streaming enabled with sceneDepth")
        } else {
            depthStreamingEnabled = false
            usingSmoothedDepth = false
            print("Depth streaming unavailable; fallback to pose-only mode")
        }

        session.run(configuration)
    }

    private var cancellables: Set<AnyCancellable> = []
    private var publishPose: Bool = false

    func subscribeToActionStream() {
        
        ARManager.shared
            .actionStream
            .sink { [weak self] action in
                switch action {
                    case .update(let ip, let port):
                        self?.publishPose = false
                        self?.socketClient.disconnect()
                        self?.hostIP = ip
                        self?.hostPort = port
                        print("Reconnecting to ZMQ Publisher: \(self!.hostIP):\(self!.hostPort)")
                        self?.socketClient.connect(hostIP: self!.hostIP, hostPort: self!.hostPort)
                        self?.publishPose = true
                }
            }
            .store(in: &cancellables)
        

    }
    
    // ARSessionDelegate method
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let transform = frame.camera.transform
        let timestamp = frame.timestamp
        let dt = timestamp - self.prevTimestamp
        let fps: Double = dt > 0 ? 1.0 / dt : 0.0
        let streamMode = depthStreamingEnabled ? (usingSmoothedDepth ? "depth_smooth" : "depth_raw") : "pose_only"

        displayString = "x: \(String(format: "%.4f", transform[3][0])), y: \(String(format: "%.4f", transform[3][1])), z: \(String(format: "%.4f", transform[3][2])), fps: \(String(format: "%.3f", fps)), mode: \(streamMode)"
        prevTimestamp = timestamp
        if publishPose {
            if depthStreamingEnabled, let depthPacket = makeDepthPacket(frame: frame, transform: transform, poseTimestamp: timestamp) {
                socketClient.sendDataV2(depthPacket)
            } else {
                let dataPacket = DataPacket(transformMatrix: transform, timestamp: timestamp)
                socketClient.sendData(dataPacket)
            }
        }
    }

    private func makeDepthPacket(frame: ARFrame, transform: simd_float4x4, poseTimestamp: Double) -> DataPacketV2? {
        let depthData: ARDepthData?
        if usingSmoothedDepth {
            depthData = frame.smoothedSceneDepth ?? frame.sceneDepth
        } else {
            depthData = frame.sceneDepth ?? frame.smoothedSceneDepth
        }

        guard let depthData else {
            return nil
        }

        let depthMap = depthData.depthMap
        let confidenceMap = depthData.confidenceMap
        guard let payload = preprocessDepth(depthMap: depthMap, confidenceMap: confidenceMap) else {
            return nil
        }

        let cameraResolution = frame.camera.imageResolution
        let cameraWidth = UInt32(max(0, Int(cameraResolution.width.rounded())))
        let cameraHeight = UInt32(max(0, Int(cameraResolution.height.rounded())))

        var depthTimestamp = frame.capturedDepthDataTimestamp
        if depthTimestamp <= 0 {
            depthTimestamp = poseTimestamp
        }

        return DataPacketV2(
            transformMatrix: transform,
            poseTimestamp: poseTimestamp,
            depthTimestamp: depthTimestamp,
            intrinsics: frame.camera.intrinsics,
            cameraWidth: cameraWidth,
            cameraHeight: cameraHeight,
            depthWidth: payload.width,
            depthHeight: payload.height,
            depthBytes: payload.depthBytes,
            confidenceBytes: payload.confidenceBytes,
            isSmoothedDepth: usingSmoothedDepth,
            depthClipMaxMeters: depthClipMaxMeters,
            depthDownsampleFactor: UInt8(max(1, min(255, depthDownsampleFactor))),
            minConfidenceLevel: minConfidenceLevel,
            isDepthClipped: payload.isClipped,
            isDepthDownsampled: payload.isDownsampled,
            isConfidenceFiltered: payload.isConfidenceFiltered,
            enableZlibCompression: enableDepthCompression
        )
    }

    private func preprocessDepth(depthMap: CVPixelBuffer, confidenceMap: CVPixelBuffer?) -> DepthPayload? {
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(depthMap, .readOnly)
        }
        if let confidenceMap {
            CVPixelBufferLockBaseAddress(confidenceMap, .readOnly)
        }
        defer {
            if let confidenceMap {
                CVPixelBufferUnlockBaseAddress(confidenceMap, .readOnly)
            }
        }

        let width = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)
        guard width > 0, height > 0 else {
            return nil
        }
        let depthBytesPerRow = CVPixelBufferGetBytesPerRow(depthMap)
        guard let depthBaseAddress = CVPixelBufferGetBaseAddress(depthMap) else {
            return nil
        }

        var confidenceBaseAddress: UnsafeMutableRawPointer?
        var confidenceBytesPerRow = 0
        if let confidenceMap {
            confidenceBaseAddress = CVPixelBufferGetBaseAddress(confidenceMap)
            confidenceBytesPerRow = CVPixelBufferGetBytesPerRow(confidenceMap)
        }

        let step = max(1, depthDownsampleFactor)
        let outputWidth = (width + step - 1) / step
        let outputHeight = (height + step - 1) / step
        let outputCount = outputWidth * outputHeight
        var depthOutput = Data(capacity: outputCount * MemoryLayout<Float32>.size)
        var confidenceOutput = Data(capacity: outputCount)

        let doClip = depthClipMaxMeters > 0
        let doConfidenceFilter = confidenceBaseAddress != nil
        var clippedAny = false

        for y in stride(from: 0, to: height, by: step) {
            let depthRow = depthBaseAddress
                .advanced(by: y * depthBytesPerRow)
                .assumingMemoryBound(to: Float32.self)
            let confidenceRow: UnsafeMutablePointer<UInt8>? = {
                guard let confidenceBaseAddress else {
                    return nil
                }
                return confidenceBaseAddress
                    .advanced(by: y * confidenceBytesPerRow)
                    .assumingMemoryBound(to: UInt8.self)
            }()

            for x in stride(from: 0, to: width, by: step) {
                var depthValue = depthRow[x]
                var confidenceValue: UInt8 = 0

                if let confidenceRow {
                    confidenceValue = confidenceRow[x]
                }

                if !depthValue.isFinite || depthValue <= 0 {
                    depthValue = 0
                } else {
                    if doConfidenceFilter && confidenceValue < minConfidenceLevel {
                        depthValue = 0
                    } else if doClip && depthValue > depthClipMaxMeters {
                        depthValue = depthClipMaxMeters
                        clippedAny = true
                    }
                }

                var depthRaw = depthValue
                Swift.withUnsafeBytes(of: &depthRaw) { bytes in
                    depthOutput.append(bytes.bindMemory(to: UInt8.self))
                }

                if doConfidenceFilter {
                    confidenceOutput.append(confidenceValue)
                }
            }
        }

        return DepthPayload(
            depthBytes: depthOutput,
            confidenceBytes: doConfidenceFilter ? confidenceOutput : Data(),
            width: UInt32(outputWidth),
            height: UInt32(outputHeight),
            isClipped: clippedAny,
            isDownsampled: step > 1,
            isConfidenceFiltered: doConfidenceFilter && minConfidenceLevel > 0
        )
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        session.pause()
    }
}
