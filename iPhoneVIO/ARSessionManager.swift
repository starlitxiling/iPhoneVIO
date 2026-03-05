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
        guard let depthBytes = pixelBufferToContiguousBytes(depthMap, bytesPerPixel: MemoryLayout<Float32>.size) else {
            return nil
        }

        var confidenceBytes = Data()
        if let confidenceMap = depthData.confidenceMap,
           let encodedConfidence = pixelBufferToContiguousBytes(confidenceMap, bytesPerPixel: MemoryLayout<UInt8>.size) {
            confidenceBytes = encodedConfidence
        }

        let cameraResolution = frame.camera.imageResolution
        let cameraWidth = UInt32(max(0, Int(cameraResolution.width.rounded())))
        let cameraHeight = UInt32(max(0, Int(cameraResolution.height.rounded())))

        let depthWidth = UInt32(CVPixelBufferGetWidth(depthMap))
        let depthHeight = UInt32(CVPixelBufferGetHeight(depthMap))
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
            depthWidth: depthWidth,
            depthHeight: depthHeight,
            depthBytes: depthBytes,
            confidenceBytes: confidenceBytes,
            isSmoothedDepth: usingSmoothedDepth
        )
    }

    private func pixelBufferToContiguousBytes(_ pixelBuffer: CVPixelBuffer, bytesPerPixel: Int) -> Data? {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
        }

        let planeCount = CVPixelBufferGetPlaneCount(pixelBuffer)
        let width: Int
        let height: Int
        let bytesPerRow: Int
        let baseAddress: UnsafeMutableRawPointer?

        if planeCount > 0 {
            width = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0)
            height = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0)
            bytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0)
            baseAddress = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0)
        } else {
            width = CVPixelBufferGetWidth(pixelBuffer)
            height = CVPixelBufferGetHeight(pixelBuffer)
            bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
            baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer)
        }

        guard let baseAddress else {
            return nil
        }
        let rowBytes = width * bytesPerPixel
        guard rowBytes > 0, bytesPerRow >= rowBytes, height > 0 else {
            return nil
        }

        var output = Data(capacity: rowBytes * height)
        for row in 0..<height {
            let rowPointer = baseAddress.advanced(by: row * bytesPerRow).assumingMemoryBound(to: UInt8.self)
            output.append(rowPointer, count: rowBytes)
        }
        return output
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        session.pause()
    }
}
