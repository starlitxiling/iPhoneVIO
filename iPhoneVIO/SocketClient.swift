import SocketIO
import simd

private extension Data {
    mutating func appendUInt8(_ value: UInt8) {
        var mutableValue = value
        Swift.withUnsafeBytes(of: &mutableValue) { bytes in
            append(bytes.bindMemory(to: UInt8.self))
        }
    }

    mutating func appendUInt16LE(_ value: UInt16) {
        var mutableValue = value.littleEndian
        Swift.withUnsafeBytes(of: &mutableValue) { bytes in
            append(bytes.bindMemory(to: UInt8.self))
        }
    }

    mutating func appendUInt32LE(_ value: UInt32) {
        var mutableValue = value.littleEndian
        Swift.withUnsafeBytes(of: &mutableValue) { bytes in
            append(bytes.bindMemory(to: UInt8.self))
        }
    }

    mutating func appendUInt64LE(_ value: UInt64) {
        var mutableValue = value.littleEndian
        Swift.withUnsafeBytes(of: &mutableValue) { bytes in
            append(bytes.bindMemory(to: UInt8.self))
        }
    }

    mutating func appendFloat32LE(_ value: Float) {
        appendUInt32LE(value.bitPattern)
    }

    mutating func appendFloat64LE(_ value: Double) {
        appendUInt64LE(value.bitPattern)
    }
}

class DataPacket {
    var transformMatrix: simd_float4x4
    var timestamp: Double

    init(transformMatrix: simd_float4x4, timestamp: Double) {
        self.transformMatrix = transformMatrix
        self.timestamp = timestamp
    }
    func toBytes() -> Data {
        var data = Data()

        // Append pose data
        for i in 0..<4 {
            for j in 0..<4 {
                var val = transformMatrix[i][j]
                data.append(Data(bytes: &val, count: MemoryLayout<Float>.size))
            }
        }

        // Append timestamp
        var timestampVal = timestamp
        data.append(Data(bytes: &timestampVal, count: MemoryLayout<Int64>.size))

        return data
    }
}

class DataPacketV2 {
    static let magicBytes = [UInt8]("IPV2".utf8)
    static let depthFormatFloat32: UInt8 = 1
    static let confidenceFormatU8: UInt8 = 1

    var transformMatrix: simd_float4x4
    var poseTimestamp: Double
    var depthTimestamp: Double
    var intrinsics: simd_float3x3
    var cameraWidth: UInt32
    var cameraHeight: UInt32
    var depthWidth: UInt32
    var depthHeight: UInt32
    var depthBytes: Data
    var confidenceBytes: Data
    var isSmoothedDepth: Bool

    init(
        transformMatrix: simd_float4x4,
        poseTimestamp: Double,
        depthTimestamp: Double,
        intrinsics: simd_float3x3,
        cameraWidth: UInt32,
        cameraHeight: UInt32,
        depthWidth: UInt32,
        depthHeight: UInt32,
        depthBytes: Data,
        confidenceBytes: Data,
        isSmoothedDepth: Bool
    ) {
        self.transformMatrix = transformMatrix
        self.poseTimestamp = poseTimestamp
        self.depthTimestamp = depthTimestamp
        self.intrinsics = intrinsics
        self.cameraWidth = cameraWidth
        self.cameraHeight = cameraHeight
        self.depthWidth = depthWidth
        self.depthHeight = depthHeight
        self.depthBytes = depthBytes
        self.confidenceBytes = confidenceBytes
        self.isSmoothedDepth = isSmoothedDepth
    }

    func toBytes() -> Data {
        var data = Data()
        data.append(contentsOf: DataPacketV2.magicBytes)
        data.appendUInt8(2) // version

        var flags: UInt8 = 0
        if !depthBytes.isEmpty {
            flags |= 1 << 0
        }
        if !confidenceBytes.isEmpty {
            flags |= 1 << 1
        }
        if isSmoothedDepth {
            flags |= 1 << 2
        }
        data.appendUInt8(flags)
        data.appendUInt16LE(0) // reserved

        data.appendFloat64LE(poseTimestamp)
        data.appendFloat64LE(depthTimestamp)

        // Keep column-major ordering to match existing transform serialization.
        for i in 0..<4 {
            for j in 0..<4 {
                data.appendFloat32LE(transformMatrix[i][j])
            }
        }

        for i in 0..<3 {
            for j in 0..<3 {
                data.appendFloat32LE(intrinsics[i][j])
            }
        }

        data.appendUInt32LE(cameraWidth)
        data.appendUInt32LE(cameraHeight)
        data.appendUInt32LE(depthWidth)
        data.appendUInt32LE(depthHeight)

        data.appendUInt8(DataPacketV2.depthFormatFloat32)
        data.appendUInt8(confidenceBytes.isEmpty ? 0 : DataPacketV2.confidenceFormatU8)
        data.appendUInt32LE(UInt32(depthBytes.count))
        data.appendUInt32LE(UInt32(confidenceBytes.count))

        data.append(depthBytes)
        data.append(confidenceBytes)
        return data
    }
}


class SocketClient{
    private var manager: SocketManager?
    private var socket: SocketIOClient?

    var ready: Bool = false
    var socketOpened: Bool
    var prevTimestamp: Double = 0
    init(){
        socketOpened = false
    }

    func connect(hostIP: String, hostPort: Int) {
        self.ready = false
        print("Connecting to \(hostIP):\(hostPort)")
        self.manager = SocketManager(socketURL: URL(string: "http://\(hostIP):\(hostPort)")!, config: [.log(true), .compress])
        usleep(100000)
        self.socket = self.manager?.defaultSocket
        self.socket?.connect()
        usleep(100000)
        self.ready = true
    }

    func sendData(_ data: DataPacket) {
        if !ready {
            print("Not ready to send")
            return
        }
        print("Start sending package, freq: \(1/(data.timestamp - prevTimestamp))Hz")
        prevTimestamp = data.timestamp
        self.ready = false
        self.socket?.emit("update", data.toBytes().base64EncodedString())
        self.ready = true
    }

    func sendDataV2(_ data: DataPacketV2) {
        if !ready {
            print("Not ready to send")
            return
        }
        let dt = data.poseTimestamp - prevTimestamp
        if dt > 0 {
            print(
                "Start sending v2 package, freq: \(1/dt)Hz, depth: \(data.depthWidth)x\(data.depthHeight)"
            )
        } else {
            print("Start sending v2 package")
        }
        prevTimestamp = data.poseTimestamp
        self.ready = false
        self.socket?.emit("update_v2", data.toBytes().base64EncodedString())
        self.ready = true
    }

    func disconnect() {
        self.ready = false
        socket?.disconnect()
        usleep(100000)
        self.socketOpened = false
    }
}
