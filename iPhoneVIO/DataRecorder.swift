import Foundation
import ARKit
import AVFoundation

class DataRecorder: ObservableObject {
    @Published var isRecording = false
    private var baseDirectory: URL?
    private var poseDataArray: [[String: Any]] = []
    
    // Writers
    private var rgbAssetWriter: AVAssetWriter?
    private var rgbPixelBufferInput: AVAssetWriterInputPixelBufferAdaptor?
    private var rgbVideoInput: AVAssetWriterInput?
    
    private var uwAssetWriter: AVAssetWriter?
    private var uwPixelBufferInput: AVAssetWriterInputPixelBufferAdaptor?
    private var uwVideoInput: AVAssetWriterInput?
    
    private var depthFileHandle: FileHandle?
    private var frameCount = 0
    private var hasStartedWriting = false
    
    // Time: use relative times for AVAssetWriter starting from 0
    private var firstFrameEpoch: Double = 0
    
    // Resolution logic
    private var videoSize: CGSize = .zero
    
    func getIsRecording() -> Bool {
        return isRecording
    }
    
    /// Write a test file to Documents on first launch to verify file sharing works
    static func writeTestFileToDocuments() {
        let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
        let documentsDir = paths[0]
        let testURL = documentsDir.appendingPathComponent("_iPhoneVIO_test.txt")
        do {
            try "iPhoneVIO file sharing is working! Documents dir: \(documentsDir.path)".write(to: testURL, atomically: true, encoding: .utf8)
            print("[DataRecorder] Test file written to: \(testURL.path)")
        } catch {
            print("[DataRecorder] FAILED to write test file: \(error)")
        }
    }
    
    func startRecording(cameraResolution: CGSize) {
        if isRecording { return }
        self.videoSize = cameraResolution
        self.poseDataArray.removeAll()
        self.frameCount = 0
        self.hasStartedWriting = false
        self.firstFrameEpoch = 0
        
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        formatter.timeZone = TimeZone(abbreviation: "UTC")
        let dateString = formatter.string(from: Date())
        
        let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
        let documentsDir = paths[0]
        let sessionDir = documentsDir.appendingPathComponent("export_\(dateString)")
        do {
            try FileManager.default.createDirectory(at: sessionDir, withIntermediateDirectories: true, attributes: nil)
            self.baseDirectory = sessionDir
            print("[DataRecorder] Created session dir: \(sessionDir.path)")
        } catch {
            print("[DataRecorder] FAILED to create recording directory: \(error)")
            return
        }
        
        setupRGBWriter()
        setupUWWriter()
        setupDepthWriter()
        
        self.isRecording = true
        print("[DataRecorder] Recording started. videoSize=\(videoSize), rgbWriter=\(String(describing: rgbAssetWriter?.status.rawValue)), uwWriter=\(String(describing: uwAssetWriter?.status.rawValue))")
    }
    
    private func setupRGBWriter() {
        guard let baseDirectory = baseDirectory else { return }
        let url = baseDirectory.appendingPathComponent("right_rgb.mp4")
        setupVideoWriter(url: url, writer: &rgbAssetWriter, videoInput: &rgbVideoInput, pixelBufferInput: &rgbPixelBufferInput)
    }
    
    private func setupUWWriter() {
        guard let baseDirectory = baseDirectory else { return }
        let url = baseDirectory.appendingPathComponent("right_ultrawidergb.mp4")
        setupVideoWriter(url: url, writer: &uwAssetWriter, videoInput: &uwVideoInput, pixelBufferInput: &uwPixelBufferInput)
    }
    
    private func setupVideoWriter(url: URL, writer: inout AVAssetWriter?, videoInput: inout AVAssetWriterInput?, pixelBufferInput: inout AVAssetWriterInputPixelBufferAdaptor?) {
        do {
            writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
            let width = Int(videoSize.width)
            let height = Int(videoSize.height)
            let outputSettings: [String: Any] = [
                AVVideoCodecKey: AVVideoCodecType.h264,
                AVVideoWidthKey: width,
                AVVideoHeightKey: height
            ]
            
            let input = AVAssetWriterInput(mediaType: .video, outputSettings: outputSettings)
            input.expectsMediaDataInRealTime = true
            
            let sourcePixelBufferAttributes: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height
            ]
            
            let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: sourcePixelBufferAttributes)
            
            if writer!.canAdd(input) {
                writer!.add(input)
            } else {
                print("[DataRecorder] WARNING: cannot add input to writer for \(url.lastPathComponent)")
            }
            
            videoInput = input
            pixelBufferInput = adaptor
            print("[DataRecorder] Video writer setup OK for \(url.lastPathComponent) (\(width)x\(height))")
            
        } catch {
            print("[DataRecorder] FAILED to initialize video writer: \(error)")
        }
    }
    
    private func setupDepthWriter() {
        guard let baseDirectory = baseDirectory else { return }
        let depthURL = baseDirectory.appendingPathComponent("right_depth.raw")
        FileManager.default.createFile(atPath: depthURL.path, contents: nil, attributes: nil)
        do {
            depthFileHandle = try FileHandle(forWritingTo: depthURL)
            print("[DataRecorder] Depth writer setup OK")
        } catch {
            print("[DataRecorder] FAILED to open depth file handle: \(error)")
        }
    }
    
    func record(frame: ARFrame, epochTimeSeconds: Double, depthPayloadFloat16: Data?) {
        guard isRecording else { return }
        
        // Use RELATIVE time from first frame for AVAssetWriter (absolute epoch is too large)
        if firstFrameEpoch == 0 {
            firstFrameEpoch = epochTimeSeconds
        }
        let relativeTime = epochTimeSeconds - firstFrameEpoch
        let presentationTime = CMTime(seconds: relativeTime, preferredTimescale: 600)
        
        if !hasStartedWriting {
            if let w = rgbAssetWriter {
                let ok = w.startWriting()
                if ok {
                    w.startSession(atSourceTime: presentationTime)
                    print("[DataRecorder] RGB writer started session. status=\(w.status.rawValue)")
                } else {
                    print("[DataRecorder] RGB writer FAILED to start: \(String(describing: w.error))")
                }
            }
            
            if let w = uwAssetWriter {
                let ok = w.startWriting()
                if ok {
                    w.startSession(atSourceTime: presentationTime)
                    print("[DataRecorder] UW writer started session. status=\(w.status.rawValue)")
                } else {
                    print("[DataRecorder] UW writer FAILED to start: \(String(describing: w.error))")
                }
            }
            hasStartedWriting = true
        }
        
        // 1. Record Main RGB
        if let rgbInput = rgbVideoInput, rgbInput.isReadyForMoreMediaData, let adaptor = rgbPixelBufferInput {
            let ok = adaptor.append(frame.capturedImage, withPresentationTime: presentationTime)
            if !ok, let w = rgbAssetWriter {
                print("[DataRecorder] RGB append FAILED at frame \(frameCount), status=\(w.status.rawValue), error=\(String(describing: w.error))")
            }
        }
        
        // 2. Record UltraWide RGB (Duplicate Main RGB as a workaround for UMI-FT assert)
        if let uwInput = uwVideoInput, uwInput.isReadyForMoreMediaData, let adaptor = uwPixelBufferInput {
            let ok = adaptor.append(frame.capturedImage, withPresentationTime: presentationTime)
            if !ok, let w = uwAssetWriter {
                print("[DataRecorder] UW append FAILED at frame \(frameCount), status=\(w.status.rawValue), error=\(String(describing: w.error))")
            }
        }
        
        // 3. Log Poses
        let matrix = frame.camera.transform
        
        // UMI-FT JSON expects 4x4 array of row-major SE3 matrices.
        // matrix[col][row] gives the element, so row `i` and col `j` is matrix[j][i].
        var rowMajor4x4: [[Float]] = []
        for row in 0..<4 {
            var rowArr: [Float] = []
            for col in 0..<4 {
                rowArr.append(matrix[col][row])
            }
            rowMajor4x4.append(rowArr)
        }
        
        // precise iso format, e.g., 2024-11-19T23:52:14.671083Z
        let date = Date(timeIntervalSince1970: epochTimeSeconds)
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSSSSS'Z'"
        formatter.timeZone = TimeZone(abbreviation: "UTC")
        let epochTimeStr = formatter.string(from: date)
        
        poseDataArray.append([
            "timeText": epochTimeStr,
            "transform": rowMajor4x4
        ])
        
        // 4. Record Depth
        if let depthBytes = depthPayloadFloat16, let handle = depthFileHandle {
            handle.write(depthBytes)
        }
        
        frameCount += 1
        if frameCount % 30 == 0 {
            print("[DataRecorder] Recorded \(frameCount) frames so far...")
        }
    }
    
    func stopRecording() {
        guard isRecording else { return }
        isRecording = false
        
        print("[DataRecorder] Stopping... \(frameCount) frames recorded")
        
        rgbVideoInput?.markAsFinished()
        uwVideoInput?.markAsFinished()
        
        let group = DispatchGroup()
        
        if let w = rgbAssetWriter, w.status == .writing {
            group.enter()
            w.finishWriting {
                print("[DataRecorder] RGB Writer finished, status=\(w.status.rawValue)")
                group.leave()
            }
        }
        
        if let w = uwAssetWriter, w.status == .writing {
            group.enter()
            w.finishWriting {
                print("[DataRecorder] UW Writer finished, status=\(w.status.rawValue)")
                group.leave()
            }
        }
        
        // wait for finish
        group.notify(queue: .main) {
            self.rgbAssetWriter = nil
            self.uwAssetWriter = nil
            print("[DataRecorder] Writers cleaned up")
        }
        
        depthFileHandle?.closeFile()
        depthFileHandle = nil
        
        saveJSON()
        
        print("[DataRecorder] Recording stopped. Saved \(frameCount) frames to \(baseDirectory?.path ?? "nil")")
        baseDirectory = nil
    }
    
    private func saveJSON() {
        guard let baseDirectory = baseDirectory else {
            print("[DataRecorder] saveJSON: baseDirectory is nil!")
            return
        }
        var times: [String] = []
        var transforms: [[[Float]]] = []
        
        for pose in poseDataArray {
            if let t = pose["timeText"] as? String, let m = pose["transform"] as? [[Float]] {
                times.append(t)
                transforms.append(m)
            }
        }
        
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        formatter.timeZone = TimeZone(abbreviation: "UTC")
        let endString = formatter.string(from: Date())
        let jsonName = "\(endString)_right.json"
        
        let jsonDict: [String: Any] = [
            "poseTimes": times,
            "poseTransforms": transforms
        ]
        
        do {
            let data = try JSONSerialization.data(withJSONObject: jsonDict, options: .prettyPrinted)
            let jsonURL = baseDirectory.appendingPathComponent(jsonName)
            try data.write(to: jsonURL)
            print("[DataRecorder] Saved JSON to \(jsonURL.path)")
            
            // Debug marker file
            let debugURL = baseDirectory.appendingPathComponent("_DEBUG_SUCCESS.txt")
            try "Saved \(frameCount) frames successfully!\nDirectory: \(baseDirectory.path)".write(to: debugURL, atomically: true, encoding: .utf8)
            print("[DataRecorder] Debug file written to \(debugURL.path)")
        } catch {
            print("[DataRecorder] FAILED to save JSON: \(error)")
        }
    }
}
