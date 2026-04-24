import AudioToolbox
import CoreMedia
import Dispatch
import Foundation
import ScreenCaptureKit

@available(macOS 13.0, *)
final class SystemAudioOutput: NSObject, SCStreamOutput, SCStreamDelegate {
    private let output = FileHandle.standardOutput
    private var reportedFormat = false
    private var reportedUnsupported = false

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        fputs("[sck] stream stopped: \(error)\n", stderr)
        exit(2)
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio, sampleBuffer.isValid else {
            return
        }
        guard let formatDescription = CMSampleBufferGetFormatDescription(sampleBuffer),
              let asbdPointer = CMAudioFormatDescriptionGetStreamBasicDescription(formatDescription) else {
            return
        }

        let asbd = asbdPointer.pointee
        var bufferList = AudioBufferList()
        var blockBuffer: CMBlockBuffer?
        let status = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sampleBuffer,
            bufferListSizeNeededOut: nil,
            bufferListOut: &bufferList,
            bufferListSize: MemoryLayout<AudioBufferList>.size,
            blockBufferAllocator: kCFAllocatorDefault,
            blockBufferMemoryAllocator: kCFAllocatorDefault,
            flags: UInt32(kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment),
            blockBufferOut: &blockBuffer
        )
        guard status == noErr else {
            return
        }

        if !reportedFormat {
            reportedFormat = true
            fputs(
                "[sck] audio format sampleRate=\(Int(asbd.mSampleRate)) channels=\(asbd.mChannelsPerFrame) bits=\(asbd.mBitsPerChannel) flags=\(asbd.mFormatFlags)\n",
                stderr
            )
        }

        let buffers = UnsafeMutableAudioBufferListPointer(&bufferList)
        for audioBuffer in buffers {
            guard let mData = audioBuffer.mData else {
                continue
            }
            writePCM(from: mData, byteCount: Int(audioBuffer.mDataByteSize), asbd: asbd)
        }
    }

    private func writePCM(from pointer: UnsafeMutableRawPointer, byteCount: Int, asbd: AudioStreamBasicDescription) {
        guard asbd.mFormatID == kAudioFormatLinearPCM else {
            reportUnsupported(asbd)
            return
        }

        let isFloat = (asbd.mFormatFlags & kAudioFormatFlagIsFloat) != 0
        let isSignedInteger = (asbd.mFormatFlags & kAudioFormatFlagIsSignedInteger) != 0

        if isFloat && asbd.mBitsPerChannel == 32 {
            let sampleCount = byteCount / MemoryLayout<Float32>.size
            let samples = pointer.bindMemory(to: Float32.self, capacity: sampleCount)
            var pcm = [Int16]()
            pcm.reserveCapacity(sampleCount)
            for idx in 0..<sampleCount {
                let clipped = min(1.0, max(-1.0, samples[idx]))
                pcm.append(Int16(clipped * Float32(Int16.max)))
            }
            pcm.withUnsafeBufferPointer { buffer in
                output.write(Data(buffer: buffer))
            }
            return
        }

        if isSignedInteger && asbd.mBitsPerChannel == 16 {
            output.write(Data(bytes: pointer, count: byteCount))
            return
        }

        reportUnsupported(asbd)
    }

    private func reportUnsupported(_ asbd: AudioStreamBasicDescription) {
        if reportedUnsupported {
            return
        }
        reportedUnsupported = true
        fputs(
            "[sck] unsupported audio format id=\(asbd.mFormatID) bits=\(asbd.mBitsPerChannel) flags=\(asbd.mFormatFlags)\n",
            stderr
        )
    }
}

@available(macOS 13.0, *)
final class CaptureRunner {
    private var stream: SCStream?
    private let output = SystemAudioOutput()

    func start() throws {
        fputs("[sck] requesting shareable content\n", stderr)
        let content = try getShareableContent()
        fputs("[sck] found \(content.displays.count) display(s)\n", stderr)
        guard let display = content.displays.first else {
            throw RuntimeError("no capturable display found")
        }

        let filter = SCContentFilter(display: display, excludingWindows: [])
        let configuration = SCStreamConfiguration()
        configuration.width = max(2, display.width)
        configuration.height = max(2, display.height)
        configuration.minimumFrameInterval = CMTime(value: 1, timescale: 1)
        configuration.capturesAudio = true
        configuration.sampleRate = 16_000
        configuration.channelCount = 1
        configuration.excludesCurrentProcessAudio = true

        let stream = SCStream(filter: filter, configuration: configuration, delegate: output)
        try stream.addStreamOutput(output, type: .audio, sampleHandlerQueue: DispatchQueue(label: "sck-audio"))
        try startCapture(stream)
        self.stream = stream

        fputs("[sck] capture started; grant Screen Recording permission if macOS prompts\n", stderr)
    }

    private func getShareableContent() throws -> SCShareableContent {
        let semaphore = DispatchSemaphore(value: 0)
        var result: SCShareableContent?
        var failure: Error?

        SCShareableContent.getExcludingDesktopWindows(false, onScreenWindowsOnly: true) { content, error in
            result = content
            failure = error
            semaphore.signal()
        }
        if semaphore.wait(timeout: .now() + 20) != .success {
            throw RuntimeError(
                "timed out while requesting shareable content; grant Screen & System Audio Recording permission and rerun"
            )
        }

        if let failure {
            throw failure
        }
        guard let result else {
            throw RuntimeError("ScreenCaptureKit returned no shareable content")
        }
        return result
    }

    private func startCapture(_ stream: SCStream) throws {
        let semaphore = DispatchSemaphore(value: 0)
        var failure: Error?

        fputs("[sck] starting stream\n", stderr)
        stream.startCapture { error in
            failure = error
            semaphore.signal()
        }
        if semaphore.wait(timeout: .now() + 10) != .success {
            throw RuntimeError("timed out while starting ScreenCaptureKit stream")
        }

        if let failure {
            throw failure
        }
    }
}

struct RuntimeError: Error, CustomStringConvertible {
    let description: String

    init(_ description: String) {
        self.description = description
    }
}

@main
struct ScreenCaptureAudioMain {
    static func main() async {
        guard #available(macOS 13.0, *) else {
            fputs("[sck] ScreenCaptureKit audio requires macOS 13.0 or newer\n", stderr)
            exit(1)
        }

        do {
            let runner = CaptureRunner()
            try runner.start()
            dispatchMain()
        } catch {
            fputs("[sck] failed to start capture: \(error)\n", stderr)
            fputs("[sck] check System Settings > Privacy & Security > Screen & System Audio Recording\n", stderr)
            exit(1)
        }
    }
}
