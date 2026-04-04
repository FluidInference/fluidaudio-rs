import Foundation
import AVFoundation
import FluidAudio
import Darwin

// MARK: - Bridge Class

/// Internal bridge class that wraps FluidAudio
/// Internal diarization segment used within the bridge.
struct BridgeDiarizationSegment {
    var speakerId: String
    var startTime: Float
    var endTime: Float
    var qualityScore: Float
}

class FluidAudioBridgeInternal {
    private var asrManager: AsrManager?
    private var asrModels: AsrModels?
    private var vadManager: VadManager?
    private var diarizerManager: OfflineDiarizerManager?
    private var streamingAsrManager: StreamingAsrManager?
    private var qwen3AsrManager: Qwen3AsrManager?
    private var qwen3StreamingManager: Qwen3StreamingManager?

    init() {}

    func initializeAsr() throws {
        let semaphore = DispatchSemaphore(value: 0)
        var initError: Error?

        Task {
            do {
                let models = try await AsrModels.downloadAndLoad()
                self.asrModels = models

                let manager = AsrManager()
                try await manager.initialize(models: models)
                self.asrManager = manager
            } catch {
                initError = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let error = initError {
            throw error
        }
    }

    func transcribeFile(_ path: String) throws -> (String, Float, Double, Double, Float) {
        guard let manager = asrManager else {
            throw BridgeError.notInitialized
        }

        let semaphore = DispatchSemaphore(value: 0)
        var result: ASRResult?
        var transcribeError: Error?

        Task {
            do {
                let url = URL(fileURLWithPath: path)
                result = try await manager.transcribe(url)
            } catch {
                transcribeError = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let error = transcribeError {
            throw error
        }

        guard let r = result else {
            throw BridgeError.noResult
        }

        return (r.text, r.confidence, r.duration, r.processingTime, r.rtfx)
    }

    func transcribeSamples(_ samples: [Float]) throws -> (String, Float, Double, Double, Float) {
        guard let manager = asrManager else {
            throw BridgeError.notInitialized
        }

        let semaphore = DispatchSemaphore(value: 0)
        var result: ASRResult?
        var transcribeError: Error?

        Task {
            do {
                result = try await manager.transcribe(samples)
            } catch {
                transcribeError = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let error = transcribeError {
            throw error
        }

        guard let r = result else {
            throw BridgeError.noResult
        }

        return (r.text, r.confidence, r.duration, r.processingTime, r.rtfx)
    }

    func isAsrAvailable() -> Bool {
        return asrManager != nil
    }

    func initializeVad(_ threshold: Float) throws {
        let semaphore = DispatchSemaphore(value: 0)
        var initError: Error?

        Task {
            do {
                let config = VadConfig(defaultThreshold: threshold)
                let manager = try await VadManager(config: config)
                self.vadManager = manager
            } catch {
                initError = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let error = initError {
            throw error
        }
    }

    func isVadAvailable() -> Bool {
        return vadManager != nil
    }

    // MARK: - Diarization

    func initializeDiarization(_ threshold: Double) throws {
        let semaphore = DispatchSemaphore(value: 0)
        var initError: Error?

        Task {
            do {
                var config = OfflineDiarizerConfig()
                config.clustering.threshold = threshold
                let manager = OfflineDiarizerManager(config: config)
                try await manager.prepareModels()
                self.diarizerManager = manager
            } catch {
                initError = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let error = initError {
            throw error
        }
    }

    func diarizeFile(_ path: String) throws -> [BridgeDiarizationSegment] {
        guard let manager = diarizerManager else {
            throw BridgeError.notInitialized
        }

        let semaphore = DispatchSemaphore(value: 0)
        var result: DiarizationResult?
        var diarizeError: Error?

        Task {
            do {
                let url = URL(fileURLWithPath: path)
                result = try await manager.process(url)
            } catch {
                diarizeError = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let error = diarizeError {
            throw error
        }

        guard let r = result else {
            throw BridgeError.noResult
        }

        return r.segments.map { segment in
            BridgeDiarizationSegment(
                speakerId: segment.speakerId,
                startTime: segment.startTimeSeconds,
                endTime: segment.endTimeSeconds,
                qualityScore: segment.qualityScore
            )
        }
    }

    func isDiarizationAvailable() -> Bool {
        return diarizerManager != nil
    }

    // MARK: - Streaming ASR

    func initializeStreamingAsr() throws {
        let semaphore = DispatchSemaphore(value: 0)
        var initError: Error?

        Task {
            do {
                let models = try await AsrModels.downloadAndLoad()
                self.asrModels = models

                let manager = StreamingAsrManager()
                self.streamingAsrManager = manager
            } catch {
                initError = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let error = initError {
            throw error
        }
    }

    func streamingAsrStart() throws {
        guard let manager = streamingAsrManager, let models = asrModels else {
            throw BridgeError.notInitialized
        }

        let semaphore = DispatchSemaphore(value: 0)
        var startError: Error?

        Task {
            do {
                try await manager.start(models: models, source: .microphone)
            } catch {
                startError = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let error = startError {
            throw error
        }
    }

    func streamingAsrFeed(_ samples: [Float]) throws {
        guard let manager = streamingAsrManager else {
            throw BridgeError.notInitialized
        }

        let semaphore = DispatchSemaphore(value: 0)

        Task {
            // Convert samples to AVAudioPCMBuffer
            let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
            let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: UInt32(samples.count))!
            buffer.frameLength = UInt32(samples.count)

            let channelData = buffer.floatChannelData![0]
            for (i, sample) in samples.enumerated() {
                channelData[i] = sample
            }

            await manager.streamAudio(buffer)
            semaphore.signal()
        }

        semaphore.wait()
    }

    func streamingAsrFinish() throws -> String {
        guard let manager = streamingAsrManager else {
            throw BridgeError.notInitialized
        }

        let semaphore = DispatchSemaphore(value: 0)
        var result: String?
        var finishError: Error?

        Task {
            do {
                result = try await manager.finish()
            } catch {
                finishError = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let error = finishError {
            throw error
        }

        return result ?? ""
    }

    func transcribeFileStreaming(_ path: String) throws -> (String, Float, Double, Double, Float) {
        guard let manager = streamingAsrManager, let models = asrModels else {
            throw BridgeError.notInitialized
        }

        let semaphore = DispatchSemaphore(value: 0)
        var text: String?
        var transcribeError: Error?
        var duration: Double = 0.0
        var processingTime: Double = 0.0

        Task {
            do {
                let url = URL(fileURLWithPath: path)

                let startTime = Date()
                try await manager.start(models: models, source: .microphone)

                // Load and stream audio file
                let audioFile = try AVAudioFile(forReading: url)
                let format = audioFile.processingFormat
                duration = Double(audioFile.length) / format.sampleRate

                let frameCount = AVAudioFrameCount(4096)
                let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)!

                while audioFile.framePosition < audioFile.length {
                    try audioFile.read(into: buffer)
                    await manager.streamAudio(buffer)
                }

                text = try await manager.finish()
                processingTime = Date().timeIntervalSince(startTime)
            } catch {
                transcribeError = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let error = transcribeError {
            throw error
        }

        let rtfx = duration > 0 ? Float(duration / processingTime) : 0.0
        return (text ?? "", 0.0, duration, processingTime, rtfx)
    }

    func isStreamingAsrAvailable() -> Bool {
        return streamingAsrManager != nil
    }

    // MARK: - Qwen3 ASR

    @available(macOS 15, iOS 18, *)
    func initializeQwen3Asr() throws {
        let semaphore = DispatchSemaphore(value: 0)
        var initError: Error?

        Task {
            do {
                let manager = Qwen3AsrManager()
                // Models are auto-downloaded from HuggingFace on first use
                let modelDir = try await Qwen3AsrModels.downloadModelIfNeeded()
                try await manager.loadModels(from: modelDir)
                self.qwen3AsrManager = manager
            } catch {
                initError = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let error = initError {
            throw error
        }
    }

    @available(macOS 15, iOS 18, *)
    func qwen3TranscribeSamples(_ samples: [Float], language: String?) throws -> (String, Float, Double, Double, Float) {
        guard let manager = qwen3AsrManager else {
            throw BridgeError.notInitialized
        }

        let semaphore = DispatchSemaphore(value: 0)
        var result: String?
        var transcribeError: Error?
        var processingTime: Double = 0.0

        Task {
            do {
                let startTime = Date()
                result = try await manager.transcribe(audioSamples: samples, language: language, maxNewTokens: 512)
                processingTime = Date().timeIntervalSince(startTime)
            } catch {
                transcribeError = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let error = transcribeError {
            throw error
        }

        let duration = Double(samples.count) / 16000.0
        let rtfx = duration > 0 ? Float(duration / processingTime) : 0.0

        return (result ?? "", 0.0, duration, processingTime, rtfx)
    }

    @available(macOS 15, iOS 18, *)
    func qwen3TranscribeFile(_ path: String, language: String?) throws -> (String, Float, Double, Double, Float) {
        guard qwen3AsrManager != nil else {
            throw BridgeError.notInitialized
        }

        let semaphore = DispatchSemaphore(value: 0)
        var samples: [Float]?
        var loadError: Error?
        var duration: Double = 0.0

        Task {
            do {
                let url = URL(fileURLWithPath: path)
                let audioFile = try AVAudioFile(forReading: url)
                let format = audioFile.processingFormat
                duration = Double(audioFile.length) / format.sampleRate

                // Convert to 16kHz mono
                let targetFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
                let converter = AVAudioConverter(from: format, to: targetFormat)!

                let capacity = UInt32(audioFile.length)
                let buffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: capacity)!

                var finished = false
                try converter.convert(to: buffer, error: nil) { _, outStatus in
                    if finished {
                        outStatus.pointee = .noDataNow
                        return nil
                    }

                    let inputBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity)!
                    try? audioFile.read(into: inputBuffer)

                    if inputBuffer.frameLength == 0 {
                        finished = true
                        outStatus.pointee = .endOfStream
                        return nil
                    }

                    outStatus.pointee = .haveData
                    return inputBuffer
                }

                let floatPtr = buffer.floatChannelData![0]
                let samplesArray = Array(UnsafeBufferPointer(start: floatPtr, count: Int(buffer.frameLength)))
                samples = samplesArray
            } catch {
                loadError = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let error = loadError {
            throw error
        }

        guard let audioSamples = samples else {
            throw BridgeError.noResult
        }

        return try qwen3TranscribeSamples(audioSamples, language: language)
    }

    @available(macOS 15, iOS 18, *)
    func isQwen3AsrAvailable() -> Bool {
        return qwen3AsrManager != nil
    }

    // MARK: - Qwen3 Streaming

    @available(macOS 15, iOS 18, *)
    func initializeQwen3Streaming() throws {
        let semaphore = DispatchSemaphore(value: 0)
        var initError: Error?

        Task {
            do {
                let asrManager = Qwen3AsrManager()
                let modelDir = try await Qwen3AsrModels.downloadModelIfNeeded()
                try await asrManager.loadModels(from: modelDir)

                let streamingManager = Qwen3StreamingManager(asrManager: asrManager)
                self.qwen3AsrManager = asrManager
                self.qwen3StreamingManager = streamingManager
            } catch {
                initError = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let error = initError {
            throw error
        }
    }

    @available(macOS 15, iOS 18, *)
    func qwen3StreamingStart(language: String?, minAudioSeconds: Double, chunkSeconds: Double, maxAudioSeconds: Double) throws {
        guard let manager = qwen3StreamingManager else {
            throw BridgeError.notInitialized
        }

        let semaphore = DispatchSemaphore(value: 0)

        Task {
            let languageEnum = language.flatMap { Qwen3AsrConfig.Language(from: $0) }
            let config = Qwen3StreamingConfig(
                minAudioSeconds: minAudioSeconds,
                chunkSeconds: chunkSeconds,
                maxAudioSeconds: maxAudioSeconds,
                language: languageEnum
            )
            await manager.configure(config)
            await manager.reset()
            semaphore.signal()
        }

        semaphore.wait()
    }

    @available(macOS 15, iOS 18, *)
    func qwen3StreamingFeed(_ samples: [Float]) throws -> String? {
        guard let manager = qwen3StreamingManager else {
            throw BridgeError.notInitialized
        }

        let semaphore = DispatchSemaphore(value: 0)
        var result: Qwen3StreamingResult?
        var feedError: Error?

        Task {
            do {
                result = try await manager.addAudio(samples)
            } catch {
                feedError = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let error = feedError {
            throw error
        }

        return result?.transcript
    }

    @available(macOS 15, iOS 18, *)
    func qwen3StreamingFinish() throws -> String {
        guard let manager = qwen3StreamingManager else {
            throw BridgeError.notInitialized
        }

        let semaphore = DispatchSemaphore(value: 0)
        var result: Qwen3StreamingResult?
        var finishError: Error?

        Task {
            do {
                result = try await manager.finish()
            } catch {
                finishError = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let error = finishError {
            throw error
        }

        return result?.transcript ?? ""
    }

    @available(macOS 15, iOS 18, *)
    func isQwen3StreamingAvailable() -> Bool {
        return qwen3StreamingManager != nil
    }

    func cleanup() {
        asrManager = nil
        asrModels = nil
        vadManager = nil
        diarizerManager = nil
        streamingAsrManager = nil
        qwen3AsrManager = nil
        qwen3StreamingManager = nil
    }
}

enum BridgeError: Error {
    case notInitialized
    case noResult
}

// MARK: - C FFI Functions

/// Storage for bridge instances (simple approach - use a single global for now)
private var globalBridge: FluidAudioBridgeInternal?

@_cdecl("fluidaudio_bridge_create")
public func fluidaudio_bridge_create() -> UnsafeMutableRawPointer? {
    let bridge = FluidAudioBridgeInternal()
    globalBridge = bridge
    return Unmanaged.passRetained(bridge).toOpaque()
}

@_cdecl("fluidaudio_bridge_destroy")
public func fluidaudio_bridge_destroy(_ ptr: UnsafeMutableRawPointer?) {
    guard let ptr = ptr else { return }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeRetainedValue()
    bridge.cleanup()
    if globalBridge === bridge {
        globalBridge = nil
    }
}

@_cdecl("fluidaudio_initialize_asr")
public func fluidaudio_initialize_asr(_ ptr: UnsafeMutableRawPointer?) -> Int32 {
    guard let ptr = ptr else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    do {
        try bridge.initializeAsr()
        return 0
    } catch {
        print("ASR init error: \(error)")
        return -1
    }
}

@_cdecl("fluidaudio_transcribe_file")
public func fluidaudio_transcribe_file(
    _ ptr: UnsafeMutableRawPointer?,
    _ path: UnsafePointer<CChar>?,
    _ outText: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?,
    _ outConfidence: UnsafeMutablePointer<Float>?,
    _ outDuration: UnsafeMutablePointer<Double>?,
    _ outProcessingTime: UnsafeMutablePointer<Double>?,
    _ outRtfx: UnsafeMutablePointer<Float>?
) -> Int32 {
    guard let ptr = ptr, let path = path else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()

    let pathString = String(cString: path)

    do {
        let (text, confidence, duration, processingTime, rtfx) = try bridge.transcribeFile(pathString)

        // Allocate and copy text
        if let outText = outText {
            let cString = strdup(text)
            outText.pointee = cString
        }

        outConfidence?.pointee = confidence
        outDuration?.pointee = duration
        outProcessingTime?.pointee = processingTime
        outRtfx?.pointee = rtfx

        return 0
    } catch {
        print("Transcribe error: \(error)")
        return -1
    }
}

@_cdecl("fluidaudio_transcribe_samples")
public func fluidaudio_transcribe_samples(
    _ ptr: UnsafeMutableRawPointer?,
    _ samples: UnsafePointer<Float>?,
    _ sampleCount: UInt32,
    _ outText: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?,
    _ outConfidence: UnsafeMutablePointer<Float>?,
    _ outDuration: UnsafeMutablePointer<Double>?,
    _ outProcessingTime: UnsafeMutablePointer<Double>?,
    _ outRtfx: UnsafeMutablePointer<Float>?
) -> Int32 {
    guard let ptr = ptr, let samples = samples else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()

    let samplesArray = Array(UnsafeBufferPointer(start: samples, count: Int(sampleCount)))

    do {
        let (text, confidence, duration, processingTime, rtfx) = try bridge.transcribeSamples(samplesArray)

        // Allocate and copy text
        if let outText = outText {
            let cString = strdup(text)
            outText.pointee = cString
        }

        outConfidence?.pointee = confidence
        outDuration?.pointee = duration
        outProcessingTime?.pointee = processingTime
        outRtfx?.pointee = rtfx

        return 0
    } catch {
        print("Transcribe samples error: \(error)")
        return -1
    }
}

@_cdecl("fluidaudio_is_asr_available")
public func fluidaudio_is_asr_available(_ ptr: UnsafeMutableRawPointer?) -> Int32 {
    guard let ptr = ptr else { return 0 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    return bridge.isAsrAvailable() ? 1 : 0
}

// MARK: - Streaming ASR FFI

@_cdecl("fluidaudio_initialize_streaming_asr")
public func fluidaudio_initialize_streaming_asr(_ ptr: UnsafeMutableRawPointer?) -> Int32 {
    guard let ptr = ptr else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    do {
        try bridge.initializeStreamingAsr()
        return 0
    } catch {
        print("Streaming ASR init error: \(error)")
        return -1
    }
}

@_cdecl("fluidaudio_streaming_asr_start")
public func fluidaudio_streaming_asr_start(_ ptr: UnsafeMutableRawPointer?) -> Int32 {
    guard let ptr = ptr else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    do {
        try bridge.streamingAsrStart()
        return 0
    } catch {
        print("Streaming ASR start error: \(error)")
        return -1
    }
}

@_cdecl("fluidaudio_streaming_asr_feed")
public func fluidaudio_streaming_asr_feed(
    _ ptr: UnsafeMutableRawPointer?,
    _ samples: UnsafePointer<Float>?,
    _ count: UInt32
) -> Int32 {
    guard let ptr = ptr, let samples = samples else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()

    let samplesArray = Array(UnsafeBufferPointer(start: samples, count: Int(count)))

    do {
        try bridge.streamingAsrFeed(samplesArray)
        return 0
    } catch {
        print("Streaming ASR feed error: \(error)")
        return -1
    }
}

@_cdecl("fluidaudio_streaming_asr_finish")
public func fluidaudio_streaming_asr_finish(
    _ ptr: UnsafeMutableRawPointer?,
    _ outText: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
    guard let ptr = ptr else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()

    do {
        let text = try bridge.streamingAsrFinish()

        if let outText = outText {
            let cString = strdup(text)
            outText.pointee = cString
        }

        return 0
    } catch {
        print("Streaming ASR finish error: \(error)")
        return -1
    }
}

@_cdecl("fluidaudio_transcribe_file_streaming")
public func fluidaudio_transcribe_file_streaming(
    _ ptr: UnsafeMutableRawPointer?,
    _ path: UnsafePointer<CChar>?,
    _ outText: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?,
    _ outConfidence: UnsafeMutablePointer<Float>?,
    _ outDuration: UnsafeMutablePointer<Double>?,
    _ outProcessingTime: UnsafeMutablePointer<Double>?,
    _ outRtfx: UnsafeMutablePointer<Float>?
) -> Int32 {
    guard let ptr = ptr, let path = path else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()

    let pathString = String(cString: path)

    do {
        let (text, confidence, duration, processingTime, rtfx) = try bridge.transcribeFileStreaming(pathString)

        if let outText = outText {
            let cString = strdup(text)
            outText.pointee = cString
        }

        outConfidence?.pointee = confidence
        outDuration?.pointee = duration
        outProcessingTime?.pointee = processingTime
        outRtfx?.pointee = rtfx

        return 0
    } catch {
        print("Streaming transcribe file error: \(error)")
        return -1
    }
}

@_cdecl("fluidaudio_is_streaming_asr_available")
public func fluidaudio_is_streaming_asr_available(_ ptr: UnsafeMutableRawPointer?) -> Int32 {
    guard let ptr = ptr else { return 0 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    return bridge.isStreamingAsrAvailable() ? 1 : 0
}

// MARK: - VAD FFI

@_cdecl("fluidaudio_initialize_vad")
public func fluidaudio_initialize_vad(_ ptr: UnsafeMutableRawPointer?, _ threshold: Float) -> Int32 {
    guard let ptr = ptr else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    do {
        try bridge.initializeVad(threshold)
        return 0
    } catch {
        print("VAD init error: \(error)")
        return -1
    }
}

@_cdecl("fluidaudio_is_vad_available")
public func fluidaudio_is_vad_available(_ ptr: UnsafeMutableRawPointer?) -> Int32 {
    guard let ptr = ptr else { return 0 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    return bridge.isVadAvailable() ? 1 : 0
}

// MARK: - Diarization FFI

@_cdecl("fluidaudio_initialize_diarization")
public func fluidaudio_initialize_diarization(_ ptr: UnsafeMutableRawPointer?, _ threshold: Double) -> Int32 {
    guard let ptr = ptr else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    do {
        try bridge.initializeDiarization(threshold)
        return 0
    } catch {
        print("Diarization init error: \(error)")
        return -1
    }
}

/// Diarize a file. Returns segment count via outCount.
/// Each segment is 4 consecutive values: speakerId (char*), startTime (float), endTime (float), qualityScore (float).
/// The flat arrays outSpeakerIds, outStartTimes, outEndTimes, outQualityScores must be freed by the caller.
@_cdecl("fluidaudio_diarize_file")
public func fluidaudio_diarize_file(
    _ ptr: UnsafeMutableRawPointer?,
    _ path: UnsafePointer<CChar>?,
    _ outSpeakerIds: UnsafeMutablePointer<UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?>?,
    _ outStartTimes: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>?,
    _ outEndTimes: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>?,
    _ outQualityScores: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>?,
    _ outCount: UnsafeMutablePointer<UInt32>?
) -> Int32 {
    guard let ptr = ptr, let path = path else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()

    let pathString = String(cString: path)

    do {
        let segments = try bridge.diarizeFile(pathString)
        let count = segments.count

        outCount?.pointee = UInt32(count)

        if count == 0 {
            outSpeakerIds?.pointee = nil
            outStartTimes?.pointee = nil
            outEndTimes?.pointee = nil
            outQualityScores?.pointee = nil
        } else {
            let ids = UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>.allocate(capacity: count)
            let starts = UnsafeMutablePointer<Float>.allocate(capacity: count)
            let ends = UnsafeMutablePointer<Float>.allocate(capacity: count)
            let scores = UnsafeMutablePointer<Float>.allocate(capacity: count)

            for (i, seg) in segments.enumerated() {
                ids[i] = strdup(seg.speakerId)
                starts[i] = seg.startTime
                ends[i] = seg.endTime
                scores[i] = seg.qualityScore
            }

            outSpeakerIds?.pointee = ids
            outStartTimes?.pointee = starts
            outEndTimes?.pointee = ends
            outQualityScores?.pointee = scores
        }

        return 0
    } catch {
        print("Diarize error: \(error)")
        return -1
    }
}

@_cdecl("fluidaudio_is_diarization_available")
public func fluidaudio_is_diarization_available(_ ptr: UnsafeMutableRawPointer?) -> Int32 {
    guard let ptr = ptr else { return 0 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    return bridge.isDiarizationAvailable() ? 1 : 0
}

@_cdecl("fluidaudio_free_diarization_result")
public func fluidaudio_free_diarization_result(
    _ speakerIds: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?,
    _ startTimes: UnsafeMutablePointer<Float>?,
    _ endTimes: UnsafeMutablePointer<Float>?,
    _ qualityScores: UnsafeMutablePointer<Float>?,
    _ count: UInt32
) {
    if let ids = speakerIds {
        for i in 0..<Int(count) {
            free(ids[i])
        }
        ids.deallocate()
    }
    startTimes?.deallocate()
    endTimes?.deallocate()
    qualityScores?.deallocate()
}

// MARK: - System Info FFI

@_cdecl("fluidaudio_get_platform")
public func fluidaudio_get_platform(_ out: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?) {
    #if os(macOS)
    let platform = "macOS"
    #elseif os(iOS)
    let platform = "iOS"
    #else
    let platform = "unknown"
    #endif

    out?.pointee = strdup(platform)
}

@_cdecl("fluidaudio_get_chip_name")
public func fluidaudio_get_chip_name(_ out: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?) {
    var size: size_t = 0
    var chipName = "Unknown"

    if sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0) == 0, size > 0 {
        var buffer = [CChar](repeating: 0, count: Int(size))
        if sysctlbyname("machdep.cpu.brand_string", &buffer, &size, nil, 0) == 0 {
            chipName = String(cString: buffer)
        }
    }

    out?.pointee = strdup(chipName)
}

@_cdecl("fluidaudio_get_memory_gb")
public func fluidaudio_get_memory_gb() -> Double {
    return Double(ProcessInfo.processInfo.physicalMemory) / (1024 * 1024 * 1024)
}

@_cdecl("fluidaudio_is_apple_silicon")
public func fluidaudio_is_apple_silicon() -> Int32 {
    return SystemInfo.isAppleSilicon ? 1 : 0
}

@_cdecl("fluidaudio_cleanup")
public func fluidaudio_cleanup(_ ptr: UnsafeMutableRawPointer?) {
    guard let ptr = ptr else { return }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    bridge.cleanup()
}

@_cdecl("fluidaudio_free_string")
public func fluidaudio_free_string(_ s: UnsafeMutablePointer<CChar>?) {
    free(s)
}

// MARK: - Qwen3 ASR FFI

@_cdecl("fluidaudio_initialize_qwen3_asr")
public func fluidaudio_initialize_qwen3_asr(_ ptr: UnsafeMutableRawPointer?) -> Int32 {
    guard let ptr = ptr else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()

    if #available(macOS 15, iOS 18, *) {
        do {
            try bridge.initializeQwen3Asr()
            return 0
        } catch {
            print("Qwen3 ASR init error: \(error)")
            return -1
        }
    } else {
        print("Qwen3 ASR requires macOS 15+ or iOS 18+")
        return -1
    }
}

@_cdecl("fluidaudio_qwen3_transcribe_samples")
public func fluidaudio_qwen3_transcribe_samples(
    _ ptr: UnsafeMutableRawPointer?,
    _ samples: UnsafePointer<Float>?,
    _ sampleCount: UInt32,
    _ language: UnsafePointer<CChar>?,
    _ outText: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?,
    _ outConfidence: UnsafeMutablePointer<Float>?,
    _ outDuration: UnsafeMutablePointer<Double>?,
    _ outProcessingTime: UnsafeMutablePointer<Double>?,
    _ outRtfx: UnsafeMutablePointer<Float>?
) -> Int32 {
    guard let ptr = ptr, let samples = samples else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()

    if #available(macOS 15, iOS 18, *) {
        let samplesArray = Array(UnsafeBufferPointer(start: samples, count: Int(sampleCount)))
        let languageString = language.map { String(cString: $0) }

        do {
            let (text, confidence, duration, processingTime, rtfx) = try bridge.qwen3TranscribeSamples(samplesArray, language: languageString)

            if let outText = outText {
                let cString = strdup(text)
                outText.pointee = cString
            }

            outConfidence?.pointee = confidence
            outDuration?.pointee = duration
            outProcessingTime?.pointee = processingTime
            outRtfx?.pointee = rtfx

            return 0
        } catch {
            print("Qwen3 transcribe samples error: \(error)")
            return -1
        }
    } else {
        print("Qwen3 ASR requires macOS 15+ or iOS 18+")
        return -1
    }
}

@_cdecl("fluidaudio_qwen3_transcribe_file")
public func fluidaudio_qwen3_transcribe_file(
    _ ptr: UnsafeMutableRawPointer?,
    _ path: UnsafePointer<CChar>?,
    _ language: UnsafePointer<CChar>?,
    _ outText: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?,
    _ outConfidence: UnsafeMutablePointer<Float>?,
    _ outDuration: UnsafeMutablePointer<Double>?,
    _ outProcessingTime: UnsafeMutablePointer<Double>?,
    _ outRtfx: UnsafeMutablePointer<Float>?
) -> Int32 {
    guard let ptr = ptr, let path = path else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()

    if #available(macOS 15, iOS 18, *) {
        let pathString = String(cString: path)
        let languageString = language.map { String(cString: $0) }

        do {
            let (text, confidence, duration, processingTime, rtfx) = try bridge.qwen3TranscribeFile(pathString, language: languageString)

            if let outText = outText {
                let cString = strdup(text)
                outText.pointee = cString
            }

            outConfidence?.pointee = confidence
            outDuration?.pointee = duration
            outProcessingTime?.pointee = processingTime
            outRtfx?.pointee = rtfx

            return 0
        } catch {
            print("Qwen3 transcribe file error: \(error)")
            return -1
        }
    } else {
        print("Qwen3 ASR requires macOS 15+ or iOS 18+")
        return -1
    }
}

@_cdecl("fluidaudio_is_qwen3_asr_available")
public func fluidaudio_is_qwen3_asr_available(_ ptr: UnsafeMutableRawPointer?) -> Int32 {
    guard let ptr = ptr else { return 0 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()

    if #available(macOS 15, iOS 18, *) {
        return bridge.isQwen3AsrAvailable() ? 1 : 0
    } else {
        return 0
    }
}

// MARK: - Qwen3 Streaming FFI

@_cdecl("fluidaudio_initialize_qwen3_streaming")
public func fluidaudio_initialize_qwen3_streaming(_ ptr: UnsafeMutableRawPointer?) -> Int32 {
    guard let ptr = ptr else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()

    if #available(macOS 15, iOS 18, *) {
        do {
            try bridge.initializeQwen3Streaming()
            return 0
        } catch {
            print("Qwen3 Streaming init error: \(error)")
            return -1
        }
    } else {
        print("Qwen3 Streaming requires macOS 15+ or iOS 18+")
        return -1
    }
}

@_cdecl("fluidaudio_qwen3_streaming_start")
public func fluidaudio_qwen3_streaming_start(
    _ ptr: UnsafeMutableRawPointer?,
    _ language: UnsafePointer<CChar>?,
    _ minAudioSeconds: Double,
    _ chunkSeconds: Double,
    _ maxAudioSeconds: Double
) -> Int32 {
    guard let ptr = ptr else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()

    if #available(macOS 15, iOS 18, *) {
        let languageString = language.map { String(cString: $0) }

        do {
            try bridge.qwen3StreamingStart(
                language: languageString,
                minAudioSeconds: minAudioSeconds,
                chunkSeconds: chunkSeconds,
                maxAudioSeconds: maxAudioSeconds
            )
            return 0
        } catch {
            print("Qwen3 Streaming start error: \(error)")
            return -1
        }
    } else {
        print("Qwen3 Streaming requires macOS 15+ or iOS 18+")
        return -1
    }
}

@_cdecl("fluidaudio_qwen3_streaming_feed")
public func fluidaudio_qwen3_streaming_feed(
    _ ptr: UnsafeMutableRawPointer?,
    _ samples: UnsafePointer<Float>?,
    _ count: UInt32,
    _ outPartialText: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
    guard let ptr = ptr, let samples = samples else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()

    if #available(macOS 15, iOS 18, *) {
        let samplesArray = Array(UnsafeBufferPointer(start: samples, count: Int(count)))

        do {
            let partialText = try bridge.qwen3StreamingFeed(samplesArray)

            if let outPartialText = outPartialText {
                if let text = partialText {
                    outPartialText.pointee = strdup(text)
                } else {
                    outPartialText.pointee = nil
                }
            }

            return 0
        } catch {
            print("Qwen3 Streaming feed error: \(error)")
            return -1
        }
    } else {
        print("Qwen3 Streaming requires macOS 15+ or iOS 18+")
        return -1
    }
}

@_cdecl("fluidaudio_qwen3_streaming_finish")
public func fluidaudio_qwen3_streaming_finish(
    _ ptr: UnsafeMutableRawPointer?,
    _ outText: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
    guard let ptr = ptr else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()

    if #available(macOS 15, iOS 18, *) {
        do {
            let text = try bridge.qwen3StreamingFinish()

            if let outText = outText {
                let cString = strdup(text)
                outText.pointee = cString
            }

            return 0
        } catch {
            print("Qwen3 Streaming finish error: \(error)")
            return -1
        }
    } else {
        print("Qwen3 Streaming requires macOS 15+ or iOS 18+")
        return -1
    }
}

@_cdecl("fluidaudio_is_qwen3_streaming_available")
public func fluidaudio_is_qwen3_streaming_available(_ ptr: UnsafeMutableRawPointer?) -> Int32 {
    guard let ptr = ptr else { return 0 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()

    if #available(macOS 15, iOS 18, *) {
        return bridge.isQwen3StreamingAvailable() ? 1 : 0
    } else {
        return 0
    }
}
