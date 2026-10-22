import Foundation
import AVFoundation
import CoreML
import CryptoKit
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
    private var asrDecoderState: TdtDecoderState?
    private var vadManager: VadManager?
    private var diarizerManager: OfflineDiarizerManager?
    private var streamingAsrManager: SlidingWindowAsrManager?
    private var kokoroManager: KokoroAneManager?
    // Model-path diarization: retain the loaded MLModel (the ~4s-to-load part) keyed by
    // model path so repeated in-process diarize calls skip the reload. Guarded by a lock
    // — the Rust bridge is Send+Sync and may be entered concurrently from multiple threads.
    private let sortformerCacheLock = NSLock()
    private var sortformerModelCache: [String: MLModel] = [:]
    // Qwen3 types require macOS 15 / iOS 18, so store as Any? and cast at call sites
    // guarded by `if #available(macOS 15, iOS 18, *)`.
    private var qwen3AsrManagerStorage: Any?
    private var qwen3StreamingManagerStorage: Any?

    init() {}

    func initializeAsr() throws {
        let semaphore = DispatchSemaphore(value: 0)
        var initError: Error?

        Task {
            do {
                let models = try await AsrModels.downloadAndLoad()
                self.asrModels = models

                let manager = AsrManager()
                try await manager.loadModels(models)
                // Construct decoder state first so a throw here does not leave
                // `asrManager` set while `asrDecoderState` is nil — that combo
                // makes `isAsrAvailable()` return true but every transcribe call
                // fail with `notInitialized`.
                self.asrDecoderState = try TdtDecoderState()
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
        // One-shot transcription: each call starts with a fresh decoder state so
        // results don't leak across utterances. The TDT decoder's LSTM hidden/cell
        // state and `lastToken` would otherwise persist, biasing the next call's
        // predictor (e.g. priming it with end-of-sentence punctuation, which
        // collapses subsequent transcripts to ".").
        // Streaming-style chunked decoding is exposed separately via the
        // streaming ASR API (`streaming_asr_*`), which manages state explicitly.
        guard let manager = asrManager else {
            throw BridgeError.notInitialized
        }
        var decoderState = try TdtDecoderState()

        let semaphore = DispatchSemaphore(value: 0)
        var result: ASRResult?
        var transcribeError: Error?

        Task {
            do {
                let url = URL(fileURLWithPath: path)
                result = try await manager.transcribe(url, decoderState: &decoderState)
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
        // See `transcribeFile` for the rationale: fresh decoder state per call.
        guard let manager = asrManager else {
            throw BridgeError.notInitialized
        }
        var decoderState = try TdtDecoderState()

        let semaphore = DispatchSemaphore(value: 0)
        var result: ASRResult?
        var transcribeError: Error?

        Task {
            do {
                result = try await manager.transcribe(samples, decoderState: &decoderState)
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

    /// Map a kesha/espeak-style language tag to a KokoroAne variant. FluidAudio
    /// 0.14.8 ships exactly two KokoroAne variants — `.english` and `.mandarin`
    /// — so `zh` selects Mandarin (its own tone-aware G2P) and everything else
    /// (en plus the Latin-script es/fr/it/pt, which synthesize acceptably through
    /// the English G2P) falls back to `.english`.
    private static func kokoroVariant(for lang: String) -> KokoroAneVariant {
        let base = lang.lowercased().split(separator: "-").first.map(String.init) ?? ""
        switch base {
        case "zh": return .mandarin
        default: return .english
        }
    }

    /// `computeUnits` maps onto `KokoroAneComputeUnits`. `.default` keeps
    /// FluidAudio's empirical per-stage mapping, which pins Albert / PostAlbert /
    /// Alignment / Vocoder to the Neural Engine. The other presets are a control
    /// for debugging and for comparing placements, not a fix for any host. Note
    /// a CoreML prediction failure does not land here: `KokoroAneManager.initialize`
    /// only downloads and loads the mlmodelcs, so this call succeeds and
    /// `synthesizeKokoro` throws `predictionFailed(stage:, ...)` on the first
    /// prediction.
    func initializeKokoro(
        defaultVoice: String, lang: String, computeUnits: TtsComputeUnitPreset = .default
    ) throws {
        let semaphore = DispatchSemaphore(value: 0)
        var initError: Error?

        Task {
            do {
                // `KokoroAneManager` feeds `speed` as a real model input tensor,
                // so `--rate` applies correctly (unlike the prior `voiceSpeed:` path).
                let variant = Self.kokoroVariant(for: lang)
                let manager = KokoroAneManager(
                    variant: variant,
                    defaultVoice: defaultVoice,
                    computeUnits: KokoroAneComputeUnits(preset: computeUnits))
                try await manager.initialize(preloadVoices: [defaultVoice])
                self.kokoroManager = manager
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

    /// Synthesize `text` and return a complete WAV byte buffer (24 kHz mono),
    /// exactly what `KokoroAneManager.synthesize` produces. `speed` (1.0 =
    /// normal) is fed to the model as a real input tensor, so it genuinely
    /// applies — unlike the removed `KokoroTtsManager.synthesize(voiceSpeed:)`.
    func synthesizeKokoro(text: String, voice: String, speed: Float) throws -> Data {
        guard let manager = kokoroManager else {
            throw BridgeError.notInitialized
        }

        let semaphore = DispatchSemaphore(value: 0)
        var result: Data?
        var synthError: Error?

        Task {
            do {
                result = try await manager.synthesize(text: text, voice: voice, speed: speed)
            } catch {
                synthError = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let error = synthError {
            throw error
        }

        guard let data = result else {
            throw BridgeError.noResult
        }

        return data
    }

    func isKokoroAvailable() -> Bool {
        return kokoroManager != nil
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

    /// Resolve the pre-staged `.mlpackage` at `modelPath` to a compiled `.mlmodelc` in a
    /// WRITABLE per-user cache — never next to the model, which may live in a read-only /
    /// air-gapped location. `MLModel.compileModel` recompiles to a throwaway temp on every
    /// call; the ~100s cost is the CoreML ANE program compile inside `MLModel(contentsOf:,
    /// .all)`, which Apple caches in `com.apple.e5rt.e5bundlecache` keyed to the compiled
    /// model's path — so a *stable* path makes the 2nd process onward ~4s (measured: cold
    /// 104.7s, warm 3.9s on M3 Pro). The compiled name fingerprints the package (path +
    /// size + mtime), so swapping a different model in at the same path recompiles.
    private static func compiledModelURL(forModelPath modelPath: String) async throws -> URL {
        let cacheDir = try compiledCacheDir()
        let key = cacheKey(forModelPath: modelPath)
        let stableURL = cacheDir.appendingPathComponent(key + ".mlmodelc")
        let fm = FileManager.default

        if !fm.fileExists(atPath: stableURL.path) {
            let compiled = try await MLModel.compileModel(at: URL(fileURLWithPath: modelPath))
            defer { try? fm.removeItem(at: compiled) }
            // Publish atomically: stage into a unique sibling of the final path (same
            // volume, so the rename below is atomic), then rename into place — a reader
            // never sees a half-written bundle. copyItem, not moveItem, from `compiled`:
            // it lives in the system temp dir, often a different volume (EXDEV).
            let staging = cacheDir.appendingPathComponent(key + ".\(UUID().uuidString).tmp")
            do {
                try fm.copyItem(at: compiled, to: staging)
                try fm.moveItem(at: staging, to: stableURL)
            } catch {
                try? fm.removeItem(at: staging)
                // Another process may have published it first; tolerate that.
                if !fm.fileExists(atPath: stableURL.path) { throw error }
            }
        }
        return stableURL
    }

    /// Writable cache dir for compiled Sortformer models: Application Support (not
    /// `.cachesDirectory`, which the OS may reclaim — recompiling costs ~100s).
    private static func compiledCacheDir() throws -> URL {
        let base = try FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true)
        let dir = base.appendingPathComponent("fluidaudio-rs/SortformerCompiled", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// SHA-256 of `<path>|<recursive size>|<newest mtime>`. Size+mtime invalidate the
    /// compiled artifact if a different model is staged at the same path; `.mlpackage` is a
    /// directory bundle, so the fingerprint walks its contents rather than stat-ing the dir.
    private static func cacheKey(forModelPath modelPath: String) -> String {
        let url = URL(fileURLWithPath: modelPath).standardizedFileURL
        var totalSize: Int64 = 0
        var newestMtime: TimeInterval = 0
        let keys: Set<URLResourceKey> = [.fileSizeKey, .contentModificationDateKey]
        if let en = FileManager.default.enumerator(at: url, includingPropertiesForKeys: Array(keys)) {
            for case let file as URL in en {
                guard let v = try? file.resourceValues(forKeys: keys) else { continue }
                totalSize += Int64(v.fileSize ?? 0)
                if let m = v.contentModificationDate?.timeIntervalSince1970, m > newestMtime {
                    newestMtime = m
                }
            }
        }
        let material = "\(url.path)|\(totalSize)|\(Int(newestMtime))"
        let digest = SHA256.hash(data: Data(material.utf8))
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    /// Return a loaded `MLModel` for `modelPath`, retained on the bridge so repeated
    /// in-process diarizations skip the ~4s `MLModel(contentsOf:)` load. Double-checked
    /// under `sortformerCacheLock`; the lock is never held across the `await` compile/load.
    /// We cache the `MLModel` (thread-safe, expensive to load) and build a fresh
    /// `SortformerModels` per call — `SortformerModels` owns mutable scratch buffers that
    /// must not be shared across concurrent diarizations.
    private func cachedSortformerModel(modelPath: String) async throws -> MLModel {
        // Key by raw path: a hit returns with zero filesystem I/O. The content
        // fingerprint (which requires walking the bundle) is computed only on the miss
        // path, inside `compiledModelURL`, to name the on-disk compiled artifact — that
        // disk cache is where content invalidation matters (across processes/versions).
        sortformerCacheLock.lock()
        if let cached = sortformerModelCache[modelPath] {
            sortformerCacheLock.unlock()
            return cached
        }
        sortformerCacheLock.unlock()

        let url = try await Self.compiledModelURL(forModelPath: modelPath)
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all
        let model = try MLModel(contentsOf: url, configuration: cfg)

        sortformerCacheLock.lock()
        defer { sortformerCacheLock.unlock() }
        if let existing = sortformerModelCache[modelPath] { return existing }
        sortformerModelCache[modelPath] = model
        return model
    }

    /// Diarize using a pre-staged Sortformer `.mlpackage` at `modelPath`. The compiled
    /// model is cached in a writable per-user dir and retained in-memory
    /// (`cachedSortformerModel`), so repeated in-process calls skip the reload; it never
    /// downloads from HuggingFace — unlike `diarizeFile`, which uses the auto-downloading
    /// OfflineDiarizerManager. Config is `.balancedV2` to match the shipped
    /// `SortformerNvidiaLow_v2.mlpackage` (fifoLen=188); a mismatched config is a hard
    /// CoreML tensor-shape error at runtime.
    func diarizeFileWithModels(audioPath: String, modelPath: String) throws -> [BridgeDiarizationSegment] {
        let semaphore = DispatchSemaphore(value: 0)
        var timeline: DiarizerTimeline?
        var diarizeError: Error?

        Task {
            do {
                let diarizer = SortformerDiarizer(
                    config: SortformerConfig.balancedV2,
                    timelineConfig: DiarizerTimelineConfig.sortformerDefault
                )
                let model = try await self.cachedSortformerModel(modelPath: modelPath)
                let models = try SortformerModels(config: SortformerConfig.balancedV2, main: model)
                diarizer.initialize(models: models)
                timeline = try diarizer.processComplete(
                    audioFileURL: URL(fileURLWithPath: audioPath),
                    keepingEnrolledSpeakers: nil,
                    finalizeOnCompletion: true,
                    progressCallback: nil
                )
            } catch {
                diarizeError = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let error = diarizeError {
            throw error
        }

        guard let tl = timeline else {
            throw BridgeError.noResult
        }

        // DiarizerTimeline.speakers: [Int: DiarizerSpeaker]; flatten finalized
        // segments across speakers and sort by start time.
        let allSegments = tl.speakers.values.flatMap { $0.finalizedSegments }
        return allSegments
            .sorted { $0.startTime < $1.startTime }
            .map { seg in
                BridgeDiarizationSegment(
                    speakerId: String(format: "SPEAKER_%02d", max(0, seg.speakerIndex)),
                    startTime: seg.startTime,
                    endTime: seg.endTime,
                    qualityScore: 1.0
                )
            }
    }

    /// Warm the diarization model: compile the `.mlpackage` into the writable per-user
    /// cache (one-time ~100s ANE compile, populates the e5rt cache) and load it once,
    /// retaining the handle in-memory. Callers warm this at install time so the first real
    /// diarize is fast — ~4s cross-process, or no reload at all in-process. No audio is
    /// processed; the caller's model directory is never written to.
    func compileDiarizationModel(modelPath: String) throws {
        let semaphore = DispatchSemaphore(value: 0)
        var compileError: Error?

        Task {
            do {
                _ = try await self.cachedSortformerModel(modelPath: modelPath)
            } catch {
                compileError = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let error = compileError {
            throw error
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

                let manager = SlidingWindowAsrManager()
                try await manager.loadModels(models)
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
        guard let manager = streamingAsrManager else {
            throw BridgeError.notInitialized
        }

        let semaphore = DispatchSemaphore(value: 0)
        var startError: Error?

        Task {
            do {
                try await manager.startStreaming(source: .microphone)
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
        guard let manager = streamingAsrManager else {
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
                try await manager.startStreaming(source: .microphone)

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
                let modelDir = try await Qwen3AsrModels.download()
                try await manager.loadModels(from: modelDir)
                self.qwen3AsrManagerStorage = manager
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
        guard let manager = qwen3AsrManagerStorage as? Qwen3AsrManager else {
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
        guard qwen3AsrManagerStorage is Qwen3AsrManager else {
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
        return qwen3AsrManagerStorage is Qwen3AsrManager
    }

    // MARK: - Qwen3 Streaming

    @available(macOS 15, iOS 18, *)
    func initializeQwen3Streaming() throws {
        let semaphore = DispatchSemaphore(value: 0)
        var initError: Error?

        Task {
            do {
                let asrManager = Qwen3AsrManager()
                let modelDir = try await Qwen3AsrModels.download()
                try await asrManager.loadModels(from: modelDir)

                let streamingManager = Qwen3StreamingManager(asrManager: asrManager)
                self.qwen3AsrManagerStorage = asrManager
                self.qwen3StreamingManagerStorage = streamingManager
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
        guard let manager = qwen3StreamingManagerStorage as? Qwen3StreamingManager else {
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
        guard let manager = qwen3StreamingManagerStorage as? Qwen3StreamingManager else {
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
        guard let manager = qwen3StreamingManagerStorage as? Qwen3StreamingManager else {
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
        return qwen3StreamingManagerStorage is Qwen3StreamingManager
    }

    // MARK: - VAD processing

    /// Returned per-chunk VAD frame data, suitable for flat C arrays.
    struct BridgeVadFrame {
        var probability: Float
        var isVoiceActive: Bool
        var processingTime: Double
    }

    func vadProcessFile(_ path: String) throws -> [BridgeVadFrame] {
        guard let manager = vadManager else {
            throw BridgeError.notInitialized
        }

        let semaphore = DispatchSemaphore(value: 0)
        var frames: [BridgeVadFrame] = []
        var processError: Error?

        Task {
            do {
                let url = URL(fileURLWithPath: path)
                let results = try await manager.process(url)
                frames = results.map {
                    BridgeVadFrame(
                        probability: $0.probability,
                        isVoiceActive: $0.isVoiceActive,
                        processingTime: $0.processingTime
                    )
                }
            } catch {
                processError = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let error = processError {
            throw error
        }

        return frames
    }

    func vadProcessSamples(_ samples: [Float]) throws -> [BridgeVadFrame] {
        guard let manager = vadManager else {
            throw BridgeError.notInitialized
        }

        let semaphore = DispatchSemaphore(value: 0)
        var frames: [BridgeVadFrame] = []
        var processError: Error?

        Task {
            do {
                let results = try await manager.process(samples)
                frames = results.map {
                    BridgeVadFrame(
                        probability: $0.probability,
                        isVoiceActive: $0.isVoiceActive,
                        processingTime: $0.processingTime
                    )
                }
            } catch {
                processError = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let error = processError {
            throw error
        }

        return frames
    }

    // MARK: - ITN (Inverse Text Normalization)

    func itnNormalize(_ input: String) -> String {
        return TextNormalizer.shared.normalize(input)
    }

    func itnNormalizeSentence(_ input: String) -> String {
        return TextNormalizer.shared.normalizeSentence(input)
    }

    func itnNormalizeSentenceMaxSpan(_ input: String, maxSpanTokens: UInt32) -> String {
        return TextNormalizer.shared.normalizeSentence(input, maxSpanTokens: maxSpanTokens)
    }

    func itnIsNativeAvailable() -> Bool {
        return TextNormalizer.shared.isNativeAvailable
    }

    func cleanup() {
        asrManager = nil
        asrModels = nil
        asrDecoderState = nil
        vadManager = nil
        diarizerManager = nil
        sortformerCacheLock.lock()
        sortformerModelCache.removeAll()
        sortformerCacheLock.unlock()
        streamingAsrManager = nil
        kokoroManager = nil
        qwen3AsrManagerStorage = nil
        qwen3StreamingManagerStorage = nil
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
        emitDiarizationSegments(
            segments,
            outSpeakerIds: outSpeakerIds,
            outStartTimes: outStartTimes,
            outEndTimes: outEndTimes,
            outQualityScores: outQualityScores,
            outCount: outCount
        )
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

// MARK: - System Info (extended)

@_cdecl("fluidaudio_is_intel_mac")
public func fluidaudio_is_intel_mac() -> Int32 {
    return SystemInfo.isIntelMac ? 1 : 0
}

// MARK: - VAD processing FFI

/// Process an audio file through VAD. Returns the number of frames via outCount.
/// Each frame contributes one entry to outProbabilities (Float), outIsVoiceActive (UInt8 0/1),
/// and outProcessingTimes (Double). Caller must free with fluidaudio_free_vad_result.
@_cdecl("fluidaudio_vad_process_file")
public func fluidaudio_vad_process_file(
    _ ptr: UnsafeMutableRawPointer?,
    _ path: UnsafePointer<CChar>?,
    _ outProbabilities: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>?,
    _ outIsVoiceActive: UnsafeMutablePointer<UnsafeMutablePointer<UInt8>?>?,
    _ outProcessingTimes: UnsafeMutablePointer<UnsafeMutablePointer<Double>?>?,
    _ outCount: UnsafeMutablePointer<UInt32>?
) -> Int32 {
    guard let ptr = ptr, let path = path else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()

    let pathString = String(cString: path)

    do {
        let frames = try bridge.vadProcessFile(pathString)
        emitVadFrames(
            frames,
            outProbabilities: outProbabilities,
            outIsVoiceActive: outIsVoiceActive,
            outProcessingTimes: outProcessingTimes,
            outCount: outCount
        )
        return 0
    } catch {
        print("VAD process file error: \(error)")
        return -1
    }
}

/// Process raw 16kHz mono Float32 samples through VAD. See fluidaudio_vad_process_file
/// for output array semantics.
@_cdecl("fluidaudio_vad_process_samples")
public func fluidaudio_vad_process_samples(
    _ ptr: UnsafeMutableRawPointer?,
    _ samples: UnsafePointer<Float>?,
    _ count: UInt32,
    _ outProbabilities: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>?,
    _ outIsVoiceActive: UnsafeMutablePointer<UnsafeMutablePointer<UInt8>?>?,
    _ outProcessingTimes: UnsafeMutablePointer<UnsafeMutablePointer<Double>?>?,
    _ outCount: UnsafeMutablePointer<UInt32>?
) -> Int32 {
    guard let ptr = ptr, let samples = samples else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()

    let samplesArray = Array(UnsafeBufferPointer(start: samples, count: Int(count)))

    do {
        let frames = try bridge.vadProcessSamples(samplesArray)
        emitVadFrames(
            frames,
            outProbabilities: outProbabilities,
            outIsVoiceActive: outIsVoiceActive,
            outProcessingTimes: outProcessingTimes,
            outCount: outCount
        )
        return 0
    } catch {
        print("VAD process samples error: \(error)")
        return -1
    }
}

@_cdecl("fluidaudio_free_vad_result")
public func fluidaudio_free_vad_result(
    _ probabilities: UnsafeMutablePointer<Float>?,
    _ isVoiceActive: UnsafeMutablePointer<UInt8>?,
    _ processingTimes: UnsafeMutablePointer<Double>?,
    _ count: UInt32
) {
    _ = count // Reserved for future per-element cleanup (none currently needed).
    probabilities?.deallocate()
    isVoiceActive?.deallocate()
    processingTimes?.deallocate()
}

private func emitVadFrames(
    _ frames: [FluidAudioBridgeInternal.BridgeVadFrame],
    outProbabilities: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>?,
    outIsVoiceActive: UnsafeMutablePointer<UnsafeMutablePointer<UInt8>?>?,
    outProcessingTimes: UnsafeMutablePointer<UnsafeMutablePointer<Double>?>?,
    outCount: UnsafeMutablePointer<UInt32>?
) {
    let count = frames.count
    outCount?.pointee = UInt32(count)

    if count == 0 {
        outProbabilities?.pointee = nil
        outIsVoiceActive?.pointee = nil
        outProcessingTimes?.pointee = nil
        return
    }

    let probs = UnsafeMutablePointer<Float>.allocate(capacity: count)
    let voice = UnsafeMutablePointer<UInt8>.allocate(capacity: count)
    let times = UnsafeMutablePointer<Double>.allocate(capacity: count)

    for (i, frame) in frames.enumerated() {
        probs[i] = frame.probability
        voice[i] = frame.isVoiceActive ? 1 : 0
        times[i] = frame.processingTime
    }

    outProbabilities?.pointee = probs
    outIsVoiceActive?.pointee = voice
    outProcessingTimes?.pointee = times
}

/// Marshal diarization segments into the four caller-owned C out-arrays (freed by
/// `fluidaudio_free_diarization_result`). Shared by the download and model-path FFI
/// entry points (the latter lives in `Diarize_ffi.swift`). `count == 0` yields NULL arrays.
func emitDiarizationSegments(
    _ segments: [BridgeDiarizationSegment],
    outSpeakerIds: UnsafeMutablePointer<UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?>?,
    outStartTimes: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>?,
    outEndTimes: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>?,
    outQualityScores: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>?,
    outCount: UnsafeMutablePointer<UInt32>?
) {
    let count = segments.count
    outCount?.pointee = UInt32(count)

    if count == 0 {
        outSpeakerIds?.pointee = nil
        outStartTimes?.pointee = nil
        outEndTimes?.pointee = nil
        outQualityScores?.pointee = nil
        return
    }

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

// MARK: - ITN (Inverse Text Normalization) FFI

/// Normalize a short ASR expression (e.g. "two hundred thirty two" -> "232").
/// The returned string must be freed via fluidaudio_free_string.
@_cdecl("fluidaudio_itn_normalize")
public func fluidaudio_itn_normalize(
    _ ptr: UnsafeMutableRawPointer?,
    _ text: UnsafePointer<CChar>?,
    _ outText: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
    guard let ptr = ptr, let text = text else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    let normalized = bridge.itnNormalize(String(cString: text))
    outText?.pointee = strdup(normalized)
    return 0
}

@_cdecl("fluidaudio_itn_normalize_sentence")
public func fluidaudio_itn_normalize_sentence(
    _ ptr: UnsafeMutableRawPointer?,
    _ text: UnsafePointer<CChar>?,
    _ outText: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
    guard let ptr = ptr, let text = text else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    let normalized = bridge.itnNormalizeSentence(String(cString: text))
    outText?.pointee = strdup(normalized)
    return 0
}

@_cdecl("fluidaudio_itn_normalize_sentence_max_span")
public func fluidaudio_itn_normalize_sentence_max_span(
    _ ptr: UnsafeMutableRawPointer?,
    _ text: UnsafePointer<CChar>?,
    _ maxSpanTokens: UInt32,
    _ outText: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
    guard let ptr = ptr, let text = text else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    let normalized = bridge.itnNormalizeSentenceMaxSpan(String(cString: text), maxSpanTokens: maxSpanTokens)
    outText?.pointee = strdup(normalized)
    return 0
}

@_cdecl("fluidaudio_itn_is_native_available")
public func fluidaudio_itn_is_native_available(
    _ ptr: UnsafeMutableRawPointer?
) -> Int32 {
    guard let ptr = ptr else { return 0 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    return bridge.itnIsNativeAvailable() ? 1 : 0
}
