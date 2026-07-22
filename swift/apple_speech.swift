// C ABI shim over the macOS 26 SpeechAnalyzer/SpeechTranscriber API.
// Compiled with a macOS 14 deployment target; every entry point checks
// #available so the newer Speech symbols stay weak-linked.

import AVFoundation
import Foundation
import Speech

private func jsonString(_ value: Any) -> String {
    guard let data = try? JSONSerialization.data(withJSONObject: value),
        let text = String(data: data, encoding: .utf8)
    else {
        return "{\"error\":\"json encoding failed\"}"
    }
    return text
}

private func errorJson(_ message: String) -> UnsafeMutablePointer<CChar> {
    strdup(jsonString(["error": message]))!
}

private func runBlocking<T: Sendable>(_ op: @escaping @Sendable () async -> T) -> T {
    let semaphore = DispatchSemaphore(value: 0)
    let box = ResultBox<T>()
    Task.detached {
        box.value = await op()
        semaphore.signal()
    }
    semaphore.wait()
    return box.value!
}

private final class ResultBox<T>: @unchecked Sendable {
    var value: T?
}

private struct GSError: Error {
    let message: String
}

@available(macOS 26.0, *)
private func resolveLocale(_ hint: String?) async -> Locale {
    let supported = await SpeechTranscriber.supportedLocales
    guard let hint, !hint.isEmpty else {
        let current = Locale.current
        if supported.contains(where: { $0.identifier(.bcp47) == current.identifier(.bcp47) }) {
            return current
        }
        return supported.first(where: { $0.language.languageCode?.identifier == current.language.languageCode?.identifier }) ?? Locale(identifier: "en-US")
    }
    if let exact = supported.first(where: { $0.identifier(.bcp47).lowercased() == hint.lowercased() }) {
        return exact
    }
    let language = hint.split(separator: "-").first.map(String.init) ?? hint
    return supported.first(where: { $0.language.languageCode?.identifier.lowercased() == language.lowercased() }) ?? Locale(identifier: hint)
}

@available(macOS 26.0, *)
private final class GSSession: @unchecked Sendable {
    let analyzer: SpeechAnalyzer
    let transcriber: SpeechTranscriber
    let continuation: AsyncStream<AnalyzerInput>.Continuation
    let analyzerFormat: AVAudioFormat
    let inputFormat: AVAudioFormat
    let converter: AVAudioConverter?
    let lock = NSLock()
    var finalizedText = ""
    var volatileText = ""
    var segments: [[String: Any]] = []
    var lastError: String?
    var resultsTask: Task<Void, Never>?

    init(
        analyzer: SpeechAnalyzer,
        transcriber: SpeechTranscriber,
        continuation: AsyncStream<AnalyzerInput>.Continuation,
        analyzerFormat: AVAudioFormat,
        inputFormat: AVAudioFormat,
        converter: AVAudioConverter?
    ) {
        self.analyzer = analyzer
        self.transcriber = transcriber
        self.continuation = continuation
        self.analyzerFormat = analyzerFormat
        self.inputFormat = inputFormat
        self.converter = converter
    }

    static func start(localeHint: String?) async -> Result<GSSession, GSError> {
        let locale = await resolveLocale(localeHint)
        let transcriber = SpeechTranscriber(
            locale: locale,
            transcriptionOptions: [],
            reportingOptions: [.volatileResults],
            attributeOptions: [.audioTimeRange]
        )
        do {
            if let request = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
                try await request.downloadAndInstall()
            }
        } catch {
            return .failure(GSError(message: "speech assets unavailable for \(locale.identifier(.bcp47)): \(error)"))
        }
        guard let analyzerFormat = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [transcriber]) else {
            return .failure(GSError(message: "no compatible audio format for \(locale.identifier(.bcp47))"))
        }
        guard let inputFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false) else {
            return .failure(GSError(message: "failed to create input format"))
        }
        let converter: AVAudioConverter?
        if inputFormat == analyzerFormat {
            converter = nil
        } else {
            guard let created = AVAudioConverter(from: inputFormat, to: analyzerFormat) else {
                return .failure(GSError(message: "failed to create audio converter"))
            }
            converter = created
        }
        let (stream, continuation) = AsyncStream.makeStream(of: AnalyzerInput.self)
        let analyzer = SpeechAnalyzer(modules: [transcriber])
        let session = GSSession(
            analyzer: analyzer,
            transcriber: transcriber,
            continuation: continuation,
            analyzerFormat: analyzerFormat,
            inputFormat: inputFormat,
            converter: converter
        )
        session.resultsTask = Task {
            do {
                for try await result in transcriber.results {
                    let text = String(result.text.characters)
                    let range = result.range
                    session.lock.withLock {
                        if result.isFinal {
                            if !text.isEmpty {
                                if !session.finalizedText.isEmpty {
                                    session.finalizedText += " "
                                }
                                session.finalizedText += text
                                session.segments.append([
                                    "start": range.start.seconds.isFinite ? range.start.seconds : 0,
                                    "end": range.end.seconds.isFinite ? range.end.seconds : 0,
                                    "text": text,
                                ])
                            }
                            session.volatileText = ""
                        } else {
                            session.volatileText = text
                        }
                    }
                }
            } catch {
                session.lock.withLock {
                    session.lastError = "\(error)"
                }
            }
        }
        do {
            try await analyzer.start(inputSequence: stream)
        } catch {
            session.resultsTask?.cancel()
            return .failure(GSError(message: "failed to start analyzer: \(error)"))
        }
        return .success(session)
    }

    func feed(samples: UnsafePointer<Float>, count: Int) -> String? {
        guard let buffer = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: AVAudioFrameCount(count)) else {
            return "failed to allocate audio buffer"
        }
        buffer.frameLength = AVAudioFrameCount(count)
        buffer.floatChannelData![0].update(from: samples, count: count)

        let outBuffer: AVAudioPCMBuffer
        if let converter {
            let ratio = analyzerFormat.sampleRate / inputFormat.sampleRate
            let capacity = AVAudioFrameCount((Double(count) * ratio).rounded(.up) + 64)
            guard let converted = AVAudioPCMBuffer(pcmFormat: analyzerFormat, frameCapacity: capacity) else {
                return "failed to allocate conversion buffer"
            }
            var fed = false
            var conversionError: NSError?
            converter.convert(to: converted, error: &conversionError) { _, status in
                if fed {
                    status.pointee = .noDataNow
                    return nil
                }
                fed = true
                status.pointee = .haveData
                return buffer
            }
            if let conversionError {
                return "audio conversion failed: \(conversionError)"
            }
            outBuffer = converted
        } else {
            outBuffer = buffer
        }
        continuation.yield(AnalyzerInput(buffer: outBuffer))
        return nil
    }

    func snapshot() -> String {
        lock.lock()
        defer { lock.unlock() }
        if volatileText.isEmpty {
            return finalizedText
        }
        if finalizedText.isEmpty {
            return volatileText
        }
        return finalizedText + " " + volatileText
    }

    func finish() async -> String {
        continuation.finish()
        do {
            try await analyzer.finalizeAndFinishThroughEndOfInput()
        } catch {
            lock.withLock {
                lastError = lastError ?? "\(error)"
            }
        }
        await resultsTask?.value
        return lock.withLock {
            if let lastError, finalizedText.isEmpty {
                return jsonString(["error": lastError])
            }
            return jsonString([
                "text": finalizedText,
                "segments": segments,
            ])
        }
    }

    func cancel() async {
        continuation.finish()
        resultsTask?.cancel()
        await analyzer.cancelAndFinishNow()
    }
}

@available(macOS 26.0, *)
private final class SessionRegistry: @unchecked Sendable {
    static let shared = SessionRegistry()
    private let lock = NSLock()
    private var next: Int64 = 1
    private var sessions: [Int64: GSSession] = [:]

    func insert(_ session: GSSession) -> Int64 {
        lock.lock()
        defer { lock.unlock() }
        let handle = next
        next += 1
        sessions[handle] = session
        return handle
    }

    func get(_ handle: Int64) -> GSSession? {
        lock.lock()
        defer { lock.unlock() }
        return sessions[handle]
    }

    func remove(_ handle: Int64) -> GSSession? {
        lock.lock()
        defer { lock.unlock() }
        return sessions.removeValue(forKey: handle)
    }
}

// Availability: 0 available, 1 unsupported OS, 2 no supported locales.
@_cdecl("gs_apple_availability")
public func gs_apple_availability() -> Int32 {
    guard #available(macOS 26.0, *) else { return 1 }
    return runBlocking {
        await SpeechTranscriber.supportedLocales.isEmpty ? 2 : 0
    }
}

// Locale status: 0 installed, 1 downloadable, 2 unsupported, 1xx errors.
@_cdecl("gs_apple_locale_status")
public func gs_apple_locale_status(_ locale: UnsafePointer<CChar>?) -> Int32 {
    guard #available(macOS 26.0, *) else { return 101 }
    let hint = locale.map { String(cString: $0) }
    return runBlocking {
        let resolved = await resolveLocale(hint)
        let supported = await SpeechTranscriber.supportedLocales
        guard supported.contains(where: { $0.identifier(.bcp47) == resolved.identifier(.bcp47) }) else {
            return 2
        }
        let installed = await SpeechTranscriber.installedLocales
        return installed.contains(where: { $0.identifier(.bcp47) == resolved.identifier(.bcp47) }) ? 0 : 1
    }
}

@_cdecl("gs_apple_stream_start")
public func gs_apple_stream_start(_ locale: UnsafePointer<CChar>?) -> Int64 {
    guard #available(macOS 26.0, *) else { return 0 }
    let hint = locale.map { String(cString: $0) }
    return runBlocking {
        switch await GSSession.start(localeHint: hint) {
        case .success(let session):
            return SessionRegistry.shared.insert(session)
        case .failure(let failure):
            NSLog("[glimpse-apple-speech] stream start failed: %@", failure.message)
            return 0
        }
    }
}

@_cdecl("gs_apple_stream_feed")
public func gs_apple_stream_feed(_ handle: Int64, _ samples: UnsafePointer<Float>?, _ count: Int) -> Int32 {
    guard #available(macOS 26.0, *) else { return 1 }
    guard let samples, count > 0, let session = SessionRegistry.shared.get(handle) else { return 1 }
    if let error = session.feed(samples: samples, count: count) {
        NSLog("[glimpse-apple-speech] feed failed: %@", error)
        return 1
    }
    return 0
}

@_cdecl("gs_apple_stream_text")
public func gs_apple_stream_text(_ handle: Int64) -> UnsafeMutablePointer<CChar>? {
    guard #available(macOS 26.0, *) else { return nil }
    guard let session = SessionRegistry.shared.get(handle) else { return nil }
    return strdup(session.snapshot())
}

// Finishes the session and returns {"text": ..., "segments": [...]} JSON.
@_cdecl("gs_apple_stream_finish")
public func gs_apple_stream_finish(_ handle: Int64) -> UnsafeMutablePointer<CChar>? {
    guard #available(macOS 26.0, *) else { return errorJson("unsupported OS") }
    guard let session = SessionRegistry.shared.remove(handle) else { return errorJson("unknown session") }
    return runBlocking {
        strdup(await session.finish())
    }
}

@_cdecl("gs_apple_stream_cancel")
public func gs_apple_stream_cancel(_ handle: Int64) {
    guard #available(macOS 26.0, *) else { return }
    guard let session = SessionRegistry.shared.remove(handle) else { return }
    runBlocking {
        await session.cancel()
    }
}

@_cdecl("gs_apple_string_free")
public func gs_apple_string_free(_ pointer: UnsafeMutablePointer<CChar>?) {
    free(pointer)
}
