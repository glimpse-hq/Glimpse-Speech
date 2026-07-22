// C ABI shim over the macOS 26 SpeechAnalyzer API.
// Compiled with a macOS 14 deployment target; every entry point checks
// #available so the newer Speech symbols stay weak-linked.
//
// Two modules: DictationTranscriber for dictation (spoken punctuation,
// emoji, custom vocabulary), SpeechTranscriber for long-form jobs that
// want fine-grained segment timestamps.

import AVFoundation
import CryptoKit
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
    let supported = await DictationTranscriber.supportedLocales
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

// Builds (and caches) a custom language model for vocabulary boosting.
// Preparation is slow the first time; the result is cached by content hash.
@available(macOS 26.0, *)
private func customVocabularyHint(locale: Locale, terms: [String]) async -> DictationTranscriber.ContentHint? {
    let terms = terms.map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
    guard !terms.isEmpty else { return nil }

    let fingerprint = locale.identifier(.bcp47) + "\n" + terms.joined(separator: "\n")
    let digest = SHA256.hash(data: Data(fingerprint.utf8))
        .map { String(format: "%02x", $0) }.joined().prefix(16)
    let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("glimpse-apple-speech/vocab-\(digest)", isDirectory: true)
    let modelURL = cacheDir.appendingPathComponent("model.bin")
    let configuration = SFSpeechLanguageModel.Configuration(languageModel: modelURL)

    if !FileManager.default.fileExists(atPath: modelURL.path) {
        do {
            try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
            let data = SFCustomLanguageModelData(
                locale: locale,
                identifier: "cc.tryglimpse.dictionary",
                version: "1"
            )
            // Bare words barely shift the model; realistic phrases work better.
            for term in terms {
                for phrase in [term, "I use \(term)", "open \(term)", "in \(term)", "with \(term)", "the \(term) app"] {
                    SFCustomLanguageModelData.PhraseCount(phrase: phrase, count: 100).insert(data: data)
                }
            }
            let exportURL = cacheDir.appendingPathComponent("data.bin")
            try await data.export(to: exportURL)
            try await SFSpeechLanguageModel.prepareCustomLanguageModel(
                for: exportURL,
                clientIdentifier: "cc.tryglimpse",
                configuration: configuration
            )
        } catch {
            NSLog("[glimpse-apple-speech] custom vocabulary preparation failed: %@", "\(error)")
            try? FileManager.default.removeItem(at: cacheDir)
            return nil
        }
    }
    return .customizedLanguage(modelConfiguration: configuration)
}

@available(macOS 26.0, *)
private enum GSModule {
    case dictation(DictationTranscriber)
    case longForm(SpeechTranscriber)

    var speechModule: any SpeechModule {
        switch self {
        case .dictation(let module): return module
        case .longForm(let module): return module
        }
    }
}

@available(macOS 26.0, *)
private final class GSSession: @unchecked Sendable {
    let analyzer: SpeechAnalyzer
    let module: GSModule
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
        module: GSModule,
        continuation: AsyncStream<AnalyzerInput>.Continuation,
        analyzerFormat: AVAudioFormat,
        inputFormat: AVAudioFormat,
        converter: AVAudioConverter?
    ) {
        self.analyzer = analyzer
        self.module = module
        self.continuation = continuation
        self.analyzerFormat = analyzerFormat
        self.inputFormat = inputFormat
        self.converter = converter
    }

    private func record(text: String, range: CMTimeRange, isFinal: Bool) {
        lock.withLock {
            if isFinal {
                if !text.isEmpty {
                    if !finalizedText.isEmpty {
                        finalizedText += " "
                    }
                    finalizedText += text
                    segments.append([
                        "start": range.start.seconds.isFinite ? range.start.seconds : 0,
                        "end": range.end.seconds.isFinite ? range.end.seconds : 0,
                        "text": text,
                    ])
                }
                volatileText = ""
            } else {
                volatileText = text
            }
        }
    }

    private func recordError(_ error: Error) {
        lock.withLock {
            lastError = "\(error)"
        }
    }

    static func start(localeHint: String?, longForm: Bool, vocabulary: [String]) async -> Result<GSSession, GSError> {
        let locale = await resolveLocale(localeHint)
        let module: GSModule
        if longForm {
            module = .longForm(
                SpeechTranscriber(
                    locale: locale,
                    transcriptionOptions: [],
                    reportingOptions: [.volatileResults],
                    attributeOptions: [.audioTimeRange]
                ))
        } else {
            var hints: Set<DictationTranscriber.ContentHint> = []
            if let vocabularyHint = await customVocabularyHint(locale: locale, terms: vocabulary) {
                hints.insert(vocabularyHint)
            }
            module = .dictation(
                DictationTranscriber(
                    locale: locale,
                    contentHints: hints,
                    transcriptionOptions: [.punctuation, .emoji],
                    reportingOptions: [.volatileResults],
                    attributeOptions: [.audioTimeRange]
                ))
        }
        do {
            if let request = try await AssetInventory.assetInstallationRequest(supporting: [module.speechModule]) {
                try await request.downloadAndInstall()
            }
        } catch {
            return .failure(GSError(message: "speech assets unavailable for \(locale.identifier(.bcp47)): \(error)"))
        }
        guard let analyzerFormat = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [module.speechModule]) else {
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
        let analyzer = SpeechAnalyzer(modules: [module.speechModule])
        let session = GSSession(
            analyzer: analyzer,
            module: module,
            continuation: continuation,
            analyzerFormat: analyzerFormat,
            inputFormat: inputFormat,
            converter: converter
        )
        session.resultsTask = Task {
            switch module {
            case .dictation(let transcriber):
                do {
                    for try await result in transcriber.results {
                        session.record(
                            text: String(result.text.characters),
                            range: result.range,
                            isFinal: result.isFinal
                        )
                    }
                } catch {
                    session.recordError(error)
                }
            case .longForm(let transcriber):
                do {
                    for try await result in transcriber.results {
                        session.record(
                            text: String(result.text.characters),
                            range: result.range,
                            isFinal: result.isFinal
                        )
                    }
                } catch {
                    session.recordError(error)
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

private func parseVocabulary(_ json: UnsafePointer<CChar>?) -> [String] {
    guard let json else { return [] }
    let raw = String(cString: json)
    guard !raw.isEmpty, let data = raw.data(using: .utf8),
        let terms = try? JSONDecoder().decode([String].self, from: data)
    else {
        return []
    }
    return terms
}

// Supported locales as a JSON array of BCP-47 identifiers.
@_cdecl("gs_apple_supported_locales")
public func gs_apple_supported_locales() -> UnsafeMutablePointer<CChar>? {
    guard #available(macOS 26.0, *) else { return strdup("[]") }
    return runBlocking {
        let locales = await DictationTranscriber.supportedLocales.map { $0.identifier(.bcp47) }
        return strdup(jsonString(locales.sorted()))
    }
}

// Availability: 0 available, 1 unsupported OS, 2 no supported locales.
@_cdecl("gs_apple_availability")
public func gs_apple_availability() -> Int32 {
    guard #available(macOS 26.0, *) else { return 1 }
    return runBlocking {
        await DictationTranscriber.supportedLocales.isEmpty ? 2 : 0
    }
}

// Locale status: 0 installed, 1 downloadable, 2 unsupported, 1xx errors.
@_cdecl("gs_apple_locale_status")
public func gs_apple_locale_status(_ locale: UnsafePointer<CChar>?) -> Int32 {
    guard #available(macOS 26.0, *) else { return 101 }
    let hint = locale.map { String(cString: $0) }
    return runBlocking {
        let resolved = await resolveLocale(hint)
        let supported = await DictationTranscriber.supportedLocales
        guard supported.contains(where: { $0.identifier(.bcp47) == resolved.identifier(.bcp47) }) else {
            return 2
        }
        let installed = await DictationTranscriber.installedLocales
        return installed.contains(where: { $0.identifier(.bcp47) == resolved.identifier(.bcp47) }) ? 0 : 1
    }
}

// long_form selects SpeechTranscriber (fine segments, no dictation extras).
// vocabulary_json is a JSON array of terms; dictation sessions only.
@_cdecl("gs_apple_stream_start")
public func gs_apple_stream_start(
    _ locale: UnsafePointer<CChar>?,
    _ longForm: Int32,
    _ vocabularyJson: UnsafePointer<CChar>?
) -> Int64 {
    guard #available(macOS 26.0, *) else { return 0 }
    let hint = locale.map { String(cString: $0) }
    let vocabulary = parseVocabulary(vocabularyJson)
    return runBlocking {
        switch await GSSession.start(localeHint: hint, longForm: longForm != 0, vocabulary: vocabulary) {
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
