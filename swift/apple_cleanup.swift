// C ABI shim over the macOS 26 FoundationModels on-device model.
// Compiled with a macOS 14 deployment target; every entry point checks
// #available so the FoundationModels symbols stay weak-linked.

import Foundation
import FoundationModels

private func jsonString(_ value: Any) -> String {
    guard let data = try? JSONSerialization.data(withJSONObject: value),
        let text = String(data: data, encoding: .utf8)
    else {
        return "{\"error\":\"json encoding failed\"}"
    }
    return text
}

private func errorJson(_ kind: String, _ message: String) -> UnsafeMutablePointer<CChar> {
    strdup(jsonString(["error": message, "kind": kind]))!
}

private final class ResultBox<T>: @unchecked Sendable {
    var value: T?
}

private final class SessionBox: @unchecked Sendable {
    let value: AnyObject
    init(_ value: AnyObject) { self.value = value }
}

// Dictation is the user's own words; the default guardrails can refuse
// to rewrite profanity or sensitive topics.
@available(macOS 26.0, *)
private let model = SystemLanguageModel(guardrails: .permissiveContentTransformations)

// One prewarmed session, taken by the next generate with matching instructions.
private let prewarmedLock = NSLock()
private var prewarmed: (instructions: String, session: AnyObject)?

@available(macOS 26.0, *)
private func takeSession(_ instructions: String) -> LanguageModelSession {
    prewarmedLock.lock()
    defer { prewarmedLock.unlock() }
    if let warm = prewarmed, warm.instructions == instructions {
        prewarmed = nil
        return warm.session as! LanguageModelSession
    }
    return LanguageModelSession(model: model, instructions: instructions)
}

// Matches by case name so macOS 26's GenerationError and macOS 27's
// LanguageModelError both map, whichever SDK built the shim.
private func errorKind(_ error: Error) -> String {
    switch Mirror(reflecting: error).children.first?.label {
    case "exceededContextWindowSize", "contextSizeExceeded": return "context"
    case "guardrailViolation", "refusal": return "refused"
    case "unsupportedLanguageOrLocale": return "language"
    case "rateLimited", "concurrentRequests": return "busy"
    case "assetsUnavailable": return "unavailable"
    case "timeout": return "timeout"
    default: return "other"
    }
}

// Availability: 0 available, 1 Apple Intelligence off, 2 model not ready, 3 unsupported.
@_cdecl("gs_cleanup_availability")
public func gs_cleanup_availability() -> Int32 {
    guard #available(macOS 26.0, *) else { return 3 }
    switch model.availability {
    case .available: return 0
    case .unavailable(.appleIntelligenceNotEnabled): return 1
    case .unavailable(.modelNotReady): return 2
    default: return 3
    }
}

// Returns the availability code; the session is only kept when it is 0.
@_cdecl("gs_cleanup_prewarm")
public func gs_cleanup_prewarm(_ instructions: UnsafePointer<CChar>) -> Int32 {
    let status = gs_cleanup_availability()
    guard status == 0, #available(macOS 26.0, *) else { return status }
    let text = String(cString: instructions)
    let session = LanguageModelSession(model: model, instructions: text)
    session.prewarm()
    prewarmedLock.lock()
    prewarmed = (text, session)
    prewarmedLock.unlock()
    return 0
}

// Returns JSON {"text": ...} or {"error": ..., "kind": ...}. temperature <= 0
// samples greedily; max_tokens <= 0 means no limit.
@_cdecl("gs_cleanup_generate")
public func gs_cleanup_generate(
    _ instructions: UnsafePointer<CChar>,
    _ prompt: UnsafePointer<CChar>,
    _ temperature: Double,
    _ maxTokens: Int32,
    _ timeoutMs: Int64
) -> UnsafeMutablePointer<CChar> {
    guard #available(macOS 26.0, *) else {
        return errorJson("unavailable", "on-device model requires macOS 26 or later")
    }
    let session = SessionBox(takeSession(String(cString: instructions)))
    let promptText = String(cString: prompt)
    let maximumResponseTokens = maxTokens > 0 ? Int(maxTokens) : nil
    // init(sampling:) is deprecated in the macOS 27 SDK; CI builds with 26.
    let options =
        temperature <= 0
        ? GenerationOptions(sampling: .greedy, maximumResponseTokens: maximumResponseTokens)
        : GenerationOptions(temperature: temperature, maximumResponseTokens: maximumResponseTokens)

    let semaphore = DispatchSemaphore(value: 0)
    let box = ResultBox<[String: String]>()
    let task = Task.detached {
        let session = session.value as! LanguageModelSession
        do {
            let response = try await session.respond(to: promptText, options: options)
            box.value = ["text": response.content]
        } catch {
            box.value = ["error": "generation failed: \(error)", "kind": errorKind(error)]
        }
        semaphore.signal()
    }
    let deadline: DispatchTime = timeoutMs > 0 ? .now() + .milliseconds(Int(timeoutMs)) : .distantFuture
    if semaphore.wait(timeout: deadline) == .timedOut {
        task.cancel()
        return errorJson("timeout", "on-device model timed out")
    }
    return strdup(jsonString(box.value!))!
}

@_cdecl("gs_cleanup_string_free")
public func gs_cleanup_string_free(_ pointer: UnsafeMutablePointer<CChar>?) {
    free(pointer)
}
