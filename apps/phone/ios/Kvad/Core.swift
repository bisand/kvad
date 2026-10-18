// The Rust core (apps/phone/core): connecting, pairing, the chat, the
// Keychain and the Network test. Three C functions and one JSON object a
// message each way; every command and event is in apps/phone/messages.json.

import Foundation

enum Core {
    /// Where events go, on the main actor. The C callback cannot carry a
    /// Swift closure, so it finds this one here.
    @MainActor private static var handler: ((Event) -> Void)?

    /// Starts the core, once. Its first event is `ready`.
    @MainActor static func start(_ handle: @escaping (Event) -> Void) {
        handler = handle
        kvad_core_start(nil, coreEvent)
    }

    @MainActor fileprivate static func deliver(_ data: Data) {
        if let event = try? JSONDecoder().decode(Event.self, from: data) {
            handler?(event)
        }
    }

    /// One command: `send("chat", ["model": id, "text": "Hi"])`.
    static func send(_ name: String, _ args: [String: Any] = [:]) {
        guard let data = try? JSONSerialization.data(withJSONObject: [name: args]),
              let text = String(data: data, encoding: .utf8) else { return }
        text.withCString { kvad_core_send($0) }
    }

    static var version: String { String(cString: kvad_core_version()) }
}

/// The core's callback, on whichever of its threads has the event. Not in
/// `start`, which is on the main actor: a closure written there would be
/// too, and Swift checks that on entry and traps on the core's thread.
private func coreEvent(_ context: UnsafeMutableRawPointer?, _ json: UnsafePointer<CChar>?) {
    // The core frees the text when this returns: copy it first.
    guard let json else { return }
    let data = Data(bytes: json, count: strlen(json))
    DispatchQueue.main.async {
        MainActor.assumeIsolated { Core.deliver(data) }
    }
}

struct ModelInfo: Decodable, Identifiable, Hashable {
    let id: String
    let kind: String
    let resident: Bool

    /// The model's name without its owner: `SmolLM2-135M-Instruct`.
    var name: String { id.split(separator: "/", maxSplits: 1).last.map(String.init) ?? id }
    var owner: String { id.contains("/") ? String(id.split(separator: "/", maxSplits: 1)[0]) : "" }
}

struct Connected: Decodable {
    let name: String
    let label: String
    let paired: Bool
    let version: String
    let who: String
    let role: String
    let path: String
    let models: [ModelInfo]
    let `default`: String
}

struct ProbeRow: Decodable, Hashable {
    let path: String
    let pings: Int
    let median: String
    let p90: String
    let max: String
}

/// The probe's table so far, with the last ping; or, at the end, its status.
struct ProbeTable: Decodable {
    var rows: [ProbeRow]?
    var totals: String?
    var last: String?
    var elapsed: Double?
    var status: String?
    var line: String?
}

/// One event from the core, by the name that is its only key.
enum Event: Decodable {
    case ready(server: String, core: String)
    case connected(Connected)
    case connectFailed(String)
    case path(String)
    case models([ModelInfo])
    case modelsFailed(String)
    case piece(String)
    case replyDone
    case chatFailed(error: String, nothing: Bool)
    case disconnected
    case unpaired
    case unpairFailed(String)
    case link(server: String, notice: String)
    case linkRefused(String)
    case probeID(String)
    case probeStarted(secs: Int, status: String)
    case probeStatus(String)
    case probePath(String)
    case probeLine(String)
    case probePing(ProbeTable)
    case probeDone(ProbeTable)
    case probeFailed(String)
    case error(String)
    /// A name this app does not know: a newer core.
    case unknown(String)

    private struct Name: CodingKey {
        var stringValue: String
        init(stringValue: String) { self.stringValue = stringValue }
        var intValue: Int? { nil }
        init?(intValue: Int) { nil }
    }
    private struct Text: Decodable { let text: String }
    private struct Failure: Decodable { let error: String }
    private struct Ready: Decodable { let server: String; let core: String }
    private struct Now: Decodable { let now: String }
    private struct Models: Decodable { let models: [ModelInfo] }
    private struct ChatFailure: Decodable { let error: String; let nothing: Bool }
    private struct Link: Decodable { let server: String; let notice: String }
    private struct ID: Decodable { let id: String }
    private struct Started: Decodable { let secs: Int; let status: String }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: Name.self)
        guard let key = c.allKeys.first else {
            throw DecodingError.dataCorrupted(.init(codingPath: [], debugDescription: "an event with no name"))
        }
        func body<T: Decodable>(_: T.Type) throws -> T { try c.decode(T.self, forKey: key) }
        switch key.stringValue {
        case "ready": let r = try body(Ready.self); self = .ready(server: r.server, core: r.core)
        case "connected": self = .connected(try body(Connected.self))
        case "connect_failed": self = .connectFailed(try body(Failure.self).error)
        case "path": self = .path(try body(Now.self).now)
        case "models": self = .models(try body(Models.self).models)
        case "models_failed": self = .modelsFailed(try body(Failure.self).error)
        case "piece": self = .piece(try body(Text.self).text)
        case "reply_done": self = .replyDone
        case "chat_failed": let f = try body(ChatFailure.self); self = .chatFailed(error: f.error, nothing: f.nothing)
        case "disconnected": self = .disconnected
        case "unpaired": self = .unpaired
        case "unpair_failed": self = .unpairFailed(try body(Failure.self).error)
        case "link": let l = try body(Link.self); self = .link(server: l.server, notice: l.notice)
        case "link_refused": self = .linkRefused(try body(Failure.self).error)
        case "probe_id": self = .probeID(try body(ID.self).id)
        case "probe_started": let s = try body(Started.self); self = .probeStarted(secs: s.secs, status: s.status)
        case "probe_status": self = .probeStatus(try body(Text.self).text)
        case "probe_path": self = .probePath(try body(Text.self).text)
        case "probe_line": self = .probeLine(try body(Text.self).text)
        case "probe_ping": self = .probePing(try body(ProbeTable.self))
        case "probe_done": self = .probeDone(try body(ProbeTable.self))
        case "probe_failed": self = .probeFailed(try body(Failure.self).error)
        case "error": self = .error(try body(Failure.self).error)
        default: self = .unknown(key.stringValue)
        }
    }
}
