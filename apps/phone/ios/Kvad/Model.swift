// What the screens show, kept up to date from the core's events, and what
// they ask the core to do. Everything here runs on the main actor; the core
// does the waiting on its own threads.

import SwiftUI
import UIKit

struct Bubble: Identifiable, Equatable {
    let id = UUID()
    let mine: Bool
    var text: String
    var streaming = false
}

enum Tab: Hashable { case chat, speak, create, models, more }

/// Where Speak is: nothing under way, recording, or the take's last words
/// being settled after the stop.
enum Recording: Equatable { case idle, starting, on, finishing }

/// The languages offered for a take, by Whisper's code; empty is the
/// model's own guess.
let speakLanguages: [(code: String, name: String)] = [
    ("", "Detect"), ("no", "Norwegian"), ("en", "English"), ("sv", "Swedish"), ("da", "Danish"),
    ("de", "German"), ("fr", "French"), ("es", "Spanish"), ("fi", "Finnish"), ("pl", "Polish"),
]

/// How Speak hears, chosen in the Speak sheet: a table of people, or one
/// voice close to the phone (docs/phone-speak-plan.md, *Hearing the far end
/// of the table*).
enum Room: String, CaseIterable {
    case meeting, near

    var name: String { self == .meeting ? "Meeting" : "Close by" }

    var hearing: Hearing {
        switch self {
        case .meeting: Hearing(level: true)
        case .near: Hearing()
        }
    }
}

#if DEBUG
/// The ways of hearing compared for *Hearing the far end of the table*, by
/// name, for the debug picker and `KVAD_PHONE_MIC`. Each records with no
/// gain of the core's: `live_session`'s `+level` puts it on afterwards, on
/// the very same take.
let hearingTrials: [(name: String, hearing: Hearing)] = [
    ("standard", Hearing()),
    ("measurement", Hearing(mode: .measurement)),
    ("voice", Hearing(voiceProcessing: true)),
    ("front", Hearing(facing: .front, everyWay: true)),
    ("back", Hearing(facing: .back, everyWay: true)),
    ("bottom", Hearing(facing: .bottom, everyWay: true)),
]
#endif

@Observable @MainActor
final class AppModel {
    // Connect
    var serverText = "127.0.0.1:5823"
    var keyText = ""
    var connecting = false
    var connectError = ""
    var connectNotice = ""
    var server: Connected?
    var path = ""
    var coreVersion = ""
    var tab: Tab = .chat

    // Chat
    var bubbles: [Bubble] = []
    var draft = ""
    var replying = false
    var chatError = ""
    var models: [ModelInfo] = []
    var model = ""
    var modelsLoading = false
    /// The conversations kept on this phone, the latest first, and the one
    /// open, once it has a first question.
    var conversations: [ChatInfo] = []
    var chatID: String?

    // The Network test
    var probeID = ""
    var probeTicket = ""
    var probeSecs = 60
    var probeRunning = false
    var probeStatus = ""
    var probeError = ""
    var probePath = ""
    var probeLast = ""
    var probeElapsed = 0.0
    var probeRows: [ProbeRow] = []
    var probeTotals = ""
    var probeLog = ""

    // Speak
    var recording: Recording = .idle
    var takeStarted: Date?
    var settled = ""
    var partial = ""
    var lag = 0.0
    /// The link as the core last said: live, reconnecting, offline, sending.
    var speakLink = ""
    var speakNotice = ""
    var speakError = ""
    /// Paused by a call or Siri, and going on after.
    var speakPaused = false
    var speakModel = ""
    var speakLanguage = "no"
    var speakRoom = Room(rawValue: UserDefaults.standard.string(forKey: "speakRoom") ?? "") ?? .meeting {
        didSet { UserDefaults.standard.set(speakRoom.rawValue, forKey: "speakRoom") }
    }
    #if DEBUG
    /// A trial's name, heard with in place of the room's way.
    var speakTrial = Script.env["KVAD_PHONE_MIC"] ?? ""
    /// What the last take heard with, as iOS had it.
    var speakRoute = ""
    #endif
    var hearing: Hearing {
        #if DEBUG
        if let trial = hearingTrials.first(where: { $0.name == speakTrial }) { return trial.hearing }
        #endif
        return speakRoom.hearing
    }
    var takes: [Take] = []
    /// Takes being transcribed again, with the text so far.
    var again: [String: String] = [:]
    private let microphone = Microphone()

    var chatModels: [ModelInfo] { models.filter { $0.kind == "chat" } }
    var speechModels: [ModelInfo] { models.filter { $0.kind == "speech" } }
    var speakModelName: String { speakModel.isEmpty ? "Choose a model" : ModelInfo(id: speakModel, kind: "", resident: false).name }

    /// Where takes are kept: Application Support/Takes, out of reach while
    /// the phone is locked but for a take already being written
    /// (`completeUnlessOpen`), and out of iCloud and computer backups. The
    /// core makes the files; they take the folder's class.
    static let takesFolder = kept("Takes")
    /// Where conversations are kept, as takes are.
    static let chatsFolder = kept("Chats")

    private static func kept(_ name: String) -> URL {
        let fm = FileManager.default
        var dir = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent(name, isDirectory: true)
        let protection: [FileAttributeKey: Any] = [.protectionKey: FileProtectionType.completeUnlessOpen]
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true, attributes: protection)
        try? fm.setAttributes(protection, ofItemAtPath: dir.path)
        var unbacked = URLResourceValues()
        unbacked.isExcludedFromBackup = true
        try? dir.setResourceValues(unbacked)
        return dir
    }
    private var folder: String { Self.takesFolder.path }

    /// The speech model live is best on: small, measured as showing a word
    /// 0.4 s after it is said (docs/speech-plan.md); then one in memory.
    static func liveModel(_ models: [ModelInfo]) -> String {
        let speech = models.filter { $0.kind == "speech" }
        return (speech.first { $0.id.hasSuffix("whisper-small") && $0.resident }
            ?? speech.first { $0.id.hasSuffix("whisper-small") }
            ?? speech.first { $0.resident }
            ?? speech.first)?.id ?? ""
    }
    var modelName: String { model.isEmpty ? "Choose a model" : (ModelInfo(id: model, kind: "", resident: false).name) }
    private var modelsWaiters: [CheckedContinuation<Void, Never>] = []

    func start() {
        Core.start { [weak self] event in self?.handle(event) }
        microphone.onPause = { [weak self] paused in self?.speakPaused = paused }
        microphone.onFailure = { [weak self] why in
            self?.speakError = why
            self?.stopTake()
        }
    }

    // MARK: What the screens ask for

    func connect() {
        connectNotice = ""
        connectError = ""
        connecting = true
        Core.send("connect", ["url": serverText, "key": keyText, "device": UIDevice.current.name])
    }

    func disconnect() {
        Core.send("disconnect")
    }

    func unpair() {
        Core.send("unpair")
    }

    func send() {
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !replying, !model.isEmpty else { return }
        bubbles.append(Bubble(mine: true, text: text))
        bubbles.append(Bubble(mine: false, text: "", streaming: true))
        draft = ""
        chatError = ""
        replying = true
        Core.send("chat", ["model": model, "text": text, "folder": Self.chatsFolder.path])
    }

    func stop() {
        Core.send("stop")
    }

    func newChat() {
        Core.send("new_chat")
        bubbles = []
        chatError = ""
        chatID = nil
    }

    func loadChats() {
        Core.send("chats", ["folder": Self.chatsFolder.path])
    }

    func openChat(_ chat: ChatInfo) {
        guard !replying else { return }
        Core.send("chat_open", ["folder": Self.chatsFolder.path, "id": chat.id])
    }

    func deleteChat(_ chat: ChatInfo) {
        Core.send("chat_delete", ["folder": Self.chatsFolder.path, "id": chat.id])
    }

    /// Asks for the model list, and returns once it is in: for pull to refresh.
    func refreshModels() async {
        modelsLoading = true
        Core.send("models")
        await withCheckedContinuation { modelsWaiters.append($0) }
    }

    func pasteLink() {
        if let text = UIPasteboard.general.string?.trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty {
            serverText = text
            Core.send("opened", ["link": text])
        }
    }

    /// A `kvad://` link opened in the app: the Camera's QR scanner, or a tap.
    func opened(_ url: URL) {
        Core.send("opened", ["link": url.absoluteString])
    }

    func networkChanged() {
        Core.send("network_changed")
    }

    func openProbe() {
        if probeID.isEmpty { Core.send("probe_open") }
    }

    func pasteTicket() {
        if let text = UIPasteboard.general.string?.trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty {
            probeTicket = text
            probeError = ""
        } else {
            probeError = "Nothing to paste. Copy the ticket on the Mac first."
        }
    }

    func copyProbeID() {
        UIPasteboard.general.string = probeID
    }

    func startProbe() {
        probeError = ""
        Core.send("probe_start", ["ticket": probeTicket, "secs": probeSecs])
    }

    func stopProbe() {
        Core.send("probe_stop")
    }

    func startTake() async {
        guard recording == .idle else { return }
        speakError = ""
        guard await Microphone.allowed() else {
            speakError = "Kvad may not use the microphone. Allow it in Settings, under Kvad."
            return
        }
        settled = ""
        partial = ""
        lag = 0
        speakLink = ""
        speakNotice = ""
        speakPaused = false
        recording = .starting
        Core.send("speak_start", ["model": speakModel, "language": speakLanguage, "folder": folder, "level": hearing.level])
    }

    func stopTake() {
        guard recording == .on || recording == .starting else { return }
        microphone.stop()
        recording = .finishing
        Core.send("speak_stop")
    }

    func cancelTake() {
        microphone.stop()
        Core.send("speak_cancel")
        recording = .idle
        takeStarted = nil
    }

    func loadTakes() {
        Core.send("takes", ["folder": folder])
    }

    func deleteTake(_ take: Take) {
        Core.send("take_delete", ["folder": folder, "id": take.id])
    }

    /// The take sent whole to the file endpoint, with the model chosen for
    /// Speak: one live could not finish, or a better model than live keeps
    /// up with.
    func transcribeAgain(_ take: Take) {
        again[take.id] = ""
        Core.send("retranscribe", ["folder": folder, "id": take.id, "model": speakModel, "language": take.language.isEmpty ? speakLanguage : take.language])
    }

    func audioURL(_ take: Take) -> URL {
        Self.takesFolder.appendingPathComponent("\(take.id).wav")
    }

    // MARK: What the core says

    private func handle(_ event: Event) {
        switch event {
        case let .ready(server, core):
            coreVersion = core
            if !server.isEmpty { serverText = server }
            #if DEBUG
            Script.atLaunch(self)
            #endif
        case let .connected(c):
            connecting = false
            server = c
            path = c.path
            models = c.models
            if model.isEmpty || !c.models.contains(where: { $0.id == model }) { model = c.default }
            if speakModel.isEmpty || !c.models.contains(where: { $0.id == speakModel }) { speakModel = Self.liveModel(c.models) }
            serverText = c.name
            loadTakes()
            loadChats()
            #if DEBUG
            Script.afterConnect(self)
            #endif
        case let .connectFailed(e):
            connecting = false
            connectError = e
        case let .path(now):
            path = now
        case let .models(m):
            models = m
            modelsDone()
        case .modelsFailed:
            modelsDone()
        case let .piece(text):
            if let i = bubbles.indices.last, !bubbles[i].mine { bubbles[i].text += text }
        case let .chatSaved(chat):
            chatID = chat.id
            conversations.removeAll { $0.id == chat.id }
            conversations.insert(chat, at: 0)
        case let .chats(list):
            conversations = list
        case let .chatOpened(id, m, turns):
            chatID = id
            chatError = ""
            bubbles = turns.map { Bubble(mine: $0.role == "user", text: $0.content) }
            // Go on with the model it was had with, if the server has it.
            if models.contains(where: { $0.id == m }) { model = m }
        case let .chatDeleted(id):
            conversations.removeAll { $0.id == id }
            if chatID == id {
                chatID = nil
                bubbles = []
            }
        case let .chatsFailed(e):
            chatError = e
        case .replyDone:
            replying = false
            if let i = bubbles.indices.last { bubbles[i].streaming = false }
        case let .chatFailed(e, nothing):
            replying = false
            chatError = e
            if let i = bubbles.indices.last, !bubbles[i].mine {
                // Nothing came back: the error says why, and an empty reply
                // would only say it less well.
                if nothing && bubbles[i].text.isEmpty { bubbles.removeLast() } else { bubbles[i].streaming = false }
            }
        case .disconnected, .unpaired:
            if case .unpaired = event { serverText = "" }
            // The core ends a take on its link; the microphone ends here.
            if recording == .on || recording == .starting {
                microphone.stop()
                recording = .finishing
            }
            server = nil
            bubbles = []
            chatID = nil
            replying = false
            keyText = ""
            model = ""
            tab = .chat
        case let .unpairFailed(e):
            chatError = e
        case let .link(link, notice):
            if server != nil {
                chatError = "A Kvad link was opened. Disconnect in More to use it."
            } else {
                serverText = link
                keyText = ""
                connectError = ""
                connectNotice = notice
            }
        case let .linkRefused(e):
            if server != nil { chatError = e } else { connectError = e }
        case let .probeID(id):
            probeID = id
        case let .probeStarted(secs, status):
            probeRunning = true
            probeSecs = secs
            probeStatus = status
            probeLog = ""
            probeRows = []
            probeTotals = ""
            probePath = ""
            probeLast = ""
            probeElapsed = 0
            UIApplication.shared.isIdleTimerDisabled = true
        case let .probeStatus(text):
            probeStatus = text
        case let .probePath(text):
            probePath = text
        case let .probeLine(text):
            addProbeLine(text)
        case let .probePing(t):
            probeRows = t.rows ?? probeRows
            probeTotals = t.totals ?? probeTotals
            probeLast = t.last ?? probeLast
            probeElapsed = t.elapsed ?? probeElapsed
        case let .probeDone(t):
            probeEnded()
            probeRows = t.rows ?? probeRows
            probeTotals = t.totals ?? probeTotals
            probeElapsed = t.elapsed ?? probeElapsed
            probeStatus = t.status ?? ""
            if let line = t.line { addProbeLine(line) }
        case let .probeFailed(e):
            probeEnded()
            probeStatus = ""
            probeError = e
        case .speakStarted:
            guard recording == .starting else { return }
            do {
                #if DEBUG
                // A scripted take is the file's audio alone.
                if Script.env["KVAD_PHONE_SPEAK"] == nil {
                    try microphone.start(hearing)
                    speakRoute = Microphone.route()
                    print("kvad-mic \(speakTrial.isEmpty ? speakRoom.rawValue : speakTrial): \(speakRoute)")
                }
                #else
                try microphone.start(hearing)
                #endif
                recording = .on
                takeStarted = Date()
                #if DEBUG
                Script.takeStarted(self)
                #endif
            } catch {
                speakError = (error as? Microphone.Failure)?.why ?? "The microphone could not start: \(error.localizedDescription)"
                cancelTake()
            }
        case let .speakSettled(text, lag):
            settled += text
            self.lag = lag
        case let .speakPartial(text, lag):
            partial = text
            self.lag = lag
        case let .speakLink(state, why):
            speakLink = state
            if !why.isEmpty { speakNotice = why }
        case let .speakDone(take):
            recording = .idle
            takeStarted = nil
            takes.removeAll { $0.id == take.id }
            takes.insert(take, at: 0)
            settled = ""
            partial = ""
        case let .speakFailed(e, kept):
            if recording == .on || recording == .starting { microphone.stop() }
            recording = .idle
            takeStarted = nil
            speakError = e
            if kept { loadTakes() }
        case .speakCancelled:
            recording = .idle
            takeStarted = nil
            settled = ""
            partial = ""
        case let .takes(list):
            takes = list
        case let .takeDeleted(id):
            takes.removeAll { $0.id == id }
        case let .retranscribePiece(id, text):
            again[id, default: ""] += text
        case let .retranscribeDone(take):
            again[take.id] = nil
            if let i = takes.firstIndex(where: { $0.id == take.id }) { takes[i] = take }
        case let .retranscribeFailed(id, e):
            again[id] = nil
            speakError = e
        case let .error(e):
            chatError = e
        case .unknown:
            break
        }
    }

    private func modelsDone() {
        modelsLoading = false
        let waiters = modelsWaiters
        modelsWaiters = []
        waiters.forEach { $0.resume() }
    }

    private func probeEnded() {
        probeRunning = false
        UIApplication.shared.isIdleTimerDisabled = false
    }

    private func addProbeLine(_ line: String) {
        probeLog += probeLog.isEmpty ? line : "\n" + line
    }
}

#if DEBUG
/// For screenshots and a scripted look at each state, in debug builds only:
/// `simctl launch` passes `SIMCTL_CHILD_KVAD_PHONE_*` through as these.
/// `KVAD_PHONE_CONNECT` is a server to connect to at launch, `KVAD_PHONE_TAB`
/// the tab to open (0 to 4), `KVAD_PHONE_MODEL` the chat model, and
/// `KVAD_PHONE_SAY` a message to send once connected. `KVAD_PHONE_SPEAK` is
/// a WAV (16 kHz mono 16-bit) to record a take from once connected, played
/// in at real time in place of the microphone's, then stopped.
/// `KVAD_PHONE_RECORD` is a number of seconds to record from the microphone
/// once connected, then stop, with the screen kept on; `KVAD_PHONE_MIC` the
/// trial to hear with (`hearingTrials`).
@MainActor
enum Script {
    static let env = ProcessInfo.processInfo.environment

    static func atLaunch(_ m: AppModel) {
        if let url = env["KVAD_PHONE_CONNECT"] {
            m.serverText = url
            m.connect()
        }
    }

    static func afterConnect(_ m: AppModel) {
        if let tab = env["KVAD_PHONE_TAB"].flatMap(Int.init) {
            m.tab = [Tab.chat, .speak, .create, .models, .more][min(max(tab, 0), 4)]
        }
        if let model = env["KVAD_PHONE_MODEL"] { m.model = model }
        if let say = env["KVAD_PHONE_SAY"] {
            m.draft = say
            m.send()
        }
        if env["KVAD_PHONE_SPEAK"] != nil {
            Task { await m.startTake() }
        }
        if let seconds = env["KVAD_PHONE_RECORD"].flatMap(Double.init) {
            m.tab = .speak
            UIApplication.shared.isIdleTimerDisabled = true
            Task {
                await m.startTake()
                try? await Task.sleep(for: .seconds(seconds))
                m.stopTake()
                UIApplication.shared.isIdleTimerDisabled = false
            }
        }
    }

    /// The file's samples, at the pace they were spoken, in place of the
    /// microphone's.
    static func takeStarted(_ m: AppModel) {
        guard let path = env["KVAD_PHONE_SPEAK"], let wav = FileManager.default.contents(atPath: path) else { return }
        let samples = Self.samples(wav)
        DispatchQueue.global().async {
            let begun = Date()
            for (i, at) in stride(from: 0, to: samples.count, by: 1600).enumerated() {
                samples[at..<min(at + 1600, samples.count)].withUnsafeBufferPointer { kvad_core_audio($0.baseAddress, $0.count) }
                Thread.sleep(until: begun.addingTimeInterval(Double(i + 1) / 10))
            }
            DispatchQueue.main.async { MainActor.assumeIsolated { m.stopTake() } }
        }
    }

    static func samples(_ wav: Data) -> [Int16] {
        let bytes = [UInt8](wav)
        var at = 12
        while at + 8 <= bytes.count {
            let size = Int(bytes[at + 4]) | Int(bytes[at + 5]) << 8 | Int(bytes[at + 6]) << 16 | Int(bytes[at + 7]) << 24
            if bytes[at..<at + 4].elementsEqual("data".utf8) {
                let end = min(at + 8 + size, bytes.count)
                return stride(from: at + 8, to: end - 1, by: 2).map { Int16(bitPattern: UInt16(bytes[$0]) | UInt16(bytes[$0 + 1]) << 8) }
            }
            at += 8 + size + size % 2
        }
        return []
    }
}
#endif
