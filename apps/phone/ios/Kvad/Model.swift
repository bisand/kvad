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

    var chatModels: [ModelInfo] { models.filter { $0.kind == "chat" } }
    var modelName: String { model.isEmpty ? "Choose a model" : (ModelInfo(id: model, kind: "", resident: false).name) }
    private var modelsWaiters: [CheckedContinuation<Void, Never>] = []

    func start() {
        Core.start { [weak self] event in self?.handle(event) }
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
        Core.send("chat", ["model": model, "text": text])
    }

    func stop() {
        Core.send("stop")
    }

    func newChat() {
        Core.send("new_chat")
        bubbles = []
        chatError = ""
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
            serverText = c.name
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
            server = nil
            bubbles = []
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
/// `KVAD_PHONE_SAY` a message to send once connected.
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
    }
}
#endif
