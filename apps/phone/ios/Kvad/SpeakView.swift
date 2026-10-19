// Speak: one button to record, the words coming up as the Mac hears them,
// and the takes kept on this phone under it. docs/phone-speak-plan.md.

import SwiftUI

struct SpeakView: View {
    @Environment(AppModel.self) private var model
    @State private var choosing = false

    var body: some View {
        NavigationStack {
            Group {
                if model.recording == .idle {
                    TakesList()
                } else {
                    LiveText()
                }
            }
            .safeAreaInset(edge: .bottom) { RecordBar() }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .principal) {
                    Button { choosing = true } label: {
                        HStack(spacing: 4) {
                            Text(model.speakModelName).font(.headline).lineLimit(1)
                            Image(systemName: "chevron.down").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                        }
                        .foregroundStyle(.primary)
                    }
                    .disabled(model.recording != .idle)
                }
            }
            .sheet(isPresented: $choosing) { SpeakSheet() }
            .onAppear { model.loadTakes() }
        }
    }
}

/// The words of the take being recorded: settled as they are, the guess
/// after them greyed, which the next guess replaces.
private struct LiveText: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        ScrollViewReader { scroller in
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    Group {
                        if model.settled.isEmpty && model.partial.isEmpty {
                            Text(model.recording == .finishing ? "…" : "Listening…").foregroundStyle(.tertiary)
                        } else {
                            Text(model.settled.trimmingLeadingSpace) + Text(model.settled.isEmpty ? model.partial.trimmingLeadingSpace : model.partial).foregroundStyle(.secondary)
                        }
                    }
                    .font(.title3)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
                    Color.clear.frame(height: 1).id("end")
                }
                .padding(20)
            }
            // The words begin at the top, and the newest stay in view.
            .defaultScrollAnchor(.top, for: .alignment)
            .defaultScrollAnchor(.bottom, for: .sizeChanges)
            .onChange(of: model.settled + model.partial) {
                scroller.scrollTo("end", anchor: .bottom)
            }
        }
    }
}

/// The record button, and how the take is going beside it.
private struct RecordBar: View {
    @Environment(AppModel.self) private var model

    private var on: Bool { model.recording == .on || model.recording == .starting }

    var body: some View {
        VStack(spacing: 10) {
            if !model.speakError.isEmpty {
                Text(model.speakError)
                    .font(.footnote)
                    .foregroundStyle(.red)
                    .multilineTextAlignment(.center)
            }
            status
            Button {
                if on { model.stopTake() } else { Task { await model.startTake() } }
            } label: {
                ZStack {
                    Circle().strokeBorder(.secondary.opacity(0.5), lineWidth: 3).frame(width: 74, height: 74)
                    RoundedRectangle(cornerRadius: on ? 7 : 31, style: .continuous)
                        .fill(Color.red)
                        .frame(width: on ? 30 : 62, height: on ? 30 : 62)
                }
                .animation(.spring(duration: 0.3), value: on)
            }
            .buttonStyle(.plain)
            .disabled(model.recording == .finishing || model.speakModel.isEmpty)
            .accessibilityLabel(on ? "Stop recording" : "Record")
            .sensoryFeedback(.start, trigger: model.recording == .on)
        }
        .padding(.horizontal, 16)
        .padding(.top, 10)
        .padding(.bottom, 8)
        .frame(maxWidth: .infinity)
        .background(.bar)
    }

    @ViewBuilder private var status: some View {
        switch model.recording {
        case .idle:
            Text(model.speakModel.isEmpty ? "The Mac has no speech model. Pull one with kvad pull." : "Tap to record. The words come up as your Mac hears them.")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        case .starting:
            Text("Starting…").font(.footnote).foregroundStyle(.secondary)
        case .on:
            VStack(spacing: 2) {
                if let started = model.takeStarted {
                    TimelineView(.periodic(from: started, by: 1)) { context in
                        Text(duration(context.date.timeIntervalSince(started)))
                            .font(.title3.monospacedDigit().weight(.medium))
                    }
                }
                Text(link).font(.footnote).foregroundStyle(.secondary).multilineTextAlignment(.center)
            }
        case .finishing:
            Text(model.speakLink == "sending" ? "Sending the take to your Mac…" : "Writing out the last words…")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
    }

    private var link: String {
        if model.speakPaused { return "Paused while the phone is busy" }
        switch model.speakLink {
        case "live": return model.lag >= 2 ? String(format: "Live · %.0f s behind", model.lag) : "Live"
        case "reconnecting": return "Reconnecting… still recording"
        case "offline": return model.speakNotice.isEmpty ? "Recording. Sent whole when you stop." : model.speakNotice
        default: return "Connecting…"
        }
    }
}

/// The takes on this phone, newest first.
private struct TakesList: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        List {
            ForEach(model.takes) { take in
                NavigationLink(value: take) { TakeRow(take: take) }
                    .swipeActions {
                        Button("Delete", role: .destructive) { model.deleteTake(take) }
                    }
            }
        }
        .navigationDestination(for: Take.self) { take in TakeView(id: take.id) }
        .overlay {
            if model.takes.isEmpty {
                ContentUnavailableView {
                    Label("Speak", systemImage: "waveform")
                } description: {
                    Text("Record, and watch the words appear as your Mac hears them. The recording stays on this phone and your Mac.")
                }
            }
        }
        .refreshable { model.loadTakes() }
    }
}

private struct TakeRow: View {
    @Environment(AppModel.self) private var model
    let take: Take

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(take.date, format: .dateTime.day().month().hour().minute())
                    .font(.subheadline.weight(.semibold))
                Spacer()
                Text(duration(take.seconds)).font(.subheadline.monospacedDigit()).foregroundStyle(.secondary)
            }
            if model.again[take.id] != nil {
                Text("Transcribing again…").font(.subheadline).foregroundStyle(.secondary)
            } else if !take.transcribed {
                Label("Not transcribed", systemImage: "exclamationmark.circle").font(.subheadline).foregroundStyle(.orange)
            } else {
                // Its paragraphs run together, for two lines of words.
                Text(take.text.isEmpty ? "Nothing was heard." : take.text.replacingOccurrences(of: "\n\n", with: " "))
                    .font(.subheadline)
                    .foregroundStyle(take.text.isEmpty ? .tertiary : .secondary)
                    .lineLimit(2)
            }
        }
        .padding(.vertical, 2)
    }
}

/// One take, whole: its text to read, select, copy and share, its
/// recording, and transcribing it again.
private struct TakeView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss
    let id: String
    @State private var confirmDelete = false

    private var take: Take? { model.takes.first { $0.id == id } }

    var body: some View {
        if let take {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    if let again = model.again[take.id] {
                        Label("Transcribing again with \(model.speakModelName)…", systemImage: "arrow.triangle.2.circlepath")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                        if !again.isEmpty { Text(again.trimmingLeadingSpace).foregroundStyle(.secondary) }
                    } else if take.transcribed {
                        Text(take.text.isEmpty ? "Nothing was heard." : take.text)
                            .font(.body)
                            .textSelection(.enabled)
                    } else {
                        Text("This take has not been transcribed. Send it to your Mac to have it written out.")
                            .foregroundStyle(.secondary)
                    }
                    Text(about(take)).font(.footnote).foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(20)
            }
            .navigationTitle(Text(take.date, format: .dateTime.day().month().hour().minute()))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                // The tab bar is where a bottom bar would be.
                ToolbarItem(placement: .topBarTrailing) {
                    ShareLink(item: take.text) { Label("Share text", systemImage: "square.and.arrow.up") }
                        .disabled(take.text.isEmpty)
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Menu {
                        Button { UIPasteboard.general.string = take.text } label: { Label("Copy text", systemImage: "doc.on.doc") }
                            .disabled(take.text.isEmpty)
                        Button { model.transcribeAgain(take) } label: {
                            Label(take.transcribed ? "Transcribe again with \(model.speakModelName)" : "Transcribe with \(model.speakModelName)", systemImage: "arrow.triangle.2.circlepath")
                        }
                        .disabled(model.again[take.id] != nil || model.speakModel.isEmpty)
                        ShareLink(item: model.audioURL(take)) { Label("Share the recording", systemImage: "waveform") }
                        Divider()
                        Button("Delete", systemImage: "trash", role: .destructive) { confirmDelete = true }
                    } label: {
                        Label("More", systemImage: "ellipsis")
                    }
                }
            }
            .confirmationDialog("Delete this take?", isPresented: $confirmDelete, titleVisibility: .visible) {
                Button("Delete", role: .destructive) {
                    model.deleteTake(take)
                    dismiss()
                }
            } message: {
                Text("Its recording and its text are removed from this phone.")
            }
        }
    }

    private func about(_ t: Take) -> String {
        let model = t.model.isEmpty ? "the Mac's speech model" : ModelInfo(id: t.model, kind: "", resident: false).name
        let how = switch t.how {
        case "live": "Written out live by \(model)"
        case "file": "Written out from the whole recording by \(model)"
        default: "Recorded"
        }
        return "\(how), \(duration(t.seconds)) long."
    }
}

/// The speech models, to choose the one Speak uses, and the language.
private struct SpeakSheet: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        @Bindable var model = model
        NavigationStack {
            List {
                Section {
                    ForEach(model.speechModels) { m in
                        Button { model.speakModel = m.id } label: {
                            HStack {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(m.name).foregroundStyle(.primary)
                                    Text(m.resident ? "\(m.owner) · in memory" : m.owner).font(.subheadline).foregroundStyle(.secondary)
                                }
                                Spacer()
                                if m.id == model.speakModel { Image(systemName: "checkmark").fontWeight(.semibold) }
                            }
                        }
                    }
                } header: {
                    Text("Model")
                } footer: {
                    Text("Small keeps up live, a word showing about half a second after it is said. Large writes better text, but is slower: use it to transcribe a take again.")
                }
                Section("Language") {
                    Picker("Language", selection: $model.speakLanguage) {
                        ForEach(speakLanguages, id: \.code) { Text($0.name).tag($0.code) }
                    }
                }
            }
            .overlay {
                if model.speechModels.isEmpty {
                    ContentUnavailableView("No speech models", systemImage: "waveform", description: Text("Pull one on the Mac, such as kvad pull NbAiLab/nb-whisper-small."))
                }
            }
            .navigationTitle("Speak")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
            .refreshable { await model.refreshModels() }
        }
        .presentationDetents([.medium, .large])
    }
}

/// 75 s as `1:15`; an hour and more as `1:02:05`.
func duration(_ seconds: Double) -> String {
    let s = Int(seconds.rounded(.down))
    return s >= 3600 ? String(format: "%d:%02d:%02d", s / 3600, s / 60 % 60, s % 60) : String(format: "%d:%02d", s / 60, s % 60)
}

private extension String {
    /// Whisper's text begins with the space before its first word.
    var trimmingLeadingSpace: String { String(drop { $0 == " " }) }
}
