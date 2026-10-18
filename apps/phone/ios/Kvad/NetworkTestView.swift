// The Network test: `relay-probe dial`, run from the phone itself, against
// a `relay-probe listen` on the Mac (tools/relay-probe). It says which path
// the phone gets, and how fast each is.

import SwiftUI

struct NetworkTestView: View {
    @Environment(AppModel.self) private var model

    private static let lengths = [(30, "30 s"), (60, "1 min"), (300, "5 min")]

    var body: some View {
        @Bindable var model = model
        Form {
            Section("Ticket") {
                TextField("kvadprobe:1:…, from relay-probe listen on the Mac", text: $model.probeTicket, axis: .vertical)
                    .font(.system(.footnote, design: .monospaced))
                    .lineLimit(2...6)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .disabled(model.probeRunning)
                Button("Paste from the Mac") { model.pasteTicket() }
                    .disabled(model.probeRunning)
            }
            Section {
                Picker("Length", selection: $model.probeSecs) {
                    ForEach(Self.lengths, id: \.0) { Text($0.1).tag($0.0) }
                }
                .pickerStyle(.segmented)
                .disabled(model.probeRunning)
                if model.probeRunning {
                    Button("Stop", role: .destructive) { model.stopProbe() }
                } else {
                    Button("Start") { model.startProbe() }
                        .disabled(model.probeTicket.isEmpty)
                }
            } footer: {
                if !model.probeError.isEmpty {
                    Text(model.probeError).foregroundStyle(.red)
                }
            }
            if model.probeRunning || !model.probeLog.isEmpty || !model.probeStatus.isEmpty {
                Section("Now") {
                    if !model.probeStatus.isEmpty {
                        Text(model.probeStatus)
                    }
                    if !model.probePath.isEmpty {
                        LabeledContent("Path", value: model.probePath)
                    }
                    if !model.probeLast.isEmpty {
                        LabeledContent("Last ping", value: model.probeLast)
                    }
                    ProgressView(value: min(model.probeElapsed, Double(model.probeSecs)), total: Double(model.probeSecs))
                }
            }
            if !model.probeRows.isEmpty {
                Section {
                    Grid(alignment: .trailing, horizontalSpacing: 12, verticalSpacing: 6) {
                        GridRow {
                            Text("path").gridColumnAlignment(.leading)
                            Text("pings")
                            Text("median")
                            Text("p90")
                            Text("max")
                        }
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                        ForEach(model.probeRows, id: \.path) { r in
                            GridRow {
                                Text(r.path).gridColumnAlignment(.leading)
                                Text("\(r.pings)")
                                Text(r.median)
                                Text(r.p90)
                                Text(r.max)
                            }
                            .font(.system(.footnote, design: .monospaced))
                        }
                    }
                } header: {
                    Text("Results, ms")
                } footer: {
                    Text(model.probeTotals)
                }
            }
            if !model.probeLog.isEmpty {
                Section("Log") {
                    Text(model.probeLog)
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                }
            }
            Section {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Probe id")
                    Text(model.probeID.isEmpty ? "…" : model.probeID)
                        .font(.system(.footnote, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
                Button("Copy id") { model.copyProbeID() }
                    .disabled(model.probeID.isEmpty)
            } header: {
                Text("This phone")
            } footer: {
                Text("The relay serves only the ids on its allow list. Add this one, once, and deploy the relay again.")
            }
        }
        .scrollDismissesKeyboard(.interactively)
        .navigationTitle("Network test")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear { model.openProbe() }
    }
}
