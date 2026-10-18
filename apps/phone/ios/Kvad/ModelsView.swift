// What the server has, grouped by what each model is for, chat first.

import SwiftUI

struct ModelsView: View {
    @Environment(AppModel.self) private var model

    private static let order = ["chat", "speech", "image", "video", "embedding"]
    private static let titles = ["chat": "Chat", "speech": "Speech", "image": "Images", "video": "Video", "embedding": "Embedding"]

    private var groups: [(kind: String, models: [ModelInfo])] {
        let rank = { (k: String) in Self.order.firstIndex(of: k) ?? Self.order.count }
        return Dictionary(grouping: model.models, by: \.kind)
            .map { (kind: $0.key, models: $0.value.sorted { $0.id < $1.id }) }
            .sorted { (rank($0.kind), $0.kind) < (rank($1.kind), $1.kind) }
    }

    var body: some View {
        NavigationStack {
            List {
                ForEach(groups, id: \.kind) { group in
                    Section(Self.titles[group.kind] ?? "Other") {
                        ForEach(group.models) { m in
                            HStack {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(m.name)
                                    if !m.owner.isEmpty {
                                        Text(m.owner).font(.subheadline).foregroundStyle(.secondary)
                                    }
                                }
                                Spacer()
                                if m.resident {
                                    Label("In memory", systemImage: "circle.fill")
                                        .labelStyle(InMemory())
                                }
                            }
                        }
                    }
                }
            }
            .overlay {
                if model.models.isEmpty {
                    ContentUnavailableView("No models yet", systemImage: "square.stack.3d.up", description: Text("Pull one on the Mac with kvad pull, or from the web UI."))
                }
            }
            .refreshable { await model.refreshModels() }
            .navigationTitle("Models")
        }
    }
}

/// "In memory", after a green dot.
private struct InMemory: LabelStyle {
    func makeBody(configuration: Configuration) -> some View {
        HStack(spacing: 6) {
            configuration.icon.font(.system(size: 8)).foregroundStyle(.green)
            configuration.title.foregroundStyle(.secondary)
        }
    }
}
