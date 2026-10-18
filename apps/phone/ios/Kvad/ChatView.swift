// The chat: the model is the title, and tapping it chooses; the reply
// streams in word by word; the composer rides on the keyboard.

import SwiftUI

struct ChatView: View {
    @Environment(AppModel.self) private var model
    @State private var choosing = false
    /// Whether the newest words are in view. While they are, the chat stays
    /// with them as they arrive; scrolled up to read, it stays put.
    @State private var atEnd = true

    var body: some View {
        NavigationStack {
            ScrollViewReader { scroller in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        ForEach(model.bubbles) { BubbleView(bubble: $0) }
                        if !model.chatError.isEmpty {
                            Text(model.chatError)
                                .font(.subheadline)
                                .foregroundStyle(.red)
                                .padding(.horizontal, 16)
                        }
                        Color.clear.frame(height: 1).id("end")
                    }
                    .padding(.vertical, 8)
                }
                .scrollDismissesKeyboard(.interactively)
                .defaultScrollAnchor(.bottom)
                .onScrollGeometryChange(for: Bool.self) { g in
                    g.contentOffset.y + g.containerSize.height >= g.contentSize.height - 40
                } action: { _, now in
                    atEnd = now
                }
                .onChange(of: model.bubbles.last?.text) {
                    if atEnd { scroller.scrollTo("end", anchor: .bottom) }
                }
                .onChange(of: model.bubbles.count) {
                    withAnimation { scroller.scrollTo("end", anchor: .bottom) }
                }
                .overlay {
                    if model.bubbles.isEmpty {
                        ContentUnavailableView {
                            Label("Ask anything", systemImage: "bubble.left.and.bubble.right")
                        } description: {
                            Text(model.model.isEmpty ? "Choose a model at the top to begin." : "It runs on your Mac. Nothing you write leaves the machines you own.")
                        }
                    }
                }
                .safeAreaInset(edge: .bottom) { Composer() }
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .principal) {
                    Button { choosing = true } label: {
                        HStack(spacing: 4) {
                            Text(model.modelName).font(.headline).lineLimit(1)
                            Image(systemName: "chevron.down").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                        }
                        .foregroundStyle(.primary)
                    }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button { model.newChat() } label: { Image(systemName: "square.and.pencil") }
                        .disabled(model.replying || model.bubbles.isEmpty)
                        .accessibilityLabel("New chat")
                }
            }
            .sheet(isPresented: $choosing) { ModelSheet() }
        }
    }
}

struct BubbleView: View {
    let bubble: Bubble

    var body: some View {
        if bubble.mine {
            Text(bubble.text)
                .foregroundStyle(Color("Ink"))
                .padding(.horizontal, 14)
                .padding(.vertical, 9)
                .background(Color.accentColor, in: RoundedRectangle(cornerRadius: 19, style: .continuous))
                .frame(maxWidth: .infinity, alignment: .trailing)
                .padding(.leading, 56)
                .padding(.trailing, 16)
                .textSelection(.enabled)
        } else {
            // Kvad's side has no bubble: a reply is read, not glanced at,
            // and gets the width of the screen.
            HStack(alignment: .top, spacing: 10) {
                MarkTile(size: 28).clipShape(Circle())
                Group {
                    if bubble.text.isEmpty && bubble.streaming {
                        Text("…")
                    } else {
                        Text(bubble.text + (bubble.streaming ? " ●" : ""))
                    }
                }
                .padding(.top, 3)
                .frame(maxWidth: .infinity, alignment: .leading)
                .textSelection(.enabled)
            }
            .padding(.horizontal, 16)
        }
    }
}

/// The message box, as tall as its lines up to five, and the send or stop
/// button beside it.
struct Composer: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        @Bindable var model = model
        HStack(alignment: .bottom, spacing: 8) {
            TextField("Message", text: $model.draft, axis: .vertical)
                .lineLimit(1...5)
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 19, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 19, style: .continuous).stroke(Color(.separator), lineWidth: 0.5))
            if model.replying {
                Button { model.stop() } label: {
                    Image(systemName: "stop.fill")
                        .font(.system(size: 14, weight: .bold))
                        .frame(width: 34, height: 34)
                        .background(.fill.secondary, in: Circle())
                }
                .foregroundStyle(.primary)
                .accessibilityLabel("Stop")
            } else {
                Button { model.send() } label: {
                    Image(systemName: "arrow.up")
                        .font(.system(size: 16, weight: .bold))
                        .frame(width: 34, height: 34)
                        .background(Color.accentColor, in: Circle())
                }
                .foregroundStyle(Color("Ink"))
                .disabled(model.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || model.model.isEmpty)
                .accessibilityLabel("Send")
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.bar)
    }
}

/// The chat models, to choose the one the chat asks.
struct ModelSheet: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List(model.chatModels) { m in
                Button {
                    model.model = m.id
                    dismiss()
                } label: {
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(m.name).foregroundStyle(.primary)
                            Text(m.resident ? "\(m.owner) · in memory" : m.owner)
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        if m.id == model.model {
                            Image(systemName: "checkmark").fontWeight(.semibold)
                        }
                    }
                }
            }
            .overlay {
                if model.chatModels.isEmpty {
                    ContentUnavailableView("No chat models", systemImage: "square.stack.3d.up", description: Text("Pull one on the Mac with kvad pull."))
                }
            }
            .navigationTitle("Chat models")
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
