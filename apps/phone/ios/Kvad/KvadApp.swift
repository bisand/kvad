// Kvad for iPhone: a client of a kvad-serve. SwiftUI for what the screen
// shows, the Rust core (Core.swift) for everything under it.
// docs/phone-swiftui-plan.md.

import SwiftUI

@main
struct KvadApp: App {
    @State private var model = AppModel()
    @Environment(\.scenePhase) private var phase

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(model)
                .tint(Color.accentColor)
                .onAppear { model.start() }
                // A `kvad://` link, from the Camera's QR scanner or a tap,
                // fills in the Server field and waits for Connect: pairing is
                // the person's act, never the link's.
                .onOpenURL { model.opened($0) }
        }
        .onChange(of: phase) { _, now in
            // Back in the foreground, maybe on another network: look for
            // paths again rather than wait for the old ones to time out.
            if now == .active { model.networkChanged() }
        }
    }
}

struct RootView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        @Bindable var model = model
        if model.server == nil {
            ConnectView()
        } else {
            TabView(selection: $model.tab) {
                ChatView()
                    .tabItem { Label("Chat", systemImage: "bubble.left.and.bubble.right") }
                    .tag(Tab.chat)
                SpeakView()
                    .tabItem { Label("Speak", systemImage: "waveform") }
                    .tag(Tab.speak)
                ComingSoon(title: "Create", symbol: "sparkles", text: "Images and video, made on your Mac and kept on it.")
                    .tabItem { Label("Create", systemImage: "sparkles") }
                    .tag(Tab.create)
                ModelsView()
                    .tabItem { Label("Models", systemImage: "square.stack.3d.up") }
                    .tag(Tab.models)
                MoreView()
                    .tabItem { Label("More", systemImage: "ellipsis") }
                    .tag(Tab.more)
            }
        }
    }
}

struct ComingSoon: View {
    let title: String
    let symbol: String
    let text: String

    var body: some View {
        NavigationStack {
            ContentUnavailableView {
                Label(title, systemImage: symbol)
            } description: {
                Text(text)
            } actions: {
                Text("Coming soon")
                    .font(.footnote.weight(.semibold))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 5)
                    .background(.fill.tertiary, in: Capsule())
            }
            .navigationTitle(title)
        }
    }
}

/// Kvad's mark, gold on its ink tile, as the app icon has it.
struct MarkTile: View {
    var size: CGFloat = 88

    var body: some View {
        Image("Mark")
            .resizable()
            .scaledToFit()
            .foregroundStyle(Color(red: 0.941, green: 0.702, blue: 0.353))
            .padding(size * 0.19)
            .frame(width: size, height: size)
            .background(Color("Ink"), in: RoundedRectangle(cornerRadius: size * 0.25, style: .continuous))
            .accessibilityHidden(true)
    }
}
