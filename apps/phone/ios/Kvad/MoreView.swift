// The server this phone is connected to, and what else there is. The rest
// of Kvad (jobs, training, settings…) comes here, screen by screen.

import SwiftUI

struct MoreView: View {
    @Environment(AppModel.self) private var model
    @State private var confirmUnpair = false

    var body: some View {
        NavigationStack {
            List {
                if let s = model.server {
                    Section("Server") {
                        LabeledContent("Address", value: s.label)
                        LabeledContent("Version", value: "kvad \(s.version)")
                        LabeledContent("Signed in as", value: s.who.isEmpty ? s.role : "\(s.who) (\(s.role))")
                        if !model.path.isEmpty {
                            LabeledContent("Path", value: model.path)
                        }
                        Button("Disconnect", role: .destructive) { model.disconnect() }
                        if s.paired {
                            Button("Unpair this phone", role: .destructive) { confirmUnpair = true }
                        }
                    }
                    if !model.chatError.isEmpty {
                        Text(model.chatError).font(.footnote).foregroundStyle(.red)
                    }
                }
                Section {
                    NavigationLink("Network test") { NetworkTestView() }
                } footer: {
                    Text("Which paths this network gives the phone to a relay-probe listener on the Mac: direct, on the LAN, or through the relay.")
                }
                Section("About") {
                    LabeledContent("Kvad for iPhone", value: Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "")
                    LabeledContent("Core", value: model.coreVersion)
                }
            }
            .navigationTitle("More")
            .confirmationDialog("Unpair this phone?", isPresented: $confirmUnpair, titleVisibility: .visible) {
                Button("Unpair", role: .destructive) { model.unpair() }
            } message: {
                Text("The server forgets this phone, and the phone forgets its keys. Pairing again takes a new link from kvad remote pair.")
            }
        }
    }
}
