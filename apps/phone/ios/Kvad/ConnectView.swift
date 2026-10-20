// The first screen: which server, and the key if it wants one. A pairing
// link from `kvad remote pair`, scanned or pasted, pairs the phone at
// Connect; a paired `kvad://` server is reached through the tunnel from
// anywhere.

import SwiftUI

struct ConnectView: View {
    @Environment(AppModel.self) private var model
    @State private var testing = false
    @FocusState private var focused: Bool

    var body: some View {
        @Bindable var model = model
        ScrollView {
            VStack(spacing: 8) {
                MarkTile()
                    .padding(.top, 56)
                    .padding(.bottom, 12)
                Text("Kvad")
                    .font(.largeTitle.bold())
                Text("Your models, on your Mac, from your phone.")
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)

                VStack(spacing: 0) {
                    field("Server") {
                        TextField("192.168.1.20:5823 or kvad://…", text: $model.serverText)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                            .keyboardType(.URL)
                    }
                    Divider().padding(.leading, 16)
                    field("Key") {
                        SecureField("Not needed on your own Mac", text: $model.keyText)
                            .onSubmit { model.connect() }
                    }
                }
                .focused($focused)
                .disabled(model.connecting)
                .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                .padding(.top, 28)

                Text("To reach your Mac from anywhere, run kvad remote pair on it and scan its QR code, here or with the Camera, or paste its link. Or type the address kvad-serve listens on, for this network only. A paired phone keeps its keys in the Keychain, on this phone alone.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 16)
                    .frame(maxWidth: .infinity, alignment: .leading)

                // The camera is asked for here, at the first Scan, and not
                // before.
                Button {
                    focused = false
                    model.scanning = true
                } label: {
                    Label("Scan a pairing code", systemImage: "qrcode.viewfinder")
                }
                .disabled(model.connecting)
                .padding(.top, 8)
                Button("Paste a pairing link") { model.pasteLink() }
                    .disabled(model.connecting)
                    .padding(.vertical, 8)

                if !model.connectNotice.isEmpty && !model.connecting {
                    Text(model.connectNotice)
                        .font(.subheadline)
                        .multilineTextAlignment(.center)
                }
                Button {
                    focused = false
                    model.connect()
                } label: {
                    HStack(spacing: 10) {
                        if model.connecting { ProgressView().tint(Color("Ink")) }
                        Text(model.connecting ? "Connecting…" : "Connect").fontWeight(.semibold)
                    }
                    .frame(maxWidth: .infinity, minHeight: 34)
                }
                .buttonStyle(.borderedProminent)
                .foregroundStyle(Color("Ink"))
                .controlSize(.large)
                .disabled(model.connecting || model.serverText.isEmpty)
                .padding(.top, 8)

                if !model.connectError.isEmpty {
                    Text(model.connectError)
                        .font(.subheadline)
                        .foregroundStyle(.red)
                        .multilineTextAlignment(.center)
                }

                // The network test needs no server: on mobile data there is
                // none to connect to until the relay is in.
                Button("Network test") { testing = true }
                    .padding(.top, 24)
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 32)
        }
        .scrollDismissesKeyboard(.interactively)
        // A form bounces only when it is taller than the screen.
        .scrollBounceBehavior(.basedOnSize)
        .background(Color(.systemGroupedBackground))
        .sheet(isPresented: $model.scanning) {
            ScannerSheet()
        }
        .sheet(isPresented: $testing) {
            NavigationStack {
                NetworkTestView()
                    .toolbar {
                        ToolbarItem(placement: .confirmationAction) {
                            Button("Done") { testing = false }
                        }
                    }
            }
        }
    }

    private func field(_ label: String, @ViewBuilder _ input: () -> some View) -> some View {
        HStack(spacing: 12) {
            Text(label).frame(width: 80, alignment: .leading)
            input()
        }
        .padding(.horizontal, 16)
        .frame(minHeight: 44)
    }
}
