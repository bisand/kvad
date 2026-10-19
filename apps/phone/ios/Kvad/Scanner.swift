// Scan a pairing code: the QR code `kvad remote pair` shows, read in the
// app rather than by the Camera. VisionKit's DataScannerViewController where
// the phone has it, AVFoundation's metadata output where it does not. A code
// is only text handed to the core's link parser (`opened`), as a link the
// Camera opens is: one that parses fills in the Server field, and nothing
// pairs until Connect. A code is never opened as a URL, and nothing the
// camera sees is kept or sent.

import AVFoundation
import SwiftUI
import VisionKit

/// The camera, as far as Kvad may use it.
enum Camera {
    case asking, allowed, refused

    /// Asks for the camera the first time, and says what the answer was.
    static func ask() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: true
        case .notDetermined: await AVCaptureDevice.requestAccess(for: .video)
        default: false
        }
    }

    /// Which scanner this phone can run: VisionKit's, AVFoundation's, or
    /// none (the simulator, which has no camera).
    @MainActor static var scanner: Scanner? {
        if DataScannerViewController.isSupported { return .vision }
        return AVCaptureDevice.default(for: .video) == nil ? nil : .capture
    }

    enum Scanner { case vision, capture }
}

struct ScannerSheet: View {
    @Environment(AppModel.self) private var model
    @State private var camera = Camera.asking

    var body: some View {
        NavigationStack {
            Group {
                switch camera {
                case .asking:
                    ProgressView()
                case .refused:
                    ContentUnavailableView {
                        Label("No camera for Kvad", systemImage: "camera")
                    } description: {
                        Text("Kvad may not use the camera, so it cannot read the pairing code. Allow it in Settings, under Kvad, or scan the code with the Camera app, or paste its link.")
                    } actions: {
                        Button("Open Settings") {
                            if let url = URL(string: UIApplication.openSettingsURLString) { UIApplication.shared.open(url) }
                        }
                        .buttonStyle(.borderedProminent)
                        .foregroundStyle(Color("Ink"))
                    }
                case .allowed:
                    if let scanner = Camera.scanner {
                        viewfinder(scanner)
                    } else if scripted {
                        viewfinder(nil)
                    } else {
                        ContentUnavailableView {
                            Label("No camera to scan with", systemImage: "qrcode.viewfinder")
                        } description: {
                            Text("This device has no camera Kvad can use. Copy the link kvad remote pair shows, and paste it on the Connect screen.")
                        }
                    }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .navigationTitle("Scan a pairing code")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { model.scanning = false }
                }
            }
        }
        .task {
            if camera == .asking { camera = await Camera.ask() ? .allowed : .refused }
        }
    }

    /// Codes from `KVAD_PHONE_SCAN`, read into a black viewfinder where
    /// there is no camera: the simulator.
    private var scripted: Bool {
        #if DEBUG
        Script.env["KVAD_PHONE_SCAN"] != nil
        #else
        false
        #endif
    }

    private func viewfinder(_ scanner: Camera.Scanner?) -> some View {
        ZStack(alignment: .bottom) {
            Group {
                switch scanner {
                case .vision: VisionScanner { model.scanned($0) }
                case .capture: CaptureScanner { model.scanned($0) }
                case nil: Color.black
                }
            }
            .ignoresSafeArea(edges: .bottom)
            VStack(spacing: 6) {
                if !model.scanNote.isEmpty {
                    Text(model.scanNote).font(.subheadline.weight(.semibold))
                }
                Text("Point the camera at the QR code kvad remote pair shows on your Mac. Nothing the camera sees is kept or sent.")
                    .font(.footnote)
            }
            .multilineTextAlignment(.center)
            .padding(14)
            .frame(maxWidth: .infinity)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .padding(16)
        }
    }
}

/// VisionKit's scanner, QR codes only. Its highlights are drawn; a tap on
/// one does nothing.
struct VisionScanner: UIViewControllerRepresentable {
    let read: (String) -> Void

    func makeUIViewController(context: Context) -> DataScannerViewController {
        let scanner = DataScannerViewController(
            recognizedDataTypes: [.barcode(symbologies: [.qr])],
            qualityLevel: .balanced,
            recognizesMultipleItems: false,
            isHighFrameRateTrackingEnabled: false,
            isPinchToZoomEnabled: true,
            isGuidanceEnabled: true,
            isHighlightingEnabled: true
        )
        scanner.delegate = context.coordinator
        return scanner
    }

    // Scanning starts once the view is on screen, which it is by now.
    func updateUIViewController(_ scanner: DataScannerViewController, context: Context) {
        context.coordinator.read = read
        if !scanner.isScanning { try? scanner.startScanning() }
    }

    static func dismantleUIViewController(_ scanner: DataScannerViewController, coordinator: Coordinator) {
        scanner.stopScanning()
    }

    func makeCoordinator() -> Coordinator { Coordinator(read: read) }

    @MainActor final class Coordinator: NSObject, DataScannerViewControllerDelegate {
        var read: (String) -> Void

        init(read: @escaping (String) -> Void) { self.read = read }

        func dataScanner(_ dataScanner: DataScannerViewController, didAdd addedItems: [RecognizedItem], allItems: [RecognizedItem]) {
            for case let .barcode(code) in addedItems {
                read(code.payloadStringValue ?? "")
            }
        }

        // Again as it moves: one read while another code was with the core
        // was passed over. The model hands each text on once.
        func dataScanner(_ dataScanner: DataScannerViewController, didUpdate updatedItems: [RecognizedItem], allItems: [RecognizedItem]) {
            for case let .barcode(code) in updatedItems {
                read(code.payloadStringValue ?? "")
            }
        }
    }
}

/// AVFoundation's scanner, for a phone VisionKit's does not run on: the
/// back camera's picture, and the QR codes in it.
struct CaptureScanner: UIViewControllerRepresentable {
    let read: (String) -> Void

    func makeUIViewController(context: Context) -> CaptureViewController {
        let camera = CaptureViewController()
        camera.read = read
        return camera
    }

    func updateUIViewController(_ camera: CaptureViewController, context: Context) {
        camera.read = read
    }
}

final class CaptureViewController: UIViewController, AVCaptureMetadataOutputObjectsDelegate {
    var read: (String) -> Void = { _ in }
    private let capture = Capture()
    private var session: AVCaptureSession { capture.session }
    private var preview: AVCaptureVideoPreviewLayer?

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        let output = AVCaptureMetadataOutput()
        guard let device = AVCaptureDevice.default(for: .video),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input), session.canAddOutput(output) else { return }
        session.addInput(input)
        session.addOutput(output)
        output.setMetadataObjectsDelegate(self, queue: .main)
        output.metadataObjectTypes = [.qr]
        let preview = AVCaptureVideoPreviewLayer(session: session)
        preview.videoGravity = .resizeAspectFill
        view.layer.addSublayer(preview)
        self.preview = preview
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        preview?.frame = view.bounds
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        capture.run(true)
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        capture.run(false)
    }

    // Every frame with a code in it: the model hands each text to the core
    // once.
    nonisolated func metadataOutput(_ output: AVCaptureMetadataOutput, didOutput metadataObjects: [AVMetadataObject], from connection: AVCaptureConnection) {
        let codes = metadataObjects.compactMap { ($0 as? AVMetadataMachineReadableCodeObject).map { $0.stringValue ?? "" } }
        MainActor.assumeIsolated {
            codes.forEach(read)
        }
    }
}

/// The capture session, started and stopped on a queue of its own, since
/// `startRunning` waits for the camera. It is set up on the main thread
/// before the first start, and only started and stopped after, in order on
/// the one queue.
private final class Capture: @unchecked Sendable {
    let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "eu.kvad.phone.camera")

    func run(_ on: Bool) {
        queue.async { on ? self.session.startRunning() : self.session.stopRunning() }
    }
}
