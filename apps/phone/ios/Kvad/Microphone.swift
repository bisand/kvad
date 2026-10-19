// The microphone, for a take: AVAudioEngine's input, converted to what
// Whisper hears (16 kHz mono 16-bit) and handed to the core a tenth of a
// second at a time, on the audio thread. The core writes it to the take's
// file before it goes anywhere else (docs/phone-speak-plan.md).
//
// The audio session is active only while a take is being recorded, so iOS's
// microphone dot goes out when it ends. A call or Siri pauses the take, and
// it carries on after; AirPods put in or taken out change the input, and
// the converter is made again for it.

import AVFoundation

final class Microphone: @unchecked Sendable {
    private let engine = AVAudioEngine()
    private let target = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 16_000, channels: 1, interleaved: true)!
    private var observers: [NSObjectProtocol] = []
    /// Told on the main queue when a take is paused (`true`) or goes on.
    var onPause: (@MainActor (Bool) -> Void)?
    /// Told on the main queue when the input cannot go on.
    var onFailure: (@MainActor (String) -> Void)?

    /// Whether the app may record, asking the first time.
    static func allowed() async -> Bool {
        switch AVAudioApplication.shared.recordPermission {
        case .granted: return true
        case .denied: return false
        default: return await AVAudioApplication.requestRecordPermission()
        }
    }

    func start() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record, mode: .default)
        try session.setActive(true)
        try listen()
        let center = NotificationCenter.default
        observers = [
            center.addObserver(forName: AVAudioSession.interruptionNotification, object: session, queue: .main) { [weak self] note in
                let began = (note.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt).flatMap(AVAudioSession.InterruptionType.init) == .began
                self?.interrupted(began: began)
            },
            // A new input (AirPods in or out) stops the engine, at a format
            // the converter was not made for.
            center.addObserver(forName: .AVAudioEngineConfigurationChange, object: engine, queue: .main) { [weak self] _ in
                self?.again()
            },
        ]
    }

    func stop() {
        observers.forEach(NotificationCenter.default.removeObserver)
        observers = []
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    /// The tap on the input, at its own format, converted for the core.
    private func listen() throws {
        let input = engine.inputNode
        input.removeTap(onBus: 0)
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else { throw Failure("This iPhone has no microphone to record with.") }
        guard let converter = AVAudioConverter(from: format, to: target) else { throw Failure("The microphone's audio could not be converted.") }
        let convert = Convert(converter: converter, target: target)
        input.installTap(onBus: 0, bufferSize: AVAudioFrameCount(format.sampleRate / 10), format: format) { buffer, _ in
            convert.send(buffer)
        }
        engine.prepare()
        try engine.start()
    }

    private func interrupted(began: Bool) {
        if began {
            MainActor.assumeIsolated { onPause?(true) }
        } else {
            again()
        }
    }

    /// Recording again after an interruption or a change of input.
    private func again() {
        do {
            try AVAudioSession.sharedInstance().setActive(true)
            try listen()
            MainActor.assumeIsolated { onPause?(false) }
        } catch {
            let why = (error as? Failure)?.why ?? "The microphone could not start again: \(error.localizedDescription)"
            MainActor.assumeIsolated { onFailure?(why) }
        }
    }

    struct Failure: Error {
        let why: String
        init(_ why: String) { self.why = why }
    }
}

/// One input format's conversion, used on the audio thread alone.
private final class Convert: @unchecked Sendable {
    let converter: AVAudioConverter
    let target: AVAudioFormat

    init(converter: AVAudioConverter, target: AVAudioFormat) {
        self.converter = converter
        self.target = target
    }

    func send(_ buffer: AVAudioPCMBuffer) {
        let ratio = target.sampleRate / buffer.format.sampleRate
        let room = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 32
        guard let out = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: room) else { return }
        var given = false
        var error: NSError?
        converter.convert(to: out, error: &error) { _, status in
            if given {
                status.pointee = .noDataNow
                return nil
            }
            given = true
            status.pointee = .haveData
            return buffer
        }
        if error == nil, out.frameLength > 0, let samples = out.int16ChannelData?[0] {
            kvad_core_audio(samples, Int(out.frameLength))
        }
    }
}
