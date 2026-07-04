//
//  SurrealControllerManager.swift
//
// Bridges OpenSurreal Surreal Touch controllers into the ALVR input pipeline.
// The manager owns the app's single SurrealControllerSession on the main actor and
// pumps its async streams into SurrealInputCache, which the render thread reads
// synchronously each frame (WorldTracker) and the event thread queries when routing
// haptics (EventHandler).
//

import Foundation
import Combine
import QuartzCore
import simd
import OpenSurreal

struct SurrealPoseSnapshot {
    var worldFromController: simd_float4x4 // ARKit world space, same as originFromAnchorTransform
    var linearVelocity: simd_float3 // world axes; includes the recovered calibration-frame velocity
    var angularVelocity: simd_float3 // world axes
    // receivedAt is CACurrentMediaTime() when the pump stored the packet — that's after
    // the BLE link plus two main-actor hops, so it's only good for link liveness
    // (staleAfter). sampleTime is the firmware timestamp mapped onto the same clock by
    // SurrealClockSync; use it for prediction dt.
    var receivedAt: Double
    var sampleTime: Double
}

struct SurrealButtonsSnapshot {
    var primary = false
    var secondary = false
    var menu = false
    var stickClick = false
    var trigger: Float = 0.0
    var grip: Float = 0.0
    var stick = SIMD2<Float>(0.0, 0.0)
    var receivedAt: Double = 0.0
}

/// Maps firmware pose timestamps onto the host `CACurrentMediaTime()` clock.
///
/// `receivedAt − deviceTime` is the true clock offset plus a non-negative delivery
/// delay (BLE link + two main-actor hops), so the minimum over a sliding window
/// tracks the fastest-delivery path and converges on the true offset. This is the
/// one-way half of the manufacturer SDK's NTP-over-BLE sync — no firmware
/// cooperation needed.
struct SurrealClockSync {
    // Firmware ticks are nanoseconds (~9.5e6 ticks between packets at ~105 Hz, per
    // decoded captures from the manufacturer's own diagnostics logs).
    static let tickToSeconds = 1e-9
    // Min-filter horizon. Crystal drift over 3 s (~50 ppm) is ~0.15 ms — negligible.
    static let windowSeconds = 3.0
    // An offset jump this large means the controller rebooted (timestamps reset).
    static let resetThreshold = 0.5
    // A sample estimated older than this means the estimator itself is wrong (e.g. a
    // firmware revision changed the tick unit) — fall back to arrival time.
    static let maxCredibleAge = 0.25

    private var samples: [(receivedAt: Double, offset: Double)] = []

    mutating func sampleTime(deviceTimestamp: UInt64, receivedAt: Double) -> Double {
        let deviceTime = Double(deviceTimestamp) * Self.tickToSeconds
        let offset = receivedAt - deviceTime
        if let last = samples.last, abs(offset - last.offset) > Self.resetThreshold {
            samples.removeAll()
        }
        samples.append((receivedAt: receivedAt, offset: offset))
        samples.removeAll { receivedAt - $0.receivedAt > Self.windowSeconds }
        let minOffset = samples.min { $0.offset < $1.offset }!.offset
        let sampleTime = deviceTime + minOffset
        guard receivedAt - sampleTime < Self.maxCredibleAge else {
            return receivedAt
        }
        return sampleTime
    }

    mutating func reset() {
        samples.removeAll()
    }
}

/// Debug-only tracking diagnostics, aggregated and printed at 1 Hz per hand.
///
/// The one-step prediction residual — predict the previous packet forward to the
/// current packet's sample time, compare against what actually arrived — quantifies
/// end-to-end prediction quality with no server or game in the loop. Flip `enabled`
/// on to A/B any tuning change against these numbers.
struct SurrealTrackingDiagnostics {
    static let enabled = false

    private let label: String
    private var windowStart = 0.0
    private var packetCount = 0
    private var tickDeltaSum = 0.0
    private var arrivalDeltaMax = 0.0
    private var confidenceMin: Float = 1.0
    private var confidenceMax: Float = 0.0
    private var confidenceSum: Float = 0.0
    private var positionErrorSquaredSum: Float = 0.0
    private var orientationErrorSquaredSum: Float = 0.0
    private var residualCount = 0
    private var previous: SurrealPoseSnapshot? = nil
    private var previousDeviceTimestamp: UInt64 = 0

    init(label: String) {
        self.label = label
    }

    mutating func record(pose: WorldPose, snapshot: SurrealPoseSnapshot) {
        guard Self.enabled else { return }
        if let previous {
            packetCount += 1
            tickDeltaSum += Double(pose.timestamp &- previousDeviceTimestamp)
            arrivalDeltaMax = max(arrivalDeltaMax, snapshot.receivedAt - previous.receivedAt)
            confidenceMin = min(confidenceMin, pose.confidence)
            confidenceMax = max(confidenceMax, pose.confidence)
            confidenceSum += pose.confidence

            let dt = Float(snapshot.sampleTime - previous.sampleTime)
            if dt > 0, dt < 0.1 {
                let predictedPosition = previous.worldFromController.columns.3.asFloat3() + previous.linearVelocity * dt
                positionErrorSquaredSum += simd_length_squared(snapshot.worldFromController.columns.3.asFloat3() - predictedPosition)

                var predictedOrientation = simd_quaternion(previous.worldFromController)
                let angularSpeed = simd_length(previous.angularVelocity)
                if angularSpeed > 0 {
                    predictedOrientation = simd_quatf(angle: angularSpeed * dt, axis: previous.angularVelocity / angularSpeed) * predictedOrientation
                }
                var angle = (predictedOrientation.inverse * simd_quaternion(snapshot.worldFromController)).angle
                angle = min(angle, 2 * .pi - angle)
                orientationErrorSquaredSum += angle * angle
                residualCount += 1
            }
        } else {
            windowStart = snapshot.receivedAt
        }
        previous = snapshot
        previousDeviceTimestamp = pose.timestamp

        if snapshot.receivedAt - windowStart >= 1.0 {
            flush(at: snapshot.receivedAt)
        }
    }

    private mutating func flush(at now: Double) {
        if packetCount > 0 {
            let seconds = now - windowStart
            let rate = Double(packetCount) / seconds
            let meanTickDelta = tickDeltaSum / Double(packetCount)
            let meanConfidence = confidenceSum / Float(packetCount)
            let posRms = residualCount > 0 ? (positionErrorSquaredSum / Float(residualCount)).squareRoot() * 1000 : 0
            let rotRms = residualCount > 0 ? (orientationErrorSquaredSum / Float(residualCount)).squareRoot() * 180 / .pi : 0
            let delay = previous.map { $0.receivedAt - $0.sampleTime } ?? 0
            print(String(format: "[SurrealDiag %@] rate=%.1fHz tickDelta=%.3g arrivalGapMax=%.1fms delay=%.1fms conf=%.2f/%.2f/%.2f residual pos=%.2fmm rot=%.3fdeg",
                         label, rate, meanTickDelta, arrivalDeltaMax * 1000, delay * 1000,
                         confidenceMin, meanConfidence, confidenceMax, posRms, rotRms))
        }
        windowStart = now
        packetCount = 0
        tickDeltaSum = 0
        arrivalDeltaMax = 0
        confidenceMin = 1.0
        confidenceMax = 0.0
        confidenceSum = 0
        positionErrorSquaredSum = 0
        orientationErrorSquaredSum = 0
        residualCount = 0
    }
}

/// Per-hand conditioning between raw OpenSurreal world poses and the snapshots the
/// render thread consumes: firmware-clock alignment, plus recovery of the velocity
/// that wrist re-registration adds on top of the controller's own reported motion.
struct SurrealPoseConditioner {
    // While the wrist is authoritative, WorldPose.linearVelocity deliberately excludes
    // the calibration frame's own motion (the per-frame re-registration onto the
    // moving wrist), so arm sweeps under-report velocity. The finite-difference
    // velocity across packets includes that motion; the smoothed difference between
    // the two recovers it.
    static let residualTau = 0.15 // s, EWMA time constant
    static let residualClamp: Float = 1.5 // m/s, bounds post-pickup snap-window noise

    private var clock = SurrealClockSync()
    private var lastPosition: simd_float3? = nil
    private var lastSampleTime = 0.0
    private var residualVelocity = simd_float3()
    private var diagnostics: SurrealTrackingDiagnostics

    init(label: String) {
        diagnostics = SurrealTrackingDiagnostics(label: label)
    }

    mutating func snapshot(for pose: WorldPose, receivedAt: Double) -> SurrealPoseSnapshot {
        let sampleTime = clock.sampleTime(deviceTimestamp: pose.timestamp, receivedAt: receivedAt)

        let position = pose.position
        let dt = sampleTime - lastSampleTime
        if lastPosition == nil || dt <= 0 || dt > 0.1 {
            // First packet, clock reset/fallback, or a delivery gap — can't difference
            // across it.
            residualVelocity = simd_float3()
        } else {
            let differenced = (position - lastPosition!) / Float(dt)
            var residual = differenced - pose.linearVelocity
            let magnitude = simd_length(residual)
            if magnitude > Self.residualClamp {
                residual *= Self.residualClamp / magnitude
            }
            residualVelocity += (residual - residualVelocity) * Float(1 - exp(-dt / Self.residualTau))
        }
        lastPosition = position
        lastSampleTime = sampleTime

        let snapshot = SurrealPoseSnapshot(
            worldFromController: pose.transform,
            linearVelocity: pose.linearVelocity + residualVelocity,
            angularVelocity: pose.angularVelocity,
            receivedAt: receivedAt,
            sampleTime: sampleTime
        )
        diagnostics.record(pose: pose, snapshot: snapshot)
        return snapshot
    }

    /// Motion can't be differenced across a set-down or pickup; the clock offset is
    /// still valid, so keep it.
    mutating func resetMotion() {
        lastPosition = nil
        residualVelocity = simd_float3()
    }

    /// Full reset for disconnects — the next connection may be a different or
    /// rebooted controller.
    mutating func reset() {
        clock.reset()
        resetMotion()
    }
}

// Mailbox between the main-actor stream pumps and the render/event threads.
final class SurrealInputCache: @unchecked Sendable {
    static let shared = SurrealInputCache()
    // Poses count as live for this long after the last packet. A paused or silent
    // controller keeps reporting its last pose (parked, zero velocity) rather than
    // releasing the hand slot; only disconnecting releases it.
    static let staleAfter = 0.25

    private let lock = NSLock()
    private var leftPose: SurrealPoseSnapshot? = nil
    private var rightPose: SurrealPoseSnapshot? = nil
    private var leftButtons: SurrealButtonsSnapshot? = nil
    private var rightButtons: SurrealButtonsSnapshot? = nil
    private var leftPaused = false
    private var rightPaused = false
    private var leftConnected = false
    private var rightConnected = false
    // Battery charge as a 0...1 gauge (ALVR's alvr_send_battery contract), or nil
    // until the first reading. Valid while the hand is connected, independent of
    // pose liveness — battery still matters for a controller that's set down.
    private var leftBattery: Float? = nil
    private var rightBattery: Float? = nil

    // MARK: Writers (stream pumps)

    func storePose(isLeft: Bool, _ snapshot: SurrealPoseSnapshot) {
        lock.lock(); defer { lock.unlock() }
        // While paused, keep the set-down pose frozen. OpenSurreal keeps emitting
        // coasting poses for a parked controller, and those can drift — or swing
        // during a grab, before the resume verdict lands — and the parked
        // controller shouldn't reproduce that.
        if isLeft {
            if leftPaused { return }
            leftPose = snapshot
        } else {
            if rightPaused { return }
            rightPose = snapshot
        }
    }

    func storeButtons(isLeft: Bool, _ snapshot: SurrealButtonsSnapshot) {
        lock.lock(); defer { lock.unlock() }
        if isLeft { leftButtons = snapshot } else { rightButtons = snapshot }
    }

    /// Stores a battery reading. `gauge` is a 0...1 charge fraction.
    func storeBattery(isLeft: Bool, gauge: Float) {
        lock.lock(); defer { lock.unlock() }
        if isLeft { leftBattery = gauge } else { rightBattery = gauge }
    }

    func setPaused(isLeft: Bool, _ paused: Bool) {
        lock.lock(); defer { lock.unlock() }
        // Pause only gates poses; buttons keep working on a set-down controller.
        // Zero the held inputs at set-down so nothing stays pressed (e.g. a
        // half-pulled trigger) — the next real button packet restores live state.
        if isLeft {
            leftPaused = paused
            if paused { leftButtons = SurrealButtonsSnapshot(receivedAt: CACurrentMediaTime()) }
        } else {
            rightPaused = paused
            if paused { rightButtons = SurrealButtonsSnapshot(receivedAt: CACurrentMediaTime()) }
        }
    }

    func setConnected(left: Bool, right: Bool) {
        lock.lock(); defer { lock.unlock() }
        if !left {
            leftPose = nil
            leftButtons = nil
            leftPaused = false
            leftBattery = nil
        }
        if !right {
            rightPose = nil
            rightButtons = nil
            rightPaused = false
            rightBattery = nil
        }
        leftConnected = left
        rightConnected = right
    }

    func clearAll() {
        lock.lock(); defer { lock.unlock() }
        leftPose = nil; rightPose = nil
        leftButtons = nil; rightButtons = nil
        leftPaused = false; rightPaused = false
        leftConnected = false; rightConnected = false
        leftBattery = nil; rightBattery = nil
    }

    // MARK: Readers (render/event threads)

    func pose(isLeft: Bool) -> SurrealPoseSnapshot? {
        lock.lock(); defer { lock.unlock() }
        guard isLeft ? leftConnected : rightConnected,
              var snapshot = isLeft ? leftPose : rightPose else {
            return nil
        }
        // A set-down (paused) or momentarily silent controller keeps reporting its
        // last pose with zeroed velocities — parked in place. Releasing the slot
        // instead would let the hand-tracking fallback drive the virtual controller
        // around with the empty hand.
        if (isLeft ? leftPaused : rightPaused) || CACurrentMediaTime() - snapshot.receivedAt >= Self.staleAfter {
            snapshot.linearVelocity = simd_float3()
            snapshot.angularVelocity = simd_float3()
        }
        return snapshot
    }

    func buttons(isLeft: Bool) -> SurrealButtonsSnapshot? {
        lock.lock(); defer { lock.unlock() }
        guard isLeft ? leftConnected : rightConnected else { return nil }
        return isLeft ? leftButtons : rightButtons
    }

    /// Latest battery charge (0...1) for a connected hand, or nil if that hand isn't a
    /// connected Surreal controller or hasn't reported yet.
    func battery(isLeft: Bool) -> Float? {
        lock.lock(); defer { lock.unlock() }
        guard isLeft ? leftConnected : rightConnected else { return nil }
        return isLeft ? leftBattery : rightBattery
    }

    /// Live tracking only — false while parked (paused) or stale, so e.g. haptics
    /// don't buzz a controller lying on a table.
    func isActive(isLeft: Bool) -> Bool {
        lock.lock(); defer { lock.unlock() }
        guard isLeft ? leftConnected : rightConnected,
              !(isLeft ? leftPaused : rightPaused),
              let snapshot = isLeft ? leftPose : rightPose else {
            return false
        }
        return CACurrentMediaTime() - snapshot.receivedAt < Self.staleAfter
    }
}

@MainActor
final class SurrealControllerManager: ObservableObject {
    static let shared = SurrealControllerManager()

    @Published private(set) var session: SurrealControllerSession? = nil
    private var pumpTasks: [Task<Void, Never>] = []
    private var leftConditioner = SurrealPoseConditioner(label: "L")
    private var rightConditioner = SurrealPoseConditioner(label: "R")

    private init() {}

    /// Creates the session and starts pumping its streams. Idempotent. The first
    /// call constructs a CBCentralManager, which triggers the Bluetooth permission
    /// prompt — only call once the user has opted into Surreal controllers.
    func start() {
        guard session == nil else { return }
        let session = SurrealControllerSession()
        self.session = session

        pumpTasks.append(Task {
            for await pose in session.worldPoseUpdates {
                guard pose.handedness != .unspecified else { continue }
                let isLeft = pose.handedness == .left
                let snapshot = isLeft
                    ? self.leftConditioner.snapshot(for: pose, receivedAt: CACurrentMediaTime())
                    : self.rightConditioner.snapshot(for: pose, receivedAt: CACurrentMediaTime())
                SurrealInputCache.shared.storePose(isLeft: isLeft, snapshot)
            }
        })

        pumpTasks.append(Task {
            for await update in session.buttonUpdates {
                guard update.handedness != .unspecified else { continue }
                SurrealInputCache.shared.storeButtons(isLeft: update.handedness == .left, SurrealButtonsSnapshot(
                    primary: update.primaryButton,
                    secondary: update.secondaryButton,
                    menu: update.menuButton,
                    stickClick: update.joystickClick,
                    trigger: update.trigger,
                    grip: update.grip,
                    stick: update.joystick,
                    receivedAt: CACurrentMediaTime()
                ))
            }
        })

        pumpTasks.append(Task {
            for await update in session.batteryUpdates {
                guard update.handedness != .unspecified else { continue }
                // OpenSurreal reports 0...100 %; ALVR's alvr_send_battery wants a
                // 0...1 gauge.
                SurrealInputCache.shared.storeBattery(isLeft: update.handedness == .left,
                                                      gauge: Float(update.level) / 100.0)
            }
        })

        pumpTasks.append(Task {
            for await event in session.stateUpdates {
                switch event {
                case .connection(let state):
                    let leftConnected = state == .leftConnected || state == .bothConnected
                    let rightConnected = state == .rightConnected || state == .bothConnected
                    SurrealInputCache.shared.setConnected(left: leftConnected, right: rightConnected)
                    if !leftConnected { self.leftConditioner.reset() }
                    if !rightConnected { self.rightConditioner.reset() }
                    if leftConnected || rightConnected {
                        // Safe to call repeatedly; the OpenSurreal hand-tracking
                        // session idles until an immersive space is open.
                        Task { await session.startSpatialTracking() }
                        if !ALVRClientApp.gStore.settings.surrealControllersEnabled {
                            ALVRClientApp.gStore.settings.surrealControllersEnabled = true
                            try? ALVRClientApp.gStore.save(settings: ALVRClientApp.gStore.settings)
                        }
                    }
                    else if state == .disconnected {
                        session.stopSpatialTracking()
                    }
                case .paused(let hand):
                    if hand != .unspecified {
                        SurrealInputCache.shared.setPaused(isLeft: hand == .left, true)
                        if hand == .left { self.leftConditioner.resetMotion() } else { self.rightConditioner.resetMotion() }
                    }
                case .resumed(let hand):
                    if hand != .unspecified {
                        SurrealInputCache.shared.setPaused(isLeft: hand == .left, false)
                        // Pickup starts the reacquisition snap window — per-frame
                        // calibration jumps that would pollute the residual estimate.
                        if hand == .left { self.leftConditioner.resetMotion() } else { self.rightConditioner.resetMotion() }
                    }
                }
            }
        })
    }

    /// Tears everything down. The session can't be restarted once stopped, so a
    /// fresh one is created on the next start().
    func stop() {
        for task in pumpTasks { task.cancel() }
        pumpTasks.removeAll()
        session?.stop()
        session = nil
        SurrealInputCache.shared.clearAll()
        leftConditioner.reset()
        rightConditioner.reset()
    }

    func vibrate(isLeft: Bool, amplitude: Float, frequency: Float, duration: Double) {
        // ALVR frequently sends frequency == 0 meaning "controller default".
        session?.vibrate(isLeft ? .left : .right,
                         amplitude: amplitude,
                         frequency: frequency > 0 ? frequency : 100,
                         duration: duration)
    }
}
