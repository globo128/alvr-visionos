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
    var linearVelocity: simd_float3 // world axes
    var angularVelocity: simd_float3 // world axes
    var receivedAt: Double // CACurrentMediaTime() at receipt; WorldPose.timestamp is firmware units
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
        }
        if !right {
            rightPose = nil
            rightButtons = nil
            rightPaused = false
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
                SurrealInputCache.shared.storePose(isLeft: pose.handedness == .left, SurrealPoseSnapshot(
                    worldFromController: pose.transform,
                    linearVelocity: pose.linearVelocity,
                    angularVelocity: pose.angularVelocity,
                    receivedAt: CACurrentMediaTime()
                ))
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
            for await event in session.stateUpdates {
                switch event {
                case .connection(let state):
                    let leftConnected = state == .leftConnected || state == .bothConnected
                    let rightConnected = state == .rightConnected || state == .bothConnected
                    SurrealInputCache.shared.setConnected(left: leftConnected, right: rightConnected)
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
                    }
                case .resumed(let hand):
                    if hand != .unspecified {
                        SurrealInputCache.shared.setPaused(isLeft: hand == .left, false)
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
    }

    func vibrate(isLeft: Bool, amplitude: Float, frequency: Float, duration: Double) {
        // ALVR frequently sends frequency == 0 meaning "controller default".
        session?.vibrate(isLeft ? .left : .right,
                         amplitude: amplitude,
                         frequency: frequency > 0 ? frequency : 100,
                         duration: duration)
    }
}
