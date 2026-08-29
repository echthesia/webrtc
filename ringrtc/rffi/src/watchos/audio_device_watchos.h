/*
 * Copyright 2026 Signal Messenger, LLC
 * SPDX-License-Identifier: AGPL-3.0-only
 */

#ifndef RFFI_WATCHOS_AUDIO_DEVICE_WATCHOS_H__
#define RFFI_WATCHOS_AUDIO_DEVICE_WATCHOS_H__

#include <cstdint>
#include <memory>
#include <optional>

#include "api/audio/audio_device.h"
#include "api/scoped_refptr.h"
#include "api/sequence_checker.h"
#include "rtc_base/thread_annotations.h"

namespace webrtc {
namespace rffi {

// Defined in the .mm: everything that touches AVFAudio. Opaque here so that
// peer_connection_factory.cc -- a plain .cc -- can include this header.
class WatchAudioEngine;

// The watch's audio device module.
//
// iOS builds its ADM in the ObjC SDK (sdk/objc/native/src/audio/), on
// AudioUnit/VoiceProcessingIO. The watch has neither: no ObjC SDK in this
// build, and no AudioToolbox in the watchOS SDK at all. What it does have is
// AVFAudio, so this ADM is AVAudioEngine underneath -- an input tap for
// capture, an AVAudioSourceNode for playout.
//
// What crosses it is mono int16 at the engine's own rate, not at a rate this
// module declares. AudioTransport takes samples_per_sec on every call and
// resamples above us, so a route change is a new rate reported to the
// transport rather than a converter rebuilt underneath one -- the same shape
// as the iOS ADM's HandleSampleRateChange. Playout and capture rates are
// tracked separately, and audio already buffered at the old rate is dropped
// rather than played out at the new one, so a route change costs a gap and
// nothing else.
//
// It is C++ rather than the Rust-callback ADM (RingRTCAudioDeviceModule) that
// the desktop uses because there is no reason for a 10 ms frame to cross the
// FFI boundary a hundred times a second when nothing on the Rust side would
// do anything but forward it. On the watch the Rust side passes an
// RffiAudioConfig with a null rust_adm_borrowed, null callbacks and a null
// free_adm_cb, and this class is created instead.
//
// Threading, mirroring RingRTCAudioDeviceModule:
//   * The module is created, used and destroyed on WebRTC's worker thread;
//     `thread_checker_` enforces that.
//   * `RegisterAudioCallback` may only change the transport while stopped
//     (WebRtcVoiceEngine::Terminate honours that: StopPlayout, StopRecording,
//     then RegisterAudioCallback(nullptr)).
//   * Nothing in AudioDeviceModule is called from the render thread; the
//     render block only drains a lock-free ring.
//   * AVAudioEngine configuration changes and audio session interruptions are
//     observed, and arrive on whatever thread AVFAudio posts them from. They
//     set a flag and wake the module's pump thread, which is what rebuilds the
//     graph; a 250 ms poll remains as the fallback. The module still never
//     sets a category or activates the session -- observing an interruption is
//     not owning one.
//
// Note that WebRTC never calls `Init()` on this route (nothing in
// WebRtcVoiceEngine or AudioState does), so the engine sets itself up lazily
// from the first Start*; `Init()` is idempotent and optional.
class WatchAudioDeviceModule : public AudioDeviceModule {
 public:
  ~WatchAudioDeviceModule() override;

  static scoped_refptr<WatchAudioDeviceModule> Create();

  // Retrieve the currently utilized audio layer
  int32_t ActiveAudioLayer(AudioLayer* audio_layer) const override;

  // Full-duplex transportation of PCM audio
  int32_t RegisterAudioCallback(AudioTransport* audio_callback) override;

  // Main initialization and termination
  int32_t Init() override;
  // Final so that calling from the destructor is safe.
  int32_t Terminate() override final;
  bool Initialized() const override;

  // Device enumeration. The watch has exactly one route at a time and the
  // system owns it; there is nothing to enumerate or select.
  int16_t PlayoutDevices() override;
  int16_t RecordingDevices() override;
  int32_t PlayoutDeviceName(uint16_t index,
                            char name[kAdmMaxDeviceNameSize],
                            char guid[kAdmMaxGuidSize]) override;
  int32_t RecordingDeviceName(uint16_t index,
                              char name[kAdmMaxDeviceNameSize],
                              char guid[kAdmMaxGuidSize]) override;

  // Device selection
  int32_t SetPlayoutDevice(uint16_t index) override;
  int32_t SetPlayoutDevice(WindowsDeviceType device) override;
  int32_t SetRecordingDevice(uint16_t index) override;
  int32_t SetRecordingDevice(WindowsDeviceType device) override;

  // Audio transport initialization
  int32_t PlayoutIsAvailable(bool* available) override;
  int32_t InitPlayout() override;
  bool PlayoutIsInitialized() const override;
  int32_t RecordingIsAvailable(bool* available) override;
  int32_t InitRecording() override;
  bool RecordingIsInitialized() const override;

  // Audio transport control
  int32_t StartPlayout() override;
  int32_t StopPlayout() override;
  bool Playing() const override;
  int32_t StartRecording() override;
  int32_t StopRecording() override;
  bool Recording() const override;

  // Audio mixer initialization
  int32_t InitSpeaker() override;
  bool SpeakerIsInitialized() const override;
  int32_t InitMicrophone() override;
  bool MicrophoneIsInitialized() const override;

  // Speaker volume controls. watchOS has no per-app output volume; the
  // hardware Digital Crown owns it.
  int32_t SpeakerVolumeIsAvailable(bool* available) override;
  int32_t SetSpeakerVolume(uint32_t volume) override;
  int32_t SpeakerVolume(uint32_t* volume) const override;
  int32_t MaxSpeakerVolume(uint32_t* max_volume) const override;
  int32_t MinSpeakerVolume(uint32_t* min_volume) const override;

  // Microphone volume controls. Likewise not settable.
  int32_t MicrophoneVolumeIsAvailable(bool* available) override;
  int32_t SetMicrophoneVolume(uint32_t volume) override;
  int32_t MicrophoneVolume(uint32_t* volume) const override;
  int32_t MaxMicrophoneVolume(uint32_t* max_volume) const override;
  int32_t MinMicrophoneVolume(uint32_t* min_volume) const override;

  // Speaker mute control
  int32_t SpeakerMuteIsAvailable(bool* available) override;
  int32_t SetSpeakerMute(bool enable) override;
  int32_t SpeakerMute(bool* enabled) const override;

  // Microphone mute control
  int32_t MicrophoneMuteIsAvailable(bool* available) override;
  int32_t SetMicrophoneMute(bool enable) override;
  int32_t MicrophoneMute(bool* enabled) const override;

  // Stereo support. The transport format is mono, always.
  int32_t StereoPlayoutIsAvailable(bool* available) const override;
  int32_t SetStereoPlayout(bool enable) override;
  int32_t StereoPlayout(bool* enabled) const override;
  int32_t StereoRecordingIsAvailable(bool* available) const override;
  int32_t SetStereoRecording(bool enable) override;
  int32_t StereoRecording(bool* enabled) const override;

  // Playout delay
  int32_t PlayoutDelay(uint16_t* delay_ms) const override;

  // Only supported on Android.
  // AVAudioEngine's voice processing (setVoiceProcessingEnabled: on the
  // input node) is an echo canceller and an AGC in the audio unit itself,
  // and it is always on in this module. Declaring them built-in is what
  // makes WebRtcVoiceEngine turn its own AEC3 and software AGC off (see
  // ApplyOptions in media/engine/webrtc_voice_engine.cc) rather than run
  // them on already-cancelled audio. Noise suppression stays with the APM.
  // Enable(false) is refused: the unit's processing cannot be switched off
  // here, so the engine keeps its own if it ever asks for none.
  bool BuiltInAECIsAvailable() const override { return true; }
  bool BuiltInAGCIsAvailable() const override { return true; }
  bool BuiltInNSIsAvailable() const override { return false; }
  int32_t EnableBuiltInAEC(bool enable) override { return enable ? 0 : -1; }
  int32_t EnableBuiltInAGC(bool enable) override { return enable ? 0 : -1; }
  int32_t EnableBuiltInNS(bool enable) override { return -1; }

  int32_t GetPlayoutUnderrunCount() const override;

  std::optional<Stats> GetStats() const override { return std::nullopt; }

// WEBRTC_IOS is defined on the watch -- it builds as an iOS variant -- so
// these are pure virtual here. Only the ObjC SDK calls them, and it is not in
// this build.
#if defined(WEBRTC_IOS)
  int GetPlayoutAudioParameters(AudioParameters* params) const override {
    return -1;
  }
  int GetRecordAudioParameters(AudioParameters* params) const override {
    return -1;
  }
#endif  // WEBRTC_IOS

 protected:
  WatchAudioDeviceModule();

 private:
  // Ensures the module is used on the thread it was constructed on.
  SequenceChecker thread_checker_;

  const std::unique_ptr<WatchAudioEngine> engine_;

  bool initialized_ RTC_GUARDED_BY(&thread_checker_) = false;
  bool playout_initialized_ RTC_GUARDED_BY(&thread_checker_) = false;
  bool recording_initialized_ RTC_GUARDED_BY(&thread_checker_) = false;
  bool playing_ RTC_GUARDED_BY(&thread_checker_) = false;
  bool recording_ RTC_GUARDED_BY(&thread_checker_) = false;
};

}  // namespace rffi
}  // namespace webrtc

#endif  // RFFI_WATCHOS_AUDIO_DEVICE_WATCHOS_H__
