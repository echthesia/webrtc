/*
 * Copyright 2026 Signal Messenger, LLC
 * SPDX-License-Identifier: AGPL-3.0-only
 */

#include "rffi/src/watchos/audio_device_watchos.h"

#import <AVFAudio/AVFAudio.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "api/make_ref_counted.h"
#include "api/units/time_delta.h"
#include "rtc_base/event.h"
#include "rtc_base/logging.h"
#include "rtc_base/platform_thread.h"
#include "rtc_base/synchronization/mutex.h"
#include "rtc_base/thread_annotations.h"
#include "rtc_base/time_utils.h"

#define TRACE_LOG RTC_LOG(LS_VERBOSE) << "WatchAudioDeviceModule::" << __func__

namespace webrtc {
namespace rffi {

namespace {

// What this module hands the AudioTransport is mono int16 -- in both
// directions, always. What it does *not* fix is the rate.
//
// AudioTransport takes samples_per_sec on every call and resamples above us
// (AudioTransportImpl keeps a PushResampler in each direction), so the honest
// thing to report is the rate the hardware is actually running at, which is
// also what iOS's ADM does when a route change moves it. Declaring a fixed
// rate and converting to it would only put a second resampler under the one
// WebRTC already has, and -- worse -- would leave converted-at-the-old-rate
// audio in the rings across a route change. So there is no rate conversion in
// this module at all: only int16 <-> float32 and the channel splat/downmix.
constexpr size_t kChannels = 1;
constexpr size_t kBytesPerFrame = kChannels * sizeof(int16_t);  // 2

// A 10 ms block is rate/100 frames, so a rate that is not a whole multiple of
// 100 has no whole-frame block; such a rate is refused rather than
// approximated. Nothing we care about is one -- 48000, 44100, 24000 and 16000
// all divide. kMaxSampleRate is what sizes the fixed 10 ms scratch buffers,
// and is likewise a refusal rather than a truncation.
constexpr int kMaxSampleRate = 96000;
constexpr size_t kMaxFramesPer10ms = kMaxSampleRate / 100;  // 960

// How far ahead of the render block the pump thread keeps the playout ring:
// deep enough that a late wakeup does not underrun, shallow enough not to add
// audible latency to a call.
constexpr double kPlayoutTargetSeconds = 0.030;

// Ring capacities, allocated once: half a second at the highest rate we
// accept. Both rings hold mono frames at the engine's current rate -- playout
// float32, capture int16 -- and both are cleared when that rate changes.
constexpr size_t kPlayoutRingFrames = kMaxSampleRate / 2;
constexpr size_t kCaptureRingFrames = kMaxSampleRate / 2;

// Largest render block, or largest tap buffer chunk, we will handle at once.
constexpr size_t kMaxRenderFrames = 8192;

// Pump cadence, and how often it re-checks a graph that is not running --
// which covers both "the audio session was not live yet" and "an interruption
// stopped the engine". The notifications (see AddObservers) make the common
// cases immediate; this is the fallback for anything that posts none.
constexpr int kPumpIntervalMs = 5;
constexpr int64_t kHealthCheckIntervalMs = 250;

// Most 10 ms blocks the pump will move in one direction in one wakeup, so a
// stall in one direction cannot starve the other.
constexpr int kMaxBlocksPerWakeup = 8;

std::string DescribeError(NSError* error) {
  if (error == nil) {
    return "none";
  }
  const char* text = error.localizedDescription.UTF8String;
  return text != nullptr ? std::string(text) : std::string("unknown");
}

// The only sample conversion left in the module: the hardware speaks float32,
// the transport speaks int16.
inline float Int16ToFloat(int16_t sample) {
  return static_cast<float>(sample) * (1.0f / 32768.0f);
}

inline int16_t FloatToInt16(float sample) {
  const float scaled = sample * 32768.0f;
  if (scaled >= 32767.0f) {
    return 32767;
  }
  if (scaled <= -32768.0f) {
    return -32768;
  }
  return static_cast<int16_t>(scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f);
}

// Whether a 10 ms block at `rate` is a whole number of frames and fits the
// fixed scratch buffers. Logs the refusal; the caller does not start that
// direction.
bool RateIsUsable(double rate, const char* direction) {
  const int hz = static_cast<int>(rate);
  if (static_cast<double>(hz) != rate || hz <= 0 || hz % 100 != 0 ||
      hz > kMaxSampleRate) {
    RTC_LOG(LS_ERROR) << "WatchAudioEngine: refusing " << direction << " rate "
                      << rate << " Hz -- needs a whole 10 ms block at no more "
                      << "than " << kMaxSampleRate << " Hz";
    return false;
  }
  return true;
}

// A single-producer/single-consumer ring. Allocation happens once, in the
// constructor; Read and Write are wait-free, which is what makes them legal
// inside an AVAudioSourceNode render block.
template <typename T>
class SpscRing {
 public:
  explicit SpscRing(size_t capacity)
      : buf_(capacity + 1), capacity_(capacity + 1) {}

  size_t Readable() const {
    const size_t r = read_.load(std::memory_order_relaxed);
    const size_t w = write_.load(std::memory_order_acquire);
    return (w >= r) ? (w - r) : (capacity_ - r + w);
  }

  size_t Writable() const { return capacity_ - 1 - Readable(); }

  // Producer side. Returns the number of elements actually written.
  size_t Write(const T* src, size_t count) {
    const size_t w = write_.load(std::memory_order_relaxed);
    const size_t n = std::min(count, Writable());
    const size_t first = std::min(n, capacity_ - w);
    std::memcpy(buf_.data() + w, src, first * sizeof(T));
    if (n > first) {
      std::memcpy(buf_.data(), src + first, (n - first) * sizeof(T));
    }
    write_.store((w + n) % capacity_, std::memory_order_release);
    return n;
  }

  // Consumer side. Returns the number of elements actually read.
  size_t Read(T* dst, size_t count) {
    const size_t r = read_.load(std::memory_order_relaxed);
    const size_t n = std::min(count, Readable());
    const size_t first = std::min(n, capacity_ - r);
    std::memcpy(dst, buf_.data() + r, first * sizeof(T));
    if (n > first) {
      std::memcpy(dst + first, buf_.data(), (n - first) * sizeof(T));
    }
    read_.store((r + n) % capacity_, std::memory_order_release);
    return n;
  }

  // Drops everything in the ring. Unlike Read and Write this is NOT safe
  // against a live counterpart: it moves both cursors, so the caller has to
  // have quiesced *both* ends first. See EnsureGraph and TeardownGraph, the
  // only callers: the pump end is the calling thread (or joined), and the
  // other end is the engine, stopped (playout) or locked out (capture).
  void Clear() {
    read_.store(0, std::memory_order_relaxed);
    write_.store(0, std::memory_order_release);
  }

 private:
  std::vector<T> buf_;
  const size_t capacity_;
  std::atomic<size_t> read_{0};
  std::atomic<size_t> write_{0};
};

// What the notification blocks capture, instead of `this`.
//
// AVFAudio posts on whatever thread it likes, and `removeObserver:` does not
// wait for a block that has already started running on another one -- so a
// block that dereferenced the engine directly could outlive it by exactly the
// window that matters. The blocks hold a shared_ptr to one of these instead;
// teardown calls Detach, which takes the same lock the block does, so when
// Detach returns no block is inside the handler and none ever will be again.
//
// (The render block, by contrast, does capture raw `this`: its lifetime is
// guaranteed structurally, by detaching the source node from a stopped engine
// before the module goes away.)
class EngineNotice {
 public:
  using Handler = std::function<void(const char*, bool)>;

  void Attach(Handler handler) {
    MutexLock lock(&mutex_);
    handler_ = std::move(handler);
  }

  void Detach() {
    MutexLock lock(&mutex_);
    handler_ = nullptr;
  }

  void Post(const char* reason, bool rebuild_now) {
    MutexLock lock(&mutex_);
    if (handler_) {
      handler_(reason, rebuild_now);
    }
  }

 private:
  Mutex mutex_;
  Handler handler_ RTC_GUARDED_BY(mutex_);
};

}  // namespace

// Everything that touches AVFAudio.
//
// The graph is the one the hardware spike (spikes/watch-call, AudioProbe)
// established as the only one that makes sound on a watch under voice
// processing:
//
//   AVAudioSourceNode(engine output format) --> engine.outputNode
//   engine.inputNode --(tap)--> us
//
// with `setVoiceProcessingEnabled:` set on the input node *before* any node is
// wired, because enabling it renegotiates the IO formats. Routing the source
// through `mainMixerNode` instead of straight to `outputNode` renders silence
// under voice processing -- measured on hardware, so the mixer is not in this
// graph.
//
// Rates. The engine's own IO rates are what the transport is told, per call,
// and playout and capture are tracked separately because nothing guarantees
// they are equal. A 10 ms block is therefore rate/100 frames rather than a
// constant, and a route change is a new rate rather than a new converter.
//
// The audio session is not ours. LiveCommunicationKit sets the category and
// activates it; this module never calls setCategory: or setActive:, and has to
// cope with being started before the session is live -- which presents as an
// output format with a zero sample rate. That is logged, not asserted, and the
// pump thread retries. It does *observe* the session's interruptions, which is
// not the same as owning it.
//
// Locks. `graph_mutex_` covers the engine and the formats it was built
// against; the render block never takes it (it only drains a lock-free ring).
// `capture_mutex_` is taken by the input tap for its whole body and by a
// rebuild while it clears the capture ring -- which is what makes that clear
// safe against a tap already in flight, and what makes it safe to call
// `removeTapOnBus:` while holding `graph_mutex_`.
class WatchAudioEngine {
 public:
  WatchAudioEngine()
      : notice_(std::make_shared<EngineNotice>()),
        playout_ring_(kPlayoutRingFrames),
        capture_ring_(kCaptureRingFrames),
        play_int16_(kMaxFramesPer10ms * kChannels),
        play_float_(kMaxFramesPer10ms * kChannels),
        record_int16_(kMaxFramesPer10ms * kChannels),
        capture_int16_(kMaxRenderFrames),
        render_scratch_(kMaxRenderFrames) {
    notice_->Attach([this](const char* reason, bool rebuild_now) {
      OnSessionEvent(reason, rebuild_now);
    });
  }

  ~WatchAudioEngine() {
    Stop(/*playout=*/true, /*recording=*/true);
    // Belt and braces: Stop has already removed the observers, and this makes
    // an in-flight block that got in first finish before the members below it
    // are destroyed.
    notice_->Detach();
  }

  WatchAudioEngine(const WatchAudioEngine&) = delete;
  WatchAudioEngine& operator=(const WatchAudioEngine&) = delete;

  void SetTransport(AudioTransport* transport) {
    transport_.store(transport, std::memory_order_release);
  }

  // Starts the pump thread if it is not already running, and tries to bring
  // the engine up. Returns 0 even when the engine could not start: the session
  // may simply not be active yet, and the pump keeps trying.
  int32_t Start(bool playout, bool recording) {
    if (playout) {
      playout_wanted_.store(true, std::memory_order_release);
    }
    if (recording) {
      recording_wanted_.store(true, std::memory_order_release);
    }
    StartPump();
    // The graph is built by the pump thread, not here: a rebuild clears the
    // rings, and the pump is the playout ring's producer and the capture
    // ring's consumer, so the cheapest way to hold both still is for the pump
    // to be the one doing the rebuilding. Asking it and returning also keeps
    // WebRTC's worker thread off a wait for a thread that calls back into the
    // transport. The pump picks this up on its next tick, microseconds away.
    rebuild_requested_.store(true, std::memory_order_release);
    pump_wake_.Set();
    return 0;
  }

  int32_t Stop(bool playout, bool recording) {
    if (playout) {
      playout_wanted_.store(false, std::memory_order_release);
    }
    if (recording) {
      recording_wanted_.store(false, std::memory_order_release);
    }
    if (!playout_wanted_.load(std::memory_order_acquire) &&
        !recording_wanted_.load(std::memory_order_acquire)) {
      StopPump();
      TeardownGraph();
    }
    return 0;
  }

  bool playout_wanted() const {
    return playout_wanted_.load(std::memory_order_acquire);
  }

  bool recording_wanted() const {
    return recording_wanted_.load(std::memory_order_acquire);
  }

  uint16_t PlayoutDelayMs() const {
    const int rate = playout_rate_.load(std::memory_order_acquire);
    if (rate <= 0) {
      return 0;
    }
    // The ring holds frames at the playout rate, so this is right at whatever
    // rate the current route settled on.
    const double buffered_ms = 1000.0 * playout_ring_.Readable() / rate;
    return static_cast<uint16_t>(
        buffered_ms + output_latency_ms_.load(std::memory_order_acquire));
  }

  int32_t underrun_count() const {
    return underruns_.load(std::memory_order_relaxed);
  }

 private:
  // --------------------------------------------------------- notifications

  // Registered when the engine object is created, removed in TeardownGraph.
  // Neither block touches `this`: see EngineNotice.
  //
  // The session is still not ours. We observe the interruption; we do not set
  // a category, and on `ended` we re-attempt the graph rather than activating
  // anything -- the app and LiveCommunicationKit own activation. `object:nil`
  // on the interruption is deliberate: there is one session on a watch, and
  // this way the module never so much as names AVAudioSession.
  void AddObservers() RTC_EXCLUSIVE_LOCKS_REQUIRED(graph_mutex_) {
    NSNotificationCenter* center = [NSNotificationCenter defaultCenter];
    std::shared_ptr<EngineNotice> notice = notice_;
    config_observer_ = [center
        addObserverForName:AVAudioEngineConfigurationChangeNotification
                    object:engine_
                     queue:nil
                usingBlock:^(NSNotification* note) {
                  notice->Post("engine configuration changed",
                               /*rebuild_now=*/true);
                }];
    interruption_observer_ = [center
        addObserverForName:AVAudioSessionInterruptionNotification
                    object:nil
                     queue:nil
                usingBlock:^(NSNotification* note) {
                  NSNumber* type =
                      note.userInfo[AVAudioSessionInterruptionTypeKey];
                  const bool ended =
                      type != nil && type.unsignedIntegerValue ==
                                         AVAudioSessionInterruptionTypeEnded;
                  // On `began` the engine is stopped under us and a rebuild
                  // would only fail; marking it not-running is the whole job,
                  // and the health check covers a missed `ended`.
                  notice->Post(ended ? "audio session interruption ended"
                                     : "audio session interrupted",
                               /*rebuild_now=*/ended);
                }];
  }

  void RemoveObservers() RTC_EXCLUSIVE_LOCKS_REQUIRED(graph_mutex_) {
    NSNotificationCenter* center = [NSNotificationCenter defaultCenter];
    if (config_observer_ != nil) {
      [center removeObserver:config_observer_];
      config_observer_ = nil;
    }
    if (interruption_observer_ != nil) {
      [center removeObserver:interruption_observer_];
      interruption_observer_ = nil;
    }
  }

  // Called from a notification block, on whatever thread posted it, with
  // EngineNotice's lock held: no AVFAudio calls, no graph lock, nothing that
  // can block. The pump does the actual rebuild on its next tick, which is
  // now rather than up to kHealthCheckIntervalMs from now.
  void OnSessionEvent(const char* reason, bool rebuild_now) {
    RTC_LOG(LS_INFO) << "WatchAudioEngine: " << reason;
    engine_running_.store(false, std::memory_order_release);
    if (rebuild_now) {
      rebuild_requested_.store(true, std::memory_order_release);
    }
    pump_wake_.Set();
  }

  // ----------------------------------------------------------------- graph

  // Brings the graph up as far as it can, one step at a time, and returns true
  // if the engine is running when it returns. Every step is conditional, so
  // this is both the first-time build and the retry: StartRecording after
  // StartPlayout adds the tap, and a re-entry after an interruption or a route
  // change rebuilds what the new formats invalidated.
  //
  // Pump thread only, and that is load-bearing rather than incidental: a
  // rebuild clears the rings, SpscRing::Clear is not safe against a live
  // counterpart, and the pump is the playout ring's producer and the capture
  // ring's consumer. Running the rebuild *on* the pump is what holds those two
  // ends still. The other two ends are held explicitly -- the engine is
  // stopped before the playout ring is cleared, and capture_mutex_ is held
  // against the tap before the capture ring is. Everything else that wants a
  // rebuild (Start, a configuration change, an interruption) sets
  // `rebuild_requested_` and wakes the pump instead of calling this.
  bool EnsureGraph() {
    MutexLock lock(&graph_mutex_);

    if (engine_ == nil) {
      engine_ = [[AVAudioEngine alloc] init];
      AddObservers();
    }

    // Before any wiring: this is the AEC (and AGC), and turning it on
    // renegotiates the IO formats, so a graph built first would be built
    // against the wrong ones. It is the only echo canceller in the path:
    // the module reports it as built-in, and the voice engine turns AEC3 off
    // in response (see the header).
    if (!voice_processing_on_) {
      NSError* error = nil;
      if ([engine_.inputNode setVoiceProcessingEnabled:YES error:&error]) {
        voice_processing_on_ = [engine_.inputNode isVoiceProcessingEnabled];
        RTC_LOG(LS_INFO) << "WatchAudioEngine: voice processing enabled="
                         << voice_processing_on_;
      } else {
        RTC_LOG(LS_WARNING)
            << "WatchAudioEngine: setVoiceProcessingEnabled failed: "
            << DescribeError(error) << " -- no AEC, expect echo";
      }
    }

    // The hardware's own format, not a synthesized one. A zero sample rate is
    // how "the audio session is not active yet" presents; that is a retry, not
    // an error.
    AVAudioFormat* output_format = [engine_.outputNode inputFormatForBus:0];
    if (output_format == nil || output_format.sampleRate <= 0) {
      RTC_LOG(LS_INFO) << "WatchAudioEngine: output format has no sample rate "
                          "(audio session not active yet); will retry";
      return false;
    }

    // A route change (or an interruption) can come back with a different
    // format. Anything built against the old one has to go -- the source node
    // carries its format from construction, so it is recreated rather than
    // reconnected -- and so does anything buffered at the old rate.
    if (built_output_format_ == nil ||
        ![built_output_format_ isEqual:output_format]) {
      RTC_LOG(LS_INFO) << "WatchAudioEngine: output format "
                       << output_format.sampleRate << " Hz, "
                       << output_format.channelCount << " ch";
      // Stopping first is what quiesces the playout ring's *consumer*, the
      // render block; its producer is this very thread. With both ends held
      // the ring can be cleared, which is the point: the ~30 ms sitting in it
      // was produced for the old rate, and rendering that through a source
      // node running at the new one is an audible pitch blip on every route
      // change. A gap is the correct outcome, and is what iOS's
      // HandleSampleRateChange leaves behind too.
      if ([engine_ isRunning]) {
        [engine_ stop];
      }
      engine_running_.store(false, std::memory_order_release);
      if (source_node_ != nil) {
        [engine_ detachNode:source_node_];
        source_node_ = nil;
      }
      playout_ring_.Clear();
      // Remembered even when the rate is refused, so the refusal is logged
      // once per format rather than at every health check.
      built_output_format_ = output_format;
      const bool usable = RateIsUsable(output_format.sampleRate, "playout");
      const int rate = usable ? static_cast<int>(output_format.sampleRate) : 0;
      playout_rate_.store(rate, std::memory_order_release);
      playout_target_frames_.store(
          static_cast<size_t>(rate * kPlayoutTargetSeconds),
          std::memory_order_release);
    }
    if (playout_rate_.load(std::memory_order_acquire) <= 0) {
      return false;  // Refused above; nothing to render into.
    }

    if (source_node_ == nil) {
      WatchAudioEngine* self = this;
      source_node_ = [[AVAudioSourceNode alloc]
          initWithFormat:output_format
             renderBlock:^OSStatus(BOOL* is_silence,
                                   const AudioTimeStamp* timestamp,
                                   AVAudioFrameCount frame_count,
                                   AudioBufferList* output_data) {
               return self->Render(is_silence, frame_count, output_data);
             }];
      [engine_ attachNode:source_node_];
      // Straight to the output node. Never mainMixerNode: under voice
      // processing the mixer path is silent. Measured on hardware.
      [engine_ connect:source_node_
                    to:engine_.outputNode
                format:output_format];
    }

    // The capture side, tracked separately: the two rates are usually equal
    // and nothing promises they will be.
    AVAudioFormat* input_format = [engine_.inputNode outputFormatForBus:0];
    if (built_input_format_ != nil &&
        ![built_input_format_ isEqual:input_format]) {
      // Same story: the tap's format is fixed at install, and the ring holds
      // frames at the old rate.
      if (tap_installed_) {
        [engine_.inputNode removeTapOnBus:0];
        tap_installed_ = false;
      }
      built_input_format_ = nil;
      capture_rate_.store(0, std::memory_order_release);
      ClearCaptureRing();
    }
    if (built_input_format_ == nil) {
      if (input_format == nil || input_format.sampleRate <= 0) {
        RTC_LOG(LS_WARNING) << "WatchAudioEngine: input format has no sample "
                               "rate; capture unavailable, will retry";
      } else {
        RTC_LOG(LS_INFO) << "WatchAudioEngine: input format "
                         << input_format.sampleRate << " Hz, "
                         << input_format.channelCount << " ch";
        built_input_format_ = input_format;
        if (RateIsUsable(input_format.sampleRate, "capture")) {
          capture_rate_.store(static_cast<int>(input_format.sampleRate),
                              std::memory_order_release);
          ClearCaptureRing();
          WatchAudioEngine* self = this;
          [engine_.inputNode
              installTapOnBus:0
                   bufferSize:1024
                       format:input_format
                        block:^(AVAudioPCMBuffer* buffer, AVAudioTime* when) {
                          self->OnCapturedBuffer(buffer);
                        }];
          tap_installed_ = true;
        }
      }
    }

    output_latency_ms_.store(engine_.outputNode.presentationLatency * 1000.0,
                             std::memory_order_release);
    input_latency_ms_.store(engine_.inputNode.presentationLatency * 1000.0,
                            std::memory_order_release);

    if ([engine_ isRunning]) {
      engine_running_.store(true, std::memory_order_release);
      return true;
    }

    [engine_ prepare];
    NSError* error = nil;
    if (![engine_ startAndReturnError:&error]) {
      RTC_LOG(LS_WARNING) << "WatchAudioEngine: engine start failed: "
                          << DescribeError(error) << "; will retry";
      engine_running_.store(false, std::memory_order_release);
      return false;
    }
    RTC_LOG(LS_INFO) << "WatchAudioEngine: engine running";
    engine_running_.store(true, std::memory_order_release);
    return true;
  }

  // Only ever called with the pump stopped (Stop stops it first), so the
  // capture ring's consumer is gone, and the tap is removed and the engine
  // stopped before either ring is cleared.
  void TeardownGraph() {
    MutexLock lock(&graph_mutex_);
    engine_running_.store(false, std::memory_order_release);
    if (engine_ == nil) {
      return;
    }
    RemoveObservers();
    if (tap_installed_) {
      // Safe under graph_mutex_: the tap block takes capture_mutex_ only.
      [engine_.inputNode removeTapOnBus:0];
      tap_installed_ = false;
    }
    [engine_ stop];
    if (source_node_ != nil) {
      [engine_ detachNode:source_node_];
      source_node_ = nil;
    }
    built_output_format_ = nil;
    built_input_format_ = nil;
    engine_ = nil;
    voice_processing_on_ = false;
    playout_rate_.store(0, std::memory_order_release);
    capture_rate_.store(0, std::memory_order_release);
    // So that a restart does not begin by rendering whatever the last route
    // left behind, at whatever rate it left it at.
    playout_ring_.Clear();
    ClearCaptureRing();
    RTC_LOG(LS_INFO) << "WatchAudioEngine: engine stopped";
  }

  // The capture ring's producer is the tap, which holds capture_mutex_ for its
  // whole body; its consumer is the pump, which is either the caller itself
  // (EnsureGraph) or already joined (TeardownGraph, which Stop calls after
  // StopPump). Both ends held, so the clear is safe.
  void ClearCaptureRing() RTC_EXCLUSIVE_LOCKS_REQUIRED(graph_mutex_) {
    MutexLock lock(&capture_mutex_);
    capture_ring_.Clear();
  }

  bool EngineIsRunning() {
    MutexLock lock(&graph_mutex_);
    return engine_ != nil && [engine_ isRunning];
  }

  // --------------------------------------------------------------- playout

  // The render block. Runs on the audio HAL's thread: no allocation, no
  // locking, nothing but a ring read and a memcpy.
  OSStatus Render(BOOL* is_silence,
                  AVAudioFrameCount frame_count,
                  AudioBufferList* output_data) {
    const size_t frames = frame_count;

    if (frames > kMaxRenderFrames) {
      // Should not happen; if it does, silence beats a buffer overrun.
      for (UInt32 index = 0; index < output_data->mNumberBuffers; ++index) {
        if (output_data->mBuffers[index].mData != nullptr) {
          std::memset(output_data->mBuffers[index].mData, 0,
                      output_data->mBuffers[index].mDataByteSize);
        }
      }
      underruns_.fetch_add(1, std::memory_order_relaxed);
      *is_silence = YES;
      return noErr;
    }

    const size_t got = playout_ring_.Read(render_scratch_.data(), frames);
    if (got < frames) {
      std::memset(render_scratch_.data() + got, 0,
                  (frames - got) * sizeof(float));
      if (playout_wanted_.load(std::memory_order_relaxed)) {
        underruns_.fetch_add(1, std::memory_order_relaxed);
      }
    }

    if (output_data->mNumberBuffers == 1 &&
        output_data->mBuffers[0].mNumberChannels > 1) {
      // Interleaved: splat the mono frame across the channels.
      const UInt32 channels = output_data->mBuffers[0].mNumberChannels;
      float* out = static_cast<float*>(output_data->mBuffers[0].mData);
      const size_t writable =
          out == nullptr ? 0
                         : output_data->mBuffers[0].mDataByteSize /
                               (sizeof(float) * channels);
      const size_t count = std::min(frames, writable);
      for (size_t frame = 0; frame < count; ++frame) {
        for (UInt32 channel = 0; channel < channels; ++channel) {
          out[frame * channels + channel] = render_scratch_[frame];
        }
      }
    } else {
      // Planar, the usual case: the same mono block into every buffer.
      for (UInt32 index = 0; index < output_data->mNumberBuffers; ++index) {
        float* out = static_cast<float*>(output_data->mBuffers[index].mData);
        if (out == nullptr) {
          continue;
        }
        const size_t writable =
            output_data->mBuffers[index].mDataByteSize / sizeof(float);
        std::memcpy(out, render_scratch_.data(),
                    std::min(frames, writable) * sizeof(float));
      }
    }

    *is_silence = (got == 0) ? YES : NO;
    return noErr;
  }

  // Pulls one 10 ms block from the transport -- at the playout rate, which is
  // the engine's own -- and pushes it into the playout ring as float. Pump
  // thread only, and takes no lock: the rate is an atomic and the ring is
  // lock-free, and a rebuild cannot be running concurrently because a rebuild
  // is this same thread.
  bool PushOnePlayoutBlock() {
    AudioTransport* transport = transport_.load(std::memory_order_acquire);
    if (transport == nullptr) {
      return false;
    }
    const int rate = playout_rate_.load(std::memory_order_acquire);
    if (rate <= 0) {
      return false;
    }
    const size_t frames = static_cast<size_t>(rate) / 100;
    size_t samples_out = 0;
    int64_t elapsed_time_ms = -1;
    int64_t ntp_time_ms = -1;
    if (transport->NeedMorePlayData(
            frames, kBytesPerFrame, kChannels, static_cast<uint32_t>(rate),
            play_int16_.data(), samples_out, &elapsed_time_ms,
            &ntp_time_ms) != 0) {
      return false;
    }
    samples_out = std::min(samples_out, play_int16_.size());
    if (samples_out == 0) {
      return false;
    }
    for (size_t index = 0; index < samples_out; ++index) {
      play_float_[index] = Int16ToFloat(play_int16_[index]);
    }
    playout_ring_.Write(play_float_.data(), samples_out);
    return true;
  }

  // --------------------------------------------------------------- capture

  // The input tap. Runs on the engine's capture thread -- not the render
  // thread, so a lock is allowed here; the transport (and with it the APM) is
  // not, which is what the ring is for.
  //
  // N channels down to mono and float32 down to int16, both by hand: there is
  // no rate change to make, because the ring and the transport are both at the
  // engine's own capture rate.
  void OnCapturedBuffer(AVAudioPCMBuffer* buffer) {
    if (buffer == nil || buffer.frameLength == 0) {
      return;
    }
    const float* const* channel_data = buffer.floatChannelData;
    const size_t channel_count =
        static_cast<size_t>(buffer.format.channelCount);
    if (channel_data == nullptr || channel_count == 0) {
      RTC_LOG(LS_WARNING) << "WatchAudioEngine: tap buffer is not float PCM; "
                             "dropping";
      return;
    }
    // `stride` is the gap between one channel's consecutive samples, which is
    // what makes this loop cover interleaved and planar buffers alike.
    const size_t stride = static_cast<size_t>(buffer.stride);
    const size_t frames = static_cast<size_t>(buffer.frameLength);
    const float gain = 1.0f / static_cast<float>(channel_count);

    MutexLock lock(&capture_mutex_);
    for (size_t done = 0; done < frames;) {
      const size_t chunk = std::min(frames - done, kMaxRenderFrames);
      for (size_t frame = 0; frame < chunk; ++frame) {
        float sum = 0.0f;
        for (size_t channel = 0; channel < channel_count; ++channel) {
          sum += channel_data[channel][(done + frame) * stride];
        }
        capture_int16_[frame] = FloatToInt16(sum * gain);
      }
      capture_ring_.Write(capture_int16_.data(), chunk);
      done += chunk;
    }
  }

  // Hands one 10 ms block to the transport, at the capture rate. Pump thread
  // only.
  bool DeliverOneCaptureBlock() {
    AudioTransport* transport = transport_.load(std::memory_order_acquire);
    if (transport == nullptr) {
      return false;
    }
    const int rate = capture_rate_.load(std::memory_order_acquire);
    if (rate <= 0) {
      return false;
    }
    const size_t frames = static_cast<size_t>(rate) / 100;
    if (capture_ring_.Readable() < frames) {
      return false;
    }
    capture_ring_.Read(record_int16_.data(), frames);

    // What the AEC needs: how long ago the far-end audio we are still holding
    // will actually be heard, plus how long ago the near-end audio we are
    // handing over was actually captured. Both terms are computed at the rate
    // the ring they describe is running at.
    const uint32_t total_delay_ms = static_cast<uint32_t>(
        PlayoutDelayMs() + input_latency_ms_.load(std::memory_order_acquire) +
        1000.0 * capture_ring_.Readable() / rate);
    uint32_t new_mic_level = 0;
    transport->RecordedDataIsAvailable(
        record_int16_.data(), frames, kBytesPerFrame, kChannels,
        static_cast<uint32_t>(rate), total_delay_ms, /*clockDrift=*/0,
        /*currentMicLevel=*/0,
        /*keyPressed=*/false, new_mic_level,
        /*estimatedCaptureTimeNS=*/std::nullopt);
    return true;
  }

  // Keeps the capture ring from filling up while nobody is listening.
  void DiscardCapturedAudio() {
    while (capture_ring_.Readable() > 0) {
      capture_ring_.Read(record_int16_.data(),
                         std::min(capture_ring_.Readable(),
                                  record_int16_.size()));
    }
  }

  // ------------------------------------------------------------------ pump

  void StartPump() {
    if (!pump_thread_.empty()) {
      return;
    }
    pump_running_.store(true, std::memory_order_release);
    pump_thread_ = PlatformThread::SpawnJoinable(
        [this] { PumpLoop(); }, "WatchAudioPump",
        ThreadAttributes().SetPriority(ThreadPriority::kRealtime));
  }

  void StopPump() {
    if (pump_thread_.empty()) {
      return;
    }
    pump_running_.store(false, std::memory_order_release);
    pump_wake_.Set();
    pump_thread_.Finalize();
  }

  // One thread drives both directions: it tops the playout ring up to the
  // target depth (so the render block never has to do more than a memcpy) and
  // drains whatever the tap captured into the transport. Both are paced by the
  // hardware -- the ring levels are the clock -- so a 5 ms poll is enough.
  //
  // It is also the only thread that ever builds or rebuilds the graph, which
  // is what makes a not-yet-active audio session survivable and what makes the
  // ring clears on a route change safe (see EnsureGraph). Start, a
  // configuration change and an interruption all just set `rebuild_requested_`
  // and wake it; failing every notification, it re-checks a graph that is not
  // running every kHealthCheckIntervalMs.
  void PumpLoop() {
    int64_t next_health_check_ms = 0;
    while (pump_running_.load(std::memory_order_acquire)) {
      const bool disturbed =
          rebuild_requested_.exchange(false, std::memory_order_acq_rel);
      const int64_t now_ms = TimeMillis();
      if (disturbed || now_ms >= next_health_check_ms) {
        next_health_check_ms = now_ms + kHealthCheckIntervalMs;
        // A configuration change can leave the engine running at a new format,
        // so a notification rebuilds unconditionally; the poll only rebuilds
        // what has actually fallen over.
        if (disturbed || !EngineIsRunning()) {
          engine_running_.store(false, std::memory_order_release);
          EnsureGraph();
        }
      }

      if (engine_running_.load(std::memory_order_acquire)) {
        if (playout_wanted()) {
          const size_t target =
              playout_target_frames_.load(std::memory_order_acquire);
          for (int block = 0; block < kMaxBlocksPerWakeup; ++block) {
            if (playout_ring_.Readable() >= target ||
                !PushOnePlayoutBlock()) {
              break;
            }
          }
        }
        if (recording_wanted()) {
          for (int block = 0; block < kMaxBlocksPerWakeup; ++block) {
            if (!DeliverOneCaptureBlock()) {
              break;
            }
          }
        } else {
          DiscardCapturedAudio();
        }
      }

      pump_wake_.Wait(TimeDelta::Millis(kPumpIntervalMs));
    }
  }

  // ----------------------------------------------------------------- state

  mutable Mutex graph_mutex_;
  AVAudioEngine* engine_ RTC_GUARDED_BY(graph_mutex_) = nil;
  AVAudioSourceNode* source_node_ RTC_GUARDED_BY(graph_mutex_) = nil;
  id config_observer_ RTC_GUARDED_BY(graph_mutex_) = nil;
  id interruption_observer_ RTC_GUARDED_BY(graph_mutex_) = nil;
  bool tap_installed_ RTC_GUARDED_BY(graph_mutex_) = false;
  bool voice_processing_on_ RTC_GUARDED_BY(graph_mutex_) = false;
  // The formats the graph was actually built against, so a route change that
  // moves them is noticed rather than silently rendered at the wrong rate.
  // Also remembered when a format is refused, so the refusal logs once.
  AVAudioFormat* built_output_format_ RTC_GUARDED_BY(graph_mutex_) = nil;
  AVAudioFormat* built_input_format_ RTC_GUARDED_BY(graph_mutex_) = nil;

  mutable Mutex capture_mutex_;

  // Outlives the observer blocks; see EngineNotice.
  const std::shared_ptr<EngineNotice> notice_;

  std::atomic<AudioTransport*> transport_{nullptr};
  std::atomic<bool> playout_wanted_{false};
  std::atomic<bool> recording_wanted_{false};
  std::atomic<bool> engine_running_{false};
  // The engine's IO rates, which are what the transport is told. Zero means
  // "not established (or refused)": that direction does not run.
  std::atomic<int> playout_rate_{0};
  std::atomic<int> capture_rate_{0};
  std::atomic<size_t> playout_target_frames_{0};
  std::atomic<double> output_latency_ms_{0};
  std::atomic<double> input_latency_ms_{0};
  std::atomic<int32_t> underruns_{0};

  // Producer: the pump thread. Consumer: the render block.
  SpscRing<float> playout_ring_;
  // Producer: the input tap. Consumer: the pump thread.
  SpscRing<int16_t> capture_ring_;

  std::vector<int16_t> play_int16_;    // pump thread only
  std::vector<float> play_float_;      // pump thread only
  std::vector<int16_t> record_int16_;  // pump thread only
  std::vector<int16_t> capture_int16_; // capture (tap) thread only
  std::vector<float> render_scratch_;  // render thread only

  std::atomic<bool> pump_running_{false};
  // Set by Start and by the notification handlers; cleared by the pump, which
  // is the only thread that rebuilds the graph.
  std::atomic<bool> rebuild_requested_{false};
  Event pump_wake_;
  PlatformThread pump_thread_;
};

// ---------------------------------------------------------------------------
// WatchAudioDeviceModule
// ---------------------------------------------------------------------------

WatchAudioDeviceModule::WatchAudioDeviceModule()
    : engine_(std::make_unique<WatchAudioEngine>()) {
  TRACE_LOG;
  RTC_DCHECK_RUN_ON(&thread_checker_);
}

WatchAudioDeviceModule::~WatchAudioDeviceModule() {
  TRACE_LOG;
  RTC_DCHECK_RUN_ON(&thread_checker_);
  Terminate();
}

// static
scoped_refptr<WatchAudioDeviceModule> WatchAudioDeviceModule::Create() {
  return make_ref_counted<WatchAudioDeviceModule>();
}

int32_t WatchAudioDeviceModule::ActiveAudioLayer(
    AudioLayer* audio_layer) const {
  RTC_DCHECK_RUN_ON(&thread_checker_);
  RTC_DCHECK(audio_layer);
  *audio_layer = kPlatformDefaultAudio;
  return 0;
}

int32_t WatchAudioDeviceModule::RegisterAudioCallback(
    AudioTransport* audio_callback) {
  TRACE_LOG;
  RTC_DCHECK_RUN_ON(&thread_checker_);
  // The same narrow race as the desktop ADM: swapping the transport while the
  // pump thread is between the load and the call would leave it calling into a
  // freed object. WebRtcVoiceEngine::Terminate stops both directions first.
  if (playing_ || recording_) {
    return -1;
  }
  engine_->SetTransport(audio_callback);
  return 0;
}

int32_t WatchAudioDeviceModule::Init() {
  TRACE_LOG;
  RTC_DCHECK_RUN_ON(&thread_checker_);
  initialized_ = true;
  return 0;
}

int32_t WatchAudioDeviceModule::Terminate() {
  TRACE_LOG;
  RTC_DCHECK_RUN_ON(&thread_checker_);
  engine_->Stop(/*playout=*/true, /*recording=*/true);
  playing_ = false;
  recording_ = false;
  playout_initialized_ = false;
  recording_initialized_ = false;
  initialized_ = false;
  return 0;
}

bool WatchAudioDeviceModule::Initialized() const {
  RTC_DCHECK_RUN_ON(&thread_checker_);
  return initialized_;
}

int16_t WatchAudioDeviceModule::PlayoutDevices() {
  RTC_DCHECK_RUN_ON(&thread_checker_);
  return 1;
}

int16_t WatchAudioDeviceModule::RecordingDevices() {
  RTC_DCHECK_RUN_ON(&thread_checker_);
  return 1;
}

int32_t WatchAudioDeviceModule::PlayoutDeviceName(
    uint16_t index,
    char name[kAdmMaxDeviceNameSize],
    char guid[kAdmMaxGuidSize]) {
  RTC_DCHECK_RUN_ON(&thread_checker_);
  if (index != 0) {
    return -1;
  }
  if (name != nullptr) {
    std::strncpy(name, "watchOS playout", kAdmMaxDeviceNameSize - 1);
    name[kAdmMaxDeviceNameSize - 1] = '\0';
  }
  if (guid != nullptr) {
    guid[0] = '\0';
  }
  return 0;
}

int32_t WatchAudioDeviceModule::RecordingDeviceName(
    uint16_t index,
    char name[kAdmMaxDeviceNameSize],
    char guid[kAdmMaxGuidSize]) {
  RTC_DCHECK_RUN_ON(&thread_checker_);
  if (index != 0) {
    return -1;
  }
  if (name != nullptr) {
    std::strncpy(name, "watchOS recording", kAdmMaxDeviceNameSize - 1);
    name[kAdmMaxDeviceNameSize - 1] = '\0';
  }
  if (guid != nullptr) {
    guid[0] = '\0';
  }
  return 0;
}

// The system owns the route on a watch: AVAudioSession picks it, the user
// changes it, and there is nothing here for WebRTC to select.
int32_t WatchAudioDeviceModule::SetPlayoutDevice(uint16_t index) {
  return 0;
}

int32_t WatchAudioDeviceModule::SetPlayoutDevice(WindowsDeviceType device) {
  return 0;
}

int32_t WatchAudioDeviceModule::SetRecordingDevice(uint16_t index) {
  return 0;
}

int32_t WatchAudioDeviceModule::SetRecordingDevice(WindowsDeviceType device) {
  return 0;
}

int32_t WatchAudioDeviceModule::PlayoutIsAvailable(bool* available) {
  RTC_DCHECK(available);
  *available = true;
  return 0;
}

int32_t WatchAudioDeviceModule::InitPlayout() {
  TRACE_LOG;
  RTC_DCHECK_RUN_ON(&thread_checker_);
  // Nothing to do until the session is live -- the hardware format is not
  // knowable before then. StartPlayout does the work, and retries.
  playout_initialized_ = true;
  return 0;
}

bool WatchAudioDeviceModule::PlayoutIsInitialized() const {
  RTC_DCHECK_RUN_ON(&thread_checker_);
  return playout_initialized_;
}

int32_t WatchAudioDeviceModule::RecordingIsAvailable(bool* available) {
  RTC_DCHECK(available);
  *available = true;
  return 0;
}

int32_t WatchAudioDeviceModule::InitRecording() {
  TRACE_LOG;
  RTC_DCHECK_RUN_ON(&thread_checker_);
  recording_initialized_ = true;
  return 0;
}

bool WatchAudioDeviceModule::RecordingIsInitialized() const {
  RTC_DCHECK_RUN_ON(&thread_checker_);
  return recording_initialized_;
}

int32_t WatchAudioDeviceModule::StartPlayout() {
  TRACE_LOG;
  RTC_DCHECK_RUN_ON(&thread_checker_);
  if (playing_) {
    return 0;
  }
  playing_ = true;
  playout_initialized_ = true;
  return engine_->Start(/*playout=*/true, /*recording=*/false);
}

int32_t WatchAudioDeviceModule::StopPlayout() {
  TRACE_LOG;
  RTC_DCHECK_RUN_ON(&thread_checker_);
  if (!playing_) {
    return 0;
  }
  playing_ = false;
  return engine_->Stop(/*playout=*/true, /*recording=*/false);
}

bool WatchAudioDeviceModule::Playing() const {
  RTC_DCHECK_RUN_ON(&thread_checker_);
  return playing_;
}

int32_t WatchAudioDeviceModule::StartRecording() {
  TRACE_LOG;
  RTC_DCHECK_RUN_ON(&thread_checker_);
  if (recording_) {
    return 0;
  }
  recording_ = true;
  recording_initialized_ = true;
  return engine_->Start(/*playout=*/false, /*recording=*/true);
}

int32_t WatchAudioDeviceModule::StopRecording() {
  TRACE_LOG;
  RTC_DCHECK_RUN_ON(&thread_checker_);
  if (!recording_) {
    return 0;
  }
  recording_ = false;
  return engine_->Stop(/*playout=*/false, /*recording=*/true);
}

bool WatchAudioDeviceModule::Recording() const {
  RTC_DCHECK_RUN_ON(&thread_checker_);
  return recording_;
}

int32_t WatchAudioDeviceModule::InitSpeaker() {
  return 0;
}

bool WatchAudioDeviceModule::SpeakerIsInitialized() const {
  return true;
}

int32_t WatchAudioDeviceModule::InitMicrophone() {
  return 0;
}

bool WatchAudioDeviceModule::MicrophoneIsInitialized() const {
  return true;
}

int32_t WatchAudioDeviceModule::SpeakerVolumeIsAvailable(bool* available) {
  RTC_DCHECK(available);
  *available = false;
  return 0;
}

int32_t WatchAudioDeviceModule::SetSpeakerVolume(uint32_t volume) {
  return -1;
}

int32_t WatchAudioDeviceModule::SpeakerVolume(uint32_t* volume) const {
  return -1;
}

int32_t WatchAudioDeviceModule::MaxSpeakerVolume(uint32_t* max_volume) const {
  return -1;
}

int32_t WatchAudioDeviceModule::MinSpeakerVolume(uint32_t* min_volume) const {
  return -1;
}

int32_t WatchAudioDeviceModule::MicrophoneVolumeIsAvailable(bool* available) {
  RTC_DCHECK(available);
  *available = false;
  return 0;
}

int32_t WatchAudioDeviceModule::SetMicrophoneVolume(uint32_t volume) {
  return -1;
}

int32_t WatchAudioDeviceModule::MicrophoneVolume(uint32_t* volume) const {
  return -1;
}

int32_t WatchAudioDeviceModule::MaxMicrophoneVolume(
    uint32_t* max_volume) const {
  return -1;
}

int32_t WatchAudioDeviceModule::MinMicrophoneVolume(
    uint32_t* min_volume) const {
  return -1;
}

int32_t WatchAudioDeviceModule::SpeakerMuteIsAvailable(bool* available) {
  RTC_DCHECK(available);
  *available = false;
  return 0;
}

int32_t WatchAudioDeviceModule::SetSpeakerMute(bool enable) {
  return -1;
}

int32_t WatchAudioDeviceModule::SpeakerMute(bool* enabled) const {
  RTC_DCHECK(enabled);
  *enabled = false;
  return 0;
}

int32_t WatchAudioDeviceModule::MicrophoneMuteIsAvailable(bool* available) {
  RTC_DCHECK(available);
  *available = false;
  return 0;
}

int32_t WatchAudioDeviceModule::SetMicrophoneMute(bool enable) {
  return -1;
}

int32_t WatchAudioDeviceModule::MicrophoneMute(bool* enabled) const {
  RTC_DCHECK(enabled);
  *enabled = false;
  return 0;
}

int32_t WatchAudioDeviceModule::StereoPlayoutIsAvailable(
    bool* available) const {
  RTC_DCHECK(available);
  *available = false;
  return 0;
}

int32_t WatchAudioDeviceModule::SetStereoPlayout(bool enable) {
  // The transport format is mono; asking for stereo fails, asking for mono is
  // a no-op.
  return enable ? -1 : 0;
}

int32_t WatchAudioDeviceModule::StereoPlayout(bool* enabled) const {
  RTC_DCHECK(enabled);
  *enabled = false;
  return 0;
}

int32_t WatchAudioDeviceModule::StereoRecordingIsAvailable(
    bool* available) const {
  RTC_DCHECK(available);
  *available = false;
  return 0;
}

int32_t WatchAudioDeviceModule::SetStereoRecording(bool enable) {
  return enable ? -1 : 0;
}

int32_t WatchAudioDeviceModule::StereoRecording(bool* enabled) const {
  RTC_DCHECK(enabled);
  *enabled = false;
  return 0;
}

int32_t WatchAudioDeviceModule::PlayoutDelay(uint16_t* delay_ms) const {
  RTC_DCHECK(delay_ms);
  *delay_ms = engine_->PlayoutDelayMs();
  return 0;
}

int32_t WatchAudioDeviceModule::GetPlayoutUnderrunCount() const {
  return engine_->underrun_count();
}

}  // namespace rffi
}  // namespace webrtc
