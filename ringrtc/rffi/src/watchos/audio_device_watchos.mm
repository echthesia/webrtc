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
#include <string>
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

// The one format this module declares to the AudioTransport, in both
// directions. WebRTC's APM resamples internally for its own processing, but
// the rate handed to (and taken from) the transport has to be the rate that is
// actually delivered, so it is fixed here and everything else is converted to
// it. 48 kHz mono is the safe declaration: a native rate for every codec
// WebRTC will pick, what the watch's own IO runs at in practice, and mono is
// all a call needs.
constexpr int kSampleRate = 48000;
constexpr size_t kChannels = 1;
constexpr size_t kFramesPer10ms = kSampleRate / 100;            // 480
constexpr size_t kBytesPerFrame = kChannels * sizeof(int16_t);  // 2

// How far ahead of the render block the pump thread keeps the playout ring:
// deep enough that a late wakeup does not underrun, shallow enough not to add
// audible latency to a call.
constexpr double kPlayoutTargetSeconds = 0.030;

// Ring capacities, allocated once. Playout holds hardware-rate float frames,
// capture holds 48 kHz int16 frames.
constexpr size_t kPlayoutRingFrames = 96000;  // >= 0.5 s at any plausible rate
constexpr size_t kCaptureRingFrames = 24000;  // 0.5 s at 48 kHz

// Largest render block or converted tap buffer we will handle.
constexpr size_t kMaxRenderFrames = 8192;

// Pump cadence, and how often it re-checks a graph that is not running --
// which covers both "the audio session was not live yet" and "an interruption
// stopped the engine".
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

 private:
  std::vector<T> buf_;
  const size_t capacity_;
  std::atomic<size_t> read_{0};
  std::atomic<size_t> write_{0};
};

}  // namespace

// Everything that touches AVFAudio.
//
// The graph is the one the hardware spike (spikes/watch-call, AudioProbe)
// established as the only one that makes sound on a watch under voice
// processing:
//
//   AVAudioSourceNode(hardware format) --> engine.outputNode
//   engine.inputNode --(tap)--> us
//
// with `setVoiceProcessingEnabled:` set on the input node *before* any node is
// wired, because enabling it renegotiates the IO formats. Routing the source
// through `mainMixerNode` instead of straight to `outputNode` renders silence
// under voice processing -- measured on hardware, so the mixer is not in this
// graph.
//
// The audio session is not ours. LiveCommunicationKit sets the category and
// activates it; this module never calls setCategory: or setActive:, and has to
// cope with being started before the session is live -- which presents as an
// output format with a zero sample rate. That is logged, not asserted, and the
// pump thread retries.
//
// Locks. `graph_mutex_` covers the engine and the playout converter; the
// render block never takes it (it only drains a lock-free ring).
// `capture_mutex_` covers the capture converter alone, and is the only lock
// the input tap takes -- which is what makes it safe to call `removeTapOnBus:`
// while holding `graph_mutex_`.
class WatchAudioEngine {
 public:
  WatchAudioEngine()
      : playout_ring_(kPlayoutRingFrames),
        capture_ring_(kCaptureRingFrames),
        play_int16_(kFramesPer10ms * kChannels),
        record_int16_(kFramesPer10ms * kChannels),
        render_scratch_(kMaxRenderFrames) {}

  ~WatchAudioEngine() { Stop(/*playout=*/true, /*recording=*/true); }

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
    EnsureGraph();
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
    const double rate = hardware_rate_.load(std::memory_order_acquire);
    if (rate <= 0) {
      return 0;
    }
    const double buffered_ms = 1000.0 * playout_ring_.Readable() / rate;
    return static_cast<uint16_t>(
        buffered_ms + output_latency_ms_.load(std::memory_order_acquire));
  }

  int32_t underrun_count() const {
    return underruns_.load(std::memory_order_relaxed);
  }

 private:
  // ----------------------------------------------------------------- graph

  // Brings the graph up as far as it can, one step at a time, and returns true
  // if the engine is running when it returns. Every step is conditional, so
  // this is both the first-time build and the retry: StartRecording after
  // StartPlayout adds the tap, and a re-entry after an interruption restarts a
  // stopped engine.
  bool EnsureGraph() {
    MutexLock lock(&graph_mutex_);

    if (engine_ == nil) {
      engine_ = [[AVAudioEngine alloc] init];
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
    AVAudioFormat* hardware_format = [engine_.outputNode inputFormatForBus:0];
    if (hardware_format == nil || hardware_format.sampleRate <= 0) {
      RTC_LOG(LS_INFO) << "WatchAudioEngine: output format has no sample rate "
                          "(audio session not active yet); will retry";
      return false;
    }

    // A route change (or an interruption) can come back with a different
    // format, and the engine stops when that happens. Anything built against
    // the old format has to go: the source node carries its format from
    // construction, so it is recreated rather than reconnected.
    const double hardware_rate = hardware_format.sampleRate;
    if (built_output_format_ == nil ||
        ![built_output_format_ isEqual:hardware_format]) {
      RTC_LOG(LS_INFO) << "WatchAudioEngine: hardware output format "
                       << hardware_rate << " Hz, "
                       << hardware_format.channelCount << " ch";
      if (source_node_ != nil) {
        [engine_ detachNode:source_node_];
        source_node_ = nil;
      }
      if (!BuildPlayoutConverter(hardware_rate)) {
        return false;
      }
      built_output_format_ = hardware_format;
      hardware_rate_.store(hardware_rate, std::memory_order_release);
      playout_target_frames_.store(
          static_cast<size_t>(hardware_rate * kPlayoutTargetSeconds),
          std::memory_order_release);
    }

    if (source_node_ == nil) {
      WatchAudioEngine* self = this;
      source_node_ = [[AVAudioSourceNode alloc]
          initWithFormat:hardware_format
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
                format:hardware_format];
    }

    AVAudioFormat* input_format = [engine_.inputNode outputFormatForBus:0];
    if (tap_installed_ && (input_format == nil ||
                           ![built_input_format_ isEqual:input_format])) {
      // Same story on the capture side: the tap's format is fixed at install.
      [engine_.inputNode removeTapOnBus:0];
      tap_installed_ = false;
      built_input_format_ = nil;
    }
    if (!tap_installed_) {
      if (input_format != nil && input_format.sampleRate > 0) {
        RTC_LOG(LS_INFO) << "WatchAudioEngine: hardware input format "
                         << input_format.sampleRate << " Hz, "
                         << input_format.channelCount << " ch";
        if (BuildCaptureConverter(input_format)) {
          WatchAudioEngine* self = this;
          [engine_.inputNode
              installTapOnBus:0
                   bufferSize:1024
                       format:input_format
                        block:^(AVAudioPCMBuffer* buffer, AVAudioTime* when) {
                          self->OnCapturedBuffer(buffer);
                        }];
          tap_installed_ = true;
          built_input_format_ = input_format;
        }
      } else {
        RTC_LOG(LS_WARNING) << "WatchAudioEngine: input format has no sample "
                               "rate; capture unavailable, will retry";
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

  void TeardownGraph() {
    MutexLock lock(&graph_mutex_);
    engine_running_.store(false, std::memory_order_release);
    if (engine_ == nil) {
      return;
    }
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
    {
      MutexLock capture_lock(&capture_mutex_);
      capture_converter_ = nil;
      capture_out_buffer_ = nil;
    }
    playout_converter_ = nil;
    playout_in_buffer_ = nil;
    playout_out_buffer_ = nil;
    built_output_format_ = nil;
    built_input_format_ = nil;
    engine_ = nil;
    voice_processing_on_ = false;
    hardware_rate_.store(0, std::memory_order_release);
    RTC_LOG(LS_INFO) << "WatchAudioEngine: engine stopped";
  }

  // 48 kHz mono int16 (what the transport hands us) -> mono float32 at the
  // hardware rate (what the render block writes out). Mono out because the
  // render block splats the same sample across however many channels the
  // output node has; a call is mono either way.
  bool BuildPlayoutConverter(double hardware_rate)
      RTC_EXCLUSIVE_LOCKS_REQUIRED(graph_mutex_) {
    AVAudioFormat* from = [[AVAudioFormat alloc]
        initWithCommonFormat:AVAudioPCMFormatInt16
                  sampleRate:kSampleRate
                    channels:static_cast<AVAudioChannelCount>(kChannels)
                 interleaved:YES];
    AVAudioFormat* to =
        [[AVAudioFormat alloc] initWithCommonFormat:AVAudioPCMFormatFloat32
                                         sampleRate:hardware_rate
                                           channels:1
                                        interleaved:NO];
    if (from == nil || to == nil) {
      RTC_LOG(LS_ERROR) << "WatchAudioEngine: could not build playout formats";
      return false;
    }
    playout_converter_ = [[AVAudioConverter alloc] initFromFormat:from
                                                         toFormat:to];
    if (playout_converter_ == nil) {
      RTC_LOG(LS_ERROR) << "WatchAudioEngine: no playout converter "
                        << kSampleRate << " -> " << hardware_rate;
      return false;
    }
    playout_in_buffer_ = [[AVAudioPCMBuffer alloc]
        initWithPCMFormat:from
            frameCapacity:static_cast<AVAudioFrameCount>(kFramesPer10ms)];
    // One 10 ms block at the hardware rate, plus slack for the resampler's own
    // framing.
    const AVAudioFrameCount out_capacity =
        static_cast<AVAudioFrameCount>(hardware_rate * 0.02) + 64;
    playout_out_buffer_ =
        [[AVAudioPCMBuffer alloc] initWithPCMFormat:to
                                      frameCapacity:out_capacity];
    return playout_in_buffer_ != nil && playout_out_buffer_ != nil;
  }

  // Whatever the input node gives us -> 48 kHz mono int16. AVAudioConverter
  // does the downmix as well as the rate change.
  bool BuildCaptureConverter(AVAudioFormat* input_format)
      RTC_EXCLUSIVE_LOCKS_REQUIRED(graph_mutex_) {
    AVAudioFormat* to = [[AVAudioFormat alloc]
        initWithCommonFormat:AVAudioPCMFormatInt16
                  sampleRate:kSampleRate
                    channels:static_cast<AVAudioChannelCount>(kChannels)
                 interleaved:YES];
    if (to == nil) {
      return false;
    }
    AVAudioConverter* converter =
        [[AVAudioConverter alloc] initFromFormat:input_format toFormat:to];
    if (converter == nil) {
      RTC_LOG(LS_ERROR) << "WatchAudioEngine: no capture converter "
                        << input_format.sampleRate << " -> " << kSampleRate;
      return false;
    }
    AVAudioPCMBuffer* buffer = [[AVAudioPCMBuffer alloc]
        initWithPCMFormat:to
            frameCapacity:static_cast<AVAudioFrameCount>(kMaxRenderFrames)];
    if (buffer == nil) {
      return false;
    }
    MutexLock capture_lock(&capture_mutex_);
    capture_converter_ = converter;
    capture_out_buffer_ = buffer;
    return true;
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

  // Pulls one 10 ms block from the transport, converts it to the hardware
  // rate, and pushes it into the playout ring. Pump thread only.
  bool PushOnePlayoutBlock() {
    AudioTransport* transport = transport_.load(std::memory_order_acquire);
    if (transport == nullptr) {
      return false;
    }
    size_t samples_out = 0;
    int64_t elapsed_time_ms = -1;
    int64_t ntp_time_ms = -1;
    if (transport->NeedMorePlayData(
            kFramesPer10ms, kBytesPerFrame, kChannels, kSampleRate,
            play_int16_.data(), samples_out, &elapsed_time_ms,
            &ntp_time_ms) != 0) {
      return false;
    }
    if (samples_out == 0) {
      return false;
    }

    MutexLock lock(&graph_mutex_);
    if (playout_converter_ == nil || playout_in_buffer_ == nil ||
        playout_out_buffer_ == nil) {
      return false;
    }
    playout_in_buffer_.frameLength =
        static_cast<AVAudioFrameCount>(samples_out);
    std::memcpy(playout_in_buffer_.int16ChannelData[0], play_int16_.data(),
                samples_out * sizeof(int16_t));

    AVAudioPCMBuffer* in_buffer = playout_in_buffer_;
    __block BOOL provided = NO;
    AVAudioConverterInputBlock input_block =
        ^AVAudioBuffer*(AVAudioPacketCount packets,
                        AVAudioConverterInputStatus* status) {
          if (provided) {
            *status = AVAudioConverterInputStatus_NoDataNow;
            return nil;
          }
          provided = YES;
          *status = AVAudioConverterInputStatus_HaveData;
          return in_buffer;
        };

    playout_out_buffer_.frameLength = playout_out_buffer_.frameCapacity;
    NSError* error = nil;
    const AVAudioConverterOutputStatus status =
        [playout_converter_ convertToBuffer:playout_out_buffer_
                                      error:&error
                         withInputFromBlock:input_block];
    if (status == AVAudioConverterOutputStatus_Error) {
      RTC_LOG(LS_WARNING) << "WatchAudioEngine: playout conversion failed: "
                          << DescribeError(error);
      return false;
    }
    const AVAudioFrameCount produced = playout_out_buffer_.frameLength;
    if (produced == 0) {
      return false;
    }
    playout_ring_.Write(playout_out_buffer_.floatChannelData[0], produced);
    return true;
  }

  // --------------------------------------------------------------- capture

  // The input tap. Runs on the engine's capture thread -- not the render
  // thread, so the converter is allowed here; the transport (and with it the
  // APM) is not, which is what the ring is for.
  void OnCapturedBuffer(AVAudioPCMBuffer* buffer) {
    if (buffer == nil || buffer.frameLength == 0) {
      return;
    }
    MutexLock lock(&capture_mutex_);
    if (capture_converter_ == nil || capture_out_buffer_ == nil) {
      return;
    }

    AVAudioPCMBuffer* in_buffer = buffer;
    __block BOOL provided = NO;
    AVAudioConverterInputBlock input_block =
        ^AVAudioBuffer*(AVAudioPacketCount packets,
                        AVAudioConverterInputStatus* status) {
          if (provided) {
            *status = AVAudioConverterInputStatus_NoDataNow;
            return nil;
          }
          provided = YES;
          *status = AVAudioConverterInputStatus_HaveData;
          return in_buffer;
        };

    capture_out_buffer_.frameLength = capture_out_buffer_.frameCapacity;
    NSError* error = nil;
    const AVAudioConverterOutputStatus status =
        [capture_converter_ convertToBuffer:capture_out_buffer_
                                      error:&error
                         withInputFromBlock:input_block];
    if (status == AVAudioConverterOutputStatus_Error) {
      RTC_LOG(LS_WARNING) << "WatchAudioEngine: capture conversion failed: "
                          << DescribeError(error);
      return;
    }
    const AVAudioFrameCount produced = capture_out_buffer_.frameLength;
    if (produced == 0) {
      return;
    }
    capture_ring_.Write(capture_out_buffer_.int16ChannelData[0], produced);
  }

  // Hands one 10 ms block to the transport. Pump thread only.
  bool DeliverOneCaptureBlock() {
    AudioTransport* transport = transport_.load(std::memory_order_acquire);
    if (transport == nullptr) {
      return false;
    }
    if (capture_ring_.Readable() < kFramesPer10ms) {
      return false;
    }
    capture_ring_.Read(record_int16_.data(), kFramesPer10ms);

    // What the AEC needs: how long ago the far-end audio we are still holding
    // will actually be heard, plus how long ago the near-end audio we are
    // handing over was actually captured.
    const uint32_t total_delay_ms = static_cast<uint32_t>(
        PlayoutDelayMs() + input_latency_ms_.load(std::memory_order_acquire) +
        1000.0 * capture_ring_.Readable() / kSampleRate);
    uint32_t new_mic_level = 0;
    transport->RecordedDataIsAvailable(
        record_int16_.data(), kFramesPer10ms, kBytesPerFrame, kChannels,
        kSampleRate, total_delay_ms, /*clockDrift=*/0, /*currentMicLevel=*/0,
        /*keyPressed=*/false, new_mic_level,
        /*estimatedCaptureTimeNS=*/std::nullopt);
    return true;
  }

  // Keeps the capture ring from filling up while nobody is listening.
  void DiscardCapturedAudio() {
    while (capture_ring_.Readable() >= kFramesPer10ms) {
      capture_ring_.Read(record_int16_.data(), kFramesPer10ms);
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
  // It is also what makes a not-yet-active audio session survivable: if the
  // graph is not running it retries every kHealthCheckIntervalMs, which covers
  // both a session that goes live after WebRTC has already asked us to start
  // and an engine stopped by an interruption.
  void PumpLoop() {
    int64_t next_health_check_ms = 0;
    while (pump_running_.load(std::memory_order_acquire)) {
      const int64_t now_ms = TimeMillis();
      if (now_ms >= next_health_check_ms) {
        next_health_check_ms = now_ms + kHealthCheckIntervalMs;
        if (!EngineIsRunning()) {
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
  AVAudioConverter* playout_converter_ RTC_GUARDED_BY(graph_mutex_) = nil;
  AVAudioPCMBuffer* playout_in_buffer_ RTC_GUARDED_BY(graph_mutex_) = nil;
  AVAudioPCMBuffer* playout_out_buffer_ RTC_GUARDED_BY(graph_mutex_) = nil;
  bool tap_installed_ RTC_GUARDED_BY(graph_mutex_) = false;
  bool voice_processing_on_ RTC_GUARDED_BY(graph_mutex_) = false;
  // The formats the graph was actually built against, so a route change that
  // moves them is noticed rather than silently rendered at the wrong rate.
  AVAudioFormat* built_output_format_ RTC_GUARDED_BY(graph_mutex_) = nil;
  AVAudioFormat* built_input_format_ RTC_GUARDED_BY(graph_mutex_) = nil;

  mutable Mutex capture_mutex_;
  AVAudioConverter* capture_converter_ RTC_GUARDED_BY(capture_mutex_) = nil;
  AVAudioPCMBuffer* capture_out_buffer_ RTC_GUARDED_BY(capture_mutex_) = nil;

  std::atomic<AudioTransport*> transport_{nullptr};
  std::atomic<bool> playout_wanted_{false};
  std::atomic<bool> recording_wanted_{false};
  std::atomic<bool> engine_running_{false};
  std::atomic<double> hardware_rate_{0};
  std::atomic<size_t> playout_target_frames_{0};
  std::atomic<double> output_latency_ms_{0};
  std::atomic<double> input_latency_ms_{0};
  std::atomic<int32_t> underruns_{0};

  // Producer: the pump thread. Consumer: the render block.
  SpscRing<float> playout_ring_;
  // Producer: the input tap. Consumer: the pump thread.
  SpscRing<int16_t> capture_ring_;

  std::vector<int16_t> play_int16_;    // pump thread only
  std::vector<int16_t> record_int16_;  // pump thread only
  std::vector<float> render_scratch_;  // render thread only

  std::atomic<bool> pump_running_{false};
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
