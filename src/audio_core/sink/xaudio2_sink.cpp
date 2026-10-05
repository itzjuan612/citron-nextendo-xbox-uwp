// SPDX-FileCopyrightText: Copyright 2025 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#ifdef HAVE_XAUDIO2

#include <array>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <xaudio2.h>

#include "audio_core/common/common.h"
#include "audio_core/sink/sink_stream.h"
#include "audio_core/sink/xaudio2_sink.h"
#include "common/logging.h"
#include "core/core.h"

namespace AudioCore::Sink {
namespace {

/// UWP exposes a single stereo output device.
constexpr u32 DeviceChannels{2};
/// Number of PCM buffers kept queued on the source voice.
constexpr u32 BufferCount{3};
/// Frames per buffer, matching cubeb's minimum latency of two target sample counts.
constexpr u32 BufferFrames{TargetSampleCount * 2};

using XAudio2CreateFn = HRESULT(STDAPICALLTYPE*)(IXAudio2**, UINT32, XAUDIO2_PROCESSOR);

} // Anonymous namespace

/**
 * XAudio2 sink stream, responsible for sinking samples to hardware.
 */
class XAudio2SinkStream final : public SinkStream, public IXAudio2VoiceCallback {
public:
    XAudio2SinkStream(IXAudio2* xaudio2_, u32 device_channels_, u32 system_channels_,
                      const std::string& name_, StreamType type_, Core::System& system_)
        : SinkStream(system_, type_), xaudio2{xaudio2_} {
        name = name_;
        device_channels = device_channels_;
        system_channels = system_channels_;

        if (type == StreamType::In || !xaudio2) {
            return;
        }

        WAVEFORMATEX format{};
        format.wFormatTag = WAVE_FORMAT_PCM;
        format.nChannels = static_cast<u16>(device_channels);
        format.nSamplesPerSec = TargetSampleRate;
        format.wBitsPerSample = 16;
        format.nBlockAlign = static_cast<u16>(device_channels * sizeof(s16));
        format.nAvgBytesPerSec = TargetSampleRate * format.nBlockAlign;

        const auto create_error = xaudio2->CreateSourceVoice(&source_voice, &format, 0,
                                                             XAUDIO2_DEFAULT_FREQ_RATIO, this);
        if (FAILED(create_error)) {
            LOG_CRITICAL(Audio_Sink, "Error creating XAudio2 source voice, error: 0x{:08X}",
                         static_cast<u32>(create_error));
            return;
        }

        for (u32 i = 0; i < BufferCount; ++i) {
            auto& buffer = buffers[i];
            buffer.samples.resize(BufferFrames * device_channels);
            buffer.desc.AudioBytes = static_cast<UINT32>(buffer.samples.size() * sizeof(s16));
            buffer.desc.pAudioData = reinterpret_cast<const BYTE*>(buffer.samples.data());
            buffer.desc.pContext = &buffer;
        }
    }

    ~XAudio2SinkStream() override {
        Finalize();
    }

    void Finalize() override {
        if (!source_voice) {
            return;
        }

        SignalPause();
        source_voice->Stop(0);
        source_voice->FlushSourceBuffers();
        source_voice->DestroyVoice();
        source_voice = nullptr;
    }

    void Start(bool resume = false) override {
        if (!source_voice || !paused) {
            return;
        }

        paused = false;
        for (u32 i = 0; i < BufferCount; ++i) {
            SubmitBuffer(i);
        }

        if (FAILED(source_voice->Start(0))) {
            LOG_CRITICAL(Audio_Sink, "Error starting XAudio2 source voice");
        }
    }

    void Stop() override {
        if (!source_voice || paused) {
            return;
        }

        SignalPause();
        source_voice->Stop(0);
        source_voice->FlushSourceBuffers();
    }

private:
    struct PcmBuffer {
        XAUDIO2_BUFFER desc{};
        std::vector<s16> samples;
    };

    void SubmitBuffer(std::size_t index) {
        auto& buffer = buffers[index];
        std::span<s16> output_buffer{buffer.samples.data(), buffer.samples.size()};
        ProcessAudioOutAndRender(output_buffer, BufferFrames);
        source_voice->SubmitSourceBuffer(&buffer.desc);
    }

    void STDMETHODCALLTYPE OnVoiceProcessingPassStart(UINT32) override {}
    void STDMETHODCALLTYPE OnVoiceProcessingPassEnd() override {}
    void STDMETHODCALLTYPE OnStreamEnd() override {}
    void STDMETHODCALLTYPE OnBufferStart(void*) override {}

    void STDMETHODCALLTYPE OnBufferEnd(void* pBufferContext) override {
        if (!source_voice || paused) {
            return;
        }

        auto* buffer = static_cast<PcmBuffer*>(pBufferContext);
        const auto index = static_cast<std::size_t>(buffer - buffers.data());
        SubmitBuffer(index);
    }

    void STDMETHODCALLTYPE OnLoopEnd(void*) override {}
    void STDMETHODCALLTYPE OnVoiceError(void*, HRESULT) override {}

    /// XAudio2 engine, owned by the sink
    IXAudio2* xaudio2{};
    /// Source voice feeding the mastering voice
    IXAudio2SourceVoice* source_voice{};
    /// PCM buffers queued on the source voice
    std::array<PcmBuffer, BufferCount> buffers{};
};

XAudio2Sink::XAudio2Sink([[maybe_unused]] std::string_view target_device_name) {
    device_channels = DeviceChannels;
    if (!xaudio2_library.Open("XAudio2_9.dll")) {
        LOG_CRITICAL(Audio_Sink, "XAudio2_9.dll failed to load, XAudio2 sink will be silent");
        return;
    }

    XAudio2CreateFn xaudio2_create{};
    if (!xaudio2_library.GetSymbol("XAudio2Create", &xaudio2_create)) {
        LOG_CRITICAL(Audio_Sink, "XAudio2Create could not be resolved");
        return;
    }

    if (FAILED(xaudio2_create(&xaudio2, 0, XAUDIO2_DEFAULT_PROCESSOR)) || !xaudio2) {
        LOG_CRITICAL(Audio_Sink, "XAudio2Create failed");
        xaudio2 = nullptr;
        return;
    }

    if (FAILED(xaudio2->CreateMasteringVoice(&mastering_voice, device_channels, TargetSampleRate,
                                              0, nullptr, nullptr, AudioCategory_GameEffects))) {
        LOG_CRITICAL(Audio_Sink, "Failed to create XAudio2 mastering voice");
        mastering_voice = nullptr;
    }
}

XAudio2Sink::~XAudio2Sink() {
    sink_streams.clear();

    if (mastering_voice) {
        mastering_voice->DestroyVoice();
        mastering_voice = nullptr;
    }

    if (xaudio2) {
        xaudio2->Release();
        xaudio2 = nullptr;
    }
}

SinkStream* XAudio2Sink::AcquireSinkStream(Core::System& system, u32 system_channels_,
                                           const std::string& name, StreamType type) {
    system_channels = system_channels_;
    SinkStreamPtr& stream = sink_streams.emplace_back(std::make_unique<XAudio2SinkStream>(
        xaudio2, device_channels, system_channels, name, type, system));

    return stream.get();
}

void XAudio2Sink::CloseStream(SinkStream* stream) {
    for (size_t i = 0; i < sink_streams.size(); i++) {
        if (sink_streams[i].get() == stream) {
            sink_streams[i].reset();
            sink_streams.erase(sink_streams.begin() + i);
            break;
        }
    }
}

void XAudio2Sink::CloseStreams() {
    sink_streams.clear();
}

f32 XAudio2Sink::GetDeviceVolume() const {
    if (sink_streams.empty() || !sink_streams[0]) {
        return 1.0f;
    }

    return sink_streams[0]->GetDeviceVolume();
}

void XAudio2Sink::SetDeviceVolume(f32 volume) {
    for (auto& stream : sink_streams) {
        stream->SetDeviceVolume(volume);
    }
}

void XAudio2Sink::SetSystemVolume(f32 volume) {
    for (auto& stream : sink_streams) {
        stream->SetSystemVolume(volume);
    }
}

std::vector<std::string> ListXAudio2SinkDevices(bool capture) {
    if (capture) {
        return {};
    }

    return {std::string(auto_device_name)};
}

bool IsXAudio2Suitable() {
    Common::DynamicLibrary library("XAudio2_9.dll");
    if (!library.IsOpen()) {
        LOG_ERROR(Audio_Sink, "XAudio2_9.dll failed to load, XAudio2 is not suitable.");
        return false;
    }

    XAudio2CreateFn xaudio2_create{};
    if (!library.GetSymbol("XAudio2Create", &xaudio2_create)) {
        LOG_ERROR(Audio_Sink, "XAudio2Create could not be resolved, XAudio2 is not suitable.");
        return false;
    }

    IXAudio2* xaudio2{};
    if (FAILED(xaudio2_create(&xaudio2, 0, XAUDIO2_DEFAULT_PROCESSOR)) || !xaudio2) {
        LOG_ERROR(Audio_Sink, "XAudio2Create failed, XAudio2 is not suitable.");
        return false;
    }

    IXAudio2MasteringVoice* mastering_voice{};
    const auto create_error = xaudio2->CreateMasteringVoice(
        &mastering_voice, DeviceChannels, TargetSampleRate, 0, nullptr, nullptr,
        AudioCategory_GameEffects);
    if (SUCCEEDED(create_error) && mastering_voice) {
        mastering_voice->DestroyVoice();
    }
    xaudio2->Release();

    if (FAILED(create_error) || !mastering_voice) {
        LOG_ERROR(Audio_Sink, "XAudio2 could not open a device, it is not suitable.");
        return false;
    }

    return true;
}

} // namespace AudioCore::Sink

#endif // HAVE_XAUDIO2
