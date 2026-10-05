// SPDX-FileCopyrightText: Copyright 2025 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#ifdef HAVE_XAUDIO2

#include <string>
#include <string_view>
#include <vector>

#include <xaudio2.h>

#include "audio_core/sink/sink.h"
#include "common/dynamic_library.h"

namespace Core {
class System;
}

namespace AudioCore::Sink {

/**
 * XAudio2 backend sink. XAudio2_9.dll is loaded dynamically (never import-linked), which is
 * required by the UWP sandbox and allows a clean fallback to Null when unavailable.
 */
class XAudio2Sink final : public Sink {
public:
    explicit XAudio2Sink(std::string_view device_id);
    ~XAudio2Sink() override;

    SinkStream* AcquireSinkStream(Core::System& system, u32 system_channels,
                                  const std::string& name, StreamType type) override;
    void CloseStream(SinkStream* stream) override;
    void CloseStreams() override;
    f32 GetDeviceVolume() const override;
    void SetDeviceVolume(f32 volume) override;
    void SetSystemVolume(f32 volume) override;

private:
    /// Dynamically loaded XAudio2_9.dll
    Common::DynamicLibrary xaudio2_library;
    /// XAudio2 engine
    IXAudio2* xaudio2{};
    /// Mastering voice shared by every stream
    IXAudio2MasteringVoice* mastering_voice{};
    /// Vector of streams managed by this sink
    std::vector<SinkStreamPtr> sink_streams{};
};

/**
 * Get a list of connected devices from XAudio2.
 *
 * @param capture - Return input (capture) devices if true, otherwise output devices.
 */
std::vector<std::string> ListXAudio2SinkDevices(bool capture);

/**
 * Check if this backend is suitable for use.
 *
 * @return True if XAudio2_9.dll loads and a device can be opened, false otherwise.
 */
bool IsXAudio2Suitable();

} // namespace AudioCore::Sink

#endif // HAVE_XAUDIO2
