// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// ffmpeg's Media Foundation code (libavcodec/mfenc.c) references IID_ICodecAPI, but
// icodecapi.h only declares it and neither mfuuid.lib nor the UWP SDK surface exports it.
// Provide the definition so the final link succeeds.

#ifdef _WIN32

#include <guiddef.h>

// {901DB4C7-31CE-41A2-85DC-8FA0BF41B8DA}
extern "C" const IID IID_ICodecAPI = {
    0x901db4c7, 0x31ce, 0x41a2, {0x85, 0xdc, 0x8f, 0xa0, 0xbf, 0x41, 0xb8, 0xda}};

#endif
