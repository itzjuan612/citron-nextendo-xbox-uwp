// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Header Dynarmic's common/x64/xbyak.h expects its embedder to provide when
// DYNARMIC_XBYAK_CUSTOM_CONTAINERS is defined (see that file for the details).
// It fixes the Xbyak container types and supplies the Common::X64 ABI helpers,
// so every translation unit bakes identical Xbyak class layouts.

#pragma once

#include "common/x64/xbyak_abi.h"
#include "common/x64/xbyak_util.h"
