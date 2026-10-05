# SPDX-FileCopyrightText: 2026 Citron Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

# The UWP (Xbox / Windows Store) frontend.

add_executable(citron-uwp
    main.cpp
    content_locator.cpp
    content_locator.h
    emu_window.cpp
    emu_window.h
    session.cpp
    session.h
)

target_link_libraries(citron-uwp PRIVATE
    audio_core
    common
    core
    input_common
    frontend_common
    network
    hid_core
    video_core
    ${PLATFORM_LIBRARIES}
    Threads::Threads
)

# ffmpeg's Media Foundation code references these GUIDs.
if(WIN32)
    target_link_libraries(citron-uwp PRIVATE mfuuid)
endif()

# UWP application properties (AppContainer, GUI subsystem, WinRT activation).
set_property(TARGET citron-uwp PROPERTY VS_APP_CONTAINER_APPLICATION TRUE)
target_link_options(citron-uwp PRIVATE /APPCONTAINER /MANIFEST:NO)
if(MSVC)
    set_target_properties(citron-uwp PROPERTIES
        LINK_FLAGS "/SUBSYSTEM:WINDOWS /ENTRY:wWinMainCRTStartup /STACK:16777216")
endif()

create_target_directory_groups(citron-uwp)