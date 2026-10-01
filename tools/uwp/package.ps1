# SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Packages the UWP frontend into a signed .appx for Device Portal deployment.
#
# Usage:
#   .\package.ps1 -BuildDir C:\citron-nextendo-xbox-uwp\build-uwp
param(
    [string]$BuildDir = "C:\citron-nextendo-xbox-uwp\build-uwp",
    [string]$Configuration = "Release",
    [string]$SdkBin = "D:\WindowsSDK\bin\10.0.28000.0\x64",
    [string]$ShaderDepsDir = "",
    [string]$DxilDll = "",
    [string]$CertThumbprint = "",
    [switch]$SkipSign
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$layout = Join-Path $root "layout"
$assets = Join-Path $layout "Assets"

New-Item -ItemType Directory -Force -Path $assets | Out-Null

# --- assets (solid-colour PNGs) ---
Add-Type -AssemblyName System.Drawing
function New-Png([string]$path, [int]$w, [int]$h) {
    $bmp = New-Object System.Drawing.Bitmap($w, $h)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.Clear([System.Drawing.Color]::FromArgb(255, 20, 90, 40))
    $g.Dispose()
    $bmp.Save($path, [System.Drawing.Imaging.ImageFormat]::Png)
    $bmp.Dispose()
}
New-Png (Join-Path $assets "StoreLogo.png") 50 50
New-Png (Join-Path $assets "Square44x44Logo.png") 44 44
New-Png (Join-Path $assets "Square150x150Logo.png") 150 150
New-Png (Join-Path $assets "Wide310x150Logo.png") 310 150

# --- payload ---
Copy-Item (Join-Path $BuildDir "bin\$Configuration\citron-uwp.exe") $layout -Force
Copy-Item (Join-Path $root "AppxManifest.xml") $layout -Force

# --- D3D12 shader toolchain (loaded at runtime via LoadPackagedLibrary) ---
# spirv_to_dxil.dll: Mesa's SPIR-V -> DXIL translator (MIT).
# dxil.dll: Microsoft's DXIL validator/signer, needed to sign shaders for the D3D12 runtime.
# Both are searched for in the package directory (see ShaderCompiler). Override the sources
# with -ShaderDepsDir / -DxilDll if your copies live elsewhere.
$shaderDepsDir = if ($ShaderDepsDir) { $ShaderDepsDir } else { "C:\uwp-deps\icd\mesa-msvc\x64" }
$dxilSource = if ($DxilDll) { $DxilDll } else { Join-Path $SdkBin "..\..\..\Redist\D3D\x64\dxil.dll" }

foreach ($dll in @("spirv_to_dxil.dll", "dxil.dll")) {
    $source = if ($dll -eq "dxil.dll" -and (Test-Path $dxilSource)) {
        $dxilSource
    } else {
        Join-Path $shaderDepsDir $dll
    }
    if (-not (Test-Path $source)) {
        throw "Missing $dll (looked at '$source'). Use -ShaderDepsDir / -DxilDll to point at it."
    }
    Copy-Item $source $layout -Force
    Write-Host "packaged: $dll ($source)"
}

# --- resources.pri (required when packing outside Visual Studio) ---
$makepri = Join-Path $SdkBin "makepri.exe"
if (Test-Path $makepri) {
    & $makepri createconfig /cf (Join-Path $layout "priconfig.xml") /dq en-US /o | Out-Null
    & $makepri new /pr $layout /cf (Join-Path $layout "priconfig.xml") /of (Join-Path $layout "resources.pri") /o | Out-Null
}

# --- pack ---
$outAppx = Join-Path $root "CitronUwp.appx"
if (Test-Path $outAppx) { Remove-Item $outAppx -Force }
& (Join-Path $SdkBin "makeappx.exe") pack /d $layout /p $outAppx /o
if ($LASTEXITCODE -ne 0) { throw "makeappx failed: $LASTEXITCODE" }
Write-Host "packed: $outAppx"

if ($SkipSign) { return }

# --- sign (reuses the dev certificate if present) ---
$cert = if ($CertThumbprint) {
    Get-ChildItem Cert:\CurrentUser\My | Where-Object { $_.Thumbprint -eq $CertThumbprint }
} else {
    Get-ChildItem Cert:\CurrentUser\My |
        Where-Object { $_.Subject -eq "CN=CitronUwpSpike" } |
        Select-Object -First 1
}
if (-not $cert) {
    $cert = New-SelfSignedCertificate -Type Custom -Subject "CN=CitronUwpSpike" `
        -KeyUsage DigitalSignature -FriendlyName "Citron UWP test cert" `
        -CertStoreLocation "Cert:\CurrentUser\My" `
        -TextExtension @("2.5.29.37={text}1.3.6.1.5.5.7.3.3", "2.5.29.19={text}")
}
& (Join-Path $SdkBin "signtool.exe") sign /fd SHA256 /sha1 $cert.Thumbprint $outAppx
if ($LASTEXITCODE -ne 0) { throw "signtool failed: $LASTEXITCODE" }
Write-Host "signed: $outAppx"
