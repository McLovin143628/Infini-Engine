# Save a PNG of one window (by title substring) or of the whole primary screen.
#
# Wave FIX1. Kept separate from `demo.ps1` so a wave can take a frame by hand
# without running the whole loop.
param(
    [Parameter(Mandatory = $true)][string]$Out,
    [string]$WindowTitle = "",
    [switch]$Foreground
)

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class InfShot {
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);
  [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr hWnd, IntPtr hdc, uint flags);
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
}
"@

$bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
if ($WindowTitle -ne "") {
    $p = Get-Process | Where-Object { $_.MainWindowTitle -like "*$WindowTitle*" } | Select-Object -First 1
    if ($p) {
        [InfShot]::ShowWindow($p.MainWindowHandle, 3) | Out-Null   # SW_MAXIMIZE
        if ($Foreground) {
            [InfShot]::SetForegroundWindow($p.MainWindowHandle) | Out-Null
            Start-Sleep -Milliseconds 500
        }
        $r = New-Object InfShot+RECT
        [InfShot]::GetWindowRect($p.MainWindowHandle, [ref]$r) | Out-Null
        if (($r.Right - $r.Left) -gt 0 -and ($r.Bottom - $r.Top) -gt 0) {
            $bounds = New-Object System.Drawing.Rectangle($r.Left, $r.Top, ($r.Right - $r.Left), ($r.Bottom - $r.Top))
        }
    }
}

# **THE WINDOW ITSELF, NOT THE SCREEN** (the VEH3f.2a audit), opt-in:
# `INF_SHOT_PRINTWINDOW=<title substring>` asks the window for its own pixels
# (`PrintWindow`, `PW_RENDERFULLCONTENT`) instead of copying the screen. Every
# frame of the wave's session carried a 400x460 "This page is having a problem /
# Out of Memory" panel over the game: another application's TOPMOST
# notification window (`any-video-downloader.exe`, "Notifications"), which a
# screen copy photographs and a window capture does not. Falls back to the
# screen copy when the window refuses.
$printed = $false
if ($env:INF_SHOT_PRINTWINDOW) {
    $pw = Get-Process | Where-Object { $_.MainWindowTitle -like "*$($env:INF_SHOT_PRINTWINDOW)*" } | Select-Object -First 1
    if ($pw) {
        $r = New-Object InfShot+RECT
        [InfShot]::GetWindowRect($pw.MainWindowHandle, [ref]$r) | Out-Null
        if (($r.Right - $r.Left) -gt 0 -and ($r.Bottom - $r.Top) -gt 0) {
            $bounds = New-Object System.Drawing.Rectangle($r.Left, $r.Top, ($r.Right - $r.Left), ($r.Bottom - $r.Top))
            $bmp = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
            $g = [System.Drawing.Graphics]::FromImage($bmp)
            $hdc = $g.GetHdc()
            $printed = [InfShot]::PrintWindow($pw.MainWindowHandle, $hdc, 2)
            $g.ReleaseHdc($hdc)
            if (-not $printed) { $g.Dispose(); $bmp.Dispose() }
        }
    }
}
if (-not $printed) {
    $bmp = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size)
}
$bmp.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $bmp.Dispose()
Write-Output "saved $Out ($($bounds.Width)x$($bounds.Height))$(if ($printed) { ' [window capture]' })"
