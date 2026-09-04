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

$bmp = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size)
$bmp.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $bmp.Dispose()
Write-Output "saved $Out ($($bounds.Width)x$($bounds.Height))"
