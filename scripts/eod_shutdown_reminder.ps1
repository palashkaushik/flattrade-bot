Add-Type -AssemblyName System.Windows.Forms

$answer = [System.Windows.Forms.MessageBox]::Show(
    "Trading day is over. Do you want to shut down the PC now?",
    "Flattrade Bot - EOD Shutdown",
    [System.Windows.Forms.MessageBoxButtons]::YesNo,
    [System.Windows.Forms.MessageBoxIcon]::Question,
    [System.Windows.Forms.MessageBoxDefaultButton]::Button2
)

if ($answer -eq [System.Windows.Forms.DialogResult]::Yes) {
    shutdown /s /t 120 2>$null
}
