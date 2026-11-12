# Windows PowerShell 5.1+; credentials are never added to this PowerShell process's environment.
[CmdletBinding()]
param(
    [ValidateSet('run', 'status', 'resume', 'cancel')][string]$Action = 'run',
    [string]$RunDir = '',
    [ValidateSet('File', 'Prompt')][string]$KeyMode = 'File',
    [ValidateRange(30, 3600)][int]$MaxSeconds = 900,
    [ValidateRange(1, 24)][int]$MaxSnapshotHours = 6
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$script:Step = 'startup'
$script:ApiKeyPattern = '(?<![A-Za-z0-9_-])sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}(?![A-Za-z0-9_-])'

function Quote-NativeArgument([string]$Value) {
    # Windows CRT quoting; no shell is used to interpret arguments.
    $result = New-Object System.Text.StringBuilder
    [void]$result.Append('"')
    $slashes = 0
    foreach ($character in $Value.ToCharArray()) {
        if ($character -eq '\') { $slashes++; continue }
        if ($character -eq '"') {
            [void]$result.Append(('\' * (2 * $slashes + 1)))
            [void]$result.Append('"')
        } else {
            [void]$result.Append(('\' * $slashes))
            [void]$result.Append($character)
        }
        $slashes = 0
    }
    [void]$result.Append(('\' * (2 * $slashes)))
    [void]$result.Append('"')
    return $result.ToString()
}

function Invoke-PilotProcess {
    param([string]$Executable, [string[]]$Arguments,
          [switch]$Quiet, [AllowNull()][string]$ApiKey = $null)
    $start = New-Object System.Diagnostics.ProcessStartInfo
    $start.FileName = $Executable
    $start.Arguments = (($Arguments | ForEach-Object { Quote-NativeArgument $_ }) -join ' ')
    $start.WorkingDirectory = $PSScriptRoot
    $start.UseShellExecute = $false
    $start.CreateNoWindow = [bool]$Quiet
    $start.RedirectStandardOutput = [bool]$Quiet
    $start.RedirectStandardError = [bool]$Quiet
    # Bootstrap processes do not inherit an API key or SDK debug logging.
    foreach ($name in @($start.EnvironmentVariables.Keys)) {
        if ($name -like 'OPENAI_*' -or $name -in @('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP', 'PYTHONINSPECT',
                                                  'PYLAUNCH_DEBUG', 'PYLAUNCHER_DEBUG', '__PYVENV_LAUNCHER__')) {
            $start.EnvironmentVariables.Remove($name)
        }
    }
    $start.EnvironmentVariables['PYTHONUTF8'] = '1'
    $start.EnvironmentVariables['PYTHONNOUSERSITE'] = '1'
    if ($ApiKey) { $start.EnvironmentVariables['OPENAI_API_KEY'] = $ApiKey }
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $start
    try {
        [void]$process.Start()
        # Process creation copies its environment. Remove the launch copy immediately.
        $start.EnvironmentVariables.Remove('OPENAI_API_KEY')
        $ApiKey = $null
        if ($Quiet) {
            $stdout = $process.StandardOutput.ReadToEndAsync()
            $stderr = $process.StandardError.ReadToEndAsync()
        }
        $process.WaitForExit()
        if ($Quiet) {
            $null = $stdout.GetAwaiter().GetResult()
            $null = $stderr.GetAwaiter().GetResult()
        }
        return $process.ExitCode
    } finally {
        $start.EnvironmentVariables.Remove('OPENAI_API_KEY')
        $ApiKey = $null
        $process.Dispose()
    }
}

function Find-PilotPython {
    $probe = 'import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)'
    foreach ($name in @('py.exe', 'python.exe')) {
        $command = Get-Command $name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($null -eq $command) { continue }
        # A Windows Store app-execution alias may open the Store instead of Python.
        if ($command.Source -match '\\Microsoft\\WindowsApps\\') { continue }
        $prefix = @()
        if ($name -eq 'py.exe') { $prefix = @('-3') }
        $code = Invoke-PilotProcess -Executable $command.Source -Arguments ($prefix + @('-I', '-S', '-X', 'utf8', '-c', $probe)) -Quiet
        if ($code -eq 0) { return @{ Path = $command.Source; Prefix = $prefix } }
    }
    throw 'PythonUnavailable'
}

function Read-PilotKey {
    $raw = $null
    $secure = $null
    $buffer = [IntPtr]::Zero
    try {
        if ($KeyMode -eq 'Prompt') {
            $secure = Read-Host 'API key (hidden; never saved)' -AsSecureString
            $buffer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
            $raw = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($buffer)
        } else {
            Add-Type -AssemblyName System.Windows.Forms
            $picker = New-Object System.Windows.Forms.OpenFileDialog
            try {
                $picker.Title = 'Choose the downloaded OpenAI API key file (kept local)'
                $picker.Filter = 'Text and JSON (*.txt;*.json)|*.txt;*.json|All files (*.*)|*.*'
                $picker.Multiselect = $false
                $downloads = Join-Path ([Environment]::GetFolderPath('UserProfile')) 'Downloads'
                if (Test-Path -LiteralPath $downloads -PathType Container) { $picker.InitialDirectory = $downloads }
                if ($picker.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { return $null }
                $file = Get-Item -LiteralPath $picker.FileName
                if ($file.Length -gt 65536 -or $file.PSIsContainer -or
                    ($file.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'InvalidKeyFile' }
                $raw = [IO.File]::ReadAllText($file.FullName)
            } finally { $picker.Dispose() }
        }
        $matchesFound = [regex]::Matches($raw, $script:ApiKeyPattern)
        if ($matchesFound.Count -ne 1) { throw 'ExpectedOneApiKey' }
        return $matchesFound[0].Value
    } finally {
        $raw = $null
        if ($buffer -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($buffer) }
        if ($null -ne $secure) { $secure.Dispose() }
    }
}

$key = $null
try {
    if (-not $RunDir) { $RunDir = Join-Path $PSScriptRoot 'prepared-run' }
    $RunDir = [IO.Path]::GetFullPath($RunDir)
    $runtime = Join-Path $PSScriptRoot 'local_runtime.py'
    $script:Step = 'finding Python 3.11 or newer'
    $python = Find-PilotPython
    $runtimeCommand = $python.Prefix + @('-I', '-S', '-X', 'utf8', $runtime)
    $common = @('--action', $Action, '--run-dir', $RunDir, '--max-seconds', [string]$MaxSeconds,
                '--max-snapshot-hours', [string]$MaxSnapshotHours)
    $script:Step = 'checking the prepared packet (offline)'
    $code = Invoke-PilotProcess -Executable $python.Path -Arguments ($runtimeCommand + @('check') + $common)
    if ($code -ne 0) { exit $code }

    $script:Step = 'preparing local Python dependencies'
    Write-Host 'Preparing local dependencies with the installed Python; the API key has not been requested.'
    $code = Invoke-PilotProcess -Executable $python.Path -Arguments ($runtimeCommand + @('prepare'))
    if ($code -ne 0) { exit $code }

    $script:Step = 'reading the locally selected credential'
    Write-Host 'The key stays on this computer and is passed only to the child Python process.'
    $key = Read-PilotKey
    if (-not $key) { Write-Host 'Cancelled before contacting the API.'; exit 0 }
    $script:Step = 'running the pilot'
    if ($Action -eq 'run') {
        Write-Host 'Starting one paid research session with the spend threshold shown above (USD cents).'
        Write-Host 'The threshold is not an exact final bill. Closing this window does not prove remote cancellation.'
    }
    $code = Invoke-PilotProcess -Executable $python.Path -Arguments ($runtimeCommand + @('execute') + $common) -ApiKey $key
    $key = $null
    Write-Host 'Finished. Receipt and any completed dossier are in the prepared run directory.'
    exit $code
} catch {
    # Deliberately omit exception text, selected filenames, process arguments and credential data.
    Write-Host ('Pilot stopped while ' + $script:Step + '. No credential details were printed.') -ForegroundColor Red
    if ($script:Step -eq 'finding Python 3.11 or newer') {
        Write-Host 'Install Python 3.11 or newer with the Python launcher, then open START.cmd again.'
    } elseif ($script:Step -eq 'preparing local Python dependencies') {
        Write-Host 'Check runtime-diagnostics.json in the pilot folder.'
    } elseif ($script:Step -eq 'reading the locally selected credential') {
        Write-Host 'Use -KeyMode Prompt if the file picker is unavailable.'
    } else {
        Write-Host 'Check Python, network access and the prepared packet.'
    }
    exit 2
} finally {
    $key = $null
}
