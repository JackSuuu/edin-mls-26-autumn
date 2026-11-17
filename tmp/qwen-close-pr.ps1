param(
    [ValidateSet('List', 'Close')][string]$Mode = 'List',
    [int]$PullNumber
)
$ErrorActionPreference = 'Stop'
$env:GIT_TERMINAL_PROMPT = '0'
$credentialOutput = "protocol=https`nhost=github.com`n`n" | git -c credential.interactive=false credential fill
if ($LASTEXITCODE -ne 0) { throw 'Existing Git authentication is unavailable.' }
$credentialFields = @{}
foreach ($entry in $credentialOutput) {
    $separator = $entry.IndexOf('=')
    if ($separator -gt 0) { $credentialFields[$entry.Substring(0, $separator)] = $entry.Substring($separator + 1) }
}
if (-not $credentialFields['password']) { throw 'No existing GitHub credential was returned.' }
$headers = @{
    Authorization = 'Bearer ' + $credentialFields['password']
    Accept = 'application/vnd.github+json'
    'X-GitHub-Api-Version' = '2022-11-28'
    'User-Agent' = 'coursework-pr-management'
}
$baseUri = 'https://api.github.com/repos/JackSuuu/edin-mls-26-autumn/pulls'
try {
    if ($Mode -eq 'List') {
        $pulls = Invoke-RestMethod -Uri ($baseUri + '?state=all&per_page=100') -Headers $headers
        ConvertTo-Json -InputObject @($pulls | ForEach-Object { [pscustomobject]@{ number = $_.number; title = $_.title; state = $_.state; merged_at = $_.merged_at; url = $_.html_url; head = $_.head.ref; base = $_.base.ref } }) -Depth 4
    } else {
        if ($PullNumber -le 0) { throw 'A positive PullNumber is required.' }
        $uri = $baseUri + '/' + $PullNumber
        $current = Invoke-RestMethod -Uri $uri -Headers $headers
        if ($current.head.ref -ne 'feat/qwen-answer-results-zero-diagnosis') { throw 'PR head does not match the current experiment branch.' }
        if ($current.state -eq 'open') {
            $current = Invoke-RestMethod -Method Patch -Uri $uri -Headers $headers -ContentType 'application/json' -Body '{"state":"closed"}'
        }
        $current | Select-Object number, title, state, merged, html_url, closed_at | ConvertTo-Json
    }
} finally {
    $headers.Clear()
    $credentialFields.Clear()
    $credentialOutput = $null
}
