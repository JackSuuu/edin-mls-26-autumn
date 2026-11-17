param(
    [ValidateSet('List', 'Create', 'Verify', 'EnableIssues')][string]$Mode = 'List',
    [string]$BodyPath,
    [int]$IssueNumber
)
$ErrorActionPreference = 'Stop'
$env:GIT_TERMINAL_PROMPT = '0'
$credentialOutput = "protocol=https`nhost=github.com`n`n" | git -c credential.interactive=false credential fill
if ($LASTEXITCODE -ne 0) { throw 'Git credential helper could not supply existing authentication.' }
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
    'User-Agent' = 'coursework-experiment-publication'
}
$baseUri = 'https://api.github.com/repos/JackSuuu/edin-mls-26-autumn/issues'
try {
    if ($Mode -eq 'EnableIssues') {
        $repoUri = 'https://api.github.com/repos/JackSuuu/edin-mls-26-autumn'
        $repo = Invoke-RestMethod -Uri $repoUri -Headers $headers
        if (-not $repo.has_issues) {
            $repo = Invoke-RestMethod -Method Patch -Uri $repoUri -Headers $headers -ContentType 'application/json' -Body '{"has_issues":true}'
        }
        $repo | Select-Object full_name, has_issues | ConvertTo-Json
    } elseif ($Mode -eq 'List') {
        $issues = Invoke-RestMethod -Uri ($baseUri + '?state=all&per_page=100') -Headers $headers
        @($issues | Select-Object number, title, state, html_url) | ConvertTo-Json -Depth 4
    } elseif ($Mode -eq 'Create') {
        $body = [IO.File]::ReadAllText((Resolve-Path -LiteralPath $BodyPath).Path)
        $payload = @{ title = '[RAG] Diagnose and eliminate avoidable zero-score answers'; body = $body } | ConvertTo-Json
        $result = Invoke-RestMethod -Method Post -Uri $baseUri -Headers $headers -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($payload))
        $result | Select-Object number, title, state, html_url | ConvertTo-Json
    } else {
        if ($IssueNumber -le 0) { throw 'IssueNumber must be positive.' }
        $result = Invoke-RestMethod -Uri ($baseUri + '/' + $IssueNumber) -Headers $headers
        $result | Select-Object number, title, state, html_url, body | ConvertTo-Json -Depth 4
    }
} finally {
    $headers.Clear()
    $credentialFields.Clear()
    $credentialOutput = $null
}
