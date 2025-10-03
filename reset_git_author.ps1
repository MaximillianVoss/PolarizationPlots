<#
.SYNOPSIS
  Переписывает историю git-репозитория, заменяя автора и коммитера на новые имя/почту.
  Может менять всех подряд или только коммиты с конкретной старой почтой.

.PARAMETER Name
  Новое имя автора/коммитера.

.PARAMETER Email
  Новый e?mail автора/коммитера.

.PARAMETER OldEmail
  (Необязательно) Старая почта. Если указана — заменяются только коммиты с этой почтой.
  Если не указана — будут переписаны *все* коммиты.

.PARAMETER Force
  Пропустить подтверждение перед переписыванием истории.

.PARAMETER DryRun
  Пробный запуск: покажет, что будет сделано, но не изменит историю.

.EXAMPLE
  .\reset_git_author.ps1 -Name "Ivan Petrov" -Email "ivan@example.com"

.EXAMPLE
  .\reset_git_author.ps1 -Name "Ivan Petrov" -Email "ivan@example.com" -OldEmail "old@mail.com"

.EXAMPLE
  .\reset_git_author.ps1 -Name "Ivan Petrov" -Email "ivan@example.com" -Force

.NOTES
  Это *переписывает историю* (rebase всего репо). После — пушить только --force-with-lease.
#>

param(
  [Parameter(Mandatory = $true)] [string] $Name,
  [Parameter(Mandatory = $true)] [string] $Email,
  [string] $OldEmail,
  [switch] $Force,
  [switch] $DryRun
)

function Fail($msg) { Write-Error $msg; exit 1 }

# 1) Предпроверки
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
  Fail "git не найден в PATH."
}

# Убедимся, что мы в репозитории
$gitTop = (& git rev-parse --show-toplevel 2>$null)
if ($LASTEXITCODE -ne 0) { Fail "Похоже, это не git?репозиторий." }
Set-Location $gitTop

# Проверим чистоту рабочего дерева
$dirty = (git status --porcelain)
if ($dirty) {
  Write-Warning "Рабочее дерево не чистое. Сохраните/закоммитьте изменения или сделайте stash."
  if (-not $Force) { Fail "Остановлено. Запустите с -Force, если уверены." }
}

# 2) Информация и подтверждение
Write-Host "Репозиторий: $gitTop"
if ($OldEmail) {
  Write-Host "Заменяем коммиты c почтой '$OldEmail' на:"
} else {
  Write-Host "Заменяем авторов/коммитеров во всей истории на:"
}
Write-Host "  Name : $Name"
Write-Host "  Email: $Email"
if ($DryRun) { Write-Host "`nDRY RUN: никаких изменений в истории сделано не будет.`n" }

if (-not $Force) {
  $answer = Read-Host "Это перепишет ВСЮ историю (branches, tags). Продолжить? (yes/no)"
  if ($answer -notin @("y","Y","yes","YES")) { Fail "Отменено пользователем." }
}

# 3) Сохраним бэкап-реф на всякий случай
$backupRef = "refs/backup/author-rewrite/$(Get-Date -f yyyyMMdd-HHmmss)"
if (-not $DryRun) {
  git update-ref $backupRef HEAD | Out-Null
  Write-Host "Создан резервный ref: $backupRef (указывает на текущий HEAD)."
}

# 4) Обновим локальную конфигурацию user.name/user.email (чтобы новые коммиты были верными)
if (-not $DryRun) {
  git config user.name  "$Name"
  git config user.email "$Email"
  Write-Host "Локальная git-конфигурация обновлена (user.name, user.email)."
} else {
  Write-Host "[DRY RUN] Пропущено обновление локальной конфигурации."
}

# 5) Попробуем использовать git filter-repo (если доступен). Иначе fallback на filter-branch.
function HasFilterRepo {
  & git filter-repo --help *>$null
  return ($LASTEXITCODE -eq 0)
}

if (HasFilterRepo) {
  Write-Host "Обнаружен: git filter-repo — используем его."
  # Построим правила замены
  # Для filter-repo укажем mapping авторов/коммитеров.
  $tmpMap = New-TemporaryFile
  try {
    if ($OldEmail) {
      # Только для указанной старой почты
      @"
author email <$OldEmail> ==> $Name <$Email>
committer email <$OldEmail> ==> $Name <$Email>
"@ | Set-Content -NoNewline -Encoding UTF8 $tmpMap
    } else {
      # Для всех — используем callback через arguments, т.к. массовая смена удобнее через env в filter-branch.
      # Но filter-repo тоже умеет "replace-text" для идентичностей:
      @"
author identity ==>* $Name <$Email>
committer identity ==>* $Name <$Email>
"@ | Set-Content -NoNewline -Encoding UTF8 $tmpMap
    }

    $commonArgs = @("--force", "--commit-callback", "pass")
    $mapArg = @("--mailmap", $tmpMap.FullName)

    if ($DryRun) {
      Write-Host "[DRY RUN] Покажем статистику потенциальных изменений:"
      git filter-repo @mapArg --dry-run
    } else {
      git filter-repo @mapArg
    }
  }
  finally {
    if (Test-Path $tmpMap) { Remove-Item $tmpMap -Force }
  }
}
else {
  Write-Host "git filter-repo не найден. Переходим на git filter-branch (медленнее)."

  # Сформируем env-filter bash-скрипт для filter-branch
  $envFilter = if ($OldEmail) {
@"
if [ "\$GIT_AUTHOR_EMAIL" = "$OldEmail" ]; then
  export GIT_AUTHOR_NAME="$Name"
  export GIT_AUTHOR_EMAIL="$Email"
fi
if [ "\$GIT_COMMITTER_EMAIL" = "$OldEmail" ]; then
  export GIT_COMMITTER_NAME="$Name"
  export GIT_COMMITTER_EMAIL="$Email"
fi
"@
  } else {
@"
export GIT_AUTHOR_NAME="$Name"
export GIT_AUTHOR_EMAIL="$Email"
export GIT_COMMITTER_NAME="$Name"
export GIT_COMMITTER_EMAIL="$Email"
"@
  }

  if ($DryRun) {
    Write-Host "[DRY RUN] Будет выполнено:"
    Write-Host "git filter-branch --env-filter '<script>' --tag-name-filter cat -- --branches --tags"
    Write-Host "`n--- env-filter preview ---`n$envFilter`n--------------------------"
  } else {
    # Запускаем filter-branch
    & git filter-branch --env-filter $envFilter --tag-name-filter cat -- --branches --tags
    if ($LASTEXITCODE -ne 0) { Fail "git filter-branch завершился с ошибкой." }
  }
}

Write-Host "`nГотово."
Write-Host "Важно: после проверки локально пушьте с перезаписью:"
Write-Host "  git push --force-with-lease --all"
Write-Host "  git push --force-with-lease --tags"
Write-Host "Если что-то пошло не так, можно вернуться к бэкапу:"
Write-Host "  git reset --hard $backupRef"
