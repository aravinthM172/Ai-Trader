' Daily go-live preflight (tools.go_live_preflight --notify) with no console window.
' Called by the scheduled task "GoldAI Preflight"; Telegram message only when a check FAILs.
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = fso.GetParentFolderName(fso.GetParentFolderName(WScript.ScriptFullName))
sh.Run "cmd /c venv\Scripts\python.exe -m tools.go_live_preflight --notify >> logs\preflight_console.log 2>&1", 0, True
