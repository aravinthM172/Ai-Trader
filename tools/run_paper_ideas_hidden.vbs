' Runs one paper (dry-run) pass of execution.paper_ideas with no console window.
' Called hourly by the Windows scheduled task "GoldAI Paper Ideas" (see start_paper_ideas.bat).
Set sh = CreateObject("WScript.Shell")
root = CreateObject("Scripting.FileSystemObject").GetParentFolderName(CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName))
sh.CurrentDirectory = root
sh.Run "cmd /c venv\Scripts\python.exe -m execution.paper_ideas >> logs\paper_ideas_console.log 2>&1", 0, True
