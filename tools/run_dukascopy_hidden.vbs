' Resumes the 20-year Dukascopy H1 download (tools.fetch_dukascopy) with no console window.
' Called nightly by the scheduled task "GoldAI Dukascopy Fetch".  Safe to re-run: cached months are skipped.
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = fso.GetParentFolderName(fso.GetParentFolderName(WScript.ScriptFullName))
sh.Run "cmd /c venv\Scripts\python.exe -m tools.fetch_dukascopy >> logs\dukascopy_console.log 2>&1", 0, True
