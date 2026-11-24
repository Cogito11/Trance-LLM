"""PyInstaller entry point shared by portable Linux and future Windows builds."""
import sys

from gui.bootstrap import explain_startup_error, main

if __name__=="__main__":
    try:
        code=main()
    except ImportError as error:
        message=explain_startup_error(error)
        if message is None: raise
        print(message,file=sys.stderr)
        code=1
    raise SystemExit(code)
