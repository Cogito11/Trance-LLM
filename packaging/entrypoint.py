"""PyInstaller entry point shared by portable Linux and future Windows builds."""
from gui.bootstrap import main

if __name__=="__main__":
    raise SystemExit(main())
