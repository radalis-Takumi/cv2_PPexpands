import time
import datetime
import sys
from watchdog.observers import Observer
from watchdog.events import RegexMatchingEventHandler
import subprocess
import time

class MyFileWatchHandler(RegexMatchingEventHandler):
    def __init__(self, app_path):
        super().__init__()
        self.app_path = app_path
        self.p = None
        self.chageflag = False

    def startProcess(self):
        print(f'Start - "', *self.app_path, '"')
        self.p = subprocess.Popen(['python', *self.app_path])
    
    def stopProcess(self):
        if not self.p.poll():
            print(f'Stop - "', *self.app_path, '"')
            self.p.kill()

    def restartProcess(self):
        if self.chageflag:
            self.stopProcess()
            self.startProcess()
            self.chageflag = False

    # ファイル作成時の動作
    def on_created(self, event):
        print(f"{datetime.datetime.now()} {event.src_path} created")
        if not self.chageflag:
            self.chageflag = True

    # ファイル変更時の動作
    def on_modified(self, event):
        print(f"{datetime.datetime.now()} {event.src_path} changed")
        if not self.chageflag:
            self.chageflag = True

    # ファイル削除時の動作
    def on_deleted(self, event):
        print(f"{datetime.datetime.now()} {event.src_path} deleted")
        if not self.chageflag:
            self.chageflag = True

    # ファイル移動時の動作
    def on_moved(self, event):
        print(f"{datetime.datetime.now()} {event.src_path} moved")
        if not self.chageflag:
            self.chageflag = True

if __name__ == "__main__":
    # 起動プログラム設定
    app_path = tuple(sys.argv[1:])

    # 対象ディレクトリ
    DIR_WATCH = '.'
    event_handler = MyFileWatchHandler(app_path)
    event_handler.startProcess()
    observer = Observer()
    observer.schedule(event_handler, DIR_WATCH, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
            event_handler.restartProcess()
    except KeyboardInterrupt:
        event_handler.stopProcess()
        observer.stop()
    observer.join()