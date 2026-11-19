# -*- coding: utf-8 -*-
"""下载静态编译版 ffmpeg 到 tools/ffmpeg/ffmpeg.exe。

环境里没有系统 ffmpeg，pip 装修不了（默认源不可达）。
giehub / gyan.dev 都太慢（约 20KB/s），改用 npmmirror 的 ffmpeg-static 二进制：
  ffmpeg-win32-x64.gz  约 29MB，解压即 exe，无依赖。
"""

from __future__ import annotations

import gzip
import os
import shutil
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "tools" / "ffmpeg"
EXE = DEST / "ffmpeg.exe"
URL = "https://registry.npmmirror.com/-/binary/ffmpeg-static/b6.1.1/ffmpeg-win32-x64.gz"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    if EXE.exists() and EXE.stat().st_size > 10_000_000:
        print(f"已存在: {EXE}")
        return

    print(f"下载 {URL}")
    t0 = time.time()
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    src = "_ffmpeg.gz"
    with urllib.request.urlopen(req, timeout=180) as r, open(src, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1024 * 256)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done/1048576:6.1f}/{total/1048576:.1f} MB  {done*100//total:3d}%",
                      end="", flush=True)
    print(f"\n  用时 {time.time()-t0:.0f}s")

    print("解压 ...")
    with gzip.open(src, "rb") as fin, EXE.open("wb") as fout:
        shutil.copyfileobj(fin, fout)
    os.chmod(EXE, 0o755)
    os.remove(src)
    print(f"完成: {EXE}  ({EXE.stat().st_size/1048576:.1f} MB)")


if __name__ == "__main__":
    main()
