"""SMTP 发送（Gmail）。凭据优先读环境变量（GitHub Actions secrets），不打印、不入仓库。

环境变量：GMAIL_USER（默认 jakeguo117@gmail.com）、GMAIL_APP_PASSWORD（Gmail 应用专用密码）、MAIL_TO（默认 394592848@qq.com）。
仅在 macOS 上作为可选回退：钥匙串 service=ashare-tactic12-smtp, account=GMAIL_USER；
再回退 $STATE_DIR/.secrets/gmail_app_password（必须 chmod 600）。
"""
from __future__ import annotations

import os
import smtplib
import subprocess
import sys
from email.header import Header
from email.mime.text import MIMEText
from pathlib import Path

SERVICE = "ashare-tactic12-smtp"
DEFAULT_SENDER = "jakeguo117@gmail.com"
DEFAULT_TO = "394592848@qq.com"
HOST, PORT = "smtp.gmail.com", 465
MAC_STATE_DIR = "/Volumes/Disk/项目/data-cache/tactic12_paper"


def mail_config(env=None) -> dict:
    env = os.environ if env is None else env
    return {"sender": env.get("GMAIL_USER") or DEFAULT_SENDER,
            "to": [x.strip() for x in (env.get("MAIL_TO") or DEFAULT_TO).split(",") if x.strip()],
            "secret_file": Path(env.get("STATE_DIR") or MAC_STATE_DIR) / ".secrets" / "gmail_app_password"}


def _secret(env=None, platform: str | None = None) -> str | None:
    env = os.environ if env is None else env
    pw = (env.get("GMAIL_APP_PASSWORD") or "").strip()
    if pw:
        return pw
    cfg = mail_config(env)
    if (platform or sys.platform) == "darwin":  # 钥匙串仅 macOS 可选回退
        try:
            r = subprocess.run(["security", "find-generic-password", "-s", SERVICE, "-a", cfg["sender"], "-w"],
                               capture_output=True, text=True)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
        except OSError:
            pass
    try:
        f = cfg["secret_file"]
        if oct(os.stat(f).st_mode & 0o777) == "0o600":
            v = f.read_text(encoding="utf-8").strip()
            return v or None
    except OSError:
        pass
    return None


def send_mail(subject: str, body: str) -> str:
    cfg = mail_config()
    pw = _secret()
    if pw is None:
        return "未发送：缺少 GMAIL_APP_PASSWORD（或 macOS 钥匙串/本地 600 文件）"
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = Header(subject, "utf-8"); msg["From"] = cfg["sender"]; msg["To"] = ", ".join(cfg["to"])
    with smtplib.SMTP_SSL(HOST, PORT, timeout=30) as s:
        s.login(cfg["sender"], pw)
        s.sendmail(cfg["sender"], cfg["to"], msg.as_string())
    return "已发送"
