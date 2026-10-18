"""SMTP 发送（Gmail）。凭据从 macOS 钥匙串读取，失败时读 chmod 600 的本地文件；不打印、不入仓库。

钥匙串条目（generic password）：service=ashare-tactic12-smtp, account=jakeguo117@gmail.com。备用文件：/Volumes/Disk/项目/data-cache/tactic12_paper/.secrets/gmail_app_password
添加：security add-generic-password -s ashare-tactic12-smtp -a jakeguo117@gmail.com -w   （交互输入 Gmail 应用专用密码）
"""
from __future__ import annotations

import smtplib
import subprocess
from email.header import Header
from email.mime.text import MIMEText

SERVICE = "ashare-tactic12-smtp"
SENDER = "jakeguo117@gmail.com"
TO = "394592848@qq.com"
HOST, PORT = "smtp.gmail.com", 465
SECRET_FILE = "/Volumes/Disk/项目/data-cache/tactic12_paper/.secrets/gmail_app_password"


def _secret() -> str | None:
    r = subprocess.run(["security", "find-generic-password", "-s", SERVICE, "-a", SENDER, "-w"], capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        return r.stdout.strip()
    try:
        import os
        if oct(os.stat(SECRET_FILE).st_mode & 0o777) == "0o600":
            v = open(SECRET_FILE, encoding="utf-8").read().strip()
            return v or None
    except OSError:
        pass
    return None


def send_mail(subject: str, body: str) -> str:
    pw = _secret()
    if pw is None:
        return f"未发送：钥匙串缺少 {SERVICE}/{SENDER}（Gmail 应用专用密码）"
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = Header(subject, "utf-8"); msg["From"] = SENDER; msg["To"] = TO
    with smtplib.SMTP_SSL(HOST, PORT, timeout=30) as s:
        s.login(SENDER, pw)
        s.sendmail(SENDER, [TO], msg.as_string())
    return "已发送"
