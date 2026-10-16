"""SMTP 发送。凭据只从 macOS 钥匙串读取，不打印、不写入文件或仓库。

钥匙串条目（generic password）：service=ashare-tactic12-smtp, account=发件邮箱。
添加：security add-generic-password -s ashare-tactic12-smtp -a 394592848@qq.com -w   （交互输入 QQ 邮箱 SMTP 授权码）
"""
from __future__ import annotations

import smtplib
import subprocess
from email.header import Header
from email.mime.text import MIMEText

SERVICE = "ashare-tactic12-smtp"
SENDER = "394592848@qq.com"
TO = "394592848@qq.com"
HOST, PORT = "smtp.qq.com", 465


def _secret() -> str | None:
    r = subprocess.run(["security", "find-generic-password", "-s", SERVICE, "-a", SENDER, "-w"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def send_mail(subject: str, body: str) -> str:
    pw = _secret()
    if pw is None:
        return f"未发送：钥匙串缺少 {SERVICE}/{SENDER}（QQ邮箱SMTP授权码）"
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = Header(subject, "utf-8"); msg["From"] = SENDER; msg["To"] = TO
    with smtplib.SMTP_SSL(HOST, PORT, timeout=30) as s:
        s.login(SENDER, pw)
        s.sendmail(SENDER, [TO], msg.as_string())
    return "已发送"
