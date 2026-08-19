"""Multi-channel alert delivery with offline queueing.

Channels: log (always), email (SMTP), SMS (Twilio or Africa's Talking), and a
DHIS2 push so the alert lands in the system officials already use.

Two behaviours matter more than the channel list:

* **nothing is lost offline** — when a channel is unreachable the alert is
  queued to disk and re-sent by the sync manager once connectivity returns
  (critical rule #6);
* **SMS is a first-class channel** — roughly 87% of Tanzanian handsets are
  feature phones, so the SMS body is a complete, actionable message, not a link
  to a dashboard the recipient cannot open.
"""

from __future__ import annotations

import json
import smtplib
from dataclasses import dataclass, field
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from config.settings import get_settings
from src.core.config_loader import load_alert_rules
from src.core.logging import get_logger
from src.core.types import Alert

log = get_logger("alerting.notify")


@dataclass
class DeliveryReport:
    """Per-channel outcome for one alert."""

    alert_id: str
    results: Dict[str, str] = field(default_factory=dict)

    @property
    def delivered(self) -> List[str]:
        return [c for c, status in self.results.items() if status == "sent"]

    @property
    def queued(self) -> List[str]:
        return [c for c, status in self.results.items() if status == "queued"]


class NotificationService:
    """Dispatch alerts across the channels configured for their severity."""

    def __init__(self, rules: Optional[dict] = None, settings=None) -> None:
        self.settings = settings or get_settings()
        self.rules = rules if rules is not None else load_alert_rules()
        self.delivery_rules = self.rules.get("delivery", {})
        self.recipients = self.rules.get("recipients", {})

    # -- queue -------------------------------------------------------------
    @property
    def queue_path(self) -> Path:
        path = Path(self.settings.data_dir) / "outbox"
        path.mkdir(parents=True, exist_ok=True)
        return path / "pending_notifications.jsonl"

    def queue(self, alert: Alert, channel: str, reason: str) -> None:
        payload = {
            "queued_at": datetime.utcnow().isoformat(timespec="seconds"),
            "channel": channel,
            "reason": reason,
            "alert": alert.model_dump(),
        }
        with open(self.queue_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")
        log.info("queued %s delivery of alert %s (%s)", channel, alert.alert_id, reason)

    def pending(self) -> List[dict]:
        if not self.queue_path.exists():
            return []
        out = []
        with open(self.queue_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return out

    def flush_queue(self) -> Dict[str, int]:
        """Retry every queued delivery; keep whatever still fails."""
        pending = self.pending()
        if not pending:
            return {"attempted": 0, "sent": 0, "still_queued": 0}
        remaining: List[dict] = []
        sent = 0
        for item in pending:
            alert = Alert.model_validate(item["alert"])
            status = self._dispatch(alert, item["channel"], queue_on_failure=False)
            if status == "sent":
                sent += 1
            else:
                remaining.append(item)
        with open(self.queue_path, "w", encoding="utf-8") as handle:
            for item in remaining:
                handle.write(json.dumps(item, default=str) + "\n")
        return {"attempted": len(pending), "sent": sent, "still_queued": len(remaining)}

    # -- dispatch ----------------------------------------------------------
    def channels_for(self, alert: Alert) -> List[str]:
        by_level = self.delivery_rules.get("by_level", {})
        return list(by_level.get(alert.risk_level, ["log"]))

    def send(self, alert: Alert) -> DeliveryReport:
        report = DeliveryReport(alert_id=alert.alert_id)
        for channel in self.channels_for(alert):
            report.results[channel] = self._dispatch(alert, channel)
        alert.delivery_status = dict(report.results)
        return report

    def send_many(self, alerts: Sequence[Alert]) -> List[DeliveryReport]:
        return [self.send(alert) for alert in alerts]

    def _dispatch(self, alert: Alert, channel: str, queue_on_failure: bool = True) -> str:
        try:
            if channel == "log":
                return self._send_log(alert)
            if channel == "email":
                return self._send_email(alert)
            if channel == "sms":
                return self._send_sms(alert)
            if channel == "dhis2":
                return self._send_dhis2(alert)
            log.warning("unknown notification channel %r", channel)
            return "unsupported"
        except Exception as exc:  # noqa: BLE001 - delivery must never crash the run
            log.warning("%s delivery failed for %s: %s", channel, alert.alert_id, exc)
            if queue_on_failure:
                self.queue(alert, channel, str(exc))
                return "queued"
            return "failed"

    # -- channels ----------------------------------------------------------
    def _send_log(self, alert: Alert) -> str:
        log.warning(
            "[%s] %s in %s (%s) week %s: %.0f cases (%.2f/1000). %s",
            alert.risk_level.upper(),
            alert.disease,
            alert.district,
            alert.region,
            alert.target_week,
            alert.predicted_cases,
            alert.predicted_incidence_per_1000,
            alert.recommendations[0].action if alert.recommendations else "no action configured",
        )
        return "sent"

    def _send_email(self, alert: Alert) -> str:
        if not self.settings.smtp_host:
            self.queue(alert, "email", "SMTP not configured")
            return "queued"
        addresses = [
            details.get("email")
            for name, details in self.recipients.items()
            if isinstance(details, dict) and details.get("email")
        ]
        if not addresses:
            self.queue(alert, "email", "no recipient addresses configured")
            return "queued"

        message = EmailMessage()
        message["Subject"] = (
            f"[AFYA-PREDICT {alert.risk_level.upper()}] {alert.disease} — "
            f"{alert.district}, week {alert.target_week}"
        )
        message["From"] = self.settings.alert_email_from
        message["To"] = ", ".join(addresses)
        message.set_content(self.render_email(alert))

        with smtplib.SMTP(self.settings.smtp_host, self.settings.smtp_port, timeout=30) as server:
            server.starttls()
            if self.settings.smtp_username:
                server.login(self.settings.smtp_username, self.settings.smtp_password or "")
            server.send_message(message)
        return "sent"

    def _send_sms(self, alert: Alert) -> str:
        numbers = [
            details.get("phone")
            for name, details in self.recipients.items()
            if isinstance(details, dict) and details.get("phone")
        ]
        if not numbers:
            self.queue(alert, "sms", "no recipient numbers configured")
            return "queued"
        body = self.render_sms(alert)

        if self.settings.africastalking_api_key and self.settings.africastalking_username:
            import requests

            response = requests.post(
                "https://api.africastalking.com/version1/messaging",
                data={"username": self.settings.africastalking_username,
                      "to": ",".join(numbers), "message": body},
                headers={"apiKey": self.settings.africastalking_api_key,
                         "Accept": "application/json"},
                timeout=30,
            )
            response.raise_for_status()
            return "sent"

        if self.settings.twilio_account_sid and self.settings.twilio_auth_token:
            from twilio.rest import Client

            client = Client(self.settings.twilio_account_sid, self.settings.twilio_auth_token)
            for number in numbers:
                client.messages.create(
                    body=body, from_=self.settings.twilio_from_number, to=number
                )
            return "sent"

        self.queue(alert, "sms", "no SMS gateway configured")
        return "queued"

    def _send_dhis2(self, alert: Alert) -> str:
        """Push the alert into DHIS2 as a message so it reaches the existing workflow."""
        if not self.settings.has_dhis2():
            self.queue(alert, "dhis2", "DHIS2 not configured")
            return "queued"
        import requests

        base = self.settings.dhis2_base_url.rstrip("/")
        payload = {
            "subject": (
                f"AFYA-PREDICT {alert.risk_level.upper()}: {alert.disease} — {alert.district}"
            ),
            "text": self.render_email(alert),
        }
        response = requests.post(
            f"{base}/api/messageConversations",
            json=payload,
            auth=(self.settings.dhis2_username, self.settings.dhis2_password),
            timeout=60,
        )
        response.raise_for_status()
        return "sent"

    # -- rendering ---------------------------------------------------------
    def render_sms(self, alert: Alert) -> str:
        from src.explainability.natural_language import sms_summary

        return sms_summary(
            disease=alert.disease,
            district=alert.district,
            risk_level=alert.risk_level,
            predicted_cases=alert.predicted_cases,
            target_week=alert.target_week,
            drivers=alert.top_drivers,
            top_action=alert.recommendations[0].action if alert.recommendations else None,
            low_confidence=alert.low_data_confidence,
        )

    def render_email(self, alert: Alert) -> str:
        lines = [
            f"AFYA-PREDICT {alert.risk_level.upper()} ALERT",
            "=" * 60,
            f"Disease        : {alert.disease}",
            f"District       : {alert.district} ({alert.region})",
            f"Target week    : {alert.target_week}  (lead time {alert.lead_time_weeks} weeks)",
            f"Forecast       : {alert.predicted_cases:,.0f} cases "
            f"({alert.predicted_incidence_per_1000:.2f} per 1,000)",
            f"Threshold      : {alert.threshold_crossed:.2f} per 1,000",
            f"Risk score     : {alert.risk_score:.2f}",
            "",
            "WHY THIS ALERT",
            "-" * 60,
            alert.explanation,
            "",
        ]
        if alert.top_drivers:
            lines.append("TOP DRIVERS")
            lines.append("-" * 60)
            for driver in alert.top_drivers[:5]:
                lines.append(
                    f"  - {driver.proxy or driver.feature}"
                    + (f" (lag {driver.lag_weeks}w)" if driver.lag_weeks else "")
                    + f": {driver.contribution_share:.0%}, {driver.direction}"
                )
            lines.append("")
        if alert.source_districts:
            lines.append("IMPORTATION SOURCES")
            lines.append("-" * 60)
            for source in alert.source_districts[:5]:
                lines.append(
                    f"  - {source.district}: {source.contributed_risk:.0%} of imported risk"
                )
            lines.append("")
        if alert.recommendations:
            lines.append("RECOMMENDED ACTIONS")
            lines.append("-" * 60)
            for i, rec in enumerate(alert.recommendations, 1):
                lines.append(f"  {i}. {rec.action}")
                lines.append(f"     within {rec.timeframe_days} days | owner: {rec.responsible}")
                if rec.quantity:
                    lines.append(f"     quantity: {rec.quantity}")
            lines.append("")
        if alert.escalation_reason:
            lines.extend(["ESCALATION", "-" * 60, alert.escalation_reason, ""])
        if alert.data_quality_flags:
            lines.extend(["DATA QUALITY NOTES", "-" * 60])
            lines.extend(f"  - {flag}" for flag in alert.data_quality_flags[:5])
            lines.append("")
        lines.append(f"Alert ID: {alert.alert_id}")
        lines.append(f"Issued  : {alert.issued_at:%Y-%m-%d %H:%M UTC}")
        return "\n".join(lines)
