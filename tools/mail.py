"""
mail.py — Gmail Tool
======================
Gmail integration via google-api-python-client with OAuth2.

Supports:
  - read_inbox: list recent inbox messages
  - read_email: read a specific email by ID
  - send_email: send an email
  - search: search emails by query

Requires:
  - pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
  - credentials.json from Google Cloud Console (OAuth2 Desktop client)
  - Gmail API enabled in your Google Cloud project

First run will open a browser for OAuth2 consent.
"""

from __future__ import annotations

import base64
import os
from email.mime.text import MIMEText
from typing import Optional

from tools.base import Tool, ToolResult


# Gmail API scopes
GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
]


# ─────────────────────────────────────────────
# GmailTool
# ─────────────────────────────────────────────
class GmailTool(Tool):
    """
    Gmail tool: read, send, and search emails via Gmail API.

    OAuth2 authentication is handled lazily on first use.
    Requires credentials.json from Google Cloud Console.

    Usage:
        tool = GmailTool(credentials_path="credentials.json")
        result = tool.execute(action="read_inbox", max_results=5)
        result = tool.execute(action="send_email", to="user@example.com",
                              subject="Hello", body="Hi there!")
    """

    def __init__(
        self,
        credentials_path: str = "credentials.json",
        token_path: str = "token.json",
    ):
        self._credentials_path = credentials_path
        self._token_path = token_path
        self._service = None  # lazy init

    @property
    def name(self) -> str:
        return "gmail"

    @property
    def description(self) -> str:
        return (
            "Read, send, and search Gmail messages. "
            "Actions: read_inbox, read_email, send_email, search."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["read_inbox", "read_email", "send_email", "search"],
                    "description": "The Gmail action to perform",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max messages to return (read_inbox, search)",
                    "default": 5,
                },
                "message_id": {
                    "type": "string",
                    "description": "Gmail message ID (read_email)",
                },
                "to": {
                    "type": "string",
                    "description": "Recipient email address (send_email)",
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject (send_email)",
                },
                "body": {
                    "type": "string",
                    "description": "Email body text (send_email)",
                },
                "query": {
                    "type": "string",
                    "description": "Gmail search query (search)",
                },
            },
            "required": ["action"],
        }

    def execute(self, **kwargs) -> ToolResult:
        """
        Execute a Gmail action.

        Args:
            action:      "read_inbox" | "read_email" | "send_email" | "search"
            max_results: max messages (for read_inbox, search)
            message_id:  message ID (for read_email)
            to:          recipient (for send_email)
            subject:     subject (for send_email)
            body:        body text (for send_email)
            query:       search query (for search)

        Returns:
            ToolResult with the action's output
        """
        action = kwargs.get("action", "")

        try:
            if action == "read_inbox":
                return self._read_inbox(kwargs.get("max_results", 5))
            elif action == "read_email":
                msg_id = kwargs.get("message_id")
                if not msg_id:
                    return ToolResult(
                        name=self.name, success=False,
                        output="Error: message_id is required for read_email"
                    )
                return self._read_email(msg_id)
            elif action == "send_email":
                to = kwargs.get("to")
                subject = kwargs.get("subject", "(no subject)")
                body = kwargs.get("body", "")
                if not to:
                    return ToolResult(
                        name=self.name, success=False,
                        output="Error: 'to' is required for send_email"
                    )
                return self._send_email(to, subject, body)
            elif action == "search":
                query = kwargs.get("query", "")
                if not query:
                    return ToolResult(
                        name=self.name, success=False,
                        output="Error: 'query' is required for search"
                    )
                return self._search(query, kwargs.get("max_results", 5))
            else:
                return ToolResult(
                    name=self.name, success=False,
                    output=f"Error: Unknown action '{action}'. "
                           f"Valid: read_inbox, read_email, send_email, search"
                )
        except Exception as e:
            return ToolResult(
                name=self.name, success=False,
                output=f"Gmail error: {type(e).__name__}: {e}"
            )

    # ── Gmail API authentication ─────────────────────────────────────────
    def _get_service(self):
        """Lazy OAuth2 authentication and Gmail service build."""
        if self._service is not None:
            return self._service

        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build
        except ImportError:
            raise ImportError(
                "Gmail tool requires: pip install "
                "google-api-python-client google-auth-httplib2 google-auth-oauthlib"
            )

        creds = None

        # Load saved token
        if os.path.exists(self._token_path):
            creds = Credentials.from_authorized_user_file(
                self._token_path, GMAIL_SCOPES
            )

        # Refresh or run OAuth flow
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self._credentials_path):
                    raise FileNotFoundError(
                        f"Gmail credentials not found at '{self._credentials_path}'. "
                        f"Download from Google Cloud Console → APIs → Credentials → "
                        f"OAuth 2.0 Client IDs → Desktop application."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    self._credentials_path, GMAIL_SCOPES
                )
                creds = flow.run_local_server(port=0)

            # Save token for future runs
            with open(self._token_path, "w") as f:
                f.write(creds.to_json())

        self._service = build("gmail", "v1", credentials=creds)
        return self._service

    # ── Action implementations ───────────────────────────────────────────
    def _read_inbox(self, max_results: int = 5) -> ToolResult:
        """Read recent inbox messages."""
        service = self._get_service()

        results = service.users().messages().list(
            userId="me",
            labelIds=["INBOX"],
            maxResults=max_results,
        ).execute()

        messages = results.get("messages", [])
        if not messages:
            return ToolResult(
                name=self.name, success=True,
                output="No messages in inbox."
            )

        output_lines = [f"Found {len(messages)} messages:\n"]

        for msg_info in messages:
            msg = service.users().messages().get(
                userId="me", id=msg_info["id"], format="metadata",
                metadataHeaders=["Subject", "From", "Date"],
            ).execute()

            headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
            snippet = msg.get("snippet", "")[:100]

            output_lines.append(
                f"  ID: {msg_info['id']}\n"
                f"  From: {headers.get('From', 'unknown')}\n"
                f"  Subject: {headers.get('Subject', '(no subject)')}\n"
                f"  Date: {headers.get('Date', 'unknown')}\n"
                f"  Preview: {snippet}\n"
            )

        return ToolResult(name=self.name, success=True, output="\n".join(output_lines))

    def _read_email(self, message_id: str) -> ToolResult:
        """Read a specific email by ID."""
        service = self._get_service()

        msg = service.users().messages().get(
            userId="me", id=message_id, format="full"
        ).execute()

        headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}

        # Extract body text
        body = self._extract_body(msg.get("payload", {}))

        output = (
            f"From: {headers.get('From', 'unknown')}\n"
            f"To: {headers.get('To', 'unknown')}\n"
            f"Subject: {headers.get('Subject', '(no subject)')}\n"
            f"Date: {headers.get('Date', 'unknown')}\n"
            f"\n{body}"
        )

        return ToolResult(name=self.name, success=True, output=output)

    def _send_email(self, to: str, subject: str, body: str) -> ToolResult:
        """Send an email."""
        service = self._get_service()

        message = MIMEText(body)
        message["to"] = to
        message["subject"] = subject

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

        sent = service.users().messages().send(
            userId="me",
            body={"raw": raw},
        ).execute()

        return ToolResult(
            name=self.name, success=True,
            output=f"Email sent successfully. Message ID: {sent.get('id', 'unknown')}"
        )

    def _search(self, query: str, max_results: int = 5) -> ToolResult:
        """Search emails by query."""
        service = self._get_service()

        results = service.users().messages().list(
            userId="me",
            q=query,
            maxResults=max_results,
        ).execute()

        messages = results.get("messages", [])
        if not messages:
            return ToolResult(
                name=self.name, success=True,
                output=f"No messages matching query: '{query}'"
            )

        output_lines = [f"Found {len(messages)} messages matching '{query}':\n"]

        for msg_info in messages:
            msg = service.users().messages().get(
                userId="me", id=msg_info["id"], format="metadata",
                metadataHeaders=["Subject", "From", "Date"],
            ).execute()

            headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
            snippet = msg.get("snippet", "")[:100]

            output_lines.append(
                f"  ID: {msg_info['id']}\n"
                f"  From: {headers.get('From', 'unknown')}\n"
                f"  Subject: {headers.get('Subject', '(no subject)')}\n"
                f"  Preview: {snippet}\n"
            )

        return ToolResult(name=self.name, success=True, output="\n".join(output_lines))

    # ── Helpers ──────────────────────────────────────────────────────────
    def _extract_body(self, payload: dict) -> str:
        """Extract plain text body from Gmail message payload."""
        # Direct body
        if payload.get("body", {}).get("data"):
            return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="replace")

        # Multipart: look for text/plain
        parts = payload.get("parts", [])
        for part in parts:
            if part.get("mimeType") == "text/plain":
                data = part.get("body", {}).get("data", "")
                if data:
                    return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")

        # Fallback: try first part
        if parts:
            data = parts[0].get("body", {}).get("data", "")
            if data:
                return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")

        return "(no readable body)"
