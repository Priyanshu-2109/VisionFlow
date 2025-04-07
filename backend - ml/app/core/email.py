"""
Email management system for the application.
Handles all email-related operations including verification, password reset, and alerts.
"""

from typing import Any, Dict, Optional
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from app.core.config import settings

# Email configuration using FastMail
conf = ConnectionConfig(
    MAIL_USERNAME=settings.MAIL_USERNAME,
    MAIL_PASSWORD=settings.MAIL_PASSWORD,
    MAIL_FROM=settings.MAIL_FROM,
    MAIL_PORT=settings.MAIL_PORT,
    MAIL_SERVER=settings.MAIL_SERVER,
    MAIL_FROM_NAME=settings.MAIL_FROM_NAME,
    MAIL_STARTTLS=settings.MAIL_STARTTLS,
    MAIL_SSL_TLS=settings.MAIL_SSL_TLS,
    USE_CREDENTIALS=True,
    VALIDATE_CERTS=True,
    TEMPLATE_FOLDER=Path(__file__).parent / 'email_templates'
)

# Template environment for rendering email templates
template_env = Environment(
    loader=FileSystemLoader(Path(__file__).parent / 'email_templates')
)

class EmailManager:
    """Manages all email operations in the application."""
    
    def __init__(self):
        self.fastmail = FastMail(conf)

    async def send_email(
        self,
        email_to: str,
        subject_template: str,
        html_template: str,
        environment: Dict[str, Any],
        subject: Optional[str] = None
    ) -> bool:
        """
        Generic method to send emails using templates.
        
        Args:
            email_to: Recipient email address
            subject_template: Template for email subject
            html_template: Template for email body
            environment: Variables to be used in templates
            subject: Optional custom subject line
            
        Returns:
            bool: True if email was sent successfully, False otherwise
        """
        try:
            # Render email templates
            subject_content = template_env.get_template(subject_template).render(**environment)
            html_content = template_env.get_template(html_template).render(**environment)

            # Create and send email message
            message = MessageSchema(
                subject=subject or subject_content,
                recipients=[email_to],
                body=html_content,
                subtype="html"
            )
            await self.fastmail.send_message(message)
            return True

        except Exception as e:
            print(f"Error sending email: {str(e)}")
            return False

    async def send_verification_email(self, email_to: str, token: str) -> bool:
        """
        Send email verification link to new users.
        
        Args:
            email_to: User's email address
            token: Verification token
            
        Returns:
            bool: True if email was sent successfully
        """
        verification_url = f"{settings.FRONTEND_URL}/verify-email?token={token}"
        environment = {
            "project_name": settings.PROJECT_NAME,
            "username": email_to.split("@")[0],
            "verification_url": verification_url,
            "valid_hours": settings.EMAIL_TOKEN_EXPIRE_HOURS
        }
        return await self.send_email(
            email_to=email_to,
            subject_template="verification_email_subject.txt",
            html_template="verification_email.html",
            environment=environment
        )

    async def send_password_reset_email(self, email_to: str, token: str) -> bool:
        """
        Send password reset link to users.
        
        Args:
            email_to: User's email address
            token: Password reset token
            
        Returns:
            bool: True if email was sent successfully
        """
        reset_url = f"{settings.FRONTEND_URL}/reset-password?token={token}"
        environment = {
            "project_name": settings.PROJECT_NAME,
            "username": email_to.split("@")[0],
            "reset_url": reset_url,
            "valid_hours": settings.EMAIL_TOKEN_EXPIRE_HOURS
        }
        return await self.send_email(
            email_to=email_to,
            subject_template="reset_password_subject.txt",
            html_template="reset_password.html",
            environment=environment
        )

    async def send_api_key_email(self, email_to: str, api_key: str) -> bool:
        """
        Send API key to users.
        
        Args:
            email_to: User's email address
            api_key: Generated API key
            
        Returns:
            bool: True if email was sent successfully
        """
        environment = {
            "project_name": settings.PROJECT_NAME,
            "username": email_to.split("@")[0],
            "api_key": api_key,
            "frontend_url": settings.FRONTEND_URL
        }
        return await self.send_email(
            email_to=email_to,
            subject_template="api_key_email_subject.txt",
            html_template="api_key_email.html",
            environment=environment
        )

    async def send_alert_email(
        self,
        email_to: str,
        alert_type: str,
        alert_message: str,
        alert_details: Dict[str, Any]
    ) -> bool:
        """
        Send alert notifications to users.
        
        Args:
            email_to: User's email address
            alert_type: Type of alert (e.g., 'critical', 'warning')
            alert_message: Alert message
            alert_details: Additional alert information
            
        Returns:
            bool: True if email was sent successfully
        """
        environment = {
            "project_name": settings.PROJECT_NAME,
            "username": email_to.split("@")[0],
            "alert_type": alert_type,
            "alert_message": alert_message,
            "alert_details": alert_details,
            "frontend_url": settings.FRONTEND_URL
        }
        return await self.send_email(
            email_to=email_to,
            subject_template="alert_email_subject.txt",
            html_template="alert_email.html",
            environment=environment
        )

# Global email manager instance
email_manager = EmailManager() 