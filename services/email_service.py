import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from Database.DBmodels.database_models import User

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_SENDER_EMAIL = "rohitvijayadapa2006@gmail.com"

class EmailService:
    def __init__(self, db_session: Optional[Session] = None):
        self.db_session = db_session
        self.sender_password = os.getenv("SMTP_PASSWORD", "")
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.use_tls = os.getenv("SMTP_USE_TLS", "true").lower() == "true"

        # Validate configuration
        if not self.sender_password:
            logger.warning("Email service not fully configured. Set SMTP_PASSWORD environment variable.")
            self.enabled = False
        else:
            self.enabled = True

    def _get_sender_email(self, user_id: Optional[int]) -> str:
        if self.db_session and user_id is not None:
            try:
                user = self.db_session.query(User).filter(User.id == user_id).first()
                if user and user.email:
                    return user.email
            except Exception as e:
                logger.warning(f"Failed to fetch email for user_id {user_id}: {e}")
        # Fallback to default email
        return DEFAULT_SENDER_EMAIL

    def send_job_notification(self, recipient_email: str, job_data: Dict[str, Any], notification_type: str, user_id: Optional[int] = None) -> bool:
        """
        Send email notification for job status updates

        Args:
            recipient_email: Email address of the recipient
            job_data: Dictionary containing job information
            notification_type: Type of notification ('submitted', 'running', 'completed', 'failed')
            user_id: Optional user ID to fetch sender email from database

        Returns:
            bool: True if email sent successfully, False otherwise
        """
        if not self.enabled:
            print(f"📧 [EMAIL DISABLED] Email service is disabled - skipping {notification_type} notification for job {job_data.get('job_id', 'Unknown')}")
            logger.info("Email service disabled - skipping notification")
            return False

        job_id = job_data.get('id', 'Unknown')
        print(f"📧 [EMAIL TRIGGERED] Sending {notification_type} notification for job {job_id} to {recipient_email}")
        logger.info(f"Email notification triggered: {notification_type} for job {job_id}")

        sender_email = self._get_sender_email(user_id)

        try:
            # Create message
            message = MIMEMultipart()
            message["From"] = sender_email
            message["To"] = recipient_email

            # Set subject based on notification type
            job_id = job_data.get('id', 'Unknown')
            device_name = job_data.get('backend', 'Unknown Device')

            subject_map = {
                'submitted': f"Quantum Job Submitted - {job_id}",
                'queued': f"Quantum Job Queued - {job_id}",
                'running': f"Quantum Job Running - {job_id}",
                'completed': f"Quantum Job Completed - {job_id}",
                'failed': f"Quantum Job Failed - {job_id}",
                'retrying': f"Quantum Job Retrying - {job_id}"
            }

            message["Subject"] = subject_map.get(notification_type, f"Quantum Job Update - {job_id}")

            # Create email body
            body = self._create_email_body(job_data, notification_type)
            message.attach(MIMEText(body, "html"))

            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            if self.use_tls:
                server.starttls()

            server.login(sender_email, self.sender_password)
            text = message.as_string()
            server.sendmail(sender_email, recipient_email, text)
            server.quit()

            print(f"📧 [EMAIL SUCCESS] ✅ Email sent successfully for {notification_type} notification - Job {job_id} to {recipient_email}")
            logger.info(f"Email notification sent successfully to {recipient_email} from {sender_email} for job {job_id}")
            return True

        except Exception as e:
            print(f"📧 [EMAIL FAILED] ❌ Failed to send {notification_type} notification for job {job_id}: {str(e)}")
            logger.error(f"Failed to send email notification: {e}")
            return False

    def _create_email_body(self, job_data: Dict[str, Any], notification_type: str) -> str:
        """Create HTML email body for job notifications"""

        job_id = job_data.get('id', 'Unknown')
        device_name = job_data.get('backend', 'Unknown Device')
        shots = job_data.get('shots', 'Unknown')
        submitted_at = job_data.get('submitted_at', 'Unknown')
        status = job_data.get('status', 'Unknown')

        # Format timestamps
        try:
            if submitted_at != 'Unknown':
                submitted_at = datetime.fromisoformat(submitted_at.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S UTC')
        except:
            pass

        # Status colors
        status_colors = {
            'submitted': '#3B82F6',  # Blue
            'queued': '#8B5CF6',     # Purple
            'running': '#F59E0B',    # Yellow/Orange
            'completed': '#10B981',  # Green
            'failed': '#EF4444',     # Red
            'retrying': '#F97316'    # Orange
        }

        color = status_colors.get(notification_type, '#6B7280')

        # Create HTML body
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Quantum Job Notification</title>
        </head>
        <body style="font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f4f4f4;">
            <div style="max-width: 600px; margin: 0 auto; background-color: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <!-- Header -->
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px 20px; text-align: center;">
                    <h1 style="color: white; margin: 0; font-size: 24px;">⚛️ Quantum Job Notification</h1>
                </div>

                <!-- Content -->
                <div style="padding: 30px 20px;">
                    <div style="text-align: center; margin-bottom: 30px;">
                        <div style="display: inline-block; padding: 10px 20px; background-color: {color}; color: white; border-radius: 20px; font-weight: bold;">
                            {notification_type.upper()} - Job {job_id}
                        </div>
                    </div>

                    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                        <h2 style="margin-top: 0; color: #333;">Job Details</h2>
                        <table style="width: 100%; border-collapse: collapse;">
                            <tr>
                                <td style="padding: 8px 0; border-bottom: 1px solid #dee2e6; font-weight: bold;">Job ID:</td>
                                <td style="padding: 8px 0; border-bottom: 1px solid #dee2e6;">{job_id}</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px 0; border-bottom: 1px solid #dee2e6; font-weight: bold;">Device:</td>
                                <td style="padding: 8px 0; border-bottom: 1px solid #dee2e6;">{device_name}</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px 0; border-bottom: 1px solid #dee2e6; font-weight: bold;">Shots:</td>
                                <td style="padding: 8px 0; border-bottom: 1px solid #dee2e6;">{shots}</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px 0; border-bottom: 1px solid #dee2e6; font-weight: bold;">Status:</td>
                                <td style="padding: 8px 0; border-bottom: 1px solid #dee2e6;">
                                    <span style="color: {color}; font-weight: bold;">{status.upper()}</span>
                                </td>
                            </tr>
                            <tr>
                                <td style="padding: 8px 0; border-bottom: 1px solid #dee2e6; font-weight: bold;">Submitted:</td>
                                <td style="padding: 8px 0; border-bottom: 1px solid #dee2e6;">{submitted_at}</td>
                            </tr>
        """

        # Add additional information based on notification type
        if notification_type == 'completed':
            completed_at = job_data.get('completed_at', 'Unknown')
            try:
                if completed_at != 'Unknown':
                    completed_at = datetime.fromisoformat(completed_at.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S UTC')
            except:
                pass

            html_body += f"""
                            <tr>
                                <td style="padding: 8px 0; border-bottom: 1px solid #dee2e6; font-weight: bold;">Completed:</td>
                                <td style="padding: 8px 0; border-bottom: 1px solid #dee2e6;">{completed_at}</td>
                            </tr>
            """

        elif notification_type == 'failed':
            error_message = job_data.get('error_message', 'Unknown error occurred')
            html_body += f"""
                            <tr>
                                <td style="padding: 8px 0; border-bottom: 1px solid #dee2e6; font-weight: bold;">Error:</td>
                                <td style="padding: 8px 0; border-bottom: 1px solid #dee2e6; color: #dc3545;">{error_message}</td>
                            </tr>
            """

        html_body += """
                        </table>
                    </div>

                    <div style="text-align: center; margin: 30px 0;">
                        <a href="#" style="background-color: #667eea; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: bold; display: inline-block;">
                            View Job Details
                        </a>
                    </div>

                    <div style="background-color: #e9ecef; padding: 15px; border-radius: 6px; margin-top: 20px;">
                        <p style="margin: 0; font-size: 14px; color: #6c757d; text-align: center;">
                            This is an automated notification from your Quantum Dashboard.<br>
                            You can manage your notification preferences in your account settings.
                        </p>
                    </div>
                </div>

                <!-- Footer -->
                <div style="background-color: #f8f9fa; padding: 20px; text-align: center; border-top: 1px solid #dee2e6;">
                    <p style="margin: 0; font-size: 12px; color: #6c757d;">
                        Quantum Dashboard | Real-time Quantum Computing Platform
                    </p>
                </div>
            </div>
        </body>
        </html>
        """

        return html_body

# Global email service instance
email_service = EmailService()
