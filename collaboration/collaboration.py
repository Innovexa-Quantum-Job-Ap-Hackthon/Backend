import secrets
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from typing import List, Optional
import logging
import asyncio
import concurrent.futures

# FIXED: Use absolute imports instead of relative
import Database.DBmodels.database_models as models
import Database.DBmodels.schemas as schemas

logger = logging.getLogger(__name__)

class CollaborationService:
    def __init__(self, db: Session):
        self.db = db

    def create_team(self, team: schemas.TeamCreate, user_id: int) -> models.Team:
        """Create a new team and add creator as admin"""
        # Check if team name already exists
        existing_team = self.db.query(models.Team).filter(
            models.Team.name == team.name
        ).first()

        if existing_team:
            raise ValueError("Team name already exists")

        db_team = models.Team(
            name=team.name,
            description=team.description,
            created_by_id=user_id
        )
        self.db.add(db_team)
        self.db.flush()

        # Add creator as admin member
        db_member = models.TeamMember(
            team_id=db_team.id,
            user_id=user_id,
            role="admin"
        )
        self.db.add(db_member)
        self.db.commit()
        self.db.refresh(db_team)

        logger.info(f"Team '{team.name}' created by user {user_id}")
        return db_team

    def invite_to_team(self, team_id: int, email: str, inviter_id: int) -> models.TeamInvitation:
        """Create a team invitation"""
        # Check if team exists
        team = self.db.query(models.Team).filter(models.Team.id == team_id).first()
        if not team:
            raise ValueError("Team not found")

        # Check if inviter is team admin
        inviter_membership = self.db.query(models.TeamMember).filter(
            models.TeamMember.team_id == team_id,
            models.TeamMember.user_id == inviter_id,
            models.TeamMember.role == "admin"
        ).first()

        if not inviter_membership:
            raise ValueError("Only team admins can invite members")

        # Check if user already has pending invitation
        existing_invitation = self.db.query(models.TeamInvitation).filter(
            models.TeamInvitation.team_id == team_id,
            models.TeamInvitation.email == email,
            models.TeamInvitation.status == "pending"
        ).first()

        if existing_invitation:
            raise ValueError("User already has a pending invitation")

        # Create invitation token
        token = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(days=7)

        db_invitation = models.TeamInvitation(
            team_id=team_id,
            email=email,
            token=token,
            invited_by_id=inviter_id,
            expires_at=expires_at
        )

        self.db.add(db_invitation)
        self.db.commit()
        self.db.refresh(db_invitation)
        logger.info(f"Invitation sent to {email} for team '{team.name}'")
        return db_invitation

    def accept_invitation(self, token: str, user_id: int) -> models.TeamMember:
        """Accept a team invitation"""
        invitation = self.db.query(models.TeamInvitation).filter(
            models.TeamInvitation.token == token,
            models.TeamInvitation.status == "pending",
            models.TeamInvitation.expires_at > datetime.utcnow()
        ).first()

        if not invitation:
            raise ValueError("Invalid or expired invitation")

        # Check if user email matches invitation
        user = self.db.query(models.User).filter(models.User.id == user_id).first()
        if user.email.lower() != invitation.email.lower():
            raise ValueError("Invitation email does not match user email")

        # Check if user is already a member
        existing_member = self.db.query(models.TeamMember).filter(
            models.TeamMember.team_id == invitation.team_id,
            models.TeamMember.user_id == user_id
        ).first()

        if existing_member:
            raise ValueError("User is already a member of this team")

        # Add user to team
        db_member = models.TeamMember(
            team_id=invitation.team_id,
            user_id=user_id,
            role="member"
        )

        # Update invitation status
        invitation.status = "accepted"

        self.db.add(db_member)
        self.db.commit()
        self.db.refresh(db_member)

        logger.info(f"User {user_id} joined team {invitation.team_id}")
        return db_member

    def get_team_jobs(self, team_id: int, days: int = 30) -> dict:
        """Get all jobs from team members grouped by user"""
        logger.info(f"🔍 SERVICE: Starting get_team_jobs for team {team_id}, days={days}")

        # Get team member user IDs
        member_ids = self.db.query(models.TeamMember.user_id).filter(
            models.TeamMember.team_id == team_id
        ).all()
        member_ids = [mid[0] for mid in member_ids]

        logger.info(f"👥 SERVICE: Found {len(member_ids)} team members: {member_ids}")

        if not member_ids:
            logger.warning(f"❌ SERVICE: No members found for team {team_id}")
            return {"users": {}, "all_jobs": []}

        # Get team creation time for filtering IBM jobs
        team = self.db.query(models.Team).filter(models.Team.id == team_id).first()
        team_creation_time = team.created_at if team else datetime.utcnow() - timedelta(days=days)
        logger.info(f"📅 SERVICE: Team creation time: {team_creation_time}")

        # Get ALL jobs from team members (not just recent ones)
        logger.info(f"📅 SERVICE: Fetching ALL jobs for team members (no time filter)")

        # First, get all jobs from team members
        all_jobs = self.db.query(models.JobLogs).filter(
            models.JobLogs.user_id.in_(member_ids)
        ).order_by(models.JobLogs.JobRasied.desc()).all()

        logger.info(f"💾 SERVICE: Found {len(all_jobs)} database jobs for team {team_id} members")

        # Group jobs by user
        users_jobs = {}
        all_job_objects = []

        # Get user info for all members
        users_info = {}
        for member_id in member_ids:
            user = self.db.query(models.User).filter(models.User.id == member_id).first()
            if user:
                users_info[member_id] = {
                    "user_name": user.name or f"User {user.id}",
                    "user_email": user.email or f"user{user.id}@unknown.com"
                }
                users_jobs[str(member_id)] = {
                    "user_name": users_info[member_id]["user_name"],
                    "user_email": users_info[member_id]["user_email"],
                    "jobs": []
                }
            else:
                users_info[member_id] = {
                    "user_name": f"User {member_id}",
                    "user_email": f"user{member_id}@unknown.com"
                }
                users_jobs[str(member_id)] = {
                    "user_name": users_info[member_id]["user_name"],
                    "user_email": users_info[member_id]["user_email"],
                    "jobs": []
                }

        # Process database jobs
        for i, job in enumerate(all_jobs):
            logger.info(f"🔄 SERVICE: Processing database job {i+1}/{len(all_jobs)}: {job.jobId}")

            user_name = None
            user_email = None
            if job.user_id and job.user_id in users_info:
                user_name = users_info[job.user_id]["user_name"]
                user_email = users_info[job.user_id]["user_email"]
                logger.info(f"✅ SERVICE: Job {job.jobId}: Found user {job.user_id} with name='{user_name}' email='{user_email}'")
            else:
                logger.warning(f"⚠️ SERVICE: Job {job.jobId}: User {job.user_id} not found in users_info")
                user_name = f"User {job.user_id}"
                user_email = f"user{job.user_id}@unknown.com"

            # Fix: Ensure job status is properly capitalized for frontend matching
            status = job.Status
            if status:
                status_lower = status.lower()
                if status_lower == 'completed':
                    status = 'Completed'
                elif status_lower == 'running':
                    status = 'Running'
                elif status_lower == 'failed':
                    status = 'Failed'
                elif status_lower == 'cancelled':
                    status = 'Cancelled'
                else:
                    status = 'Pending'
            else:
                status = 'Pending'

            job_dict = {
                'jobId': job.jobId,
                'JobRasied': job.JobRasied,
                'Device': job.Device,
                'Status': status,
                'Shots': job.Shots,
                'JobCompletion': job.JobCompletion,
                'user_id': job.user_id,
                'user_name': user_name,
                'user_email': user_email,
                'project_id': job.project_id
            }

            # Add to user's job list
            if str(job.user_id) in users_jobs:
                users_jobs[str(job.user_id)]["jobs"].append(job_dict)

            logger.info(f"✅ SERVICE: Job {job.jobId}: Final job_dict user_name='{user_name}' user_email='{user_email}'")
            all_job_objects.append(job_dict)

        # Fetch IBM Quantum jobs for all team members
        logger.info(f"🔄 SERVICE: Fetching IBM jobs for team {team_id}")
        ibm_jobs = self._get_team_ibm_jobs(team_id, member_ids, team_creation_time)
        logger.info(f"💻 SERVICE: Found {len(ibm_jobs)} IBM jobs for team {team_id}")

        # Group IBM jobs by user
        for ibm_job in ibm_jobs:
            user_id = str(ibm_job.get('user_id'))
            if user_id in users_jobs:
                users_jobs[user_id]["jobs"].append(ibm_job)
            all_job_objects.append(ibm_job)

        logger.info(f"📊 SERVICE: Returning jobs grouped by {len(users_jobs)} users with {len(all_job_objects)} total jobs for team {team_id}")
        return {
            "users": users_jobs,
            "all_jobs": all_job_objects
        }

    def _get_team_ibm_jobs(self, team_id: int, member_ids: List[int], team_creation_time: datetime) -> List[dict]:
        """Fetch IBM Quantum jobs for all team members after team creation time"""
        ibm_jobs = []

        try:
            # Import IBM utilities
            from services.ibm_utils import get_ibm_service_for_user, fetch_ibm_jobs_with_timeout

            # Get all team members with their IBM credentials in a single query
            team_members = self.db.query(models.User).filter(
                models.User.id.in_(member_ids),
                models.User.ibm_api_key_encrypted.isnot(None),
                models.User.ibm_instance_key_encrypted.isnot(None)
            ).all()

            if not team_members:
                logger.info(f"No team members with IBM credentials found for team {team_id}")
                return []

            logger.info(f"Fetching IBM jobs for {len(team_members)} team members after {team_creation_time}")

            # Fetch jobs for each member concurrently with optimized batch processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(team_members), 3)) as executor:
                # Submit all jobs at once for better concurrency
                futures = [
                    executor.submit(self._fetch_user_ibm_jobs, user, team_creation_time)
                    for user in team_members
                ]

                # Collect results as they complete
                for future in concurrent.futures.as_completed(futures):
                    try:
                        user_jobs = future.result()
                        if user_jobs:  # Only extend if we have jobs
                            ibm_jobs.extend(user_jobs)
                            logger.info(f"Fetched {len(user_jobs)} IBM jobs for a team member")
                    except Exception as e:
                        logger.warning(f"Error fetching IBM jobs for a team member: {e}")

        except Exception as e:
            logger.error(f"Error fetching team IBM jobs: {e}")

        logger.info(f"Total IBM jobs fetched for team {team_id}: {len(ibm_jobs)}")
        return ibm_jobs

    def _fetch_user_ibm_jobs(self, user: models.User, team_creation_time: datetime) -> List[dict]:
        try:
            logger.info(f"🔍 [COLLAB] Fetching IBM jobs for user {user.email} (ID: {user.id})")
            from services.ibm_utils import get_ibm_service_for_user, fetch_ibm_jobs_with_timeout

            # Get IBM service for user
            user_service = get_ibm_service_for_user(user, skip_validation=True)
            if not user_service:
                logger.warning(f"🔍 [COLLAB] No IBM service available for user {user.email}")
                return []

            # Fetch recent jobs
            logger.info(f"🔍 [COLLAB] Calling fetch_ibm_jobs_with_timeout for user {user.email}")
            recent_jobs = fetch_ibm_jobs_with_timeout(user_service, user.id)
            logger.info(f"🔍 [COLLAB] Fetched {len(recent_jobs)} IBM jobs for user {user.email}")

            # Format all jobs for frontend (no time filtering)
            filtered_jobs = []
            for job in recent_jobs:
                try:
                    job_timestamp = datetime.fromisoformat(job.get('timestamp', '').replace('Z', '+00:00'))

                    # Format job to match frontend expectations
                    formatted_job = {
                        'jobId': job.get('job_id', job.get('id', 'unknown')),
                        'JobRasied': job_timestamp,
                        'Device': job.get('backend', 'unknown'),
                        'Status': job.get('status', 'pending').capitalize(),
                        'Shots': job.get('shots', 1024),
                        'JobCompletion': None,  # Will be set if job is completed
                        'user_id': user.id,
                        'user_name': user.name,
                        'user_email': user.email,
                        'project_id': None
                    }

                    # Set completion time if available
                    if job.get('completed_at'):
                        try:
                            completion_time = datetime.fromisoformat(job['completed_at'].replace('Z', '+00:00'))
                            formatted_job['JobCompletion'] = completion_time
                        except:
                            pass

                    filtered_jobs.append(formatted_job)
                except Exception as e:
                    logger.warning(f"Error processing job timestamp: {e}")
                    continue

            return filtered_jobs

        except Exception as e:
            logger.error(f"Error fetching IBM jobs for user {user.email}: {e}")
            return []

    def create_project(self, project: schemas.ProjectCreate, user_id: int) -> models.Project:
        """Create a new project within a team"""
        # Verify user is member of the team
        membership = self.db.query(models.TeamMember).filter(
            models.TeamMember.team_id == project.team_id,
            models.TeamMember.user_id == user_id
        ).first()

        if not membership:
            raise ValueError("User is not a member of this team")

        db_project = models.Project(
            name=project.name,
            description=project.description,
            team_id=project.team_id,
            created_by_id=user_id
        )

        self.db.add(db_project)
        self.db.commit()
        self.db.refresh(db_project)

        logger.info(f"Project '{project.name}' created in team {project.team_id}")
        return db_project

    def get_user_teams(self, user_id: int) -> List[models.Team]:
        """Get all teams a user belongs to"""
        memberships = self.db.query(models.TeamMember).filter(
            models.TeamMember.user_id == user_id
        ).all()

        teams = []
        for membership in memberships:
            team = self.db.query(models.Team).filter(
                models.Team.id == membership.team_id
            ).first()
            if team:
                teams.append(team)

        return teams

    def get_team_members(self, team_id: int) -> List[models.TeamMember]:
    
        members = self.db.query(models.TeamMember).filter(
            models.TeamMember.team_id == team_id
        ).all()
        # Eager load user details for each member
        for member in members:
            def get_project_job_count(self, project_id: int) -> int:
                return members

    def get_team_projects(self, team_id: int) -> List[models.Project]:
        """Get all projects for a team"""
        return self.db.query(models.Project).filter(
            models.Project.team_id == team_id
        ).all()
        """Get job count for a project"""
        # For now, return 0 since project_id column doesn't exist yet
        # This will be updated once the database is properly migrated
        return 0
